"""Optional accelerated kernel for the counting loops of ``longitudinal_modularity``.

Those two loops -- intra-community interactions and community switches -- are
around half of the metric's cost, are O(time-edges), and are the only part of it
a compiled language can meaningfully improve: the rest is either an O(n) closed
form or bookkeeping. They are also the only part that moves across a language
boundary *exactly*, because they accumulate weights in a fixed order and do no
other arithmetic.

Where the boundary is, and why:

* **The topology is flattened once per link stream and cached.** Flattening
  costs O(time-edges) -- the same as the loop it replaces -- so a kernel that
  rebuilt it per call would gain nothing. Scoring many partitions of one stream,
  which is what LAGO and any evaluation loop does, pays it once.
* **Only the labels change between calls**, and they are O(time-nodes).
* **Neighbours are stored in the order the Python loop visits them**, so weights
  are summed in the same sequence and the result is bit-identical rather than
  merely close.

Two behaviours of the Python loops are folded into the flat form rather than
reproduced in the kernel:

* An undirected self-loop is counted twice. Its target is the leaf itself, hence
  always in the leaf's own community, so the doubling is unconditional and is
  applied to the stored weight at build time.
* The directed ``* 2`` applies to every community equally and is left to the
  caller.

The pure-Python :func:`_reference_kernel` below is the contract; backends are
looked up at import and fall back to it. ``benchmarks/compare_backends.py`` is
what holds up the claim that they agree.
"""

from __future__ import annotations

import contextlib
import importlib
import os
from array import array
from bisect import bisect_left, bisect_right
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lago.core.linkstream import LinkStream

__all__ = [
    "Topology",
    "available_backends",
    "backend_name",
    "build_topology",
    "core_name",
    "count_intra_and_switches",
    "is_accelerated",
    "should_accelerate",
]

# Backends, fastest first. Each exposes ``count_intra_and_switches`` with the
# signature of _reference_kernel. The Cython kernel is built with the package
# (setup.py) from lago/_accel_kernel.pyx; it is simply absent from a pure
# install. A Rust/PyO3 backend would be one more line here; see
# docs/ACCELERATED_BACKENDS.md for why it is not built yet.
_BACKENDS = (("cython", "lago._accel_kernel"),)


# =============================================================================
# Flattened topology
# =============================================================================


class Topology:
    """The parts of a link stream the counting loops touch, as flat arrays.

    Rows are time-nodes in ``leaves_dict`` order, and each row's neighbours are
    in ``leaf.topo_neighbors`` iteration order, so a kernel walking the arrays
    in order visits edges in exactly the order the Python loop does.

    Everything here depends on the stream alone, so it is built once per stream
    (see :func:`build_topology`). What depends on the partition -- the label
    array -- is built per call by :meth:`labels_from_segments`, in O(time-nodes)
    integer writes.

    Attributes:
        n: Number of time-nodes.
        directed: Whether the stream is directed.
        indptr: Row starts into ``target`` / ``weight``, length ``n + 1``.
        target: Neighbour row of each incident edge.
        weight: Weight of each incident edge, doubled for undirected self-loops.
        left: Previous active time-node of the same node, or -1.
        right: Next active time-node of the same node, or -1.
        times_by_node: Node -> its active times, sorted.
        rows_by_node: Node -> the row of each of those times, in the same order.
    """

    __slots__ = (
        "_nb_edges",
        "_source",
        "_unlabelled",
        "directed",
        "indptr",
        "left",
        "n",
        "right",
        "rows_by_node",
        "target",
        "times_by_node",
        "weight",
    )

    def __init__(self, linkstream: LinkStream) -> None:
        leaves_dict = linkstream.leaves_dict
        leaves = list(leaves_dict.values())
        n = self.n = len(leaves)
        self.directed = linkstream.directed
        # Holding the dict itself is what makes the cache check below sound: a
        # live reference cannot have its address recycled under a replacement.
        self._source = leaves_dict
        self._nb_edges = linkstream.nb_edges

        # Rows go on the leaves themselves: Leaf.__hash__ is a Python method, so
        # a Leaf-keyed dict would cost a Python call per edge below.
        for row, leaf in enumerate(leaves):
            leaf._accel_row = row

        undirected = not self.directed
        indptr = array("q", bytes(8 * (n + 1)))
        target = array("i")
        weight = array("d")
        target_append = target.append
        weight_append = weight.append

        position = 0
        for row, leaf in enumerate(leaves):
            indptr[row] = position
            for edge in leaf.topo_neighbors:
                other = edge.target
                target_append(other._accel_row)
                # An undirected self-loop is counted twice, to stay consistent
                # with every other interaction being seen from both ends. The
                # target is the leaf itself, so it always shares the leaf's
                # community and the doubling is unconditional.
                if undirected and other is leaf:
                    weight_append(2 * edge.weight)
                else:
                    weight_append(edge.weight)
                position += 1
        indptr[n] = position

        # A trailing sentinel keeps the edge arrays non-empty for streams with
        # no links, so a backend can take a typed buffer without special-casing.
        # indptr bounds every read, so it is never visited.
        target_append(0)
        weight_append(0.0)

        self.indptr = indptr
        self.target = target
        self.weight = weight

        left = array("i", bytes(4 * n)) if n else array("i", [0])
        right = array("i", bytes(4 * n)) if n else array("i", [0])
        for row, leaf in enumerate(leaves):
            neighbour = leaf.left_time_active_neighbor
            left[row] = neighbour._accel_row if neighbour is not None else -1
            neighbour = leaf.right_time_active_neighbor
            right[row] = neighbour._accel_row if neighbour is not None else -1
        self.left = left
        self.right = right

        # Per-node sorted time index, so a segment [start, end] becomes a slice
        # of rows by binary search. This is what _build_leaf_labels rebuilds on
        # every call; here it is paid once per stream.
        times_by_node: dict[int, list[int]] = {}
        rows_by_node: dict[int, list[int]] = {}
        for row, (node, time) in enumerate(leaves_dict):
            times_by_node.setdefault(node, []).append(time)
            rows_by_node.setdefault(node, []).append(row)
        for node, times in times_by_node.items():
            if times != sorted(times):
                order = sorted(range(len(times)), key=times.__getitem__)
                times_by_node[node] = [times[i] for i in order]
                rows_by_node[node] = [rows_by_node[node][i] for i in order]
        self.times_by_node = times_by_node
        self.rows_by_node = rows_by_node

        # Template copied per call; copying an array is a memcpy.
        self._unlabelled = array("i", [-1]) * max(n, 1)

    def labels_from_segments(
        self,
        communities_segments: dict[Any, dict[int, tuple[tuple[int, int], ...]]],
    ) -> tuple[array, list[Any]]:
        """Dense label array for a partition given as per-node inclusive runs.

        Mirrors ``lago.metrics.modularity._build_leaf_labels``: only rows that
        exist in the stream are written, and where modules overlap (possible on
        continuous streams, where a split edge runs one instant past its gap)
        the later module wins, as it did in the expanded (node, time) form.

        Communities are numbered in ``communities_segments`` order. The number
        only indexes the result; the summation order within a community is the
        row order, which is fixed by the stream.

        Args:
            communities_segments: Mapping from label to per-node inclusive runs.

        Returns:
            ``(label, communities)`` -- the per-row community index with -1 for
            unlabelled rows, and the community label for each index.
        """
        label = array("i", self._unlabelled)
        communities: list[Any] = []
        times_by_node = self.times_by_node
        rows_by_node = self.rows_by_node

        for number, (community, segments) in enumerate(communities_segments.items()):
            communities.append(community)
            for node, runs in segments.items():
                times = times_by_node.get(node)
                if times is None:
                    continue
                rows = rows_by_node[node]
                for start, end in runs:
                    for index in range(bisect_left(times, start), bisect_right(times, end)):
                        label[rows[index]] = number
        return label, communities


def _cached_topology(linkstream: LinkStream) -> Topology | None:
    """The stream's cached topology if it still describes the stream.

    Invalid once the stream's ``leaves_dict`` is replaced (what
    ``_split_continuous_linkstream`` does) or its edge count changes (what
    ``add_links`` does), which together cover every mutation the class performs
    after construction.
    """
    cached: Topology | None = getattr(linkstream, "_accel_topology", None)
    if (
        cached is not None
        and cached._source is linkstream.leaves_dict
        and cached._nb_edges == linkstream.nb_edges
    ):
        return cached
    return None


def should_accelerate(linkstream: LinkStream) -> bool:
    """Whether this scoring of ``linkstream`` should take the compiled path.

    Building the flat topology costs about as much as scoring once with the
    Python loops, so a stream scored a single time gains nothing from the
    kernel and would pay the build for no return. The rule is therefore: the
    compiled path from the **second** scoring of a stream on (or whenever a
    valid topology is already cached). The second scoring breaks even and every
    later one runs several times faster; a single scoring is never slower than
    pure Python. Both paths return the same value, bit for bit.
    """
    if not is_accelerated():
        return False
    if _cached_topology(linkstream) is not None:
        return True
    count = getattr(linkstream, "_accel_scorings", 0) + 1
    with contextlib.suppress(AttributeError):
        linkstream._accel_scorings = count
    return count >= 2


def build_topology(linkstream: LinkStream) -> Topology:
    """Flat topology for a stream, cached on the stream itself (see :func:`_cached_topology`)."""
    cached = _cached_topology(linkstream)
    if cached is not None:
        return cached
    topology = Topology(linkstream)
    with contextlib.suppress(AttributeError):  # a LinkStream with __slots__, one day
        linkstream._accel_topology = topology
    return topology


# =============================================================================
# Reference kernel
# =============================================================================


def _reference_kernel(
    n: int,
    label: array,
    indptr: array,
    target: array,
    weight: array,
    left: array,
    right: array,
    n_communities: int,
) -> tuple[list[float], int]:
    """Pure-Python kernel. The contract every backend must reproduce exactly.

    Args:
        n: Number of time-nodes.
        label: Community index per time-node, -1 for unlabelled.
        indptr: Row starts into ``target`` / ``weight``, length ``n + 1``.
        target: Neighbour row of each incident edge.
        weight: Weight of each incident edge.
        left: Previous active time-node, or -1.
        right: Next active time-node, or -1.
        n_communities: Number of distinct communities.

    Returns:
        ``(intra, switches)`` -- the per-community interaction sum, and twice the
        number of community switches.
    """
    intra = [0.0] * n_communities
    switches = 0

    for row in range(n):
        community = label[row]
        if community < 0:
            continue

        total = intra[community]
        for position in range(indptr[row], indptr[row + 1]):
            if label[target[position]] == community:
                total += weight[position]
        intra[community] = total

        neighbour = left[row]
        if neighbour >= 0:
            other = label[neighbour]
            if other >= 0 and other != community:
                switches += 1
        neighbour = right[row]
        if neighbour >= 0:
            other = label[neighbour]
            if other >= 0 and other != community:
                switches += 1

    return intra, switches


# =============================================================================
# Backend selection
# =============================================================================


def _load_backend() -> tuple[str, Any]:
    """Pick the fastest importable backend.

    ``LAGO_ACCEL`` forces one by name, ``python`` included. An explicit choice
    that cannot be imported raises: silently running something other than what
    was asked for would make a benchmark meaningless.
    """
    requested = os.environ.get("LAGO_ACCEL", "").strip().lower()
    if requested == "python":
        return "python", _reference_kernel
    if requested:
        known = dict(_BACKENDS)
        if requested not in known:
            names = ", ".join(["python", *known])
            msg = f"LAGO_ACCEL={requested!r}; expected one of: {names}"
            raise ValueError(msg)
        return requested, importlib.import_module(known[requested]).count_intra_and_switches

    for name, module_name in _BACKENDS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        return name, module.count_intra_and_switches
    return "python", _reference_kernel


backend_name, _kernel = _load_backend()


def is_accelerated() -> bool:
    """Whether a compiled kernel is in use."""
    return backend_name != "python"


def core_name() -> str:
    """``"compiled"`` when the core data model runs as a Cython extension, else ``"python"``.

    The compiled modules are built with the package (``setup.py``; in a checkout
    ``python setup.py build_ext --inplace``) and picked up by the import system
    on their own; ``LAGO_CORE=python`` forces the ``.py`` sources.
    """
    from lago.algorithm._internal import _leaf

    return "compiled" if _leaf.__file__.endswith((".so", ".pyd")) else "python"


def available_backends() -> list[str]:
    """Which backends can be imported in this interpreter."""
    found = ["python"]
    for name, module_name in _BACKENDS:
        try:
            importlib.import_module(module_name)
        except ImportError:
            continue
        found.append(name)
    return found


def count_intra_and_switches(
    topology: Topology,
    label: array,
    n_communities: int,
) -> tuple[list[float], int]:
    """Run the active backend's kernel. See :func:`_reference_kernel`."""
    if topology.n == 0:
        return [0.0] * n_communities, 0
    return _kernel(
        topology.n,
        label,
        topology.indptr,
        topology.target,
        topology.weight,
        topology.left,
        topology.right,
        n_communities,
    )
