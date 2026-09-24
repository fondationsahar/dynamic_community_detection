"""Internal module class used by the LAGO algorithm.

This module contains the _LagoModule class, which is an internal mutable data
structure used during the LAGO algorithm execution. It stores sets of Leaf
objects and supports hierarchical module structures.

Note:
    This is an internal class. For user-facing results, see TimeModule and
    TimeModules in lago/time_modules.py.

Relationship to TimeModule:
    - _LagoModule: Mutable, stores Leaf objects, used during algorithm
    - TimeModule: Immutable view, stores (node, time) tuples, for user queries

See Also:
    - lago.time_modules.TimeModule: The public API for accessing module results
    - lago.time_modules.TimeModules: Container for all detected modules
"""

from __future__ import annotations

import copy
import itertools
from bisect import bisect_left, insort
from typing import TYPE_CHECKING, Self, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._leaf import Leaf

Leaves: TypeAlias = set


class _JMAggregate:
    """What the Joint-Membership expectation needs of a module, kept current.

    JM scores a module by the sum of its nodes' degrees and by its time span
    (first start to last end of any interaction). Both are exact integers of
    the module's content, so they are maintained as time-nodes come and go
    rather than recomputed over the whole module per candidate move:

    * ``node_counts`` -- time-nodes per node, so a node's degree leaves the sum
      exactly when its last time-node leaves the module;
    * ``start_counts`` / ``end_counts`` -- time-nodes per interaction start and
      end instant, with the distinct instants kept sorted in ``starts`` /
      ``ends`` so the span is read at the ends and the span *without* a few
      time-nodes is found by walking inwards over the exhausted instants.

    ``sums`` caches the degree aggregates derived from ``node_counts``
    (``DeltaLongitudinalModularityComputer._jm_sums``); it is dropped on any
    change and recomputed, never shifted, so it cannot drift.
    """

    __slots__ = ("end_counts", "ends", "node_counts", "size", "start_counts", "starts", "sums")

    def __init__(self, leaves: Iterable[Leaf]) -> None:
        self.node_counts: dict[int, int] = {}
        self.start_counts: dict[int, int] = {}
        self.end_counts: dict[int, int] = {}
        self.size = 0
        for leaf in leaves:
            self._count(leaf, 1)
        self.starts = sorted(self.start_counts)
        self.ends = sorted(self.end_counts)
        self.sums = None

    def _count(self, leaf: Leaf, sign: int) -> None:
        counts = self.node_counts
        node = leaf.node
        count = counts.get(node, 0) + sign
        if count:
            counts[node] = count
        else:
            del counts[node]
        # A time-node with no interaction covers no time (get_module_duration
        # skips it); every time-node of a stream has one, but keep the rule.
        if leaf.topo_neighbors or leaf.topo_neighbors_from:
            start = leaf.time
            count = self.start_counts.get(start, 0) + sign
            if count:
                self.start_counts[start] = count
            else:
                del self.start_counts[start]
            end = start + leaf.edge_duration
            count = self.end_counts.get(end, 0) + sign
            if count:
                self.end_counts[end] = count
            else:
                del self.end_counts[end]
        self.size += sign

    def add(self, leaves: Iterable[Leaf]) -> None:
        """Account for time-nodes that joined the module."""
        for leaf in leaves:
            before_start = len(self.start_counts)
            before_end = len(self.end_counts)
            self._count(leaf, 1)
            if len(self.start_counts) != before_start:
                insort(self.starts, leaf.time)
            if len(self.end_counts) != before_end:
                insort(self.ends, leaf.time + leaf.edge_duration)
        self.sums = None

    def remove(self, leaves: Iterable[Leaf]) -> None:
        """Account for time-nodes that left the module."""
        for leaf in leaves:
            before_start = len(self.start_counts)
            before_end = len(self.end_counts)
            self._count(leaf, -1)
            if len(self.start_counts) != before_start:
                del self.starts[bisect_left(self.starts, leaf.time)]
            if len(self.end_counts) != before_end:
                del self.ends[bisect_left(self.ends, leaf.time + leaf.edge_duration)]
        self.sums = None

    def span(self) -> tuple[int | None, int | None]:
        """First interaction start and last interaction end, or ``(None, None)``."""
        if not self.starts:
            return None, None
        return self.starts[0], self.ends[-1]

    def span_without(self, start_counts: dict[int, int], end_counts: dict[int, int]):
        """The span once the time-nodes counted in the arguments are removed.

        Walks inwards from each end over the instants those time-nodes exhaust;
        in practice that is zero or one step.
        """
        first = None
        for instant in self.starts:
            if self.start_counts[instant] - start_counts.get(instant, 0) > 0:
                first = instant
                break
        last = None
        for instant in reversed(self.ends):
            if self.end_counts[instant] - end_counts.get(instant, 0) > 0:
                last = instant
                break
        return first, last

# Monotonic creation index, used as the hash of a module so that the iteration
# order of a set of modules is a function of the order in which the algorithm
# created them, not of the memory allocator. It MUST be restarted at the
# beginning of each LAGO run: a process-global counter gives the second run in a
# process different indices for the same modules, hence a different set layout,
# hence a different exploration order.
_module_counter = itertools.count()


def reset_module_counter() -> None:
    """Restart the module creation index. Called once per LAGO run."""
    global _module_counter
    _module_counter = itertools.count()


class _LagoModule:
    """Internal module representation for LAGO algorithm.

    This class is used internally by the LAGO optimization algorithm to track
    module membership during the greedy search process. It is mutable and
    contains references to Leaf objects.

    Attributes:
        leaves: Set of Leaf objects belonging to this module.
        submodules: List of child modules in the hierarchy.
        parent: Parent module in the hierarchy (None if root).
        neighbors: Set of neighboring modules in the graph.

    Note:
        This is an internal class. Users should work with TimeModule objects
        returned by lago_modules().
    """

    def __init__(self, leaves: Leaves) -> None:
        """Initialize a _LagoModule with a set of leaves.

        Args:
            leaves: Set of Leaf objects to include in this module.
        """
        self.leaves: Leaves = leaves
        self.submodules: list[_LagoModule] = []
        self.parent: _LagoModule | None = None
        self.neighbors: set[_LagoModule] = set()
        # Creation index: stable identity for hashing and for cache keys.
        # Modules are compared by identity (the default __eq__), and distinct
        # modules always get distinct indices, so this is a consistent hash.
        self.index: int = next(_module_counter)
        # Memoised per-node durations of this module's leaves, and the degree
        # aggregates derived from them. Every candidate move needs both, and a
        # module only changes when a move is accepted -- hundreds of evaluations
        # per accepted move -- so they are recomputed on mutation rather than
        # maintained incrementally, which keeps them free of drift.
        # See invalidate_durations().
        self._durations: dict[int, float] | None = None
        self._durations_size: int = -1
        self._mm_sums: tuple | None = None
        # Joint-Membership aggregates, built on first use and maintained by the
        # moves (see jm_add / jm_remove). None until JM asks for them.
        self._jm: _JMAggregate | None = None

    def __hash__(self) -> int:
        return self.index

    def get_jm(self) -> _JMAggregate:
        """The JM aggregates of this module's leaves, memoised and maintained."""
        aggregate = self._jm
        if aggregate is None or aggregate.size != len(self.leaves):
            aggregate = self._jm = _JMAggregate(self.leaves)
        return aggregate

    def jm_add(self, leaves: Iterable[Leaf]) -> None:
        """Keep the JM aggregates current after ``leaves`` joined this module."""
        if self._jm is not None:
            self._jm.add(leaves)

    def jm_remove(self, leaves: Iterable[Leaf]) -> None:
        """Keep the JM aggregates current after ``leaves`` left this module."""
        if self._jm is not None:
            self._jm.remove(leaves)

    def get_durations(self, compute) -> dict[int, float]:
        """Per-node durations of this module's leaves, memoised.

        Args:
            compute: Callable taking the leaves set and returning the durations.

        Returns:
            The durations mapping. Do not mutate it -- it is shared.
        """
        if self._durations is None or self._durations_size != len(self.leaves):
            self._durations = compute(self.leaves)
            self._durations_size = len(self.leaves)
            self._mm_sums = None
        return self._durations

    def invalidate_durations(self) -> None:
        """Drop the memoised durations. Call after changing ``leaves``."""
        self._durations = None
        self._durations_size = -1
        self._mm_sums = None

    def apply_duration_changes(self, changes: dict[int, list[float]]) -> None:
        """Bring the memoised durations up to date after a move, without recomputing.

        ``changes`` maps a node to its ``[before, after]`` duration as produced
        by the evaluation of the move that was just applied, against this very
        memo. The values are exact integers (see ``utils.duration_delta_on_add``),
        so applying them gives exactly what ``get_nodes_durations`` would
        recompute -- a run always lasts at least one step, so a node reaching 0
        has no time-node left and is dropped, as a recompute would leave it out.

        Recomputing instead costs O(|module|) per accepted move, which is the
        term that grows with module size on wide streams. Call after ``leaves``
        has been updated.

        Args:
            changes: ``{node: [before, after]}`` for the nodes whose duration
                changed. Nodes absent from it are unchanged.
        """
        durations = self._durations
        if durations is None:
            # Nothing memoised to update; the next reader recomputes.
            self.invalidate_durations()
            return
        for node, (_, after) in changes.items():
            if after:
                durations[node] = after
            else:
                durations.pop(node, None)
        self._durations_size = len(self.leaves)
        # Derived aggregates are recomputed from the durations, never shifted
        # incrementally: repeated float updates would drift from a recompute.
        self._mm_sums = None

    def compute_neighbors(self, subset: set = set()) -> None:
        """Compute neighboring modules based on leaf connections.

        Updates self.neighbors with all modules that share edges (topological
        or temporal) with this module's leaves.

        Args:
            subset: Optional set of leaves to restrict neighbor search to.
                If empty, all neighbors are considered.
        """
        leaves_neighbors: set[Leaf] = set()
        for leaf in self.leaves:
            leaves_neighbors |= {tmp_neighbor.target for tmp_neighbor in leaf.topo_neighbors}
            right_time_neighb = leaf.right_time_active_neighbor
            if right_time_neighb is not None:
                leaves_neighbors.add(right_time_neighb)
            left_time_neighb = leaf.left_time_active_neighbor
            if left_time_neighb is not None:
                leaves_neighbors.add(left_time_neighb)

        if subset:
            self.neighbors = {leaf.module for leaf in leaves_neighbors & subset if leaf.module is not None}
        else:
            self.neighbors = {leaf.module for leaf in leaves_neighbors if leaf.module is not None}

    def duplicates(self) -> Self:
        """Create a shallow copy of this module.

        Returns:
            A new _LagoModule with a copy of the leaves set.
            Note: The leaves themselves are not copied.
        """
        duplicated_module = _LagoModule(copy.copy(self.leaves))
        return duplicated_module  # type: ignore
