"""Contract tests for ``lago.accel``.

The compiled backend is optional, so most of these run without it: they hold the
pure-Python reference kernel and the flat topology to the Python loops in
``lago.metrics.modularity``, which are the definition of the metric. The one
test that needs a compiled module skips when none is importable.

Everything is compared with ``==`` on floats, never with a tolerance. The whole
point of the design is that the accelerated path produces the same double.
"""

from __future__ import annotations

import importlib
import random
from array import array

import pytest

from lago import LexType, LinkStream, accel, longitudinal_modularity
from lago.accel import _reference_kernel, build_topology
from lago.core.time_modules import TimeModules
from lago.metrics.modularity import (
    _build_leaf_labels,
    _count_community_switches,
    _count_intra_community_interactions,
)

# =============================================================================
# Random cases over every stream option
# =============================================================================

MODES = ("instant", "continuous", "delayed")


def _random_case(rng: random.Random, mode: str, directed: bool, weighted: bool, partite: bool):
    """A small stream in the given mode, plus a partial random partition of it.

    Partial: roughly one time-node in seven is left out of every community, and
    one community label in three owns nothing at all, so both the unlabelled
    and the empty-community branches are exercised.
    """
    n_nodes = rng.randint(3, 12)
    n_times = rng.randint(1, 6)
    mapping = None
    if partite:
        mapping = {n: rng.randint(0, 1) for n in range(n_nodes) if rng.random() < 0.8}

    kwargs: dict = {"directed": directed, "partite_mapping": mapping}
    if mode == "continuous":
        kwargs["continuous"] = True
    elif mode == "delayed":
        kwargs["delayed"] = True
    stream = LinkStream(**kwargs)

    links = []
    seen = set()
    for _ in range(rng.randint(2, 30)):
        a, b = rng.randrange(n_nodes), rng.randrange(n_nodes)
        # Self-loops only in instantaneous mode, where they are a supported input
        # and the one place the folded doubling can be wrong.
        if a == b and mode != "instant":
            continue
        if not directed and a > b:
            a, b = b, a
        t = rng.randrange(n_times)
        if mode == "instant":
            key = (a, b, t)
        elif mode == "continuous":
            key = (a, b, t, rng.randint(1, 3))
        else:
            key = (a, b, t, t + rng.randint(0, 1))
        if key in seen:
            continue
        seen.add(key)
        links.append((*key, round(rng.uniform(0.2, 3.0), 3)) if weighted else key)
    if not links:
        return None
    stream.add_links(links)

    n_comms = rng.randint(1, 4)
    communities: dict[int, set] = {label: set() for label in range(n_comms)}
    for node, t in stream.leaves_dict:
        if rng.random() < 1 / 7:
            continue
        communities[rng.randrange(n_comms)].add((node, t))
    if rng.random() < 1 / 3:
        communities[n_comms] = set()  # a label owning no time-node
    return stream, communities


def _build_cases():
    rng = random.Random(20260923)
    cases = []
    for mode in MODES:
        for directed in (False, True):
            for weighted in (False, True):
                # k-partite mappings are only generated for instantaneous
                # streams here, matching what the benchmark corpus covers.
                for partite in ((False, True) if mode == "instant" else (False,)):
                    for repeat in range(3):
                        case = None
                        while case is None:
                            case = _random_case(rng, mode, directed, weighted, partite)
                        name = f"{mode}-{'dir' if directed else 'und'}-{'w' if weighted else 'u'}"
                        name += f"-{'kp' if partite else 'np'}-{repeat}"
                        cases.append(pytest.param(case, id=name))
    return cases


CASES = _build_cases()


def _segments(communities):
    return {label: TimeModules._members_to_segments(m) for label, m in communities.items()}


def _python_counts(stream, segments):
    """What the metric's Python loops produce, unchanged."""
    leaf_labels = _build_leaf_labels(stream, segments)
    return (
        leaf_labels,
        _count_intra_community_interactions(stream, leaf_labels),
        _count_community_switches(leaf_labels),
    )


# =============================================================================
# Reference kernel and flat topology against the Python loops
# =============================================================================


class TestReferenceKernel:
    """The pure-Python kernel is the contract; it must match the loops exactly."""

    @pytest.mark.parametrize("case", CASES)
    def test_intra_and_switches_match_loops(self, case) -> None:
        stream, communities = case
        segments = _segments(communities)
        _, expected_intra, expected_switches = _python_counts(stream, segments)

        topology = build_topology(stream)
        label, order = topology.labels_from_segments(segments)
        intra, switches = accel._reference_kernel(
            topology.n,
            label,
            topology.indptr,
            topology.target,
            topology.weight,
            topology.left,
            topology.right,
            len(order),
        )
        scale = 2 if stream.directed else 1
        got = {community: intra[i] * scale for i, community in enumerate(order)}

        # Labels missing from the Python dict contributed nothing; the kernel
        # reports them as 0.0. Both read as 0 downstream.
        for community in set(order) | set(expected_intra):
            assert got.get(community, 0.0) == expected_intra.get(community, 0)
        assert switches / 2 == expected_switches

    @pytest.mark.parametrize("case", CASES)
    def test_labels_from_segments_match_build_leaf_labels(self, case) -> None:
        stream, communities = case
        segments = _segments(communities)
        leaf_labels = _build_leaf_labels(stream, segments)

        topology = build_topology(stream)
        label, order = topology.labels_from_segments(segments)

        for leaf, community in leaf_labels.items():
            assert order[label[leaf._accel_row]] == community
        labelled_rows = sum(1 for value in label[: topology.n] if value >= 0)
        assert labelled_rows == len(leaf_labels)

    def test_rows_follow_leaves_dict_order(self) -> None:
        stream = LinkStream()
        stream.add_links([(0, 1, 0), (1, 2, 1), (0, 2, 1)])
        topology = build_topology(stream)
        for row, leaf in enumerate(stream.leaves_dict.values()):
            assert leaf._accel_row == row
        assert topology.n == len(stream.leaves_dict)
        # One trailing sentinel beyond the last edge
        assert len(topology.target) == topology.indptr[topology.n] + 1


# =============================================================================
# The wired metric
# =============================================================================


def _force_python(monkeypatch) -> None:
    monkeypatch.setattr(accel, "backend_name", "python")
    monkeypatch.setattr(accel, "_kernel", _reference_kernel)


def _force_accelerated(monkeypatch) -> None:
    """Drive the accelerated path of the metric with the reference kernel."""
    monkeypatch.setattr(accel, "backend_name", "reference-forced")
    monkeypatch.setattr(accel, "_kernel", _reference_kernel)


PARAMS = ((1.0, 2.0, 5), (0.0, 2.0, 5), (0.5, 0.0, 5), (2.0, 1.0, 12))


class TestWiredMetric:
    """``longitudinal_modularity`` gives the same result on either path."""

    @pytest.mark.parametrize("case", CASES)
    def test_same_value_and_penalty(self, case, monkeypatch) -> None:
        stream, communities = case

        _force_python(monkeypatch)
        assert not accel.is_accelerated()
        expected = [
            longitudinal_modularity(stream, communities, lex=lex, gamma=g, omega=o, ndigits=nd)
            for lex in LexType
            for g, o, nd in PARAMS
        ]

        _force_accelerated(monkeypatch)
        assert accel.is_accelerated()
        got = [
            longitudinal_modularity(stream, communities, lex=lex, gamma=g, omega=o, ndigits=nd)
            for lex in LexType
            for g, o, nd in PARAMS
        ]

        for before, after in zip(expected, got, strict=True):
            assert after.value == before.value
            assert after.time_penalty == before.time_penalty

    def test_time_modules_input(self, monkeypatch) -> None:
        """A TimeModules hands its segments over directly."""
        stream = LinkStream()
        stream.add_links([(0, 1, 0), (1, 2, 0), (0, 1, 1), (2, 3, 1), (3, 4, 2)])
        modules = TimeModules({0: {(0, 0), (1, 0), (0, 1), (1, 1)}, 1: {(2, 0), (2, 1), (3, 1)}})

        _force_python(monkeypatch)
        expected = longitudinal_modularity(stream, modules, lex="MM")
        _force_accelerated(monkeypatch)
        got = longitudinal_modularity(stream, modules, lex="MM")
        assert got.value == expected.value
        assert got.time_penalty == expected.time_penalty

    def test_nothing_labelled(self, monkeypatch) -> None:
        stream = LinkStream()
        stream.add_links([(0, 1, 0), (1, 2, 1)])
        communities = {0: set(), 1: set()}

        _force_python(monkeypatch)
        expected = longitudinal_modularity(stream, communities, lex="CM")
        _force_accelerated(monkeypatch)
        got = longitudinal_modularity(stream, communities, lex="CM")
        assert got.value == expected.value


# =============================================================================
# Cache
# =============================================================================


class TestTopologyCache:
    def test_reused_while_stream_unchanged(self) -> None:
        stream = LinkStream()
        stream.add_links([(0, 1, 0), (1, 2, 1)])
        first = build_topology(stream)
        assert build_topology(stream) is first

    def test_invalidated_by_add_links(self, monkeypatch) -> None:
        stream = LinkStream()
        stream.add_links([(0, 1, 0), (1, 2, 1)])
        communities = {0: {(0, 0), (1, 0), (1, 1)}, 1: {(2, 1)}}

        _force_accelerated(monkeypatch)
        # The compiled path starts at the second scoring of a stream.
        longitudinal_modularity(stream, communities, lex="MM")
        longitudinal_modularity(stream, communities, lex="MM")
        stale = stream._accel_topology

        # Adds an edge between two existing time-nodes: leaves_dict is
        # unchanged, so only the edge count can tell the cache it is stale.
        stream.add_links([(0, 2, 1)])
        got = longitudinal_modularity(stream, communities, lex="MM")
        assert stream._accel_topology is not stale

        _force_python(monkeypatch)
        expected = longitudinal_modularity(stream, communities, lex="MM")
        assert got.value == expected.value

    def test_empty_stream(self) -> None:
        stream = LinkStream()
        topology = build_topology(stream)
        assert topology.n == 0
        assert accel.count_intra_and_switches(topology, array("i", [0]), 2) == ([0.0, 0.0], 0)

    def test_compiled_path_starts_at_the_second_scoring(self, monkeypatch) -> None:
        """One scoring gains nothing from a topology that costs a scoring to build."""
        stream = LinkStream()
        stream.add_links([(0, 1, 0), (1, 2, 1), (0, 2, 1)])
        communities = {0: {(0, 0), (1, 0), (0, 1)}, 1: {(1, 1), (2, 1)}}

        _force_accelerated(monkeypatch)
        first = longitudinal_modularity(stream, communities, lex="MM")
        assert getattr(stream, "_accel_topology", None) is None
        second = longitudinal_modularity(stream, communities, lex="MM")
        assert stream._accel_topology is not None
        assert second.value == first.value

        _force_python(monkeypatch)
        assert longitudinal_modularity(stream, communities, lex="MM").value == first.value


# =============================================================================
# Compiled backend against the reference kernel
# =============================================================================

_COMPILED = [name for name in accel.available_backends() if name != "python"]


@pytest.mark.skipif(not _COMPILED, reason="no compiled lago.accel backend importable")
@pytest.mark.parametrize("backend", _COMPILED)
def test_compiled_kernel_matches_reference(backend: str) -> None:
    """Random flat inputs, float weights, bit-exact agreement.

    Independent of any LinkStream: this checks the kernel as a function of its
    arrays, including inputs no stream produces (arbitrary left/right rows).
    """
    kernel = importlib.import_module(dict(accel._BACKENDS)[backend]).count_intra_and_switches
    rng = random.Random(0)

    for _ in range(300):
        n = rng.randint(0, 40)
        n_communities = rng.randint(0, 5)
        size = max(n, 1)

        label = array("i", [-1]) * size
        if n_communities:
            for row in range(n):
                label[row] = rng.randrange(-1, n_communities)

        indptr = array("q", [0]) * (n + 1)
        target = array("i")
        weight = array("d")
        for row in range(n):
            indptr[row] = len(target)
            for _ in range(rng.randint(0, 6)):
                target.append(rng.randrange(n))
                # Mix of integers and floats with inexact binary expansions
                weight.append(rng.choice((1.0, 2.0, 0.1, 0.7, 1.3, rng.uniform(0.01, 9.0))))
        indptr[n] = len(target)
        target.append(0)
        weight.append(0.0)

        left = array("i", [rng.randrange(-1, n) for _ in range(size)]) if n else array("i", [0])
        right = array("i", [rng.randrange(-1, n) for _ in range(size)]) if n else array("i", [0])

        args = (n, label, indptr, target, weight, left, right, n_communities)
        assert kernel(*args) == _reference_kernel(*args)
