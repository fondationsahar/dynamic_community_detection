"""Per-module durations maintained across moves instead of recomputed.

A move's evaluation already produces the exact per-node duration changes of the
two modules involved; ``_LagoModule.apply_duration_changes`` applies them. What
must hold, exactly:

* after every accepted move the memo equals a fresh ``get_nodes_durations`` of
  the module's leaves -- same keys, same values;
* the MM aggregates derived from the memo do not depend on the order the nodes
  entered it, so maintained and recomputed durations give the same floats;
* consequently the whole search gives the same partition with maintenance as
  with recomputation.
"""

from __future__ import annotations

import random

import pytest

import lago.core.utils as tls
from lago import LinkStream, lago_modules
from lago.algorithm._internal._lago_module import _LagoModule
from lago.algorithm._internal.delta_lm import DeltaLongitudinalModularityComputer as Computer

# =============================================================================
# Streams
# =============================================================================


def _undirected():
    rng = random.Random(11)
    links = [
        (i, j, t)
        for t in range(6)
        for i in range(18)
        for j in range(i + 1, 18)
        if rng.random() < (0.7 if i % 3 == j % 3 else 0.05)
    ]
    ls = LinkStream()
    ls.add_links(links)
    return ls


def _directed_weighted():
    rng = random.Random(12)
    links = [
        (i, j, t, round(rng.uniform(0.5, 2.5), 2))
        for t in range(5)
        for i in range(14)
        for j in range(14)
        if i != j and rng.random() < (0.6 if i % 2 == j % 2 else 0.06)
    ]
    ls = LinkStream(directed=True)
    ls.add_links(links)
    return ls


def _continuous():
    rng = random.Random(13)
    links = []
    for _ in range(240):
        i, j = rng.randrange(14), rng.randrange(14)
        if i == j or rng.random() > (0.8 if i % 2 == j % 2 else 0.1):
            continue
        links.append((min(i, j), max(i, j), rng.randrange(0, 12), rng.randrange(1, 4)))
    ls = LinkStream(continuous=True)
    ls.add_links(links)
    return ls


def _bipartite():
    rng = random.Random(14)
    links = [
        (i, 10 + j, t)
        for t in range(5)
        for i in range(10)
        for j in range(8)
        if rng.random() < (0.6 if (i % 2) == (j % 2) else 0.05)
    ]
    ls = LinkStream(partite_mapping={n: (0 if n < 10 else 1) for n in range(18)})
    ls.add_links(links)
    return ls


STREAMS = {
    "undirected": _undirected,
    "directed_weighted": _directed_weighted,
    "continuous": _continuous,
    "bipartite": _bipartite,
}


def _fingerprint(modules) -> tuple:
    return tuple(sorted(tuple(sorted(members)) for members in modules._raw_modules.values()))


# =============================================================================
# Tests
# =============================================================================


class TestMemoMatchesRecomputation:
    @pytest.mark.parametrize("refinement", ["STEM", None])
    @pytest.mark.parametrize("name", sorted(STREAMS))
    def test_after_every_move(self, name: str, refinement: str | None, monkeypatch) -> None:
        original = _LagoModule.apply_duration_changes
        applied = {"n": 0, "with_memo": 0}

        def checked(self, changes):
            applied["n"] += 1
            had_memo = self._durations is not None
            original(self, changes)
            if had_memo:
                applied["with_memo"] += 1
                fresh = tls.get_nodes_durations(self.leaves)
                assert dict(self._durations) == dict(fresh)
                assert self._durations_size == len(self.leaves)
                assert self._mm_sums is None

        monkeypatch.setattr(_LagoModule, "apply_duration_changes", checked)
        lago_modules(STREAMS[name](), lex="MM", refinement=refinement, seed=1)
        assert applied["n"] > 0
        assert applied["with_memo"] > 0

    def test_node_reaching_zero_is_dropped(self) -> None:
        module = _LagoModule(set())
        module._durations = {1: 3, 2: 5}
        module._durations_size = 0
        module.apply_duration_changes({1: [3, 0], 2: [5, 7], 3: [0, 2]})
        assert dict(module._durations) == {2: 7, 3: 2}

    def test_without_memo_it_just_invalidates(self) -> None:
        module = _LagoModule(set())
        module.apply_duration_changes({1: [0, 2]})
        assert module._durations is None
        assert module._durations_size == -1


class TestJMAggregatesMatchRebuild:
    """The maintained JM aggregates equal a rebuild from the leaves after every move."""

    @pytest.mark.parametrize("refinement", ["STEM", None])
    @pytest.mark.parametrize("name", sorted(STREAMS))
    def test_after_every_move(self, name: str, refinement: str | None, monkeypatch) -> None:
        from lago.algorithm._internal._lago_module import _JMAggregate

        checked = {"n": 0}

        def check(module):
            aggregate = module._jm
            if aggregate is None:
                return
            fresh = _JMAggregate(module.leaves)
            assert aggregate.node_counts == fresh.node_counts
            assert aggregate.start_counts == fresh.start_counts
            assert aggregate.end_counts == fresh.end_counts
            assert aggregate.starts == fresh.starts
            assert aggregate.ends == fresh.ends
            assert aggregate.size == len(module.leaves)
            assert aggregate.sums is None
            checked["n"] += 1

        original_add, original_remove = _LagoModule.jm_add, _LagoModule.jm_remove

        def add(self, leaves):
            original_add(self, leaves)
            check(self)

        def remove(self, leaves):
            original_remove(self, leaves)
            check(self)

        monkeypatch.setattr(_LagoModule, "jm_add", add)
        monkeypatch.setattr(_LagoModule, "jm_remove", remove)
        lago_modules(STREAMS[name](), lex="JM", refinement=refinement, seed=1)
        assert checked["n"] > 0

    def test_span_without(self) -> None:
        from lago.algorithm._internal._lago_module import _JMAggregate

        ls = _continuous()
        leaves = list(ls.leaves_dict.values())
        aggregate = _JMAggregate(leaves)
        first_leaf = min(leaves, key=lambda leaf: leaf.time)
        # Removing every time-node at the first instant moves the span start.
        at_first = [leaf for leaf in leaves if leaf.time == first_leaf.time]
        starts = {first_leaf.time: len(at_first)}
        ends = {}
        for leaf in at_first:
            ends[leaf.time + leaf.edge_duration] = ends.get(leaf.time + leaf.edge_duration, 0) + 1
        rest = _JMAggregate([leaf for leaf in leaves if leaf.time != first_leaf.time])
        assert aggregate.span_without(starts, ends) == rest.span()
        # Removing nothing changes nothing.
        assert aggregate.span_without({}, {}) == aggregate.span()


class TestAggregatesAreOrderIndependent:
    @pytest.mark.parametrize("name", ["undirected", "directed_weighted", "bipartite"])
    def test_mm_sums_do_not_depend_on_insertion_order(self, name: str) -> None:
        stream = STREAMS[name]()
        computer = Computer(stream, lex="MM", gamma=1, omega=2)
        nodes = sorted(stream.degrees or stream.degrees_out)
        rng = random.Random(0)
        durations = {node: rng.randint(1, 9) for node in nodes}

        reference = computer._mm_sums(durations)
        for _ in range(5):
            items = list(durations.items())
            rng.shuffle(items)
            assert computer._mm_sums(dict(items)) == reference


class TestSamePartitionAsRecomputation:
    @pytest.mark.parametrize("lex", ["MM", "JM"])
    @pytest.mark.parametrize("refinement", ["STEM", None])
    @pytest.mark.parametrize("name", sorted(STREAMS))
    def test_identical(self, name: str, refinement: str | None, lex: str, monkeypatch) -> None:
        maintained = _fingerprint(lago_modules(STREAMS[name](), lex=lex, refinement=refinement, seed=1))

        monkeypatch.setattr(
            _LagoModule,
            "apply_duration_changes",
            lambda self, changes: _LagoModule.invalidate_durations(self),
        )
        recomputed = _fingerprint(lago_modules(STREAMS[name](), lex=lex, refinement=refinement, seed=1))
        assert maintained == recomputed
