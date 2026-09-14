"""`seed` selects a greedy trajectory; `nb_iter` tries several and keeps the best.

The exploration order -- which candidate move is considered first -- is the only
thing that varies between runs on the same input, and the greedy search is
sensitive to it. These pin the three properties that follow:

* the canonical order (no seed) is stable, so published results stay reproducible;
* a given seed is reproducible, so an experiment can be repeated;
* different seeds reach different optima, so `nb_iter` has something to gain.

The third is not decoration. Before the exploration order became seedable,
`nb_iter=5` ran the identical search five times at five times the cost.
"""

from __future__ import annotations

import random

import pytest

from lago import LinkStream, lago_modules, longitudinal_modularity
from lago.algorithm._internal.exploration import ExplorationQueue


def _planted(n_nodes: int = 24, n_times: int = 6, n_comms: int = 3, seed: int = 0) -> LinkStream:
    """Small planted-partition stream, structured enough for order to matter."""
    rng = random.Random(seed)
    comm = {node: node % n_comms for node in range(n_nodes)}
    links = [
        (i, j, t)
        for t in range(n_times)
        for i in range(n_nodes)
        for j in range(i + 1, n_nodes)
        if rng.random() < (0.45 if comm[i] == comm[j] else 0.06)
    ]
    stream = LinkStream()
    stream.add_links(links)
    return stream


def _fingerprint(modules) -> str:
    """Canonical, label-invariant identity of a partition."""
    return repr(sorted(tuple(sorted(members)) for members in modules._raw_modules.values()))


# =============================================================================
# The queue itself
# =============================================================================


class TestExplorationQueue:
    def test_unseeded_matches_a_plain_set(self) -> None:
        """Without a generator the queue must behave exactly like set.pop().

        This is what keeps default runs reproducing previously published results.
        """
        items = [f"item-{i}" for i in range(50)]
        queue = ExplorationQueue(items)
        plain = set(items)
        while plain:
            assert queue.pop() == plain.pop()
        assert not queue

    def test_seeded_is_a_permutation_of_the_same_items(self) -> None:
        items = list(range(50))
        popped = []
        queue = ExplorationQueue(items, rng=random.Random(7))
        while queue:
            popped.append(queue.pop())
        assert sorted(popped) == items

    def test_seeded_order_is_reproducible(self) -> None:
        def drain(seed):
            queue = ExplorationQueue(range(50), rng=random.Random(seed))
            return [queue.pop() for _ in range(50)]

        assert drain(3) == drain(3)
        assert drain(3) != drain(4)

    def test_items_added_back_are_served_once(self) -> None:
        """The loops re-add neighbours mid-iteration; nothing may be served twice."""
        for rng in (None, random.Random(1)):
            queue = ExplorationQueue([1, 2, 3], rng=rng)
            first = queue.pop()
            queue.add(first)  # put it back
            queue.add(first)  # and again -- still a set
            seen = []
            while queue:
                seen.append(queue.pop())
            assert sorted(seen) == [1, 2, 3]

    def test_update_and_membership(self) -> None:
        queue = ExplorationQueue([1, 2], rng=random.Random(0))
        queue |= [2, 3, 4]
        assert len(queue) == 4
        assert 3 in queue and 9 not in queue
        assert sorted(iter(queue)) == [1, 2, 3, 4]


# =============================================================================
# seed
# =============================================================================


class TestSeedSelectsATrajectory:
    def test_canonical_order_is_stable(self) -> None:
        """No seed: the same input always gives the same modules."""
        results = {_fingerprint(lago_modules(_planted(), lex="MM")) for _ in range(3)}
        assert len(results) == 1

    @pytest.mark.parametrize("seed", [1, 2, 42])
    def test_a_seed_is_reproducible(self, seed: int) -> None:
        results = {
            _fingerprint(lago_modules(_planted(), lex="MM", seed=seed)) for _ in range(3)
        }
        assert len(results) == 1

    def test_different_seeds_reach_different_optima(self) -> None:
        """Otherwise nb_iter has nothing to explore.

        Asserted over a set of seeds rather than a specific pair: which seeds
        diverge is a property of the stream, not something worth pinning.
        """
        stream_seeds = (0, 1, 2)
        found = set()
        for stream_seed in stream_seeds:
            found.update(
                _fingerprint(lago_modules(_planted(seed=stream_seed), lex="MM", seed=s))
                for s in (None, 1, 2, 3, 4)
            )
            if len(found) > len(stream_seeds):
                return
        assert len(found) > len(stream_seeds), (
            "no seed changed the outcome on any stream -- the exploration order "
            "is not reaching the search"
        )


# =============================================================================
# nb_iter
# =============================================================================


class TestNbIterExplores:
    def test_more_iterations_never_score_worse(self) -> None:
        """nb_iter keeps the best run, so its objective is monotone in nb_iter."""
        stream = _planted()
        scores = {}
        for nb_iter in (1, 3, 6):
            modules = lago_modules(stream, lex="MM", nb_iter=nb_iter)
            scores[nb_iter] = longitudinal_modularity(stream, modules, ndigits=10).value
        assert scores[3] >= scores[1]
        assert scores[6] >= scores[3]

    def test_nb_iter_is_reproducible(self) -> None:
        stream = _planted()
        results = {_fingerprint(lago_modules(stream, lex="MM", nb_iter=4)) for _ in range(2)}
        assert len(results) == 1

    def test_first_iteration_is_the_canonical_run(self) -> None:
        """nb_iter=1 unseeded must be exactly the default single run.

        This is the no-regression guarantee: whatever nb_iter gains, it may not
        change what a plain call returns.
        """
        stream = _planted()
        assert _fingerprint(lago_modules(stream, lex="MM", nb_iter=1)) == _fingerprint(
            lago_modules(stream, lex="MM")
        )

    def test_nb_iter_actually_explores(self) -> None:
        """At least one stream must show nb_iter finding something better.

        If this ever fails everywhere, nb_iter has silently become a no-op again
        -- which is exactly the regression this file exists to catch.
        """
        for stream_seed in (0, 1, 2, 3):
            stream = _planted(n_nodes=30, n_times=8, seed=stream_seed)
            one = longitudinal_modularity(
                stream, lago_modules(stream, lex="MM", nb_iter=1), ndigits=10
            ).value
            many = longitudinal_modularity(
                stream, lago_modules(stream, lex="MM", nb_iter=6), ndigits=10
            ).value
            if many > one:
                return
        pytest.fail("nb_iter improved nothing on any stream -- restarts are not exploring")
