"""Tests for lago.core.utils, focused on order-independence invariants.

These functions take a *set* of leaves. Python set iteration order depends on
the contents and on insertion history, so any result that depends on which
element ``pop()`` returns first is a bug: the same module would get different
durations on different runs.
"""

from __future__ import annotations

import pytest

from lago import LinkStream
from lago.core.utils import get_nodes_durations, leaf_edge_duration


def _leaves(linkstream: LinkStream, node: int, times: tuple[int, ...]) -> set:
    return {linkstream.leaves_dict[(node, t)] for t in times}


class TestGetNodesDurations:
    """get_nodes_durations must be a pure function of its input set."""

    def test_instantaneous_single_leaf(self) -> None:
        ls = LinkStream()
        ls.add_links([(0, 1, 0)])
        assert dict(get_nodes_durations(_leaves(ls, 0, (0,)))) == {0: 1}

    def test_instantaneous_contiguous_run(self) -> None:
        ls = LinkStream()
        ls.add_links([(0, 1, 0), (0, 1, 1), (0, 1, 2)])
        assert dict(get_nodes_durations(_leaves(ls, 0, (0, 1, 2)))) == {0: 3}

    def test_gap_between_active_times_is_still_one_run(self) -> None:
        """A node stays in the module between two consecutive activities.

        Time neighbours link a node's consecutive *active* times, not consecutive
        integers, so leaves at t=0, 1 and 5 form a single run covering [0, 6).
        """
        ls = LinkStream()
        ls.add_links([(0, 1, 0), (0, 1, 1), (0, 1, 5)])
        assert dict(get_nodes_durations(_leaves(ls, 0, (0, 1, 5)))) == {0: 6}

    def test_broken_run(self) -> None:
        """Leaving a middle leaf out of the set really does split the run."""
        ls = LinkStream()
        ls.add_links([(0, 1, 0), (0, 1, 1), (0, 1, 5)])
        # drop (0, 1): two runs, [0, 1) and [5, 6)
        assert dict(get_nodes_durations(_leaves(ls, 0, (0, 5)))) == {0: 2}

    @pytest.mark.parametrize("order", [(30, 31), (31, 30)])
    def test_continuous_run_is_order_independent(self, order: tuple[int, int]) -> None:
        """The result must not depend on which leaf of a run is visited first.

        Regression test: the segment used to be extended by the duration of an
        edge at the leaf *after* it, which is only reached when the rightward
        walk runs -- i.e. only when pop() happened to return the leftmost leaf.
        """
        ls = LinkStream(continuous=True)
        ls.add_links([(14, 1, 30, 1), (14, 2, 31, 1), (14, 3, 32, 5)])
        ls._split_continuous_linkstream()

        leaves = [ls.leaves_dict[(14, t)] for t in order]
        # Build the same set from both insertion orders
        result_a = dict(get_nodes_durations({leaves[0], leaves[1]}))
        result_b = dict(get_nodes_durations({leaves[1], leaves[0]}))
        assert result_a == result_b

    def test_continuous_run_uses_own_last_leaf_duration(self) -> None:
        """A run [30, 31] spans until 31 + duration(31), not until the next leaf."""
        ls = LinkStream(continuous=True)
        ls.add_links([(14, 1, 30, 1), (14, 2, 31, 1), (14, 3, 32, 5)])
        ls._split_continuous_linkstream()

        subset = _leaves(ls, 14, (30, 31))
        right = ls.leaves_dict[(14, 31)]
        expected = 31 - 30 + leaf_edge_duration(right)
        assert dict(get_nodes_durations(subset)) == {14: expected}

    def test_matches_across_many_orderings(self) -> None:
        """Exhaustive check over a larger continuous stream."""
        ls = LinkStream(continuous=True)
        ls.add_links([(0, 1, t, d) for t, d in ((0, 2), (2, 1), (3, 4), (7, 1))])
        ls._split_continuous_linkstream()

        node0 = [leaf for key, leaf in ls.leaves_dict.items() if key[0] == 0]
        reference = dict(get_nodes_durations(set(node0)))
        for rotation in range(len(node0)):
            rotated = node0[rotation:] + node0[:rotation]
            assert dict(get_nodes_durations(set(rotated))) == reference


class TestLeafEdgeDuration:
    def test_instantaneous_is_one(self) -> None:
        ls = LinkStream()
        ls.add_links([(0, 1, 0)])
        assert leaf_edge_duration(ls.leaves_dict[(0, 0)]) == 1

    def test_directed_target_only_leaf(self) -> None:
        """A leaf that is only ever a target still reports its edge duration."""
        ls = LinkStream(directed=True)
        ls.add_links([(0, 1, 0)])
        target = ls.leaves_dict[(1, 0)]
        assert not target.topo_neighbors
        assert leaf_edge_duration(target) == 1
