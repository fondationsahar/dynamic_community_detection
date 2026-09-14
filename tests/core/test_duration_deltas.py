"""The incremental duration update must agree with full recomputation.

``duration_delta_on_add`` / ``duration_delta_on_remove`` are what make a
candidate move cost O(|M0|) instead of O(|Mx|); if they drift from
``get_nodes_durations`` the whole objective drifts with them.
"""

from __future__ import annotations

import itertools
import random

import pytest

from lago import LinkStream
from lago.core.utils import (
    duration_delta_on_add,
    duration_delta_on_remove,
    get_nodes_durations,
)


def _streams():
    """A few streams whose runs have non-trivial structure."""
    instantaneous = LinkStream()
    instantaneous.add_links(
        [(0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 5), (1, 2, 5), (0, 1, 6), (2, 3, 3)]
    )
    yield "instantaneous", instantaneous

    directed = LinkStream(directed=True)
    directed.add_links([(0, 1, 0), (1, 0, 1), (0, 2, 2), (2, 1, 4), (1, 2, 5)])
    yield "directed", directed

    continuous = LinkStream(continuous=True)
    continuous.add_links([(0, 1, 0, 2), (0, 1, 2, 1), (0, 2, 3, 4), (1, 2, 5, 3), (0, 1, 9, 1)])
    continuous._split_continuous_linkstream()
    yield "continuous", continuous

    delayed = LinkStream(delayed=True)
    delayed.add_links([(0, 1, 0, 2), (1, 2, 2, 3), (0, 2, 1, 4), (2, 0, 5, 6)])
    yield "delayed", delayed


def _random_stream(seed: int) -> LinkStream:
    rng = random.Random(seed)
    ls = LinkStream(continuous=True)
    links = []
    for _ in range(rng.randint(3, 10)):
        a, b = rng.randrange(4), rng.randrange(4)
        if a == b:
            continue
        links.append((min(a, b), max(a, b), rng.randrange(0, 8), rng.randrange(1, 4)))
    if not links:
        links = [(0, 1, 0, 1)]
    ls.add_links(links)
    ls._split_continuous_linkstream()
    return ls


@pytest.mark.parametrize("name,stream", list(_streams()), ids=lambda v: v if isinstance(v, str) else "")
def test_add_matches_recomputation_on_every_subset(name: str, stream: LinkStream) -> None:
    """For every subset and every leaf outside it, the delta must be exact."""
    leaves = list(stream.leaves_dict.values())
    assert len(leaves) <= 20, "exhaustive test: keep the streams small"

    for size in range(len(leaves) + 1):
        for subset in itertools.combinations(leaves, size):
            base = set(subset)
            before = get_nodes_durations(base)
            for leaf in leaves:
                if leaf in base:
                    continue
                expected = get_nodes_durations(base | {leaf})
                delta = duration_delta_on_add(leaf, base.__contains__)
                assert delta == expected[leaf.node] - before.get(leaf.node, 0), (
                    f"{name}: adding ({leaf.node}, {leaf.time}) to "
                    f"{sorted((x.node, x.time) for x in base)}"
                )


@pytest.mark.parametrize("name,stream", list(_streams()), ids=lambda v: v if isinstance(v, str) else "")
def test_remove_matches_recomputation_on_every_subset(name: str, stream: LinkStream) -> None:
    """Removal is the exact inverse of addition."""
    leaves = list(stream.leaves_dict.values())
    for size in range(1, len(leaves) + 1):
        for subset in itertools.combinations(leaves, size):
            base = set(subset)
            before = get_nodes_durations(base)
            for leaf in subset:
                remaining = base - {leaf}
                expected = get_nodes_durations(remaining)
                delta = duration_delta_on_remove(leaf, remaining.__contains__)
                assert delta == expected.get(leaf.node, 0) - before[leaf.node], (
                    f"{name}: removing ({leaf.node}, {leaf.time}) from "
                    f"{sorted((x.node, x.time) for x in base)}"
                )


@pytest.mark.parametrize("seed", range(25))
def test_batch_add_matches_recomputation(seed: int) -> None:
    """Adding several leaves one at a time telescopes to the right total."""
    rng = random.Random(seed + 1000)
    stream = _random_stream(seed)
    leaves = list(stream.leaves_dict.values())
    rng.shuffle(leaves)
    split = rng.randint(0, len(leaves))
    base, extra = set(leaves[:split]), leaves[split:]

    durations = dict(get_nodes_durations(base))
    grown = set(base)
    for leaf in extra:
        durations[leaf.node] = durations.get(leaf.node, 0) + duration_delta_on_add(
            leaf, grown.__contains__
        )
        grown.add(leaf)

    expected = dict(get_nodes_durations(grown))
    assert {k: v for k, v in durations.items() if v} == {k: v for k, v in expected.items() if v}


@pytest.mark.parametrize("seed", range(25))
def test_batch_remove_matches_recomputation(seed: int) -> None:
    """Removing several leaves one at a time telescopes to the right total."""
    rng = random.Random(seed + 2000)
    stream = _random_stream(seed)
    leaves = list(stream.leaves_dict.values())
    rng.shuffle(leaves)
    split = rng.randint(0, len(leaves))
    victims = leaves[:split]

    remaining = set(leaves)
    durations = dict(get_nodes_durations(remaining))
    for leaf in victims:
        remaining.discard(leaf)
        durations[leaf.node] = durations.get(leaf.node, 0) + duration_delta_on_remove(
            leaf, remaining.__contains__
        )

    expected = dict(get_nodes_durations(remaining))
    assert {k: v for k, v in durations.items() if v} == {k: v for k, v in expected.items() if v}
