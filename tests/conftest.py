"""Pytest configuration and shared fixtures for lago tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from lago import LinkStream


# =============================================================================
# Basic Link Fixtures (Instantaneous)
# =============================================================================


@pytest.fixture
def simple_triangle_links() -> list[tuple[int, int, int]]:
    """Basic triangle at single timestep: nodes 0, 1, 2 at t=0."""
    return [
        (0, 1, 0),
        (1, 2, 0),
        (0, 2, 0),
    ]


@pytest.fixture
def temporal_path_links() -> list[tuple[int, int, int]]:
    """Simple path across multiple timesteps."""
    return [
        (0, 1, 0),
        (1, 2, 1),
        (2, 3, 2),
    ]


@pytest.fixture
def weighted_links() -> list[tuple[int, int, int, float]]:
    """Links with explicit weights."""
    return [
        (0, 1, 0, 2.0),
        (1, 2, 0, 0.5),
        (0, 2, 1, 1.5),
    ]


@pytest.fixture
def self_loop_link() -> list[tuple[int, int, int]]:
    """Single self-loop (if supported)."""
    return [(0, 0, 0)]


@pytest.fixture
def duplicate_links() -> list[tuple[int, int, int]]:
    """Same edge appearing multiple times at same timestamp."""
    return [
        (0, 1, 0),
        (0, 1, 0),  # Duplicate
        (0, 1, 0),  # Duplicate
    ]


@pytest.fixture
def empty_links() -> list[tuple[int, int, int]]:
    """Empty link list."""
    return []


@pytest.fixture
def single_link() -> list[tuple[int, int, int]]:
    """Single link."""
    return [(0, 1, 0)]


@pytest.fixture
def large_timestamp_links() -> list[tuple[int, int, int]]:
    """Links with very large timestamps."""
    return [
        (0, 1, 1000000),
        (1, 2, 2000000),
    ]


@pytest.fixture
def zero_weight_links() -> list[tuple[int, int, int, float]]:
    """Links with zero weight."""
    return [
        (0, 1, 0, 0.0),
        (1, 2, 0, 0.0),
    ]


@pytest.fixture
def negative_weight_links() -> list[tuple[int, int, int, float]]:
    """Links with negative weight (edge case)."""
    return [
        (0, 1, 0, -1.0),
    ]


# =============================================================================
# Continuous Link Fixtures
# =============================================================================


@pytest.fixture
def simple_continuous_links() -> list[tuple[int, int, int, int]]:
    """Basic continuous links with duration."""
    return [
        (0, 1, 0, 3),  # Link from t=0 to t=3 (duration=3)
        (1, 2, 1, 2),  # Link from t=1 to t=3 (duration=2)
    ]


@pytest.fixture
def overlapping_continuous_links() -> list[tuple[int, int, int, int]]:
    """Continuous links that overlap in time."""
    return [
        (0, 1, 0, 5),  # t=0 to t=5
        (0, 2, 2, 4),  # t=2 to t=6, overlaps with first
        (1, 2, 3, 3),  # t=3 to t=6, overlaps with both
    ]


@pytest.fixture
def weighted_continuous_links() -> list[tuple[int, int, int, int, float]]:
    """Continuous links with weights."""
    return [
        (0, 1, 0, 3, 2.0),
        (1, 2, 1, 2, 0.5),
    ]


@pytest.fixture
def zero_duration_continuous_links() -> list[tuple[int, int, int, int]]:
    """Continuous links with zero duration (edge case)."""
    return [
        (0, 1, 0, 0),
    ]


# =============================================================================
# Delayed Link Fixtures
# =============================================================================


@pytest.fixture
def simple_delayed_links() -> list[tuple[int, int, int, int]]:
    """Basic delayed links with different source/target times."""
    return [
        (0, 1, 0, 2),  # Source at t=0, target at t=2
        (1, 2, 1, 3),  # Source at t=1, target at t=3
    ]


@pytest.fixture
def weighted_delayed_links() -> list[tuple[int, int, int, int, float]]:
    """Delayed links with weights."""
    return [
        (0, 1, 0, 2, 1.5),
        (1, 2, 1, 3, 2.5),
    ]


@pytest.fixture
def same_time_delayed_links() -> list[tuple[int, int, int, int]]:
    """Delayed links where source and target have same time (edge case)."""
    return [
        (0, 1, 5, 5),  # Same time = effectively instantaneous
    ]


@pytest.fixture
def reverse_time_delayed_links() -> list[tuple[int, int, int, int]]:
    """Delayed links where target time < source time (edge case)."""
    return [
        (0, 1, 5, 2),  # Target time before source time
    ]


# =============================================================================
# Directed Graph Fixtures
# =============================================================================


@pytest.fixture
def directed_triangle_links() -> list[tuple[int, int, int]]:
    """Directed triangle (cycle)."""
    return [
        (0, 1, 0),  # 0 -> 1
        (1, 2, 0),  # 1 -> 2
        (2, 0, 0),  # 2 -> 0
    ]


@pytest.fixture
def directed_star_links() -> list[tuple[int, int, int]]:
    """Directed star from center."""
    return [
        (0, 1, 0),  # 0 -> 1
        (0, 2, 0),  # 0 -> 2
        (0, 3, 0),  # 0 -> 3
    ]


@pytest.fixture
def bidirectional_links() -> list[tuple[int, int, int]]:
    """Bidirectional edges (both directions)."""
    return [
        (0, 1, 0),
        (1, 0, 0),  # Reverse direction
    ]


# =============================================================================
# Complex/Integration Fixtures
# =============================================================================


@pytest.fixture
def two_communities_links() -> list[tuple[int, int, int]]:
    """Two clear communities connected by a bridge."""
    return [
        # Community 1: nodes 0, 1, 2
        (0, 1, 0),
        (1, 2, 0),
        (0, 2, 0),
        # Community 2: nodes 3, 4, 5
        (3, 4, 0),
        (4, 5, 0),
        (3, 5, 0),
        # Bridge
        (2, 3, 1),
    ]


@pytest.fixture
def evolving_community_links() -> list[tuple[int, int, int]]:
    """Communities that evolve over time."""
    return [
        # Time 0: single community 0-1-2
        (0, 1, 0),
        (1, 2, 0),
        (0, 2, 0),
        # Time 1: community splits
        (0, 1, 1),
        (2, 3, 1),
        # Time 2: new community forms
        (3, 4, 2),
        (4, 5, 2),
    ]


# =============================================================================
# Factory Fixtures
# =============================================================================


@pytest.fixture
def linkstream_factory() -> Callable[..., LinkStream]:
    """Factory for creating LinkStream with various configurations."""
    from lago import LinkStream

    def _factory(
        continuous: bool = False,
        directed: bool = False,
        delayed: bool = False,
        partite_mapping: dict[int, int] | None = None,
    ) -> LinkStream:
        return LinkStream(
            continuous=continuous,
            directed=directed,
            delayed=delayed,
            partite_mapping=partite_mapping if partite_mapping else {},
        )

    return _factory


@pytest.fixture
def populated_linkstream(
    linkstream_factory: Callable[..., LinkStream],
    simple_triangle_links: list[tuple[int, int, int]],
) -> LinkStream:
    """Pre-populated linkstream with simple triangle."""
    ls = linkstream_factory()
    ls.add_links(simple_triangle_links)
    return ls


@pytest.fixture
def populated_directed_linkstream(
    linkstream_factory: Callable[..., LinkStream],
    directed_triangle_links: list[tuple[int, int, int]],
) -> LinkStream:
    """Pre-populated directed linkstream."""
    ls = linkstream_factory(directed=True)
    ls.add_links(directed_triangle_links)
    return ls


@pytest.fixture
def populated_continuous_linkstream(
    linkstream_factory: Callable[..., LinkStream],
    simple_continuous_links: list[tuple[int, int, int, int]],
) -> LinkStream:
    """Pre-populated continuous linkstream."""
    ls = linkstream_factory(continuous=True)
    ls.add_continous_links(simple_continuous_links)  # Note: typo in original code
    return ls


@pytest.fixture
def populated_delayed_linkstream(
    linkstream_factory: Callable[..., LinkStream],
    simple_delayed_links: list[tuple[int, int, int, int]],
) -> LinkStream:
    """Pre-populated delayed linkstream."""
    ls = linkstream_factory(delayed=True)
    ls.add_delayed_links(simple_delayed_links)
    return ls
