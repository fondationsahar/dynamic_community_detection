"""Longitudinal modularity computation for temporal community detection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

from lago.core.enums import LexType
from lago.core.time_modules import TimeModules
from lago.core.utils import (
    get_module_duration,
    get_nodes_durations,
    get_nodes_times,
)

if TYPE_CHECKING:
    from lago.algorithm._internal._leaf import Leaf
    from lago.core.linkstream import LinkStream


# =============================================================================
# Result Dataclass
# =============================================================================


@dataclass(frozen=True)
class ModularityResult:
    """Result of longitudinal modularity computation.

    Attributes:
        value: The total modularity value (including time penalty).
        time_penalty: The time penalty term.
        lex_type: The expectation type used for computation.
        ndigits: Number of decimal places used for rounding.
    """

    value: float
    time_penalty: float
    lex_type: LexType
    ndigits: int = 5

    @property
    def modularity_without_penalty(self) -> float:
        """Modularity value without time penalty."""
        return round(self.value - self.time_penalty, self.ndigits)


# =============================================================================
# Type Aliases
# =============================================================================

# Community label can be string or int
CommunityLabel = str | int
# Labels maps (node, time) to community label
LabelsDict = dict[tuple[int, int], CommunityLabel]


# =============================================================================
# Degree Cache for Performance
# =============================================================================


class _DegreeCache:
    """Cache for degree contribution computations.

    Avoids redundant dictionary lookups and arithmetic for frequently
    accessed node pairs.
    """

    def __init__(self, linkstream: LinkStream) -> None:
        self._linkstream = linkstream
        self._cache: dict[tuple[int, int], float] = {}

    def get_contribution(self, source: int, target: int) -> float:
        """Get cached degree contribution for a node pair.

        Args:
            source: Source node identifier.
            target: Target node identifier.

        Returns:
            The degree contribution value.
        """
        # Normalize key order for undirected graphs
        if not self._linkstream.directed and source > target:
            source, target = target, source

        key = (source, target)
        if key not in self._cache:
            self._cache[key] = self._compute_contribution(source, target)
        return self._cache[key]

    def _compute_contribution(self, source: int, target: int) -> float:
        """Compute degree contribution for a node pair."""
        multiplier = 2 if source != target else 1
        ls = self._linkstream

        if ls.directed:
            return multiplier * (
                ls.degrees_in.get(source, 0) * ls.degrees_out.get(target, 0)
                + ls.degrees_out.get(source, 0) * ls.degrees_in.get(target, 0)
            )

        return multiplier * ls.degrees.get(source, 0) * ls.degrees.get(target, 0)


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_expected_value(
    degrees_contribution: float,
    time_factor: float,
    network_duration: int,
    total_weight: float,
) -> float:
    """Compute expected value from degree contribution and time factor.

    Args:
        degrees_contribution: The degree-based contribution.
        time_factor: The time-based factor (duration, geo_mean, or coexistence).
        network_duration: Total duration of the network.
        total_weight: Total weight of all edges.

    Returns:
        The expected value for this node pair.
    """
    return degrees_contribution * (time_factor / network_duration) / (2 * total_weight) ** 2


def _validate_lex_type(lex_type: LexType | str) -> LexType:
    """Validate and convert lex_type to LexType enum.

    Args:
        lex_type: LexType enum or string ('CM', 'JM', 'MM').

    Returns:
        LexType enum value.

    Raises:
        ValueError: If lex_type string is invalid.
        TypeError: If lex_type is not a LexType enum or string.
    """
    if isinstance(lex_type, LexType):
        return lex_type

    if isinstance(lex_type, str):
        lex_str = lex_type.upper()
        if lex_str not in ("CM", "JM", "MM"):
            msg = f'Invalid lex_type string "{lex_type}". Must be "CM", "JM", or "MM".'
            raise ValueError(msg)
        return LexType[lex_str]

    msg = (
        f"lex_type must be a LexType enum or string ('CM', 'JM', 'MM'), "
        f"got {type(lex_type).__name__}"
    )
    raise TypeError(msg)


# =============================================================================
# Main Function
# =============================================================================


def longitudinal_modularity(
    linkstream: LinkStream,
    communities: dict[CommunityLabel, set[tuple[int, int]]] | TimeModules,
    lex: LexType | str = LexType.MM,
    alpha: float = 1.0,
    omega: float = 2.0,
    ndigits: int = 5,
) -> ModularityResult:
    """Compute longitudinal modularity for temporal communities.

    Args:
        linkstream: The temporal network.
        communities: Either a TimeModules object, or a mapping from community
            label to set of (node, time) tuples.
        lex: Expectation type. Can be LexType.CM, LexType.JM, LexType.MM, or
            strings "CM"/"JM"/"MM" for backward compatibility.
            - CM (Coexistence): uses intersection of node active times.
            - JM (Joint-Membership): uses community duration.
            - MM (Mean-Membership): uses geometric mean of node durations.
            Defaults to LexType.MM.
        alpha: Weight for expectation term. Default 1.0.
        omega: Weight for time penalty term. Default 2.0.
        ndigits: Number of decimal places for rounding. Default 5.

    Returns:
        ModularityResult containing the modularity value, time penalty, and lex.

    Raises:
        ValueError: If lex string is invalid.
        TypeError: If lex is not a LexType enum or valid string.

    Note:
        This function does NOT modify the linkstream. Community labels are
        stored in a separate internal dictionary.

    Example:
        ```python
        from lago import LinkStream, longitudinal_modularity, LexType
        from lago.core.time_modules import TimeModules

        ls = LinkStream()
        ls.add_links([(0, 1, 0), (1, 2, 0)])

        # Using LexType enum (recommended):
        result = longitudinal_modularity(ls, communities, lex=LexType.MM)

        # Using string (backward compatible):
        result = longitudinal_modularity(ls, communities, lex="MM")

        # Using TimeModules:
        tm = TimeModules({0: {(0, 0), (1, 0)}, 1: {(2, 0)}})
        result = longitudinal_modularity(ls, tm, lex=LexType.MM)
        ```
    """
    # Validate and normalize lex
    lex = _validate_lex_type(lex)

    # Handle TimeModules input - use its pre-computed structures
    if isinstance(communities, TimeModules):
        labels = communities.to_flat_labels()
        communities_dict = communities.to_communities_dict()
        communities_nodes: dict[CommunityLabel, set[int]] = {
            label: set(communities.get_module_nodes(label)) for label in communities.modules
        }
        communities_leaves: dict[CommunityLabel, set[Leaf]] = {}
        for label, members in communities_dict.items():
            communities_leaves[label] = {
                linkstream.leaves_dict[key] for key in members if key in linkstream.leaves_dict
            }
    else:
        # Build labels dict and communities data structures from raw dict
        labels: LabelsDict = {}
        communities_leaves: dict[CommunityLabel, set[Leaf]] = {}
        communities_nodes: dict[CommunityLabel, set[int]] = {}
        communities_dict = communities

        for label, members in communities.items():
            communities_leaves[label] = set()
            communities_nodes[label] = set()
            for leaf_key in members:
                if leaf_key not in linkstream.leaves_dict:
                    continue
                leaf = linkstream.leaves_dict[leaf_key]
                communities_leaves[label].add(leaf)
                communities_nodes[label].add(leaf.node)
                labels[leaf_key] = label

    # 1 - Count intra-community interactions
    communities_nb_interactions = _count_intra_community_interactions(linkstream, labels)

    # 2 - Compute expectations (LAZY: skip if alpha == 0)
    if alpha == 0:
        communities_expectations: dict[CommunityLabel, float] = dict.fromkeys(communities, 0.0)
    else:
        # Create degree cache for performance
        degree_cache = _DegreeCache(linkstream)

        expectation_functions = {
            LexType.CM: _compute_coexistence_expectations,
            LexType.JM: _compute_joint_expectations,
            LexType.MM: _compute_mean_expectations,
        }
        communities_expectations = expectation_functions[lex](
            linkstream, communities_leaves, communities_nodes, degree_cache
        )

    # 3 - Time penalty (LAZY: skip if omega == 0)
    if omega == 0:
        time_penalty = 0.0
    else:
        switch_count = _count_community_switches(linkstream, labels)
        time_penalty = -omega / (2 * linkstream.weight) * switch_count

    # 4 - Aggregation
    lm_modularity = 0.0
    for community, expectation in communities_expectations.items():
        nb_links = communities_nb_interactions.get(community, 0)
        lm_modularity += nb_links / (2 * linkstream.weight) - alpha * expectation

    lm_modularity += time_penalty

    return ModularityResult(
        value=round(lm_modularity, ndigits=ndigits),
        time_penalty=round(time_penalty, ndigits=ndigits),
        lex_type=lex,
        ndigits=ndigits,
    )


# =============================================================================
# Internal Functions
# =============================================================================


def _count_intra_community_interactions(
    linkstream: LinkStream,
    labels: LabelsDict,
) -> dict[CommunityLabel, float]:
    """Count interactions within each community.

    Args:
        linkstream: The temporal network.
        labels: Mapping from (node, time) to community label.

    Returns:
        Dictionary mapping community labels to interaction counts.
    """
    communities_nb_interactions: dict[CommunityLabel, float] = {}

    for (node, time), leaf in linkstream.leaves_dict.items():
        community = labels.get((node, time))
        if community is None:
            continue

        if community not in communities_nb_interactions:
            communities_nb_interactions[community] = 0

        neighbors = leaf.topo_neighbors | leaf.topo_neighbors_from
        for neighbor in neighbors:
            neighbor_key = (neighbor.target.node, neighbor.target.time)
            if labels.get(neighbor_key) != community:
                continue
            # Avoid double counting self-loops
            weight_multiplier = 2 if neighbor.target == leaf else 1
            communities_nb_interactions[community] += weight_multiplier * neighbor.weight

    return communities_nb_interactions


def _compute_joint_expectations(
    linkstream: LinkStream,
    communities_leaves: dict[CommunityLabel, set[Leaf]],
    communities_nodes: dict[CommunityLabel, set[int]],
    degree_cache: _DegreeCache,
) -> dict[CommunityLabel, float]:
    """Compute Joint Modularity Expectation (JM) for communities.

    JM uses the community duration as the time factor.

    Args:
        linkstream: The temporal network.
        communities_leaves: Mapping from community label to set of Leaf objects.
        communities_nodes: Pre-computed mapping from community label to node IDs.
        degree_cache: Cache for degree contribution lookups.

    Returns:
        Dictionary mapping community labels to expectation values.
    """
    communities_expectations: dict[CommunityLabel, float] = {}

    for community, leaves in communities_leaves.items():
        expectation = 0.0
        community_nodes = communities_nodes[community]
        community_duration = get_module_duration(leaves)

        for source, target in combinations_with_replacement(community_nodes, 2):
            degrees_part = degree_cache.get_contribution(source, target)
            expected_value = _compute_expected_value(
                degrees_part,
                community_duration,
                linkstream.network_duration,
                linkstream.weight,
            )
            expectation += expected_value

        communities_expectations[community] = expectation

    return communities_expectations


def _compute_mean_expectations(
    linkstream: LinkStream,
    communities_leaves: dict[CommunityLabel, set[Leaf]],
    communities_nodes: dict[CommunityLabel, set[int]],
    degree_cache: _DegreeCache,
) -> dict[CommunityLabel, float]:
    """Compute Mean Modularity Expectation (MM) for communities.

    MM uses the geometric mean of node durations as the time factor.

    Args:
        linkstream: The temporal network.
        communities_leaves: Mapping from community label to set of Leaf objects.
        communities_nodes: Pre-computed mapping from community label to node IDs.
        degree_cache: Cache for degree contribution lookups.

    Returns:
        Dictionary mapping community labels to expectation values.
    """
    communities_expectations: dict[CommunityLabel, float] = {}

    for community, leaves in communities_leaves.items():
        expectation = 0.0
        nodes_durations = get_nodes_durations(module_leaves=leaves)
        community_nodes = communities_nodes[community]

        for source, target in combinations_with_replacement(community_nodes, 2):
            geo_mean = (nodes_durations.get(source, 0) * nodes_durations.get(target, 0)) ** 0.5

            degrees_part = degree_cache.get_contribution(source, target)
            expected_value = _compute_expected_value(
                degrees_part,
                geo_mean,
                linkstream.network_duration,
                linkstream.weight,
            )
            expectation += expected_value

        communities_expectations[community] = expectation

    return communities_expectations


def _compute_coexistence_expectations(
    linkstream: LinkStream,
    communities_leaves: dict[CommunityLabel, set[Leaf]],
    communities_nodes: dict[CommunityLabel, set[int]],
    degree_cache: _DegreeCache,
) -> dict[CommunityLabel, float]:
    """Compute Coexistence Modularity Expectation (CM) for communities.

    CM uses the coexistence time (intersection of active times) as the time factor.

    Args:
        linkstream: The temporal network.
        communities_leaves: Mapping from community label to set of Leaf objects.
        communities_nodes: Pre-computed mapping from community label to node IDs.
        degree_cache: Cache for degree contribution lookups.

    Returns:
        Dictionary mapping community labels to expectation values.
    """
    communities_expectations: dict[CommunityLabel, float] = {}

    for community, leaves in communities_leaves.items():
        expectation = 0.0
        community_nodes_sorted = sorted(communities_nodes[community])
        nodes_times = get_nodes_times(leaves)

        for idx, source in enumerate(community_nodes_sorted):
            source_times = nodes_times[source]

            for target in community_nodes_sorted[idx:]:
                target_times = nodes_times[target]
                coexistence = len(source_times & target_times)

                if not coexistence:
                    continue

                degrees_part = degree_cache.get_contribution(source, target)
                expected_value = _compute_expected_value(
                    degrees_part,
                    coexistence,
                    linkstream.network_duration,
                    linkstream.weight,
                )
                expectation += expected_value

        communities_expectations[community] = expectation

    return communities_expectations


def _count_community_switches(
    linkstream: LinkStream,
    labels: LabelsDict,
) -> float:
    """Count community switches across time.

    A community switch occurs when a node changes community between
    consecutive time steps.

    Args:
        linkstream: The temporal network.
        labels: Mapping from (node, time) to community label.

    Returns:
        Number of community switches (divided by 2 to avoid double counting).
    """
    switch_count = 0

    for (node, time), leaf in linkstream.leaves_dict.items():
        community = labels.get((node, time))
        if community is None:
            continue

        # Check left neighbor
        left_neighbor = leaf.left_time_active_neighbor
        if left_neighbor:
            left_key = (left_neighbor.node, left_neighbor.time)
            left_community = labels.get(left_key)
            if left_community is not None and left_community != community:
                switch_count += 1

        # Check right neighbor
        right_neighbor = leaf.right_time_active_neighbor
        if right_neighbor:
            right_key = (right_neighbor.node, right_neighbor.time)
            right_community = labels.get(right_key)
            if right_community is not None and right_community != community:
                switch_count += 1

    # Switches are counted twice (once from each side)
    return switch_count / 2
