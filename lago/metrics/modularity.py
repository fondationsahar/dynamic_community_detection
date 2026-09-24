"""Longitudinal modularity computation for temporal community detection."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import floor, isfinite, ulp
from typing import TYPE_CHECKING

from lago import accel
from lago.core.enums import LexType
from lago.core.time_modules import TimeModules
from lago.core.utils import (
    get_module_duration_from_members,
    get_module_duration_from_segments,
    get_nodes_durations_from_members,
    get_nodes_durations_from_segments,
    get_nodes_times_from_members,
    iter_coexistence_spans,
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
        lex: The expectation type used for computation.
        ndigits: Number of decimal places used for rounding.
    """

    value: float
    time_penalty: float
    lex: LexType
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

        # In k-partite networks, only interactions between different partites
        # are expected, so a within-partite pair contributes nothing.
        if ls.partite_mapping.get(source, -1) == ls.partite_mapping.get(target, -2):
            return 0

        if ls.directed:
            if source == target:
                return ls.degrees_out.get(source, 0) * ls.degrees_in.get(target, 0)
            return ls.degrees_out.get(source, 0) * ls.degrees_in.get(
                target, 0
            ) + ls.degrees_out.get(target, 0) * ls.degrees_in.get(source, 0)

        return multiplier * ls.degrees.get(source, 0) * ls.degrees.get(target, 0)


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_expected_value(
    degrees_contribution: float,
    time_factor: float,
    network_duration: int,
    total_weight: float,
    directed: bool,
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
    denom_direct_factor = 2 ** (not directed)

    return (
        degrees_contribution
        / (denom_direct_factor * total_weight) ** 2
        * (time_factor / network_duration)
    )


# Sentinel defaults used by the k-partite mask. A partite id equal to one of
# these makes the mask asymmetric, which the closed forms cannot reproduce.
_PARTITE_SENTINELS = (-1, -2)


def _closed_form_is_exact(linkstream: LinkStream) -> bool:
    """Whether the closed-form expectations reproduce the pair loop exactly.

    The k-partite mask is ``partite_mapping.get(a, -1) == partite_mapping.get(b, -2)``,
    which is symmetric in ``a`` and ``b`` -- and therefore a real grouping the
    closed form can factor out -- unless some node carries one of the sentinel
    values itself.
    """
    mapping = linkstream.partite_mapping
    if not mapping:
        return True
    return not any(value in _PARTITE_SENTINELS for value in mapping.values())


def _partite_groups(
    nodes: set[int],
    partite_mapping: dict[int, int],
) -> list[list[int]]:
    """Group the mapped nodes by partite; unmapped nodes belong to no group.

    Pairs inside a group are masked out (they contribute nothing), including the
    diagonal. Unmapped nodes are masked against nothing at all -- not even
    themselves -- which is why they are simply left out here.
    """
    if not partite_mapping:
        return []
    groups: dict[int, list[int]] = {}
    for node in nodes:
        partite = partite_mapping.get(node)
        if partite is not None:
            groups.setdefault(partite, []).append(node)
    return list(groups.values())


def _masked_square(
    nodes: set[int],
    weights: dict[int, float],
    groups: list[list[int]],
    second_weights: dict[int, float] | None = None,
) -> float:
    """Sum of ``w[i] * w2[j]`` over every ordered pair not masked by the grouping.

    The pair loop computes an upper-triangular sum whose ``2 ** (i != j)``
    multiplier makes it exactly the full ordered sum, and that factors:

        sum over all ordered (i, j)  -  sum over ordered pairs inside a group

    Args:
        nodes: The community's nodes.
        weights: Per-node factor for the first element of the pair.
        groups: Partite groups, from :func:`_partite_groups`.
        second_weights: Per-node factor for the second element, for directed
            networks where the two differ. Defaults to ``weights``.

    Returns:
        The masked sum.
    """
    other = weights if second_weights is None else second_weights
    total = sum(weights[node] for node in nodes) * sum(other[node] for node in nodes)
    for group in groups:
        total -= sum(weights[node] for node in group) * sum(other[node] for node in group)
    return total


def _near_rounding_tie(value: float, ndigits: int, tol: float = 1e-4) -> bool:
    """Whether rounding ``value`` to ``ndigits`` is a coin flip.

    The closed forms divide once at the end where the pair loop divides every
    term, so the two differ in the last bits. That is invisible except when the
    exact value sits precisely on a half-way boundary -- which happens for
    integer-weight networks, whose exact modularity is a rational with a
    ``2**a * 5**b`` denominator. Those calls fall back to the pair loop.

    The probe ``value * 10 ** ndigits`` has a resolution of its own that degrades
    as ``ndigits`` grows, so the band widens with it. Once the band would cover
    the whole interval the probe carries no information and every call falls
    back.
    """
    if not isfinite(value):
        return False
    scaled = value * 10**ndigits
    if not isfinite(scaled):
        return False
    band = max(tol, 64 * ulp(scaled))
    if band >= 0.5:
        # Cannot tell a tie from anything else; use the exact loop.
        return True
    return abs(scaled - floor(scaled) - 0.5) < band


def _aggregate_modularity(
    linkstream: LinkStream,
    communities_expectations: dict[CommunityLabel, float],
    communities_nb_interactions: dict[CommunityLabel, float],
    gamma: float,
    time_penalty: float,
) -> float:
    """Combine the three terms into the (unrounded) longitudinal modularity."""
    lm_modularity = 0.0
    for community, expectation in communities_expectations.items():
        nb_links = communities_nb_interactions.get(community, 0)
        lm_modularity += nb_links / (2 * linkstream.weight) - gamma * expectation
    return lm_modularity + time_penalty


def _validate_lex(lex: LexType | str) -> LexType:
    """Validate and convert lex to LexType enum.

    Args:
        lex: LexType enum or string ('CM', 'JM', 'MM').

    Returns:
        LexType enum value.

    Raises:
        ValueError: If lex string is invalid.
        TypeError: If lex is not a LexType enum or string.
    """
    if isinstance(lex, LexType):
        return lex

    if isinstance(lex, str):
        lex_str = lex.upper()
        if lex_str not in ("CM", "JM", "MM"):
            msg = f'Invalid lex string "{lex}". Must be "CM", "JM", or "MM".'
            raise ValueError(msg)
        return LexType[lex_str]

    msg = f"lex must be a LexType enum or string ('CM', 'JM', 'MM'), got {type(lex).__name__}"
    raise TypeError(msg)


# =============================================================================
# Main Function
# =============================================================================


def longitudinal_modularity(
    linkstream: LinkStream,
    communities: dict[CommunityLabel, set[tuple[int, int]]] | TimeModules,
    lex: LexType | str = LexType.MM,
    gamma: float = 1.0,
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
        gamma: Weight for expectation term. Default 1.0.
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
    lex = _validate_lex(lex)
    # Segments are the compact form of a community: a module covering a long
    # interval has far more (node, time) members than runs. A TimeModules
    # already stores them; a raw dict is compressed once, here.
    if isinstance(communities, TimeModules):
        communities_segments = communities.segments
    else:
        communities_segments = {
            label: TimeModules._members_to_segments(members)
            for label, members in communities.items()
        }

    communities_nodes: dict[CommunityLabel, set[int]] = {
        label: set(segments) for label, segments in communities_segments.items()
    }

    # 1 - Count intra-community interactions. With a compiled kernel the labels
    # go into a flat array over the stream's cached topology and one pass counts
    # both the interactions and the switches of step 3, since both walk the same
    # time-nodes. Without one -- or on the first scoring of a stream, where the
    # topology build would cost as much as it saves -- the labels are indexed by
    # Leaf object (going through the (node, time) key would build a tuple per
    # neighbour per leaf) and the two Python loops below stay in charge, unchanged.
    switch_count: float | None = None
    if accel.should_accelerate(linkstream):
        topology = accel.build_topology(linkstream)
        label, community_order = topology.labels_from_segments(communities_segments)
        intra, switches = accel.count_intra_and_switches(topology, label, len(community_order))
        scale = 2 if linkstream.directed else 1
        communities_nb_interactions = {
            community: intra[index] * scale for index, community in enumerate(community_order)
        }
        switch_count = switches / 2
    else:
        leaf_labels = _build_leaf_labels(linkstream, communities_segments)
        communities_nb_interactions = _count_intra_community_interactions(linkstream, leaf_labels)

    def _expectations_from_loops() -> dict[CommunityLabel, float]:
        """Fallback pair loops, which need the expanded (node, time) members."""
        dense = {
            label: TimeModules._segments_to_members(segments)
            for label, segments in communities_segments.items()
        }
        return _EXPECTATION_LOOPS[lex](
            linkstream, dense, communities_nodes, _DegreeCache(linkstream)
        )

    # 2 - Compute expectations (LAZY: skip if gamma == 0)
    use_closed_form = gamma != 0 and _closed_form_is_exact(linkstream)
    if gamma == 0:
        communities_expectations: dict[CommunityLabel, float] = dict.fromkeys(
            communities_segments, 0.0
        )
    elif use_closed_form:
        communities_expectations = _EXPECTATION_CLOSED[lex](
            linkstream, communities_segments, communities_nodes
        )
    else:
        communities_expectations = _expectations_from_loops()

    # 3 - Time penalty (LAZY: skip if omega == 0)
    if omega == 0:
        time_penalty = 0.0
    else:
        if switch_count is None:
            switch_count = _count_community_switches(leaf_labels)
        time_penalty = -omega / (2 * linkstream.nb_edges) * switch_count

    # 4 - Aggregation
    lm_modularity = _aggregate_modularity(
        linkstream, communities_expectations, communities_nb_interactions, gamma, time_penalty
    )

    # The closed forms divide once where the loops divide per term, so the two
    # can land on opposite sides of an exact rounding tie. Rare (~0.2% of
    # integer-weight runs), and only detectable at the rounding boundary -- fall
    # back to the loops there so the reported value is unchanged.
    if use_closed_form and _near_rounding_tie(lm_modularity, ndigits):
        communities_expectations = _expectations_from_loops()
        lm_modularity = _aggregate_modularity(
            linkstream, communities_expectations, communities_nb_interactions, gamma, time_penalty
        )

    return ModularityResult(
        value=float(round(lm_modularity, ndigits=ndigits)),
        time_penalty=float(round(time_penalty, ndigits=ndigits)),
        lex=lex,
        ndigits=ndigits,
    )


# =============================================================================
# Internal Functions
# =============================================================================


def _build_leaf_labels(
    linkstream: LinkStream,
    communities_segments: dict[CommunityLabel, dict[int, tuple[tuple[int, int], ...]]],
) -> dict[Leaf, CommunityLabel]:
    """Index community labels by Leaf object rather than by (node, time).

    Only time-nodes that exist in the stream can ever be looked up, so the
    segments are searched per leaf rather than expanded: a module covering a long
    interval has far more members than the stream has time-nodes.

    Built in ``leaves_dict`` order so that consumers iterating it visit leaves in
    the same order as before, which keeps their summation order unchanged.

    Args:
        linkstream: The temporal network.
        communities_segments: Mapping from label to per-node inclusive runs.

    Returns:
        Mapping from Leaf to community label, for the leaves that have one.
    """
    # Index the stream's time-nodes by node, sorted, so a run can be turned into
    # a slice of them by binary search.
    times_by_node: dict[int, list[int]] = {}
    leaves_by_node: dict[int, list[Leaf]] = {}
    for (node, time), leaf in linkstream.leaves_dict.items():
        times_by_node.setdefault(node, []).append(time)
        leaves_by_node.setdefault(node, []).append(leaf)
    for node, times in times_by_node.items():
        if times != sorted(times):
            order = sorted(range(len(times)), key=times.__getitem__)
            times_by_node[node] = [times[i] for i in order]
            leaves_by_node[node] = [leaves_by_node[node][i] for i in order]

    # Modules can claim the same time-node: on continuous streams a split edge
    # runs one instant past its gap, so a module's last instant can coincide with
    # the next module's first. Assigning in iteration order, letting later
    # modules win, is what the expanded (node, time) form did.
    assigned: dict[Leaf, CommunityLabel] = {}
    for label, segments in communities_segments.items():
        for node, runs in segments.items():
            times = times_by_node.get(node)
            if times is None:
                continue
            leaves = leaves_by_node[node]
            for start, end in runs:
                for index in range(bisect_left(times, start), bisect_right(times, end)):
                    assigned[leaves[index]] = label

    if not assigned:
        return {}
    # Re-emit in leaves_dict order: consumers accumulate per community as they
    # iterate, so their summation order must not depend on the module order.
    return {leaf: assigned[leaf] for leaf in linkstream.leaves_dict.values() if leaf in assigned}


def _degree_factors(
    linkstream: LinkStream,
    nodes: set[int],
    time_factors: dict[int, float] | None = None,
) -> tuple[dict[int, float], dict[int, float] | None]:
    """Per-node weights for the closed forms: degree times its time factor.

    Returns a single mapping for undirected networks, and an (out, in) pair for
    directed ones, mirroring the two shapes of
    :meth:`_DegreeCache._compute_contribution`.
    """
    if time_factors is None:
        factor = dict.fromkeys(nodes, 1.0)
    else:
        factor = {node: time_factors.get(node, 0) for node in nodes}

    if linkstream.directed:
        degrees_out = linkstream.degrees_out
        degrees_in = linkstream.degrees_in
        return (
            {node: degrees_out.get(node, 0) * factor[node] for node in nodes},
            {node: degrees_in.get(node, 0) * factor[node] for node in nodes},
        )
    degrees = linkstream.degrees
    return {node: degrees.get(node, 0) * factor[node] for node in nodes}, None


def _scale_expectation(linkstream: LinkStream, total: float, time_factor: float) -> float:
    """Apply the normalisation of :func:`_compute_expected_value` once."""
    denom_direct_factor = 2 ** (not linkstream.directed)
    return (
        total
        / (denom_direct_factor * linkstream.weight) ** 2
        * (time_factor / linkstream.network_duration)
    )


def _count_intra_community_interactions(
    linkstream: LinkStream,
    leaf_labels: dict[Leaf, CommunityLabel],
) -> dict[CommunityLabel, float]:
    """Count interactions within each community.

    Args:
        linkstream: The temporal network.
        leaf_labels: Mapping from Leaf to community label.

    Returns:
        Dictionary mapping community labels to interaction counts.
    """
    communities_nb_interactions: dict[CommunityLabel, float] = {}
    directed = linkstream.directed
    get = leaf_labels.get

    for leaf, community in leaf_labels.items():
        if community not in communities_nb_interactions:
            communities_nb_interactions[community] = 0
        total = communities_nb_interactions[community]
        for neighbor in leaf.topo_neighbors:
            target = neighbor.target
            if get(target) != community:
                continue
            # Avoid double count self-loops to remain consistant with other interactions already counted twice
            if target is leaf and not directed:
                total += 2 * neighbor.weight
            else:
                total += neighbor.weight
        communities_nb_interactions[community] = total

    # NOTE reformulate that maybe
    if linkstream.directed:
        communities_nb_interactions = {
            key: val * 2 for key, val in communities_nb_interactions.items()
        }
    return communities_nb_interactions


def _compute_joint_expectations(
    linkstream: LinkStream,
    communities_members: dict[CommunityLabel, set[tuple[int, int]]],
    communities_nodes: dict[CommunityLabel, set[int]],
    degree_cache: _DegreeCache,
) -> dict[CommunityLabel, float]:
    """Compute Joint Modularity Expectation (JM) for communities.

    JM uses the community duration as the time factor.

    Args:
        linkstream: The temporal network.
        communities_members: Mapping from community label to set of (node, time) tuples.
        communities_nodes: Pre-computed mapping from community label to node IDs.
        degree_cache: Cache for degree contribution lookups.

    Returns:
        Dictionary mapping community labels to expectation values.
    """
    communities_expectations: dict[CommunityLabel, float] = {}

    for community, members in communities_members.items():
        expectation = 0.0
        community_nodes = communities_nodes[community]
        community_duration = get_module_duration_from_members(members)

        for source, target in combinations_with_replacement(community_nodes, 2):
            degrees_part = degree_cache.get_contribution(source, target)
            expected_value = _compute_expected_value(
                degrees_part,
                community_duration,
                linkstream.network_duration,
                linkstream.weight,
                linkstream.directed,
            )
            expectation += expected_value

        communities_expectations[community] = expectation

    return communities_expectations


def _compute_mean_expectations(
    linkstream: LinkStream,
    communities_members: dict[CommunityLabel, set[tuple[int, int]]],
    communities_nodes: dict[CommunityLabel, set[int]],
    degree_cache: _DegreeCache,
) -> dict[CommunityLabel, float]:
    """Compute Mean Modularity Expectation (MM) for communities.

    MM uses the geometric mean of node durations as the time factor.

    Args:
        linkstream: The temporal network.
        communities_members: Mapping from community label to set of (node, time) tuples.
        communities_nodes: Pre-computed mapping from community label to node IDs.
        degree_cache: Cache for degree contribution lookups.

    Returns:
        Dictionary mapping community labels to expectation values.
    """
    communities_expectations: dict[CommunityLabel, float] = {}

    for community, members in communities_members.items():
        expectation = 0.0
        nodes_durations = get_nodes_durations_from_members(members)
        community_nodes = communities_nodes[community]

        for source, target in combinations_with_replacement(community_nodes, 2):
            geo_mean = (nodes_durations.get(source, 0) * nodes_durations.get(target, 0)) ** 0.5
            degrees_part = degree_cache.get_contribution(source, target)
            expected_value = _compute_expected_value(
                degrees_part,
                geo_mean,
                linkstream.network_duration,
                linkstream.weight,
                linkstream.directed,
            )
            expectation += expected_value

        communities_expectations[community] = expectation

    return communities_expectations


def _compute_coexistence_expectations(
    linkstream: LinkStream,
    communities_members: dict[CommunityLabel, set[tuple[int, int]]],
    communities_nodes: dict[CommunityLabel, set[int]],
    degree_cache: _DegreeCache,
) -> dict[CommunityLabel, float]:
    """Compute Coexistence Modularity Expectation (CM) for communities.

    CM uses the coexistence time (intersection of active times) as the time factor.

    Args:
        linkstream: The temporal network.
        communities_members: Mapping from community label to set of (node, time) tuples.
        communities_nodes: Pre-computed mapping from community label to node IDs.
        degree_cache: Cache for degree contribution lookups.

    Returns:
        Dictionary mapping community labels to expectation values.
    """
    communities_expectations: dict[CommunityLabel, float] = {}

    for community, members in communities_members.items():
        expectation = 0.0
        community_nodes_sorted = sorted(communities_nodes[community])
        nodes_times = get_nodes_times_from_members(members)

        for idx, source in enumerate(community_nodes_sorted):
            source_times = nodes_times.get(source, set())

            for target in community_nodes_sorted[idx:]:
                target_times = nodes_times.get(target, set())
                coexistence = len(source_times & target_times)

                if not coexistence:
                    continue

                degrees_part = degree_cache.get_contribution(source, target)
                expected_value = _compute_expected_value(
                    degrees_part,
                    coexistence,
                    linkstream.network_duration,
                    linkstream.weight,
                    linkstream.directed,
                )
                expectation += expected_value

        communities_expectations[community] = expectation

    return communities_expectations


def _count_community_switches(
    leaf_labels: dict[Leaf, CommunityLabel],
) -> float:
    """Count community switches across time.

    A community switch occurs when a node changes community between
    consecutive time steps.

    Args:
        leaf_labels: Mapping from Leaf to community label.

    Returns:
        Number of community switches (divided by 2 to avoid double counting).
    """
    switch_count = 0
    get = leaf_labels.get

    for leaf, community in leaf_labels.items():
        # Check left neighbor
        left_neighbor = leaf.left_time_active_neighbor
        if left_neighbor is not None:
            left_community = get(left_neighbor)
            if left_community is not None and left_community != community:
                switch_count += 1

        # Check right neighbor
        right_neighbor = leaf.right_time_active_neighbor
        if right_neighbor is not None:
            right_community = get(right_neighbor)
            if right_community is not None and right_community != community:
                switch_count += 1

    # Switches are counted twice (once from each side)
    return switch_count / 2


# =============================================================================
# Closed-form Expectations
# =============================================================================
#
# The pair loops above sum, over combinations_with_replacement(nodes, 2), a term
# whose 2 ** (source != target) multiplier makes the upper-triangular sum equal
# to the full ordered sum. That factors:
#
#   undirected:  sum_{i,j} d_i d_j f_i f_j            = (sum_i d_i f_i) ** 2
#   directed:    sum_{i,j} out_i in_j f_i f_j         = (sum_i out_i f_i)(sum_i in_i f_i)
#
# with the k-partite mask removed by subtracting the same quantity per partite
# group (see _masked_square). f_i is the per-node time factor: sqrt(duration)
# for MM, 1 for JM. CM's factor depends on the pair, so it is decomposed over
# time instead: |T_i n T_j| = sum_t [t in T_i][t in T_j].
#
# These are O(n) / O(#members) where the loops are O(n**2); they are exact in
# real arithmetic, and differ from the loops only in floating-point rounding.


def _compute_joint_expectations_closed(
    linkstream: LinkStream,
    communities_segments: dict[CommunityLabel, dict[int, tuple[tuple[int, int], ...]]],
    communities_nodes: dict[CommunityLabel, set[int]],
    degree_cache: _DegreeCache | None = None,
) -> dict[CommunityLabel, float]:
    """Closed-form Joint Modularity Expectation (JM). See module notes above."""
    communities_expectations: dict[CommunityLabel, float] = {}
    partite_mapping = linkstream.partite_mapping

    for community, members in communities_segments.items():
        nodes = communities_nodes[community]
        if not nodes:
            communities_expectations[community] = 0.0
            continue

        community_duration = get_module_duration_from_segments(members)
        weights, second = _degree_factors(linkstream, nodes)
        groups = _partite_groups(nodes, partite_mapping)
        total = _masked_square(nodes, weights, groups, second)
        communities_expectations[community] = _scale_expectation(
            linkstream, total, community_duration
        )

    return communities_expectations


def _compute_mean_expectations_closed(
    linkstream: LinkStream,
    communities_segments: dict[CommunityLabel, dict[int, tuple[tuple[int, int], ...]]],
    communities_nodes: dict[CommunityLabel, set[int]],
    degree_cache: _DegreeCache | None = None,
) -> dict[CommunityLabel, float]:
    """Closed-form Mean Modularity Expectation (MM). See module notes above."""
    communities_expectations: dict[CommunityLabel, float] = {}
    partite_mapping = linkstream.partite_mapping

    for community, members in communities_segments.items():
        nodes = communities_nodes[community]
        if not nodes:
            communities_expectations[community] = 0.0
            continue

        durations = get_nodes_durations_from_segments(members)
        # The geometric mean sqrt(D_i * D_j) factors into sqrt(D_i) * sqrt(D_j)
        roots = {node: durations.get(node, 0) ** 0.5 for node in nodes}
        weights, second = _degree_factors(linkstream, nodes, roots)
        groups = _partite_groups(nodes, partite_mapping)
        total = _masked_square(nodes, weights, groups, second)
        communities_expectations[community] = _scale_expectation(linkstream, total, 1)

    return communities_expectations


def _compute_coexistence_expectations_closed(
    linkstream: LinkStream,
    communities_segments: dict[CommunityLabel, dict[int, tuple[tuple[int, int], ...]]],
    communities_nodes: dict[CommunityLabel, set[int]],
    degree_cache: _DegreeCache | None = None,
) -> dict[CommunityLabel, float]:
    """Closed-form Coexistence Modularity Expectation (CM). See module notes above."""
    communities_expectations: dict[CommunityLabel, float] = {}
    partite_mapping = linkstream.partite_mapping

    for community, members in communities_segments.items():
        nodes = communities_nodes[community]
        if not nodes:
            communities_expectations[community] = 0.0
            continue

        # The coexistence count of a pair is the number of instants where both
        # appear, so summing over instants avoids the pairwise intersection
        # entirely -- and the active set only changes at a segment boundary, so
        # the instants between two boundaries are handled in one step.
        weights, second = _degree_factors(linkstream, nodes)
        total = 0.0
        for length, present in iter_coexistence_spans(members):
            groups = _partite_groups(present, partite_mapping)
            total += length * _masked_square(present, weights, groups, second)

        communities_expectations[community] = _scale_expectation(linkstream, total, 1)

    return communities_expectations


_EXPECTATION_LOOPS = {
    LexType.CM: _compute_coexistence_expectations,
    LexType.JM: _compute_joint_expectations,
    LexType.MM: _compute_mean_expectations,
}

_EXPECTATION_CLOSED = {
    LexType.CM: _compute_coexistence_expectations_closed,
    LexType.JM: _compute_joint_expectations_closed,
    LexType.MM: _compute_mean_expectations_closed,
}
