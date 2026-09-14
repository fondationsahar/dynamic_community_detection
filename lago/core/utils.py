from collections import defaultdict

from lago.algorithm._internal import lago_tools as lts
from lago.algorithm._internal._leaf import Leaf

# ===========================================================================
# Logging Helpers
# ===========================================================================


def log_info(message: str, verbose: int | bool) -> None:
    """Print info message if verbose >= 1.

    Args:
        message: Message to print.
        verbose: Verbosity level (0=silent, 1=info, 2=debug).
    """
    v = int(verbose) if isinstance(verbose, bool) else verbose
    if v >= 1:
        print(message)


def log_debug(message: str, verbose: int | bool) -> None:
    """Print debug message if verbose >= 2.

    Args:
        message: Message to print.
        verbose: Verbosity level (0=silent, 1=info, 2=debug).
    """
    v = int(verbose) if isinstance(verbose, bool) else verbose
    if v >= 2:
        print(message)


# ===========================================================================
# Module Utilities
# ===========================================================================


def get_module_duration(module_leaves: set[Leaf]) -> float:
    """Compute the duration of a module.

    Note that it does not take into account
    time breaks for module existence.

    Args:
        module_leaves: Set of Leaf (time-node) objects.

    Returns:
        Duration of the module.
    """
    # Only the extremes matter, so track them directly instead of materialising
    # the set of every instant the module covers. A time-node with no incident
    # edge covers nothing and is skipped.
    min_time: int | None = None
    max_time: int | None = None
    for leaf in module_leaves:
        if leaf.topo_neighbors:
            edge_duration = next(iter(leaf.topo_neighbors)).duration
        elif leaf.topo_neighbors_from:
            edge_duration = next(iter(leaf.topo_neighbors_from)).duration
        else:
            continue

        start = leaf.time
        end = start + edge_duration
        if min_time is None or start < min_time:
            min_time = start
        if max_time is None or end > max_time:
            max_time = end

    if min_time is None:
        return 0

    # Approximation allowed for LAGO, no resurgences expected
    duration: float = max_time - min_time + 1

    return duration


def get_nodes_times(module_leaves: set[Leaf]) -> dict[int, set[int]]:
    """Compute the times when each node belongs to the module.

    Args:
        module_leaves: Set of Leaf (time-node) objects.

    Returns:
        Dictionary mapping node ID to set of time points.
    """
    nodes_times = {}
    for leaf in module_leaves:
        node = leaf.node
        if node not in nodes_times:
            nodes_times[node] = set()
        nodes_times[node].add(leaf.time)
        right_time_active_neighbor = leaf.right_time_active_neighbor
        if not (right_time_active_neighbor and right_time_active_neighbor in module_leaves):
            continue

        nodes_times[node] |= set(range(leaf.time, right_time_active_neighbor.time))
    return nodes_times


def leaf_edge_duration(leaf: Leaf) -> int:
    """Duration of the interactions carried by a time-node.

    All edges incident to a leaf share the same duration: continuous links are
    split on the same global set of time instants, so every edge starting at
    ``leaf`` ends at the same next instant.

    Args:
        leaf: The time-node.

    Returns:
        The duration, or 1 for a leaf with no incident edge.
    """
    if leaf.topo_neighbors:
        return next(iter(leaf.topo_neighbors)).duration
    if leaf.topo_neighbors_from:
        return next(iter(leaf.topo_neighbors_from)).duration
    return 1


def duration_delta_on_add(leaf: Leaf, contains) -> int:
    """Change in ``leaf.node``'s duration when ``leaf`` joins a set of leaves.

    A node's duration is the total length of its maximal runs of consecutive
    active time-nodes, and a run is delimited by ``left/right_time_active_neighbor``
    -- so adding one time-node only ever extends a run, bridges two, or starts a
    new one. Which of the three it is depends solely on whether the leaf's two
    time-neighbours are in the set, and in every case the run's far endpoints
    cancel out of the difference. That makes this O(1), exact, and integral.

    Args:
        leaf: The time-node being added; must not already be in the set.
        contains: Membership predicate for the set **before** the addition.

    Returns:
        The signed change in the node's duration.
    """
    left = leaf.left_time_active_neighbor
    right = leaf.right_time_active_neighbor
    has_left = left is not None and contains(left)
    has_right = right is not None and contains(right)

    if has_left:
        if has_right:
            # Two runs merge and the leaf fills the gap between them.
            return right.time - left.time - leaf_edge_duration(left)
        # The run ending at `left` now ends at `leaf`.
        return leaf.time - left.time - leaf_edge_duration(left) + leaf_edge_duration(leaf)
    if has_right:
        # The run starting at `right` now starts at `leaf`.
        return right.time - leaf.time
    # A new run holding just this leaf.
    return leaf_edge_duration(leaf)


def duration_delta_on_remove(leaf: Leaf, contains) -> int:
    """Change in ``leaf.node``'s duration when ``leaf`` leaves a set of leaves.

    Removal is the exact inverse of the corresponding addition: the leaf's two
    time-neighbours cannot be the leaf itself, so their membership is the same
    before and after.

    Args:
        leaf: The time-node being removed; must currently be in the set.
        contains: Membership predicate for the set (with or without ``leaf``).

    Returns:
        The signed change in the node's duration.
    """
    return -duration_delta_on_add(leaf, contains)


def get_nodes_durations(module_leaves: set[Leaf]) -> dict[int, float]:
    """Compute how long each node belongs to the module.

    A node's duration is the total length of the maximal runs of consecutive
    active time-nodes it has inside ``module_leaves``; a run ending at leaf ``r``
    extends to ``r.time + leaf_edge_duration(r)``, the same convention as
    :func:`get_expanded_module`.

    Args:
        module_leaves: Set of Leaf (time-node) objects.

    Returns:
        Dictionary mapping node ID to duration.
    """

    nodes_durations: dict[int, float] = defaultdict(float)

    leaves_set = set(module_leaves)
    contains = module_leaves.__contains__
    discard = leaves_set.remove
    pop = leaves_set.pop
    while leaves_set:
        # Select leaf. Any leaf of a run yields the same run, so the result does
        # not depend on which one pop() happens to return.
        left_leaf = pop()
        right_leaf = left_leaf

        # If necessary, init dictionnary key corresponding to node id
        if left_leaf.node not in nodes_durations:
            nodes_durations[left_leaf.node] = 0

        # Extend segment on the right until right neighbor (next time occurence of the node)
        # does not exist or belong to another module
        right_time_active_neighbor = right_leaf.right_time_active_neighbor
        while right_time_active_neighbor is not None and contains(right_time_active_neighbor):
            right_leaf = right_time_active_neighbor
            discard(right_leaf)
            right_time_active_neighbor = right_leaf.right_time_active_neighbor

        # Extend duration on the left until left neighbor (previous time occurence of the node)
        # does not exist or belong to another module
        left_time_active_neighbor = left_leaf.left_time_active_neighbor
        while left_time_active_neighbor is not None and contains(left_time_active_neighbor):
            left_leaf = left_time_active_neighbor
            discard(left_leaf)
            left_time_active_neighbor = left_leaf.left_time_active_neighbor

        # leaf_edge_duration inlined: this is the innermost loop of LAGO
        if right_leaf.topo_neighbors:
            edge_duration = next(iter(right_leaf.topo_neighbors)).duration
        elif right_leaf.topo_neighbors_from:
            edge_duration = next(iter(right_leaf.topo_neighbors_from)).duration
        else:
            edge_duration = 1

        # The run covers [left_leaf.time, right_leaf.time + duration of right_leaf)
        nodes_durations[left_leaf.node] += right_leaf.time - left_leaf.time + edge_duration

    return nodes_durations


# ===========================================================================
# Module Member Utilities (work directly with (node, time) tuples)
# ===========================================================================


def get_module_duration_from_members(members: set[tuple[int, int]]) -> int:
    """Compute the duration of a module from its members.

    Args:
        members: Set of (node, time) tuples representing module membership.

    Returns:
        Duration of the module (max_time - min_time + 1).
    """
    if not members:
        return 0
    times = {time for _, time in members}
    return len(times)


def get_nodes_times_from_members(members: set[tuple[int, int]]) -> dict[int, set[int]]:
    """Compute the times when each node belongs to the module from members.

    Args:
        members: Set of (node, time) tuples representing module membership.

    Returns:
        Dictionary mapping node ID to set of time points.
    """
    nodes_times: dict[int, set[int]] = {}
    for node, time in members:
        if node not in nodes_times:
            nodes_times[node] = set()
        nodes_times[node].add(time)
    return nodes_times


def get_nodes_durations_from_members(members: set[tuple[int, int]]) -> dict[int, int]:
    """Compute how long each node belongs to the module from members.

    Args:
        members: Set of (node, time) tuples representing module membership.

    Returns:
        Dictionary mapping node ID to duration (number of time points).
    """
    if isinstance(members, (set, frozenset, dict)):
        # The (node, time) pairs are already unique, so the duration of a node is
        # just how many pairs mention it -- no need to materialise its time set.
        durations: dict[int, int] = {}
        for node, _time in members:
            durations[node] = durations.get(node, 0) + 1
        return durations

    # Any other container may repeat a pair; deduplicate through the time sets.
    nodes_times = get_nodes_times_from_members(members)
    return {node: len(times) for node, times in nodes_times.items()}


def get_module_segments(backbone_leaves: set[Leaf]) -> dict[int, tuple[tuple[int, int], ...]]:
    """Describe a module as per-node inclusive ``[start, end]`` runs.

    The runs are the module itself; :func:`get_expanded_module` just enumerates
    the instants they cover, which can be orders of magnitude more values.

    Args:
        backbone_leaves: Module active time-nodes.

    Returns:
        Mapping from node to its inclusive (start, end) runs.
    """
    module_segments = lts.get_nodes_segment(
        module_leaves=backbone_leaves,
    )
    runs: dict[int, tuple[tuple[int, int], ...]] = {}
    for node, segments in module_segments.items():
        node_runs = []
        for segment in segments:
            first = segment[0]
            last = segment[-1]
            # A run reaches to the end of its last time-node's interactions,
            # and the (node, time) view stops one instant short of that.
            node_runs.append((first.time, last.time + leaf_edge_duration(last) - 1))
        runs[node] = tuple(node_runs)
    return runs


def get_expanded_module(backbone_leaves: set[Leaf]) -> set[tuple[int, int]]:
    """Expand a module to all (node, time) tuples from active time-nodes.

    Args:
        backbone_leaves: Module active time-nodes.

    Returns:
        Set of (node, time) tuples belonging to the module.
    """
    time_module = set[tuple[int, int]]()
    for node, runs in get_module_segments(backbone_leaves).items():
        for start, end in runs:
            time_module.update((node, time) for time in range(start, end + 1))
    return time_module


# ===========================================================================
# Segment Utilities (work with per-node inclusive [start, end] runs)
# ===========================================================================
#
# A module's segments carry the same information as its (node, time) members --
# expanding one gives the other -- but a module spanning a long interval has far
# more members than segments. These compute directly on the segments, so their
# cost follows the number of runs rather than the number of instants covered.


def get_nodes_durations_from_segments(
    segments: dict[int, tuple[tuple[int, int], ...]],
) -> dict[int, int]:
    """How long each node belongs to the module, from its segments.

    Args:
        segments: Mapping from node to inclusive (start, end) runs.

    Returns:
        Dictionary mapping node ID to duration.
    """
    return {
        node: sum(end - start + 1 for start, end in runs) for node, runs in segments.items()
    }


def get_module_duration_from_segments(
    segments: dict[int, tuple[tuple[int, int], ...]],
) -> int:
    """Number of distinct instants the module covers, from its segments.

    Nodes overlap in time, so the runs are merged before measuring.

    Args:
        segments: Mapping from node to inclusive (start, end) runs.

    Returns:
        The count of covered instants.
    """
    runs = sorted(run for node_runs in segments.values() for run in node_runs)
    if not runs:
        return 0

    total = 0
    current_start, current_end = runs[0]
    for start, end in runs[1:]:
        if start > current_end + 1:
            total += current_end - current_start + 1
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    return total + current_end - current_start + 1


def iter_coexistence_spans(
    segments: dict[int, tuple[tuple[int, int], ...]],
):
    """Sweep the segments, yielding maximal spans over which the active set is fixed.

    Coexistence sums the same quantity over every instant, and the set of active
    nodes only changes at a segment boundary -- so the instants in between can be
    handled in one step instead of one at a time.

    Args:
        segments: Mapping from node to inclusive (start, end) runs.

    Yields:
        ``(length, active_nodes)`` pairs, where ``length`` is the number of
        instants and ``active_nodes`` the set active throughout them. The set is
        reused between yields; copy it if you need to keep it.
    """
    events: dict[int, list[tuple[int, bool]]] = {}
    for node, runs in segments.items():
        for start, end in runs:
            events.setdefault(start, []).append((node, True))
            events.setdefault(end + 1, []).append((node, False))

    active: set[int] = set()
    previous: int | None = None
    for boundary in sorted(events):
        if previous is not None and active:
            yield boundary - previous, active
        for node, entering in events[boundary]:
            if entering:
                active.add(node)
            else:
                active.discard(node)
        previous = boundary
