from collections import defaultdict

import numpy as np  # type: ignore

from .lago_src import lago_tools as lts
from .leaf import Leaf


def get_module_duration(module_leaves: set[Leaf]):
    """Compute the duration of a module.
    Note that it does not take into account
    time breaks for module existence.

    Args:
        module_leaves (set): set of time nodes

    Returns:
        float: duration
    """
    # all_times = [*{leaf.time for leaf in module_leaves}]
    all_times = set()
    for leaf in module_leaves:
        all_times |= set(
            range(
                leaf.time,
                leaf.time
                + list(leaf.topo_neighbors | leaf.topo_neighbors_from)[0].duration
                + 1,
            )
        )
    all_times = list(all_times)
    if not all_times:
        return 0
    duration: float = np.max(all_times) - np.min(all_times) + 1

    return duration


def get_nodes_times(module_leaves: set[Leaf]):
    """Compute the times nodes belong to module.

    Args:
        module_leaves (set): set of time nodes

    Returns:
        dict: nodes times for belonging to the module
    """
    nodes_times = {}
    for leaf in module_leaves:
        node = leaf.node
        if node not in nodes_times:
            nodes_times[node] = set()
        nodes_times[node].add(leaf.time)
        right_time_active_neighbor = leaf.right_time_active_neighbor
        if not (
            right_time_active_neighbor and right_time_active_neighbor in module_leaves
        ):
            continue

        nodes_times[node] |= set(range(leaf.time, right_time_active_neighbor.time))
    return nodes_times


def get_nodes_durations(
    module_leaves: set[Leaf],
):
    """Compute nodes membership durations to the module.

    Args:
        module_leaves (set): set of time nodes

    Returns:
        dict: {node: duration, ...}
    """

    nodes_durations: dict[int, float] = defaultdict(float)

    leaves_set = set(module_leaves)
    while leaves_set:
        # Select leaf
        left_leaf = leaves_set.pop()
        right_leaf = left_leaf

        # If necessary, init dictionnary key corresponding to node id
        if left_leaf.node not in nodes_durations:
            nodes_durations[left_leaf.node] = 0

        # Extend segment on the right until right neighbor (next time occurence of the node)
        # does not exist or belong to another module
        right_time_active_neighbor = right_leaf.right_time_active_neighbor
        # Extend to the right for the amount of the last edge duration
        right_duration = 1
        while (
            right_time_active_neighbor and right_time_active_neighbor in module_leaves
        ):
            right_leaf = right_time_active_neighbor
            leaves_set.remove(right_leaf)
            right_time_active_neighbor = right_leaf.right_time_active_neighbor
            # Select random topological edge (if not None), they are all supposed to have the same duration.
            if right_time_active_neighbor:
                right_duration = list(
                    right_time_active_neighbor.topo_neighbors
                    | right_time_active_neighbor.topo_neighbors_from
                )[0].duration

        # Extend duration on the left until left neighbor (previous time occurence of the node)
        # does not exist or belong to another module
        left_time_active_neighbor = left_leaf.left_time_active_neighbor
        while left_time_active_neighbor and left_time_active_neighbor in module_leaves:
            left_leaf = left_time_active_neighbor
            leaves_set.remove(left_leaf)
            left_time_active_neighbor = left_leaf.left_time_active_neighbor

        nodes_durations[left_leaf.node] += (
            right_leaf.time - left_leaf.time + right_duration
        )

    return nodes_durations


def get_expanded_module(
    backbone_leaves: set[Leaf],
):
    """Compute total time nodes of the module from only the active time nodes.

    Args:
        backbone_leaves (set): module active time nodes

    Returns:
        set: tuple of (node, time) that belong to the module
    """
    time_module = set[tuple[int, int]]()

    module_segments = lts.get_nodes_segment(
        module_leaves=backbone_leaves,
    )
    for node, segments in module_segments.items():
        for segment in segments:
            if len(segment) == 1:
                segment = [segment[0], segment[0]]
                # time_module |= set([(node, segment[0].time)])
                # continue
            time1 = segment[0].time
            time2 = (
                segment[1].time
                + list(segment[1].topo_neighbors | segment[1].topo_neighbors_from)[
                    0
                ].duration
            )
            time_module |= set(zip([node] * (time2 - time1), range(time1, time2)))

    return time_module
