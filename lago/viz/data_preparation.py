"""
Data preparation module for longitudinal community plotting.

This module provides functions for preparing, filtering, and processing community data
for longitudinal visualization. It handles community filtering based on focus criteria,
time range calculations, community sorting, and data structure transformations.

Key Functions:
    - prepare_communities_for_display: Prepare communities with focus filtering
    - calculate_node_time_ranges: Calculate node activity time ranges
    - filter_and_sort_communities: Filter and sort communities by size
    - create_time_node_community_mapping: Create community membership mapping
    - get_communities_to_focus: Identify communities matching focus criteria
    - sort_communities_by_size: Sort communities by membership size
"""

from typing import Any, Collection, Dict, List, Optional, Set, Tuple


def prepare_communities_for_display(
    communities: Dict,
    node_focus: Optional[list],
    time_focus: Optional[int],
    node_OR_time_focus: bool,
) -> Tuple[Dict, Dict]:
    """
    Prepare communities for display by filtering based on focus criteria.

    Args:
        communities: Original communities dictionary
        node_focus: Node to focus on (optional)
        time_focus: Time to focus on (optional)
        node_OR_time_focus: Whether to use OR or AND logic for focus

    Returns:
        Tuple of (focused_communities, monochrome_communities)
    """
    if node_focus != [] or time_focus is not None:
        communities_to_focus = get_communities_to_focus(
            communities, node_focus, time_focus, node_OR_time_focus
        )
        communities_monochrome = {
            label: communities[label]
            for label in communities.keys()
            if label not in communities_to_focus
        }
        communities_to_display = {
            label: communities[label] for label in communities_to_focus
        }
    else:
        communities_to_display = communities
        communities_monochrome = {}

    return communities_to_display, communities_monochrome


def calculate_node_time_ranges(
    nodes: Collection[Any], time_links: List[Any], network_duration: int, trim: bool
) -> Tuple[Dict[Any, int], Dict[Any, int]]:
    """
    Calculate start and end times for each node.

    Args:
        nodes: Set of nodes
        time_links: List of (source, target, time, weight) tuples
        network_duration: Total duration of the network
        trim: Whether to trim to actual activity periods

    Returns:
        Tuple of (start_times, end_times) dictionaries
    """
    if trim:
        start_nodes = {}
        end_nodes = {}
        for source, target, time, _ in time_links:
            for node in [source, target]:
                start_nodes[node] = min(time, start_nodes.get(node, network_duration))
                end_nodes[node] = max(time, end_nodes.get(node, 0))
    else:
        start_nodes = {node: 0 for node in nodes}
        end_nodes = {node: network_duration for node in nodes}

    return start_nodes, end_nodes


def filter_and_sort_communities(
    communities: Dict, max_shown_communities: int, hide_self_communities: bool
) -> Dict:
    """
    Filter and sort communities based on size and display criteria.

    Args:
        communities: Communities dictionary
        max_shown_communities: Maximum number of communities to show
        hide_self_communities: Whether to hide single-node communities

    Returns:
        Filtered and sorted communities dictionary
    """
    if max_shown_communities > -1:
        communities = sort_communities_by_size(communities, max_shown_communities)

    if hide_self_communities:
        return {
            label: community
            for label, community in communities.items()
            if len(community) > 1
        }
    return communities


def create_time_node_community_mapping(communities: Dict) -> Dict:
    """
    Create mapping from (node, time) tuples to community labels.

    Args:
        communities: Communities dictionary

    Returns:
        Dictionary mapping (node, time) to community label
    """
    time_node_community_mapping = {}
    for community_label, community in communities.items():
        for time_node in community:
            time_node_community_mapping[time_node] = community_label
    return time_node_community_mapping


def get_communities_to_focus(
    communities: Dict,
    node_focus: Optional[List] = None,
    time_focus: Optional[int] = None,
    node_OR_time_focus: bool = False,
) -> Set:
    """
    Get set of community labels that should be focused based on node/time criteria.

    Args:
        communities: Communities dictionary
        node_focus: Node to focus on
        time_focus: Time to focus on
        node_OR_time_focus: Whether to use OR logic for focus

    Returns:
        Set of community labels to focus on
    """
    focused_communities = set()

    for community_label, members in communities.items():
        for node, time in members:
            if node_focus is None or len(node_focus) == 0:
                if time == time_focus:
                    focused_communities.add(community_label)
                    break
            elif time_focus is None:
                if node in node_focus:
                    focused_communities.add(community_label)
                    break
            else:
                if node_OR_time_focus:
                    if time == time_focus or node in node_focus:
                        focused_communities.add(community_label)
                        break
                else:
                    if time == time_focus and node in node_focus:
                        focused_communities.add(community_label)
                        break

    return focused_communities


def sort_communities_by_size(communities: Dict, top_n: int = 20) -> Dict:
    """
    Sort communities by size and return top N.

    Args:
        communities: Communities dictionary
        top_n: Number of top communities to return

    Returns:
        Dictionary of top N communities sorted by size
    """
    # Create list of (community_index, size) tuples
    community_sizes = {lab: len(community) for lab, community in communities.items()}

    # Sort by size in descending order
    sorted_sizes = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[
        :top_n
    ]
    return dict(sorted_sizes)
    # Return top N communities
    # result = {}
    # for i in range(min(top_n, len(sorted_sizes))):
    #     original_index = sorted_sizes[i][0]
    #     result[original_index] = list(communities.values())[original_index]

    # return result
