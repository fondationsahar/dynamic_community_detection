"""
Data preparation module for longitudinal module plotting.

This module provides functions for preparing, filtering, and processing module data
for longitudinal visualization. It handles module filtering based on focus criteria,
time range calculations, module sorting, and data structure transformations.

Key Functions:
    - prepare_modules_for_display: Prepare modules with focus filtering
    - calculate_node_time_ranges: Calculate node activity time ranges
    - filter_and_sort_modules: Filter and sort modules by size
    - create_time_node_module_mapping: Create module membership mapping
    - get_modules_to_focus: Identify modules matching focus criteria
    - sort_modules_by_size: Sort modules by membership size
"""

from typing import Any, Collection, Dict, List, Optional, Set, Tuple


def prepare_modules_for_display(
    modules: Dict,
    node_focus: Optional[list],
    time_focus: Optional[int],
    node_OR_time_focus: bool,
) -> Tuple[Dict, Dict]:
    """
    Prepare modules for display by filtering based on focus criteria.

    Args:
        modules: Original modules dictionary
        node_focus: Node to focus on (optional)
        time_focus: Time to focus on (optional)
        node_OR_time_focus: Whether to use OR or AND logic for focus

    Returns:
        Tuple of (focused_modules, monochrome_modules)
    """
    if node_focus != [] or time_focus is not None:
        modules_to_focus = get_modules_to_focus(
            modules, node_focus, time_focus, node_OR_time_focus
        )
        modules_monochrome = {
            label: modules[label]
            for label in modules.keys()
            if label not in modules_to_focus
        }
        modules_to_display = {
            label: modules[label] for label in modules_to_focus
        }
    else:
        modules_to_display = modules
        modules_monochrome = {}

    return modules_to_display, modules_monochrome


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


def filter_and_sort_modules(
    modules: Dict, max_shown_modules: int, hide_self_modules: bool
) -> Dict:
    """
    Filter and sort modules based on size and display criteria.

    Args:
        modules: Modules dictionary
        max_shown_modules: Maximum number of modules to show
        hide_self_modules: Whether to hide single-node modules

    Returns:
        Filtered and sorted modules dictionary
    """
    if max_shown_modules > -1:
        modules = sort_modules_by_size(modules, max_shown_modules)

    if hide_self_modules:
        return {
            label: module
            for label, module in modules.items()
            if len(module) > 1
        }
    return modules


def create_time_node_module_mapping(modules: Dict) -> Dict:
    """
    Create mapping from (node, time) tuples to module labels.

    Args:
        modules: Modules dictionary

    Returns:
        Dictionary mapping (node, time) to module label
    """
    time_node_module_mapping = {}
    for module_label, module in modules.items():
        for time_node in module:
            time_node_module_mapping[time_node] = module_label
    return time_node_module_mapping


def get_modules_to_focus(
    modules: Dict,
    node_focus: Optional[List] = None,
    time_focus: Optional[int] = None,
    node_OR_time_focus: bool = False,
) -> Set:
    """
    Get set of module labels that should be focused based on node/time criteria.

    Args:
        modules: Modules dictionary
        node_focus: Node to focus on
        time_focus: Time to focus on
        node_OR_time_focus: Whether to use OR logic for focus

    Returns:
        Set of module labels to focus on
    """
    focused_modules = set()

    for module_label, members in modules.items():
        for node, time in members:
            if node_focus is None or len(node_focus) == 0:
                if time == time_focus:
                    focused_modules.add(module_label)
                    break
            elif time_focus is None:
                if node in node_focus:
                    focused_modules.add(module_label)
                    break
            else:
                if node_OR_time_focus:
                    if time == time_focus or node in node_focus:
                        focused_modules.add(module_label)
                        break
                else:
                    if time == time_focus and node in node_focus:
                        focused_modules.add(module_label)
                        break

    return focused_modules


def sort_modules_by_size(modules: Dict, top_n: int = 20) -> Dict:
    """
    Sort modules by size and return top N.

    Args:
        modules: Modules dictionary
        top_n: Number of top modules to return

    Returns:
        Dictionary of top N modules sorted by size
    """
    # Create list of (module_index, size) tuples
    module_sizes = {lab: len(module) for lab, module in modules.items()}

    # Sort by size in descending order
    sorted_sizes = sorted(module_sizes.items(), key=lambda x: x[1], reverse=True)[
        :top_n
    ]
    return dict(sorted_sizes)
