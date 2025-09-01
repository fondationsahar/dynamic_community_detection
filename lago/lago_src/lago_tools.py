from typing import List

from lago.leaf import Leaf
from lago.module import Module


def get_nodes_segment(
    module_leaves: set[Leaf],
):
    """Compute time segments of module membership for each node.

    Args:
        module_leaves (set): set of time nodes

    Returns:
        dict: {node: [[left, right], [left, right], ...], ...}
    """
    nodes_segments: dict[int, List[List[Leaf]]] = {}

    leaves_set = set(module_leaves)
    while leaves_set:
        # Start segment with random leaf
        left_leaf = leaves_set.pop()
        right_leaf = left_leaf

        # If necessary, init dictionnary key corresponding to node id
        if left_leaf.node not in nodes_segments:
            nodes_segments[left_leaf.node] = []

        # Extend segment on the right until right neighbor (next time occurence of the node)
        # does not exist or belong to another module
        right_time_active_neighbor = right_leaf.right_time_active_neighbor
        while (
            right_time_active_neighbor and right_time_active_neighbor in module_leaves
        ):
            right_leaf = right_time_active_neighbor
            leaves_set.remove(right_leaf)
            right_time_active_neighbor = right_leaf.right_time_active_neighbor

        # Extend segment on the left until left neighbor (previous time occurence of the node)
        # does not exist or belong to another module
        left_time_active_neighbor = left_leaf.left_time_active_neighbor
        while left_time_active_neighbor and left_time_active_neighbor in module_leaves:
            left_leaf = left_time_active_neighbor
            leaves_set.remove(left_leaf)
            left_time_active_neighbor = left_leaf.left_time_active_neighbor

        # Sometimes time segement have only one time step
        if left_leaf == right_leaf:
            segment = [left_leaf]
        else:
            segment = [left_leaf, right_leaf]

        # Add built segment to the list
        nodes_segments[left_leaf.node].append(segment)

    return nodes_segments


def move_submodule(
    submodule: Module,
    from_module: Module,
    to_module: Module,
):
    """Update affiliations of the submodule moving from a module to another.

    Args:
        submodule (Module): Submodule to move
        from_module (Module): Source module from which submodule
        to_module (Module): Target module joined by submodule
    """
    from_module.submodules.remove(submodule)
    for leaf in submodule.leaves:
        from_module.leaves.remove(leaf)
    to_module.submodules.append(submodule)
    to_module.leaves |= submodule.leaves
    submodule.parent = to_module


def update_fast_iteration_exploration_set(
    child_module: Module, exploration_set: set[Module]
):
    """Udpate exploration set by adding neighbors of the child module.
        Neighbors with the same parent neighbors are ignore.

    Args:
        child_module (Module): Sub module that have been moved to a better module.
        exploration_set (set): child modules to explore as candidates for moving.

    Returns:
        set: child modules to explore as candidates for moving.
    """
    for neighbor_module in child_module.neighbors:
        if neighbor_module.parent == child_module.parent:
            continue
        exploration_set.add(neighbor_module)
    return exploration_set


def create_module_from_leaves(leaves: tuple[Leaf, Leaf]):
    """Build a Module from leaves with matching parent module affiliation.

    Args:
        leaves (tuple): time nodes

    Returns:
        Module: Module built from leaves
    """
    child_edge = [*{*leaves}]
    parent_module = child_edge[0].module
    C0_leaves = set(child_edge)

    module = Module(C0_leaves)
    module.parent = parent_module

    return module


def get_neighbors_modules_parents(module: Module, time=True, topo=True):
    """Compute the parents of the neighbors of the time nodes of the module.

    Args:
        module (Module): Module from which looking for neighbors
        time (bool, optional): Whether to select time neighbors or not.
            Defaults to True.
        topo (bool, optional): Whether to select topological neighbors or not.
            Defaults to True.

    Returns:
        set: modules
    """
    neighbors_modules = set[Module]()
    for leaf in module.leaves:
        if topo:
            for tmp_leaf in leaf.topo_neighbors:
                neighbors_modules.add(tmp_leaf.module)
        if time:
            if (
                leaf.left_time_active_neighbor is not None
                and leaf.left_time_active_neighbor.module is not None
            ):
                neighbors_modules.add(leaf.left_time_active_neighbor.module)

            if (
                leaf.right_time_active_neighbor is not None
                and leaf.right_time_active_neighbor.module is not None
            ):
                neighbors_modules.add(leaf.right_time_active_neighbor.module)
    return neighbors_modules
