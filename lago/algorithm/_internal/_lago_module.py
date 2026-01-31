"""Internal module class used by the LAGO algorithm.

This module contains the _LagoModule class, which is an internal mutable data
structure used during the LAGO algorithm execution. It stores sets of Leaf
objects and supports hierarchical module structures.

Note:
    This is an internal class. For user-facing results, see TimeModule and
    TimeModules in lago/time_modules.py.

Relationship to TimeModule:
    - _LagoModule: Mutable, stores Leaf objects, used during algorithm
    - TimeModule: Immutable view, stores (node, time) tuples, for user queries

See Also:
    - lago.time_modules.TimeModule: The public API for accessing module results
    - lago.time_modules.TimeModules: Container for all detected modules
"""

import copy
from typing import TYPE_CHECKING, Self, TypeAlias

if TYPE_CHECKING:
    from ._leaf import Leaf

Leaves: TypeAlias = set


class _LagoModule:
    """Internal module representation for LAGO algorithm.

    This class is used internally by the LAGO optimization algorithm to track
    module membership during the greedy search process. It is mutable and
    contains references to Leaf objects.

    Attributes:
        leaves: Set of Leaf objects belonging to this module.
        submodules: List of child modules in the hierarchy.
        parent: Parent module in the hierarchy (None if root).
        neighbors: Set of neighboring modules in the graph.

    Note:
        This is an internal class. Users should work with TimeModule objects
        returned by lago_modules().
    """

    def __init__(self, leaves: Leaves) -> None:
        """Initialize a _LagoModule with a set of leaves.

        Args:
            leaves: Set of Leaf objects to include in this module.
        """
        self.leaves: Leaves = leaves
        self.submodules: list[_LagoModule] = []
        self.parent: _LagoModule | None = None
        self.neighbors: set[_LagoModule] = set()

    def compute_neighbors(self, subset: set = set()) -> None:
        """Compute neighboring modules based on leaf connections.

        Updates self.neighbors with all modules that share edges (topological
        or temporal) with this module's leaves.

        Args:
            subset: Optional set of leaves to restrict neighbor search to.
                If empty, all neighbors are considered.
        """
        leaves_neighbors: set[Leaf] = set()
        for leaf in self.leaves:
            leaves_neighbors |= set([tmp_neighbor.target for tmp_neighbor in leaf.topo_neighbors])
            right_time_neighb = leaf.right_time_active_neighbor
            if right_time_neighb is not None:
                leaves_neighbors.add(right_time_neighb)
            left_time_neighb = leaf.left_time_active_neighbor
            if left_time_neighb is not None:
                leaves_neighbors.add(left_time_neighb)

        if subset:
            self.neighbors = set([leaf.module for leaf in leaves_neighbors & subset])
        else:
            self.neighbors = set([leaf.module for leaf in leaves_neighbors])
        for neighbor in self.neighbors:
            if neighbor is None:
                self.neighbors.remove(neighbor)

    def duplicates(self) -> Self:
        """Create a shallow copy of this module.

        Returns:
            A new _LagoModule with a copy of the leaves set.
            Note: The leaves themselves are not copied.
        """
        duplicated_module = _LagoModule(copy.copy(self.leaves))
        return duplicated_module  # type: ignore
