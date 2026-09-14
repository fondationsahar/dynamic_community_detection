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
import itertools
from typing import TYPE_CHECKING, Self, TypeAlias

if TYPE_CHECKING:
    from ._leaf import Leaf

Leaves: TypeAlias = set

# Monotonic creation index, used as the hash of a module so that the iteration
# order of a set of modules is a function of the order in which the algorithm
# created them, not of the memory allocator. It MUST be restarted at the
# beginning of each LAGO run: a process-global counter gives the second run in a
# process different indices for the same modules, hence a different set layout,
# hence a different exploration order.
_module_counter = itertools.count()


def reset_module_counter() -> None:
    """Restart the module creation index. Called once per LAGO run."""
    global _module_counter
    _module_counter = itertools.count()


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
        # Creation index: stable identity for hashing and for cache keys.
        # Modules are compared by identity (the default __eq__), and distinct
        # modules always get distinct indices, so this is a consistent hash.
        self.index: int = next(_module_counter)
        # Memoised per-node durations of this module's leaves, and the degree
        # aggregates derived from them. Every candidate move needs both, and a
        # module only changes when a move is accepted -- hundreds of evaluations
        # per accepted move -- so they are recomputed on mutation rather than
        # maintained incrementally, which keeps them free of drift.
        # See invalidate_durations().
        self._durations: dict[int, float] | None = None
        self._durations_size: int = -1
        self._mm_sums: tuple | None = None

    def __hash__(self) -> int:
        return self.index

    def get_durations(self, compute) -> dict[int, float]:
        """Per-node durations of this module's leaves, memoised.

        Args:
            compute: Callable taking the leaves set and returning the durations.

        Returns:
            The durations mapping. Do not mutate it -- it is shared.
        """
        if self._durations is None or self._durations_size != len(self.leaves):
            self._durations = compute(self.leaves)
            self._durations_size = len(self.leaves)
            self._mm_sums = None
        return self._durations

    def invalidate_durations(self) -> None:
        """Drop the memoised durations. Call after changing ``leaves``."""
        self._durations = None
        self._durations_size = -1
        self._mm_sums = None

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
            leaves_neighbors |= {tmp_neighbor.target for tmp_neighbor in leaf.topo_neighbors}
            right_time_neighb = leaf.right_time_active_neighbor
            if right_time_neighb is not None:
                leaves_neighbors.add(right_time_neighb)
            left_time_neighb = leaf.left_time_active_neighbor
            if left_time_neighb is not None:
                leaves_neighbors.add(left_time_neighb)

        if subset:
            self.neighbors = {leaf.module for leaf in leaves_neighbors & subset if leaf.module is not None}
        else:
            self.neighbors = {leaf.module for leaf in leaves_neighbors if leaf.module is not None}

    def duplicates(self) -> Self:
        """Create a shallow copy of this module.

        Returns:
            A new _LagoModule with a copy of the leaves set.
            Note: The leaves themselves are not copied.
        """
        duplicated_module = _LagoModule(copy.copy(self.leaves))
        return duplicated_module  # type: ignore
