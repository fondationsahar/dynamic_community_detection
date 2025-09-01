import copy
from typing import List, Self, Set, TypeAlias  # type: ignore

Leaves: TypeAlias = set


class Module:
    def __init__(self, leaves: Leaves):
        self.leaves: Leaves = leaves
        self.submodules: List[Module] = []
        self.parent: Module | None = None

        self.neighbors: Set[Module] = set()

    def compute_neighbors(self, subset: set = set()):
        leaves_neighbors = set()
        for leaf in self.leaves:
            leaves_neighbors |= set(
                [tmp_neighbor for tmp_neighbor in leaf.topo_neighbors]
            )
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
        duplicated_module = Module(copy.copy(self.leaves))
        return duplicated_module  # type: ignore
