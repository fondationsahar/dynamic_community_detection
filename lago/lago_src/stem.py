import copy

import lago.lago_src.lago_tools as lts
from lago.lago_src.delta_lm import (
    DeltaLongitudinalModularityComputer,
)
from lago.lago_src.find_best_move import (
    find_best_module_for_submodule,
)
from lago.leaf import Leaf
from lago.linkstream import LinkStream
from lago.module import Module


class SingleTimeEdgeMover:
    kind: str = "STEM"

    def __init__(
        self,
        linkstream: LinkStream,
        fast_exploration: bool,
        modules: set[Module],
        delta_lm_computer: DeltaLongitudinalModularityComputer,
    ) -> None:
        self.linkstream = linkstream  # NOTE Maybe overkill to have it here
        self.fast_exploration = fast_exploration
        self.modules = modules
        self.delta_lm_computer = delta_lm_computer

    def run(self) -> float:
        """Single Time Edge Movements refinement strategy:
            1 - Each time nodes couple that interact in the same module are selected as
            submodules candidates for moving. It also includes single active time nodes.
            2 - submodules are moved between parents

        Returns:
            float: Δ L-Modularity between initial state and final state.
        """
        edges_iterator = self._build_stem_iterator()

        delta_longitudinal_modularity = 0
        move = True
        while move:
            move = False
            tmp_edges = copy.copy(edges_iterator)
            while tmp_edges:
                child_edge = tmp_edges.pop()

                # Only move edges that are inside the same module
                # Condition must remain here because tmp_edges may overlap
                # and time node may change affiliation during the exploration loop
                if child_edge[0].module != child_edge[1].module:
                    continue

                child_module = lts.create_module_from_leaves(child_edge)

                neighbors_modules = list(
                    lts.get_neighbors_modules_parents(child_module)
                )
                best_module, delta_lm = find_best_module_for_submodule(
                    self.delta_lm_computer,
                    child_module,
                    neighbors_modules,
                )

                if not best_module or not delta_lm:
                    continue

                delta_longitudinal_modularity += delta_lm

                self._update_affiliation_after_stem(child_module, best_module)

                if self.fast_exploration:
                    tmp_edges |= self._update_fast_iteration_exploration_set_for_stem(
                        child_edge, best_module
                    )
                else:
                    move = True

        return delta_longitudinal_modularity

    def _build_stem_iterator(self) -> set[tuple[Leaf, Leaf]]:
        """Prepare exploration set for the STEM refinement strategy.
            Time nodes and couple of active time nodes that interact
            together are added to the exploration set.

            NOTE Impossible to manipulate submodules as Module instances
            here because actual submodules overlap which results in
            affiliations collides during exploration.

        Returns:
            set: submodules set on which iterate for the STEM refinement
        """

        leaves = set(self.linkstream.leaves_dict.values())
        stem_iterator = set()

        while leaves:
            leaf = leaves.pop()
            # Add self time node
            stem_iterator.add((leaf, leaf))
            # Add all topological neighbors
            for topo_neighbor in leaf.topo_neighbors:
                tmp_edge = [leaf, topo_neighbor]
                # Sort to avoid duplicates
                tmp_edge.sort(key=id)
                stem_iterator.add(tuple(tmp_edge))

        return stem_iterator

    def _update_fast_iteration_exploration_set_for_stem(
        self,
        child_edge,
        best_module,
    ) -> set[tuple[Leaf, Leaf]]:
        """Udpate exploration set by adding neighbors of the child module.

        Args:
            child_edge (tuple): time nodes couples that have been moved
            best_module (_type_): _description_

        Returns:
            set: child edges to explore as candidates for moving.
        """
        other_edges = set()
        for leaf in child_edge:
            for neighbor in leaf.neighbors:
                # Ignore edge leaf
                if neighbor in child_edge:
                    continue
                # Ignore neighbor in same module
                neighbor_module = neighbor.module
                if neighbor_module == best_module:
                    continue
                # Add self time node as edge to explore
                new_edge = [neighbor, neighbor]
                other_edges.add(tuple(new_edge))
                # Add all time edges
                for topo_neighbor in neighbor.topo_neighbors:
                    new_edge = [neighbor, topo_neighbor]
                    # Sort to avoid duplicates
                    new_edge.sort(key=id)
                    other_edges.add(tuple(new_edge))

        return other_edges

    def _update_affiliation_after_stem(self, child_module, affiliation_module) -> None:
        """Update affiliations after a time edge move.

        Args:
            child_module (Module): submodule to change affiliation.
            affiliation_module (_type_): new affiliation module for child_module.
        """
        for leaf in child_module.leaves:
            child_module.parent.leaves.remove(leaf)
            leaf.module = affiliation_module
            affiliation_module.leaves.add(leaf)
