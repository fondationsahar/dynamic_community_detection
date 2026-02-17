import copy

from lago.core.linkstream import LinkStream

from . import lago_tools as lts
from ._lago_module import _LagoModule
from ._leaf import Leaf
from .delta_lm import (
    DeltaLongitudinalModularityComputer,
)
from .find_best_move import (
    find_best_module_for_submodule,
)


class SingleTimeEdgeMover:
    kind: str = "STEM"

    def __init__(
        self,
        linkstream: LinkStream,
        fast_exploration: bool,
        modules: set[_LagoModule],
        delta_lm_computer: DeltaLongitudinalModularityComputer,
        stopping_criterion: float = 0.0,
    ) -> None:
        self.linkstream = linkstream  # NOTE Maybe overkill to have it here
        self.fast_exploration = fast_exploration
        self.modules = modules
        self.delta_lm_computer = delta_lm_computer
        self.stopping_criterion = stopping_criterion

    def run(self, verbose: bool | int = 0) -> float:
        """Single Time Edge Movements refinement strategy:
            1 - Each time nodes couple that interact in the same module are selected as
            submodules candidates for moving. It also includes single active time nodes.
            2 - submodules are moved between parents

        Args:
            verbose: Verbosity level for loop tracking (3+ enables tracking).

        Returns:
            float: Δ L-Modularity between initial state and final state.
        """
        edges_iterator = self._build_stem_iterator(verbose)

        delta_longitudinal_modularity = 0
        move = True
        outer_loop_iteration = 0
        max_iterations_warning = 10000  # Higher threshold for inner exploration loops
        moves_made = 0  # Track number of actual moves

        # Cache to prevent oscillating moves: (module_id, from_parent, to_parent) -> gain
        move_cache: dict[tuple[int, int, int], float] = {}
        blocked_moves = 0  # Count of moves blocked by cache

        while move:
            outer_loop_iteration += 1

            # Track potential infinite loops when verbose >= 3
            if verbose >= 3:
                if outer_loop_iteration == 1:
                    print("[LOOP TRACKING] STEM exploration outer loop starting")
                elif outer_loop_iteration == 10:
                    print(
                        "[WARNING] STEM outer loop reached 10 iterations - moves may be getting undone/redone"
                    )
                elif outer_loop_iteration == 50:
                    print(
                        "[WARNING] STEM outer loop reached 50 iterations - likely stuck in repetitive moves!"
                    )
                elif outer_loop_iteration % 500 == 0:
                    print(
                        f"[LOOP TRACKING] STEM exploration outer loop: iteration {outer_loop_iteration}"
                    )
                if outer_loop_iteration >= max_iterations_warning:
                    print(
                        f"[WARNING] STEM exploration outer loop exceeded {max_iterations_warning} iterations - possible infinite loop!"
                    )

            move = False
            tmp_edges = copy.copy(edges_iterator)
            inner_loop_iteration = 0
            iteration_moves = 0  # Track moves in this iteration

            if verbose >= 3:
                print(
                    f"[LOOP TRACKING] STEM inner loop starting with {len(tmp_edges)} edges to process"
                )

            while tmp_edges:
                inner_loop_iteration += 1

                # Heartbeat every 5000 edges to show progress
                if verbose >= 3 and inner_loop_iteration % 5000 == 0:
                    print(
                        f"[LOOP TRACKING] STEM inner loop heartbeat: processed {inner_loop_iteration} edges, {len(tmp_edges)} remaining, {iteration_moves} moves made"
                    )

                child_edge = tmp_edges.pop()

                # Only move edges that are inside the same module
                # Condition must remain here because tmp_edges may overlap
                # and time node may change affiliation during the exploration loop
                # for tedg in child_edge:
                #     if type(tedg) is not Leaf:
                #         print(tedg)
                #         print(tedg.target)
                if child_edge[0].module != child_edge[1].module:
                    continue

                child_module = lts.create_module_from_leaves(child_edge)

                neighbors_modules = list(lts.get_neighbors_modules_parents(child_module))
                best_module, delta_lm = find_best_module_for_submodule(
                    self.delta_lm_computer,
                    child_module,
                    self.linkstream.partite_mapping,
                    neighbors_modules,
                    stopping_criterion=self.stopping_criterion * self.linkstream.weight,
                )

                if not best_module or not delta_lm:
                    continue

                # Check cache to prevent oscillating moves
                old_parent_id = id(child_module.parent) if child_module.parent else 0
                move_key = (id(child_module), old_parent_id, id(best_module))
                reverse_key = (id(child_module), id(best_module), old_parent_id)

                # If reverse move was made previously, only allow this move if gain is strictly higher
                if reverse_key in move_cache:
                    previous_reverse_gain = move_cache[reverse_key]
                    if delta_lm <= previous_reverse_gain:
                        # Block this move - it would just oscillate
                        blocked_moves += 1
                        if verbose >= 3:
                            print(
                                f"[CACHE] Blocked move: Edge[id:{id(child_module)}] from {old_parent_id} to {id(best_module)}, "
                                f"gain {delta_lm:.6e} <= previous reverse gain {previous_reverse_gain:.6e}"
                            )
                        continue

                # Log the move details when verbose >= 3
                if verbose >= 3:
                    # Create identifier for the edge/module
                    edge_identifier = (
                        f"Edge[{len(child_module.leaves)} leaves, id:{id(child_module)}]"
                    )
                    print(
                        f"[MOVE] {edge_identifier} from parent {old_parent_id} -> parent {id(best_module)}, delta_lm: {delta_lm:.6e}"
                    )

                delta_longitudinal_modularity += delta_lm
                iteration_moves += 1
                moves_made += 1

                # Update cache with this move
                move_cache[move_key] = delta_lm

                self._update_affiliation_after_stem(child_module, best_module)

                if self.fast_exploration:
                    tmp_edges |= self._update_fast_iteration_exploration_set_for_stem(
                        child_edge, best_module
                    )
                else:
                    move = True

            if verbose >= 3:
                print(
                    f"[LOOP TRACKING] STEM exploration inner loop processed {inner_loop_iteration} edges, made {iteration_moves} moves"
                )
                if blocked_moves > 0:
                    print(f"[CACHE] Blocked {blocked_moves} oscillating moves so far")

        if verbose >= 3:
            print(
                f"[LOOP TRACKING] STEM exploration outer loop completed after {outer_loop_iteration} iterations, total moves: {moves_made}"
            )
            print(
                f"[LOOP TRACKING] STEM total delta L-Modularity: {delta_longitudinal_modularity:.10e}"
            )
            print(
                f"[CACHE] Total moves in cache: {len(move_cache)}, total blocked: {blocked_moves}"
            )

        return delta_longitudinal_modularity

    def _build_stem_iterator(self, verbose: bool | int = 0) -> set[tuple[Leaf, Leaf]]:
        """Prepare exploration set for the STEM refinement strategy.
            Time nodes and couple of active time nodes that interact
            together are added to the exploration set.

            NOTE Impossible to manipulate submodules as Module instances
            here because actual submodules overlap which results in
            affiliations collides during exploration.

        Args:
            verbose: Verbosity level for loop tracking (3+ enables tracking).

        Returns:
            set: submodules set on which iterate for the STEM refinement
        """

        leaves = set(self.linkstream.leaves_dict.values())
        stem_iterator = set()
        iteration_count = 0

        while leaves:
            iteration_count += 1
            leaf = leaves.pop()
            # Add self time node
            stem_iterator.add((leaf, leaf))
            # Add all topological neighbors
            for topo_neighbor in leaf.topo_neighbors:
                tmp_edge = [leaf, topo_neighbor.target]
                # Sort to avoid duplicates
                tmp_edge.sort(key=id)
                stem_iterator.add(tuple(tmp_edge))

        if verbose >= 3:
            print(
                f"[LOOP TRACKING] STEM iterator built with {iteration_count} leaves, {len(stem_iterator)} edges"
            )

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
                    new_edge = [neighbor, topo_neighbor.target]
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
