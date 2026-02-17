import copy

from . import lago_tools as lts
from ._lago_module import _LagoModule
from .delta_lm import (
    DeltaLongitudinalModularityComputer,
)
from .find_best_move import (
    find_best_module_for_submodule,
    get_find_best_stats,
    reset_find_best_stats,
)


class TimeModuleMover:
    def __init__(
        self,
        fast_exploration: bool,
        modules: set[_LagoModule],
        delta_lm_computer: DeltaLongitudinalModularityComputer,
        partite_mapping: dict[int, int],
        stopping_criterion: float = 0.0,
    ) -> None:
        self.fast_exploration = fast_exploration
        self.modules = modules
        self.delta_lm_computer = delta_lm_computer
        self.partite_mapping = partite_mapping
        self.stopping_criterion = stopping_criterion

    def run(self, verbose: bool | int = 0) -> float:
        """Optimize one level of Recursive Time Module Mover.
            For each module, a similar parent module is created,
            with affiliation. Then, each module is moved to the
            parent neighbor module that best increases L-Modularity.

        Args:
            verbose: Verbosity level for loop tracking (3+ enables tracking).

        Returns:
            float: Δ L-Modularity between initial state and final state.
        """
        self._create_parent_modules()
        delta_longitudinal_modularity = self._exploration_loop(self.modules, verbose)

        self._update_modules()

        return delta_longitudinal_modularity

    def _exploration_loop(self, submodules: set[_LagoModule], verbose: bool | int = 0) -> float:
        delta_longitudinal_modularity = 0
        move = True
        outer_loop_iteration = 0
        max_iterations_warning = 10000  # Higher threshold for inner exploration loops
        moves_made = 0  # Track number of actual moves

        # Track module processing to detect repeated processing
        module_process_count: dict[int, int] = {}  # module id -> count

        # Cache to prevent oscillating moves: (module_id, from_parent, to_parent) -> gain
        move_cache: dict[tuple[int, int, int], float] = {}
        blocked_moves = 0  # Count of moves blocked by cache

        # Track moves to detect cyclic behavior (only when verbose >= 3)
        if verbose >= 3:
            move_history: list[
                tuple[int, int, int]
            ] = []  # (module_id, from_parent_id, to_parent_id)
            reverse_move_count = 0  # Count of moves that reverse a previous move

        while move:
            outer_loop_iteration += 1

            # Reset stats for this iteration
            if verbose >= 3:
                reset_find_best_stats()

            # Track potential infinite loops when verbose >= 3
            if verbose >= 3:
                if outer_loop_iteration == 1:
                    print("[LOOP TRACKING] TMM exploration outer loop starting")
                elif outer_loop_iteration == 10:
                    print(
                        "[WARNING] TMM outer loop reached 10 iterations - moves may be getting undone/redone"
                    )
                elif outer_loop_iteration == 50:
                    print(
                        "[WARNING] TMM outer loop reached 50 iterations - likely stuck in repetitive moves!"
                    )
                elif outer_loop_iteration % 500 == 0:
                    print(
                        f"[LOOP TRACKING] TMM exploration outer loop: iteration {outer_loop_iteration}"
                    )
                if outer_loop_iteration >= max_iterations_warning:
                    print(
                        f"[WARNING] TMM exploration outer loop exceeded {max_iterations_warning} iterations - possible infinite loop!"
                    )

            move = False
            tmp_leaves_modules = copy.copy(submodules)
            inner_loop_iteration = 0
            iteration_moves = 0  # Track moves in this iteration

            if verbose >= 3:
                print(
                    f"[LOOP TRACKING] TMM inner loop starting with {len(tmp_leaves_modules)} modules to process"
                )

            while tmp_leaves_modules:
                inner_loop_iteration += 1

                child_module = tmp_leaves_modules.pop()

                # Track how many times each module is processed
                module_id = id(child_module)
                module_process_count[module_id] = module_process_count.get(module_id, 0) + 1

                # Heartbeat every 1000 modules to show progress
                if verbose >= 3 and inner_loop_iteration % 1000 == 0:
                    # Calculate how many modules have been processed multiple times
                    reprocessed = sum(1 for count in module_process_count.values() if count > 1)
                    max_process = max(module_process_count.values()) if module_process_count else 0
                    print(
                        f"[LOOP TRACKING] TMM inner loop heartbeat: processed {inner_loop_iteration} modules, {len(tmp_leaves_modules)} remaining, {iteration_moves} moves made"
                    )
                    print(
                        f"[LOOP TRACKING] TMM reprocessed modules: {reprocessed}/{len(module_process_count)}, max times: {max_process}"
                    )

                best_module, delta_lm = find_best_module_for_submodule(
                    self.delta_lm_computer,
                    child_module,
                    self.partite_mapping,
                    verbose=verbose,
                    stopping_criterion=self.stopping_criterion,
                )

                if not best_module or not delta_lm or not child_module.parent:
                    continue

                old_parent = child_module.parent

                # Check cache to prevent oscillating moves
                move_key = (id(child_module), id(old_parent), id(best_module))
                reverse_key = (id(child_module), id(best_module), id(old_parent))

                # If reverse move was made previously, only allow this move if gain is strictly higher
                if reverse_key in move_cache:
                    previous_reverse_gain = move_cache[reverse_key]
                    if delta_lm <= previous_reverse_gain:
                        # Block this move - it would just oscillate
                        blocked_moves += 1
                        if verbose >= 3:
                            print(
                                f"[CACHE] Blocked move: Module[id:{id(child_module)}] from {id(old_parent)} to {id(best_module)}, "
                                f"gain {delta_lm:.6e} <= previous reverse gain {previous_reverse_gain:.6e}"
                            )
                        continue

                # Log the move details and detect cyclic behavior when verbose >= 3
                if verbose >= 3:
                    # Create a simple identifier for the module using its leaves
                    module_identifier = (
                        f"Module[{len(child_module.leaves)} leaves, id:{id(child_module)}]"
                    )
                    move_tuple = (id(child_module), id(old_parent), id(best_module))

                    # Check if this is a reverse of a recent move
                    reverse_move = (id(child_module), id(best_module), id(old_parent))
                    if reverse_move in move_history:
                        reverse_move_count += 1
                        print(
                            f"[MOVE] {module_identifier} from parent {id(old_parent)} -> parent {id(best_module)}, delta_lm: {delta_lm:.6e} [REVERSE MOVE #{reverse_move_count}]"
                        )
                    else:
                        print(
                            f"[MOVE] {module_identifier} from parent {id(old_parent)} -> parent {id(best_module)}, delta_lm: {delta_lm:.6e}"
                        )

                    move_history.append(move_tuple)

                lts.move_submodule(child_module, child_module.parent, best_module)

                # Update cache with this move
                move_cache[move_key] = delta_lm

                delta_longitudinal_modularity += delta_lm
                iteration_moves += 1
                moves_made += 1

                if self.fast_exploration:
                    set_size_before = len(tmp_leaves_modules)
                    tmp_leaves_modules = lts.update_fast_iteration_exploration_set(
                        child_module, tmp_leaves_modules
                    )
                    added_count = len(tmp_leaves_modules) - set_size_before

                    # Warn if we're adding more modules than we're removing
                    if verbose >= 3 and inner_loop_iteration % 1000 == 0:
                        print(
                            f"[LOOP TRACKING] TMM last move: module moved from parent {id(old_parent)} to {id(best_module)}, added {added_count} neighbors back"
                        )
                else:
                    move = True

            if verbose >= 3:
                reprocessed = sum(1 for count in module_process_count.values() if count > 1)
                max_process = max(module_process_count.values()) if module_process_count else 0
                avg_process = (
                    sum(module_process_count.values()) / len(module_process_count)
                    if module_process_count
                    else 0
                )
                print(
                    f"[LOOP TRACKING] TMM exploration inner loop processed {inner_loop_iteration} modules, made {iteration_moves} moves"
                )
                print(
                    f"[LOOP TRACKING] TMM iteration summary: {reprocessed}/{len(module_process_count)} modules reprocessed, max: {max_process}, avg: {avg_process:.1f}"
                )
                if blocked_moves > 0:
                    print(f"[CACHE] Blocked {blocked_moves} oscillating moves in this iteration")

                # Report move history stats
                if iteration_moves > 0:
                    reverse_pct = 100 * reverse_move_count / iteration_moves
                    print(
                        f"[LOOP TRACKING] TMM reverse moves: {reverse_move_count}/{iteration_moves} ({reverse_pct:.1f}%)"
                    )
                    if reverse_move_count > 0:
                        print(
                            f"[WARNING] TMM: Detected {reverse_move_count} reverse moves - modules moving back and forth!"
                        )

                # Report find_best_module_for_submodule stats
                stats = get_find_best_stats()
                print(
                    f"[LOOP TRACKING] TMM find_best stats: {stats['success']} successful moves, {stats['no_improvement']} no improvement, {stats['no_modules']} no candidates"
                )
                if stats["calls"] > 0:
                    success_rate = 100 * stats["success"] / stats["calls"]
                    print(
                        f"[LOOP TRACKING] TMM success rate: {success_rate:.1f}% ({stats['success']}/{stats['calls']} calls)"
                    )

                # Warn if excessive reprocessing detected
                if max_process > 100:
                    print(
                        f"[WARNING] TMM: Some modules processed {max_process} times - likely cyclic moves!"
                    )
                elif avg_process > 10:
                    print(
                        f"[WARNING] TMM: Average {avg_process:.1f} processes per module - excessive reprocessing!"
                    )

        if verbose >= 3:
            total_unique = len(module_process_count)
            total_processes = sum(module_process_count.values())
            print(
                f"[LOOP TRACKING] TMM exploration outer loop completed after {outer_loop_iteration} iterations, total moves: {moves_made}"
            )
            print(
                f"[LOOP TRACKING] TMM total modules: {total_unique} unique, {total_processes} total processes (avg: {total_processes / total_unique:.1f}x per module)"
            )
            print(
                f"[LOOP TRACKING] TMM total delta L-Modularity: {delta_longitudinal_modularity:.10e}"
            )
            # print(
            #     f"[CACHE] Total moves in cache: {len(move_cache)}, total blocked: {blocked_moves}"
            # )

        return delta_longitudinal_modularity

    def _create_parent_modules(self) -> None:
        """Creates parent modules for each module,
        and affiliates each module to its parent.
        """
        for module in self.modules:
            parent_module = module.duplicates()
            module.parent = parent_module
            parent_module.submodules.append(module)

    def _update_modules(self) -> None:
        """Turns parent modules into regular modules,
        and update affiliations.
        Applied after a level optimisation.
        """
        parent_modules = set(
            [module.parent for module in self.modules if module.parent is not None]
        )
        for module in parent_modules:
            # Remove former modules
            for submodule in module.submodules:
                if submodule in self.modules:
                    self.modules.remove(submodule)
            # Parent modules become new modules
            module.submodules = []
            for leaf in module.leaves:
                leaf.module = module
                self.modules.add(module)
        for module in self.modules:
            module.compute_neighbors()
