import copy

import lago.lago_src.lago_tools as lts
from lago.lago_src.delta_lm import (
    DeltaLongitudinalModularityComputer,
)
from lago.lago_src.find_best_move import (
    find_best_module_for_submodule,
)
from lago.module import Module


class TimeModuleMover:
    def __init__(
        self,
        fast_exploration: bool,
        modules: set[Module],
        delta_lm_computer: DeltaLongitudinalModularityComputer,
    ) -> None:
        self.fast_exploration = fast_exploration
        self.modules = modules
        self.delta_lm_computer = delta_lm_computer

    def run(self) -> float:
        """Optimize one level of Recursive Time Module Mover.
            For each module, a similar parent module is created,
            with affiliation. Then, each module is moved to the
            parent neighbor module that best increases L-Modularity.

        Returns:
            float: Δ L-Modularity between initial state and final state.
        """
        self._create_parent_modules()
        delta_longitudinal_modularity = self._exploration_loop(self.modules)

        self._update_modules()

        return delta_longitudinal_modularity

    def _exploration_loop(self, submodules: set[Module]) -> float:
        delta_longitudinal_modularity = 0
        move = True
        while move:
            move = False
            tmp_leaves_modules = copy.copy(submodules)
            while tmp_leaves_modules:
                child_module = tmp_leaves_modules.pop()

                best_module, delta_lm = find_best_module_for_submodule(
                    self.delta_lm_computer, child_module
                )

                if not best_module or not delta_lm or not child_module.parent:
                    continue

                lts.move_submodule(child_module, child_module.parent, best_module)

                delta_longitudinal_modularity += delta_lm

                if self.fast_exploration:
                    tmp_leaves_modules = lts.update_fast_iteration_exploration_set(
                        child_module, tmp_leaves_modules
                    )
                else:
                    move = True

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
