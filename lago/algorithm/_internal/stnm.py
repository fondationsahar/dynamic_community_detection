from ._lago_module import _LagoModule
from .delta_lm import (
    DeltaLongitudinalModularityComputer,
)
from .tmm import TimeModuleMover


class SingleTimeNodeMover(TimeModuleMover):
    kind: str = "STNM"

    def __init__(
        self,
        fast_exploration: bool,
        modules: set[_LagoModule],
        delta_lm_computer: DeltaLongitudinalModularityComputer,
        partite_mapping: dict[int, int],
        stopping_criterion: float = 0.0,
    ) -> None:
        super().__init__(
            fast_exploration,
            modules,
            delta_lm_computer,
            partite_mapping,
            stopping_criterion,
        )

    def run(
        self,
        verbose: bool | int = 0,
    ) -> float:
        """Single Time Node Movements refinement strategy:
            1 - each active time node is assign to its own module and
            affiliated to its previous module as a parent module.
            2 - submodules (active time nodes) are moved between parent
            modules as long as it optimizes L-Modularity

        Args:
            verbose: Verbosity level for loop tracking (3+ enables tracking).

        Returns:
            float: Δ L-Modularity between initial state and final state.
        """

        leaves_modules = self._stnm_iterator()
        delta_longitudinal_modularity = self._exploration_loop(leaves_modules, verbose)

        self._update_modules_after_stnm(leaves_modules)

        return delta_longitudinal_modularity

    def _stnm_iterator(self) -> set[_LagoModule]:
        """Initiate Single Time Node Movements strategy by
                1 - preparing time nodes (leaves) set on which iterate;
                2 - ensuring appropriate affiliations to parent
                    modules for movements.

        Returns:
            set: time nodes on which iterate for
                Single Time Node Movements strategy.
        """
        leaves_modules = set[_LagoModule]()
        for module in self.modules:
            for leaf in module.leaves:
                # Create new module for single time node 'leaf'
                tmp_module = _LagoModule({leaf})
                # Assign new module to the parent module
                tmp_module.parent = module
                # Assign the leaf to its own module
                leaf.module = tmp_module
                # Add the leaf module to the submodules
                # list of its parent module
                module.submodules.append(tmp_module)
                leaves_modules.add(tmp_module)

        # Compute leaves modules neighbors
        # to fasten exploration loops
        for module in leaves_modules:
            module.compute_neighbors()

        return leaves_modules

    def _update_modules_after_stnm(self, leaves_modules) -> None:
        """Update modules after Single Time Node Movements refinement strategy.
            Parent modules are reset and leaves affiliation are updated.

        Args:
            leaves_modules (set): ative time nodes as modules
        """
        self.modules = set[_LagoModule]()
        for module in leaves_modules:
            for leaf in module.leaves:
                leaf.module = module.parent
            self.modules.add(module.parent)
        for module in self.modules:
            module.compute_neighbors()
