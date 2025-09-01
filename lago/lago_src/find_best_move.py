from typing import List

import lago.lago_src.lago_tools as lts
from lago.lago_src.delta_lm import (
    DeltaLongitudinalModularityComputer,
)
from lago.module import Module


def find_best_module_for_submodule(
    delta_lm_computer: DeltaLongitudinalModularityComputer,
    submodule: Module,
    modules: List[Module] | None = None,
):
    """Find best module to move the submodule to.
    First compute the gain of moving Submodule M0 from its module M1,
    then the gain for it to join each candidate module M2.
    The move that increases L-Modularity the best is applied.

    Args:
        submodule (Module): Submodule to move
        modules (List[Module], optional): Candidates
            parent modules between which to move submodule.
            If not specified, parents of the submodule
            neighbors are considered. Defaults to None.

    Returns:
        tuple: (
            Module: best module to which move submodule
            float: Δ L-Modularity gain from the move
        )
    """
    if modules is None:
        # Get parents of submodule neighbors
        neighbors = submodule.neighbors
        modules = list(
            set([module.parent for module in neighbors if module.parent is not None])
        )

    # Exclude self parent from move options
    if modules and submodule.parent in modules:
        modules.remove(submodule.parent)

    if not modules:
        return None, None

    M0_leaves = submodule.leaves
    M0_time_segments = lts.get_nodes_segment(
        module_leaves=M0_leaves,
    )
    if not submodule.parent:
        return None, None

    M1_leaves = submodule.parent.leaves - M0_leaves

    delta_lm_M0_leaving_M1 = -delta_lm_computer.M0_to_Mx(
        M0_leaves=M0_leaves,
        M0_time_segments=M0_time_segments,
        Mx_leaves=M1_leaves,
    )

    candidates_delta_lm: dict[Module, float] = {}

    for module in modules:
        M2_leaves = module.leaves - M0_leaves

        if M1_leaves == M2_leaves:
            continue

        delta_lm_M0_joining_M2 = delta_lm_computer.M0_to_Mx(
            M0_leaves=M0_leaves,
            M0_time_segments=M0_time_segments,
            Mx_leaves=M2_leaves,
        )

        candidates_delta_lm[module] = delta_lm_M0_leaving_M1 + delta_lm_M0_joining_M2

    if not candidates_delta_lm:
        return None, None

    delta_lm = max(candidates_delta_lm.values())
    if delta_lm <= 0:
        # No move improves LM, continue exploring...
        return None, None

    best_module = [
        module for module, dlm in candidates_delta_lm.items() if dlm == delta_lm
    ][0]

    return best_module, delta_lm
