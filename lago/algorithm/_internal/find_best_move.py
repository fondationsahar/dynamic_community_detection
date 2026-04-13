from . import lago_tools as lts
from ._lago_module import _LagoModule
from .delta_lm import (
    DeltaLongitudinalModularityComputer,
)

# Global counters for tracking (only used when verbose >= 4)
_find_best_stats = {
    "calls": 0,
    "no_modules": 0,
    "no_improvement": 0,
    "success": 0,
}


def reset_find_best_stats():
    """Reset statistics counters."""
    global _find_best_stats
    _find_best_stats = {
        "calls": 0,
        "no_modules": 0,
        "no_improvement": 0,
        "success": 0,
    }


def get_find_best_stats():
    """Get current statistics."""
    return _find_best_stats.copy()


def find_best_module_for_submodule(
    delta_lm_computer: DeltaLongitudinalModularityComputer,
    submodule: _LagoModule,
    partite_mapping: dict[int, int] = {},
    modules: list[_LagoModule] | None = None,
    verbose: bool | int = 0,
    stopping_criterion: float = 0.0,
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
        verbose (bool | int, optional): Verbosity level for tracking.

    Returns:
        tuple: (
            Module: best module to which move submodule
            float: Δ L-Modularity gain from the move
        )
    """
    global _find_best_stats
    _find_best_stats["calls"] += 1

    if modules is None:
        # Get parents of submodule neighbors
        neighbors = submodule.neighbors
        modules = list({module.parent for module in neighbors if module.parent is not None})

    # Exclude self parent from move options
    if modules and submodule.parent in modules:
        modules.remove(submodule.parent)

    if not modules:
        _find_best_stats["no_modules"] += 1
        return None, None

    M0_leaves = submodule.leaves
    M0_time_segments = lts.get_nodes_segment(
        module_leaves=M0_leaves,
    )
    if not submodule.parent:
        _find_best_stats["no_modules"] += 1
        return None, None

    M1_leaves = submodule.parent.leaves - M0_leaves

    # NOTE maybe could be optimized because if M1_leaves is empty
    # no computing is needed. Check that.
    delta_lm_M0_leaving_M1 = -delta_lm_computer.M0_to_Mx(
        M0_leaves=M0_leaves,
        M0_time_segments=M0_time_segments,
        Mx_leaves=M1_leaves,
        partite_mapping=partite_mapping,
    )

    candidates_delta_lm: dict[_LagoModule, float] = {}

    for module in modules:
        M2_leaves = module.leaves - M0_leaves

        if M1_leaves == M2_leaves:
            continue

        delta_lm_M0_joining_M2 = delta_lm_computer.M0_to_Mx(
            M0_leaves=M0_leaves,
            M0_time_segments=M0_time_segments,
            Mx_leaves=M2_leaves,
            partite_mapping=partite_mapping,
        )

        candidates_delta_lm[module] = delta_lm_M0_leaving_M1 + delta_lm_M0_joining_M2

    if not candidates_delta_lm:
        _find_best_stats["no_improvement"] += 1
        return None, None

    delta_lm = max(candidates_delta_lm.values())
    if delta_lm <= stopping_criterion:
        # No move improves LM beyond stopping criterion, continue exploring...
        _find_best_stats["no_improvement"] += 1
        if verbose >= 4:
            m0_nodes = sorted({leaf.node for leaf in M0_leaves})
            print(
                f"    [find_best] submodule nodes={m0_nodes}, delta_leaving={delta_lm_M0_leaving_M1:.6f}"
            )
            for mod, dlm in sorted(candidates_delta_lm.items(), key=lambda x: x[1], reverse=True):
                mod_nodes = sorted({leaf.node for leaf in mod.leaves})
                print(
                    f"      candidate {mod_nodes}: delta_lm={dlm:.6f} (rejected, <= {stopping_criterion})"
                )
        return None, None

    best_module = [module for module, dlm in candidates_delta_lm.items() if dlm == delta_lm][0]
    _find_best_stats["success"] += 1

    if verbose >= 4:
        m0_nodes = sorted({leaf.node for leaf in M0_leaves})
        m1_nodes = sorted({leaf.node for leaf in M1_leaves})
        print(
            f"    [find_best] submodule nodes={m0_nodes}, parent(remaining)={m1_nodes}, delta_leaving={delta_lm_M0_leaving_M1:.6f}"
        )
        for mod, dlm in sorted(candidates_delta_lm.items(), key=lambda x: x[1], reverse=True):
            mod_nodes = sorted({leaf.node for leaf in mod.leaves})
            marker = " <-- best" if mod is best_module else ""
            print(f"      candidate {mod_nodes}: delta_lm={dlm:.6f}{marker}")

    return best_module, delta_lm
