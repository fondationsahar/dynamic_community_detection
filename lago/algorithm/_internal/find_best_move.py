from typing import NamedTuple

from . import lago_tools as lts
from ._lago_module import _LagoModule
from .delta_lm import (
    DeltaLongitudinalModularityComputer,
)
from .leaf_set import LeafDifference

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


class MoveChanges(NamedTuple):
    """Per-node duration changes a chosen move implies, as already evaluated.

    Either side is ``None`` when the evaluation path did not compute it (JM,
    k-partite sentinel values, or the reference path).
    """

    leaving: dict | None
    joining: dict | None


def find_best_module_for_submodule(
    delta_lm_computer: DeltaLongitudinalModularityComputer,
    submodule: _LagoModule,
    partite_mapping: dict[int, int] = {},
    modules: list[_LagoModule] | None = None,
    verbose: bool | int = 0,
    stopping_criterion: float = 0.0,
    level: str = "parent",
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
        level (str): Which module owns a leaf at this stage of the search:
            ``"parent"`` for ``leaf.module.parent`` (TMM / STNM levels),
            ``"module"`` for ``leaf.module`` (STEM).

    Returns:
        tuple: (
            Module: best module to which move submodule
            float: Δ L-Modularity gain from the move
            MoveChanges: the duration changes of that move, or None
        )
    """
    global _find_best_stats
    _find_best_stats["calls"] += 1

    if not delta_lm_computer.use_batch:
        return _find_best_reference(
            delta_lm_computer, submodule, partite_mapping, modules, verbose, stopping_criterion, level
        )

    parent = submodule.parent
    if parent is None:
        _find_best_stats["no_modules"] += 1
        return None, None, None

    M0_leaves = submodule.leaves
    M0_time_segments = lts.get_nodes_segment(module_leaves=M0_leaves)

    if modules is None and level == "parent":
        # A TMM/STNM level takes its candidates from the neighbours recorded by
        # compute_neighbors when the level started, not from the live adjacency.
        # The two differ once a STEM pass has moved leaves between modules, and
        # the recorded set is what the search has always used -- deriving the
        # candidates live changes 6 of 54 corpus partitions.
        modules = {module.parent for module in submodule.neighbors if module.parent is not None}

    candidates, leaving_delta, leaving_changes, evaluations = (
        delta_lm_computer.evaluate_candidates(
            M0_leaves, M0_time_segments, parent, level == "parent", partite_mapping, modules
        )
    )

    if not candidates:
        _find_best_stats["no_modules"] += 1
        return None, None, None
    if not evaluations:
        _find_best_stats["no_improvement"] += 1
        return None, None, None

    # Best gain; on an exact tie the candidate with the smallest creation index,
    # which is what taking the first maximum of an index-sorted list gave.
    best_module = None
    best_delta = None
    best_changes = None
    for module, joining_delta, joining_changes in evaluations:
        delta = leaving_delta + joining_delta
        if (
            best_module is None
            or delta > best_delta
            or (delta == best_delta and module.index < best_module.index)
        ):
            best_module, best_delta, best_changes = module, delta, joining_changes

    if best_delta <= stopping_criterion:
        # No move improves LM beyond stopping criterion, continue exploring...
        _find_best_stats["no_improvement"] += 1
        if verbose >= 4:
            _print_candidates(M0_leaves, parent, leaving_delta, evaluations, None, stopping_criterion)
        return None, None, None

    _find_best_stats["success"] += 1
    if verbose >= 4:
        _print_candidates(M0_leaves, parent, leaving_delta, evaluations, best_module, None)

    return best_module, best_delta, MoveChanges(leaving_changes, best_changes)


def _print_candidates(M0_leaves, parent, leaving_delta, evaluations, best_module, stopping_criterion):
    m0_nodes = sorted({leaf.node for leaf in M0_leaves})
    m1_nodes = sorted({leaf.node for leaf in parent.leaves if leaf not in M0_leaves})
    print(
        f"    [find_best] submodule nodes={m0_nodes}, parent(remaining)={m1_nodes}, "
        f"delta_leaving={leaving_delta:.6f}"
    )
    ranked = sorted(evaluations, key=lambda item: leaving_delta + item[1], reverse=True)
    for module, joining_delta, _ in ranked:
        mod_nodes = sorted({leaf.node for leaf in module.leaves})
        delta = leaving_delta + joining_delta
        if best_module is None:
            print(f"      candidate {mod_nodes}: delta_lm={delta:.6f} (rejected, <= {stopping_criterion})")
        else:
            marker = " <-- best" if module is best_module else ""
            print(f"      candidate {mod_nodes}: delta_lm={delta:.6f}{marker}")


def _find_best_reference(
    delta_lm_computer: DeltaLongitudinalModularityComputer,
    submodule: _LagoModule,
    partite_mapping: dict[int, int],
    modules: list[_LagoModule] | None,
    verbose: bool | int,
    stopping_criterion: float,
    level: str,
):
    """The per-candidate formulation, kept as the reference.

    This is the original body of :func:`find_best_module_for_submodule`: one
    :meth:`M0_to_Mx` call per candidate. ``benchmarks/compare_candidates.py``
    runs it against the batch path on every call of a real run.
    """
    if modules is None:
        # Get parents of submodule neighbors.
        # Sorted by creation index so that the candidate order -- and therefore
        # which module wins an exact tie below -- is a function of the data.
        if level == "parent":
            neighbors = submodule.neighbors
            modules = sorted(
                {module.parent for module in neighbors if module.parent is not None},
                key=lambda m: m.index,
            )
        else:
            modules = sorted(lts.get_neighbors_modules_parents(submodule), key=lambda m: m.index)

    # Exclude self parent from move options
    if modules and submodule.parent in modules:
        modules.remove(submodule.parent)

    if not modules:
        _find_best_stats["no_modules"] += 1
        return None, None, None

    M0_leaves = submodule.leaves
    M0_time_segments = lts.get_nodes_segment(
        module_leaves=M0_leaves,
    )
    if not submodule.parent:
        _find_best_stats["no_modules"] += 1
        return None, None, None

    # A view rather than a copy: the parent is the largest set in sight and the
    # leaving move only ever tests membership in it. See leaf_set.LeafDifference.
    M1_leaves = LeafDifference(submodule.parent.leaves, M0_leaves)

    # M0 is part of its parent, so M1 U M0 is exactly the parent's leaves: the
    # parent's memoised durations serve as the "union" side of the leaving move.
    delta_lm_M0_leaving_M1 = -delta_lm_computer.M0_to_Mx(
        M0_leaves=M0_leaves,
        M0_time_segments=M0_time_segments,
        Mx_leaves=M1_leaves,
        partite_mapping=partite_mapping,
        union_module=submodule.parent,
    )

    candidates_delta_lm: dict[_LagoModule, float] = {}

    for module in modules:
        # Candidate modules are other parents, and the modules partition the
        # leaves, so M0 is normally disjoint from them and the difference is a
        # plain copy. Checking costs O(|M0|); the copy costs O(|module|).
        disjoint = module.leaves.isdisjoint(M0_leaves)
        M2_leaves = module.leaves if disjoint else module.leaves - M0_leaves

        # Parents partition the leaves, so M1 and M2 are disjoint and can only be
        # equal when both are empty.
        if not M2_leaves and not M1_leaves:
            continue

        delta_lm_M0_joining_M2 = delta_lm_computer.M0_to_Mx(
            M0_leaves=M0_leaves,
            M0_time_segments=M0_time_segments,
            Mx_leaves=M2_leaves,
            partite_mapping=partite_mapping,
            # Only when M2 is the module untouched: otherwise the leaf set we
            # pass is not the one the module memoised.
            mx_module=module if disjoint else None,
        )

        candidates_delta_lm[module] = delta_lm_M0_leaving_M1 + delta_lm_M0_joining_M2

    if not candidates_delta_lm:
        _find_best_stats["no_improvement"] += 1
        return None, None, None

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
        return None, None, None

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

    return best_module, delta_lm, None
