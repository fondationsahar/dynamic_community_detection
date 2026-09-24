"""Per-call check of the batch candidate evaluation against the reference.

``find_best_module_for_submodule`` evaluates every candidate of a move in one
pass over M0 (``DeltaLongitudinalModularityComputer.evaluate_candidates``). The
original formulation -- one ``M0_to_Mx`` call per candidate, candidates found
through ``compute_neighbors`` / ``get_neighbors_modules_parents`` -- is kept as
the reference. This runs both on every call of a real run and requires:

* the same candidate set,
* the same leaving delta and the same total per candidate, **bit for bit** --
  the batch path performs the same operations in the same order, so equality
  is exact and any tolerance would hide a reordering,
* the same winner and the same gain from the two ``find_best`` entry points.

Both formulations are pure functions of the search state, so running both on
each call does not alter the trajectory.

    python3.11 benchmarks/compare_candidates.py                                # MM, exact
    python3.11 benchmarks/compare_candidates.py --lex JM --tol 1e-9            # JM: float-weighted
                                                                               # streams differ in the
                                                                               # last bits (sorted sums)
"""

from __future__ import annotations

import argparse
import sys

from _common import REPO_ROOT  # noqa: F401  (fixes sys.path)
from streams import GENERATORS

import lago.algorithm._internal.find_best_move as fbm
import lago.algorithm._internal.lago_tools as lts
import lago.algorithm._internal.stem as stem
import lago.algorithm._internal.tmm as tmm
from lago import lago_modules
from lago.algorithm._internal.delta_lm import DeltaLongitudinalModularityComputer as Computer
from lago.algorithm._internal.leaf_set import LeafDifference

DEFAULT_STREAMS = [
    "planted_small",
    "planted_medium",
    "weighted_medium",
    "directed_medium",
    "bipartite_medium",
    "bipartite_directed_medium",
    "continuous_medium",
    "delayed_medium",
    "fixture",
]

_batch_find_best = fbm.find_best_module_for_submodule


def reference_evaluation(computer, submodule, partite_mapping, modules, level):
    """Candidates and per-candidate totals the way the original code found them."""
    if modules is None:
        if level == "parent":
            modules = sorted(
                {m.parent for m in submodule.neighbors if m.parent is not None},
                key=lambda m: m.index,
            )
        else:
            modules = sorted(lts.get_neighbors_modules_parents(submodule), key=lambda m: m.index)
    parent = submodule.parent
    modules = [m for m in modules if m is not parent]
    if not modules or parent is None:
        return None, {}

    M0 = submodule.leaves
    segments = lts.get_nodes_segment(module_leaves=M0)
    M1 = LeafDifference(parent.leaves, M0)
    leaving = -computer.M0_to_Mx(
        M0_leaves=M0,
        M0_time_segments=segments,
        Mx_leaves=M1,
        partite_mapping=partite_mapping,
        union_module=parent,
    )
    totals = {}
    for module in modules:
        disjoint = module.leaves.isdisjoint(M0)
        M2 = module.leaves if disjoint else module.leaves - M0
        if not M2 and not M1:
            continue
        totals[module] = leaving + computer.M0_to_Mx(
            M0_leaves=M0,
            M0_time_segments=segments,
            Mx_leaves=M2,
            partite_mapping=partite_mapping,
            mx_module=module if disjoint else None,
        )
    return leaving, totals


def batch_evaluation(computer, submodule, partite_mapping, modules, level):
    parent = submodule.parent
    if parent is None:
        return None, {}
    M0 = submodule.leaves
    segments = lts.get_nodes_segment(module_leaves=M0)
    if modules is None and level == "parent":
        # As find_best_module_for_submodule does: the neighbours recorded at
        # the start of the level, not the live adjacency.
        modules = {m.parent for m in submodule.neighbors if m.parent is not None}
    candidates, leaving, _, evaluations = computer.evaluate_candidates(
        M0, segments, parent, level == "parent", partite_mapping, modules
    )
    if not candidates:
        return None, {}
    return leaving, {module: leaving + joining for module, joining, _ in evaluations}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default=",".join(DEFAULT_STREAMS))
    ap.add_argument("--lex", default="MM", help="comma-separated; JM wants --tol 1e-9 on weighted streams")
    ap.add_argument("--refinement", default="STEM,None")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="relative tolerance on deltas; 0 (default) demands bit-identity. The JM "
        "aggregates sum degrees in a canonical order, so float-weighted streams under "
        "JM differ in the last bits -- use 1e-9 there, as for the MM closed form.",
    )
    args = ap.parse_args()

    stats = {"calls": 0, "candidate_sets": 0, "values": 0, "winners": 0, "max_abs": 0.0, "reference_only": 0}

    def differs(a, b) -> bool:
        if a is None or b is None:
            return a is not b
        return abs(a - b) > args.tol * max(1.0, abs(a))

    def shadow(computer, submodule, partite_mapping={}, modules=None, verbose=0, stopping_criterion=0.0, level="parent"):  # noqa: B006 - mirrors the signature it replaces
        if not computer.use_batch:
            # A computer pinned to the reference path (STNM, see runner.py):
            # nothing to compare, the batch evaluator is not used.
            stats["reference_only"] += 1
            return _batch_find_best(computer, submodule, partite_mapping, modules, verbose, stopping_criterion, level)
        stats["calls"] += 1

        ref_leaving, ref_totals = reference_evaluation(computer, submodule, partite_mapping, modules, level)
        new_leaving, new_totals = batch_evaluation(computer, submodule, partite_mapping, modules, level)

        if set(ref_totals) != set(new_totals):
            stats["candidate_sets"] += 1
        elif ref_totals:
            if differs(ref_leaving, new_leaving):
                stats["values"] += 1
            stats["max_abs"] = max(stats["max_abs"], abs(ref_leaving - new_leaving))
            for module, value in ref_totals.items():
                other = new_totals[module]
                if differs(value, other):
                    stats["values"] += 1
                stats["max_abs"] = max(stats["max_abs"], abs(value - other))

        # The two entry points, on the same state.
        result = _batch_find_best(computer, submodule, partite_mapping, modules, verbose, stopping_criterion, level)
        Computer.use_batch = False
        try:
            reference = _batch_find_best(computer, submodule, partite_mapping, modules, verbose, stopping_criterion, level)
        finally:
            Computer.use_batch = True
        # The reference counted this call too; keep the counters as one call.
        fbm._find_best_stats["calls"] -= 1
        if result[0] is not reference[0] or differs(result[1], reference[1]):
            stats["winners"] += 1
        return result

    tmm.find_best_module_for_submodule = shadow
    stem.find_best_module_for_submodule = shadow

    failures = 0
    print(f"{'stream':26s} {'lex':4s} {'ref':5s} {'calls':>8s} {'cand.sets':>9s} {'values':>7s} {'winners':>8s}")
    try:
        for sname in args.streams.split(","):
            for lex in args.lex.split(","):
                for token in args.refinement.split(","):
                    refinement = None if token == "None" else token
                    for key in stats:
                        stats[key] = 0 if key != "max_abs" else 0.0
                    ls = GENERATORS[sname]()
                    lago_modules(ls, lex=lex, refinement=refinement, seed=args.seed)
                    bad = stats["candidate_sets"] + stats["values"] + stats["winners"]
                    failures += bad
                    note = ""
                    if stats["max_abs"]:
                        note = f"   max |diff| {stats['max_abs']:.3e}"
                    if stats["reference_only"]:
                        note += f"   ({stats['reference_only']} calls on the reference path only)"
                    print(
                        f"{sname:26s} {lex:4s} {refinement!s:5s} {stats['calls']:8d} "
                        f"{stats['candidate_sets']:9d} {stats['values']:7d} {stats['winners']:8d}{note}",
                        flush=True,
                    )
    finally:
        tmm.find_best_module_for_submodule = _batch_find_best
        stem.find_best_module_for_submodule = _batch_find_best

    print(f"\n{'FAIL' if failures else 'OK'}: {failures} differences between batch and reference")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
