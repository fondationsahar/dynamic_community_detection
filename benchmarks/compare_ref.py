"""Compare the working copy against a reference snapshot of the package.

The reference is a plain copy of ``lago/`` taken before the change, with its
internal imports rewritten to ``lago_ref``, so both can be imported in the same
process. Create one with ``make_ref.py``.

    python3.11 benchmarks/compare_ref.py --ref-dir /path/containing/lago_ref

Reports, per stream: wall time of both, and whether the resulting partition
(canonical, label-invariant) and the L-modularity value are identical.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from _common import fingerprint_modules
from streams import GENERATORS

from lago import lago_modules, longitudinal_modularity

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True, help="directory containing the reference package")
    ap.add_argument("--ref-name", default="lago_ref", help="reference package name")
    ap.add_argument("--streams", default=",".join(DEFAULT_STREAMS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None,
                    help="exploration order; omit for the canonical order")
    ap.add_argument("--lex", default="MM")
    ap.add_argument("--refinement", default="STEM")
    args = ap.parse_args()

    sys.path.insert(0, args.ref_dir)
    import importlib

    lago_ref = importlib.import_module(args.ref_name)
    ref_lago_modules = lago_ref.lago_modules
    ref_lm = lago_ref.longitudinal_modularity

    print(f"reference: {lago_ref.__file__}")
    refinement = None if args.refinement in ("None", "none", "") else args.refinement

    header = (
        f"{'stream':20s} {'ref(s)':>8s} {'new(s)':>8s} {'x':>6s} "
        f"{'ref stable':>10s} {'new stable':>10s} {'same part.':>10s} {'same LM':>8s}"
    )
    print(header)
    n_same = n_total = 0
    for sname in args.streams.split(","):
        results = {}
        for label, modules_fn, lm_fn in (
            ("ref", ref_lago_modules, ref_lm),
            ("new", lago_modules, longitudinal_modularity),
        ):
            times, fps, lms = [], [], []
            for _ in range(args.reps):
                ls = GENERATORS[sname]()
                t0 = time.perf_counter()
                tm = modules_fn(ls, lex=args.lex, refinement=refinement, seed=args.seed)
                times.append(time.perf_counter() - t0)
                fps.append(fingerprint_modules(tm))
                lms.append(lm_fn(ls, tm, ndigits=12).value)
            results[label] = (statistics.median(times), fps, lms)

        (t_ref, fp_ref, lm_ref) = results["ref"]
        (t_new, fp_new, lm_new) = results["new"]
        ref_stable = len(set(fp_ref)) == 1
        new_stable = len(set(fp_new)) == 1
        same_part = set(fp_ref) == set(fp_new) and ref_stable and new_stable
        same_lm = set(lm_ref) == set(lm_new)
        n_total += 1
        n_same += same_part
        print(
            f"{sname:20s} {t_ref:8.3f} {t_new:8.3f} {t_ref / t_new:6.2f} "
            f"{ref_stable!s:>10s} {new_stable!s:>10s} {same_part!s:>10s} "
            f"{same_lm!s:>8s}",
            flush=True,
        )

    print(f"\nidentical partition on {n_same}/{n_total} streams")
    return 0


if __name__ == "__main__":
    sys.exit(main())
