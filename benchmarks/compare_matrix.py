"""Full-matrix partition equality check against a reference snapshot.

Every stream x lex x refinement x fast_exploration x refinement_in x seed must
return exactly the same partition as the reference. This is the acceptance test
for any change that is supposed to be output-preserving.

    python3.11 benchmarks/compare_matrix.py --ref-dir /tmp/refs --ref-name lago_step3
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

from _common import fingerprint_modules, fingerprint_partition
from streams import GENERATORS

from lago import lago_modules

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
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--ref-name", default="lago_ref")
    ap.add_argument("--streams", default=",".join(DEFAULT_STREAMS))
    ap.add_argument("--seeds", default="", help="comma-separated; empty = canonical order only")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.ref_dir)
    ref = importlib.import_module(args.ref_name)
    ref_modules = ref.lago_modules
    print(f"reference: {ref.__file__}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [None]
    flag_combos = [(True, True)] if not args.full else [
        (True, True), (True, False), (False, True), (False, False)
    ]

    checked = mismatched = 0
    t_ref = t_new = 0.0
    failures = []
    # Configurations the reference cannot run at all are reported, not hidden:
    # a fixed crash is not a mismatch, but it is not a clean comparison either.
    ref_broken: list[str] = []
    for sname in args.streams.split(","):
        for lex in ("MM", "JM"):
            for refinement in ("STEM", "STNM", None):
                for fast, ref_in in flag_combos:
                    for seed in seeds:
                        kwargs = {
                            "lex": lex,
                            "refinement": refinement,
                            "fast_exploration": fast,
                            "refinement_in": ref_in,
                            "seed": seed,
                        }
                        ls = GENERATORS[sname]()
                        t0 = time.perf_counter()
                        try:
                            old = ref_modules(ls, **kwargs)
                        except Exception as exc:
                            ref_broken.append(
                                f"    {sname} lex={lex} ref={refinement} fast={fast} "
                                f"in={ref_in} seed={seed}: reference raises "
                                f"{type(exc).__name__}"
                            )
                            continue
                        finally:
                            t_ref += time.perf_counter() - t0

                        ls = GENERATORS[sname]()
                        t0 = time.perf_counter()
                        new = lago_modules(ls, **kwargs)
                        t_new += time.perf_counter() - t0

                        fp_old = fingerprint_partition(old._raw_modules.values())
                        fp_new = fingerprint_modules(new)
                        checked += 1
                        if fp_old != fp_new:
                            mismatched += 1
                            failures.append(
                                f"    {sname} lex={lex} ref={refinement} fast={fast} "
                                f"in={ref_in} seed={seed}: {fp_old} != {fp_new}"
                            )
        print(f"  {sname}: {checked} checked, {mismatched} mismatched", flush=True)

    print(f"\n{checked - mismatched}/{checked} configurations produced an identical partition")
    print(f"reference {t_ref:.1f}s, new {t_new:.1f}s, speedup {t_ref / max(t_new, 1e-9):.2f}x")
    for line in failures[:20]:
        print(line)
    if ref_broken:
        print(f"\n{len(ref_broken)} configurations the reference cannot run (not compared):")
        for line in ref_broken[:10]:
            print(line)
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
