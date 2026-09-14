"""Acceptance test for a frozen exploration order.

Runs the full configuration matrix (streams x lex x refinement x fast_exploration
x refinement_in) several times and reports, per configuration, whether
``lago_modules`` returned the same partition every time.

    python3.11 benchmarks/check_determinism.py                  # default matrix
    python3.11 benchmarks/check_determinism.py --seed 1 --reps 3
    python3.11 benchmarks/check_determinism.py --out run_a.txt  # then diff two runs
                                                               # to cover separate processes

Exit status is 0 only when every configuration is stable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import fingerprint_modules
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


def configurations(streams, full: bool):
    for sname in streams:
        for lex in ("MM", "JM"):
            for refinement in ("STEM", "STNM", None):
                if full:
                    for fast in (True, False):
                        for ref_in in (True, False):
                            yield sname, lex, refinement, fast, ref_in
                else:
                    yield sname, lex, refinement, True, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default=",".join(DEFAULT_STREAMS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None,
                    help="exploration order; omit for the canonical order")
    ap.add_argument("--full", action="store_true", help="also vary fast_exploration/refinement_in")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    lines = []
    unstable = 0
    total = 0
    for sname, lex, refinement, fast, ref_in in configurations(args.streams.split(","), args.full):
        fps = []
        for _ in range(args.reps):
            ls = GENERATORS[sname]()
            tm = lago_modules(
                ls,
                lex=lex,
                refinement=refinement,
                fast_exploration=fast,
                refinement_in=ref_in,
                seed=args.seed,
            )
            fps.append(fingerprint_modules(tm))
        stable = len(set(fps)) == 1
        total += 1
        unstable += not stable
        line = (
            f"{sname:18s} {lex} {refinement!s:5s} fast={int(fast)} in={int(ref_in)} "
            f"stable={stable!s:5s} fp={fps[0]}"
        )
        lines.append(line)
        print(line, flush=True)

    summary = f"\n{total - unstable}/{total} configurations stable over {args.reps} repeats"
    lines.append(summary)
    print(summary)

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")

    return 1 if unstable else 0


if __name__ == "__main__":
    sys.exit(main())
