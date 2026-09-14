"""Per-call numerical check of the MM expectation delta.

Partition fingerprints are not enough to validate a change of *formula*: a
systematic error can leave the greedy choices untouched on a given corpus. This
runs the closed form and the pair loop side by side on every real call of a
genuine run and reports the distribution of the difference.

    python3.11 benchmarks/compare_delta.py
"""

from __future__ import annotations

import argparse
import sys

from _common import REPO_ROOT  # noqa: F401  (fixes sys.path)
from streams import GENERATORS

from lago import lago_modules
from lago.algorithm._internal.delta_lm import DeltaLongitudinalModularityComputer as Computer

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

_impl = Computer._get_expectation_mm_part


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default=",".join(DEFAULT_STREAMS))
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    stats: dict[str, dict] = {}

    def shadow(self, M0_leaves, Mx_leaves, partite_mapping, mx_module=None, union_module=None):
        # The real implementation, whichever path it takes...
        fast_value = _impl(self, M0_leaves, Mx_leaves, partite_mapping, mx_module, union_module)

        # ...against the O(n**2) pair loop on the same inputs. Clearing the
        # module hints as well as the closed-form flag forces the original
        # enumeration, with no cache and no algebraic shortcut.
        saved, self._mm_closed_form = self._mm_closed_form, False
        try:
            loop_value = _impl(self, M0_leaves, Mx_leaves, partite_mapping, None, None)
        finally:
            self._mm_closed_form = saved

        entry = stats[shadow.tag]
        entry["n"] += 1
        if loop_value != fast_value:
            entry["differ"] += 1
            rel = abs(loop_value - fast_value) / max(abs(loop_value), 1e-300)
            entry["max_rel"] = max(entry["max_rel"], rel)
            if rel > args.tol:
                entry["significant"] += 1
                if loop_value and len(entry["ratios"]) < 5:
                    entry["ratios"].add(round(fast_value / loop_value, 6))

        # Follow the reference trajectory so both arms always see identical state.
        return loop_value
        return loop_value

    Computer._get_expectation_mm_part = shadow

    print(f"{'stream':20s} {'calls':>9s} {'differ':>9s} {'>tol':>7s} {'max rel':>11s}  ratios")
    worst = 0
    for sname in args.streams.split(","):
        shadow.tag = sname
        stats[sname] = {"n": 0, "differ": 0, "significant": 0, "max_rel": 0.0, "ratios": set()}
        ls = GENERATORS[sname]()
        lago_modules(ls, lex="MM", refinement="STEM", seed=1)
        entry = stats[sname]
        worst += entry["significant"]
        print(
            f"{sname:20s} {entry['n']:9d} {entry['differ']:9d} {entry['significant']:7d} "
            f"{entry['max_rel']:11.3e}  {sorted(entry['ratios'])}",
            flush=True,
        )

    Computer._get_expectation_mm_part = _impl
    print(f"\n{'FAIL' if worst else 'OK'}: {worst} calls differ by more than {args.tol}")
    return 1 if worst else 0


if __name__ == "__main__":
    sys.exit(main())
