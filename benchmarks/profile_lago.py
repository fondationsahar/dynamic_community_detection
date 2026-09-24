"""Where the wall-clock time of ``lago_modules`` goes, per function, at two sizes.

Not cProfile. cProfile charges every Python-level call the same fixed overhead,
which triples the run time here and makes tiny hot functions (``Leaf.__hash__``,
set membership) look several times dearer than they are. This wraps a dozen
functions of interest with ``perf_counter`` and reports their share of the
*unprofiled* run, so the percentages add up to something real. The wrapper costs
about 0.3 us per call -- a few percent on the most-called functions -- and the
report says so.

Two sizes per shape, because the interesting question is not "what is slow" but
"what grows": a term that is 8 % at 30k edges and 25 % at 80k is the one that
decides how far the algorithm scales.

    python3.11 benchmarks/profile_lago.py                 # wide 30k/80k, MM
    python3.11 benchmarks/profile_lago.py --lex JM --sizes wide:10k deep:10k
    python3.11 benchmarks/profile_lago.py --sizes wide:30k deep:30k
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import TYPE_CHECKING

from _common import REPO_ROOT  # noqa: F401  (fixes sys.path)

if TYPE_CHECKING:
    from collections.abc import Callable
from scale import sparse_planted

import lago.algorithm._internal.delta_lm as delta_lm
import lago.algorithm._internal.lago_tools as lago_tools
import lago.algorithm._internal.stem as stem
import lago.algorithm._internal.tmm as tmm
import lago.core.utils as utils
from lago import LinkStream, lago_modules

# shape -> {size label: (nodes, timesteps, edges per timestep)}
SIZES = {
    "wide": {"10k": (700, 10, 1000), "30k": (2000, 10, 3000), "80k": (6000, 10, 8000)},
    "deep": {"10k": (40, 250, 40), "30k": (40, 750, 40), "80k": (40, 2000, 40)},
}


class Tally:
    """Accumulated wall time and call count per wrapped name."""

    def __init__(self) -> None:
        self.data: dict[str, list[float]] = {}

    def wrap(self, name: str, function: Callable) -> Callable:
        entry = self.data.setdefault(name, [0.0, 0])

        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            result = function(*args, **kwargs)
            entry[0] += time.perf_counter() - start
            entry[1] += 1
            return result

        return wrapped

    def reset(self) -> None:
        for entry in self.data.values():
            entry[0] = 0.0
            entry[1] = 0


def instrument() -> Tally:
    """Patch the functions of interest in place. Idempotent for one process."""
    tally = Tally()
    computer = delta_lm.DeltaLongitudinalModularityComputer

    utils.get_nodes_durations = tally.wrap("get_nodes_durations (recompute)", utils.get_nodes_durations)

    for name in (
        "M0_to_Mx",
        "_get_expectation_mm_part",
        "_get_expectation_jm_part",
        "_get_weight_diff",
        "_get_csc_diff",
        "_shift_mm_sums",
        "_mm_from_sums",
        "_mm_sums",
    ):
        setattr(computer, name, tally.wrap(name, getattr(computer, name)))
    for name in ("_duration_changes_adding", "_duration_changes_removing"):
        # Read through the class so the staticmethod is unwrapped, then re-wrap
        # it as one; a plain function would receive ``self``.
        setattr(computer, name, staticmethod(tally.wrap(name, getattr(computer, name))))
    if hasattr(computer, "evaluate_candidates"):
        computer.evaluate_candidates = tally.wrap("evaluate_candidates", computer.evaluate_candidates)

    find_best = tally.wrap("find_best_module_for_submodule", tmm.find_best_module_for_submodule)
    tmm.find_best_module_for_submodule = find_best
    stem.find_best_module_for_submodule = find_best

    for name in ("get_neighbors_modules_parents", "get_nodes_segment", "create_module_from_leaves", "move_submodule"):
        setattr(lago_tools, name, tally.wrap(name, getattr(lago_tools, name)))
    return tally


def report(label: str, stream: LinkStream, total: float, tally: Tally) -> None:
    print(
        f"\n=== {label}: {int(stream.nb_edges)} time-edges, {len(stream.leaves_dict)} time-nodes, "
        f"{total:.1f}s total, {total / stream.nb_edges * 1e6:.0f} us/edge (wrapper overhead included)"
    )
    print(f"  {'function':36s} {'seconds':>8s} {'share':>7s} {'calls':>11s} {'us/call':>9s}")
    for name, (seconds, calls) in sorted(tally.data.items(), key=lambda item: -item[1][0]):
        if not calls:
            continue
        print(f"  {name:36s} {seconds:8.2f} {100 * seconds / total:6.1f}% {calls:11,} {seconds / calls * 1e6:9.1f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lex", default="MM", choices=("MM", "JM"))
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["wide:30k", "wide:80k"],
        help="shape:size pairs, shapes wide|deep, sizes 10k|30k|80k",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    tally = instrument()
    for spec in args.sizes:
        shape, size = spec.split(":")
        n_nodes, n_times, edges_per_time = SIZES[shape][size]
        stream = LinkStream()
        stream.add_links(sparse_planted(n_nodes, n_times, edges_per_time, seed=args.seed))

        tally.reset()
        start = time.perf_counter()
        modules = lago_modules(stream, lex=args.lex, verbose=False)
        total = time.perf_counter() - start
        report(f"{shape} {size} {args.lex} ({len(modules.segments)} modules)", stream, total, tally)
    return 0


if __name__ == "__main__":
    sys.exit(main())
