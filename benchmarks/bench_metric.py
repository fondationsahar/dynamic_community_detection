"""Timing benchmark for longitudinal_modularity, against a reference snapshot.

    python3.11 benchmarks/bench_metric.py --ref-dir /tmp/refs --ref-name lago_step2
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
import time

from _common import REPO_ROOT  # noqa: F401  (fixes sys.path)
from streams import GENERATORS

from lago import LexType, LinkStream, lago_modules, longitudinal_modularity


def ground_truth(n_nodes: int, n_comms: int, n_times: int, p_in: float, p_out: float, seed=0):
    """A planted partition, handed to the metric directly.

    Exercises the O(sum n_c^2) term that LAGO's own output rarely reaches.
    """
    rng = random.Random(seed)
    comm = {n: n % n_comms for n in range(n_nodes)}
    links = [
        (i, j, t)
        for t in range(n_times)
        for i in range(n_nodes)
        for j in range(i + 1, n_nodes)
        if rng.random() < (p_in if comm[i] == comm[j] else p_out)
    ]
    ls = LinkStream()
    ls.add_links(links)
    communities: dict[int, set] = {}
    for t in range(n_times):
        for n in range(n_nodes):
            communities.setdefault(comm[n], set()).add((n, t))
    return ls, communities


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="")
    ap.add_argument("--ref-name", default="lago_ref")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    ref_lm = ref_lex = None
    if args.ref_dir:
        sys.path.insert(0, args.ref_dir)
        ref = importlib.import_module(args.ref_name)
        ref_lm, ref_lex = ref.longitudinal_modularity, ref.LexType
        print(f"reference: {ref.__file__}")

    cases = []
    for sname in ("planted_medium", "weighted_medium", "directed_medium", "bipartite_medium",
                  "continuous_medium", "delayed_medium", "fixture"):
        ls = GENERATORS[sname]()
        # As a plain dict: the reference package has its own TimeModules class,
        # so an isinstance check there would not recognise ours.
        cases.append((f"lago:{sname}", ls, lago_modules(ls, seed=42).to_communities_dict()))
    cases.append(("truth:n240_c4", *ground_truth(240, 4, 10, 0.08, 0.002)))
    cases.append(("truth:n600_c2", *ground_truth(600, 2, 5, 0.03, 0.0005)))

    print(f"{'case':22s} {'lex':4s} {'ref(ms)':>9s} {'new(ms)':>9s} {'x':>7s} {'same':>5s}")
    for name, ls, communities in cases:
        for lex in (LexType.MM, LexType.JM, LexType.CM):
            t0 = time.perf_counter()
            for _ in range(args.reps):
                new = longitudinal_modularity(ls, communities, lex=lex)
            t_new = (time.perf_counter() - t0) / args.reps * 1000

            if ref_lm is None:
                print(f"{name:22s} {lex.name:4s} {'-':>9s} {t_new:9.3f} {'-':>7s} {'-':>5s}")
                continue

            rlex = getattr(ref_lex, lex.name)
            t0 = time.perf_counter()
            for _ in range(args.reps):
                old = ref_lm(ls, communities, lex=rlex)
            t_old = (time.perf_counter() - t0) / args.reps * 1000
            same = old.value == new.value and old.time_penalty == new.time_penalty
            print(
                f"{name:22s} {lex.name:4s} {t_old:9.3f} {t_new:9.3f} "
                f"{t_old / max(t_new, 1e-9):7.2f} {same!s:>5s}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
