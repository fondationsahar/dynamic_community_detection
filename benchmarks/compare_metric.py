"""Exhaustive equivalence check for longitudinal_modularity against a reference.

Sized for the rounding-tie rate documented in docs/PERFORMANCE_DIAGNOSIS.md: a
closed-form expectation can land on the other side of an exact half-way tie in
roughly 0.1-0.3% of integer-weight runs, so a few dozen cases prove nothing.
Run thousands.

    python3.11 benchmarks/compare_metric.py --ref-dir /tmp/refs --ref-name lago_step2 -n 5000
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
import time

from _common import REPO_ROOT  # noqa: F401  (fixes sys.path)

from lago import LexType, LinkStream, longitudinal_modularity


def random_case(rng: random.Random):
    """A small random link stream plus a random partition of its time-nodes."""
    directed = rng.random() < 0.4
    weighted = rng.random() < 0.35
    n_nodes = rng.randint(3, 14)
    n_times = rng.randint(1, 5)
    partite = rng.random() < 0.15

    mapping = None
    if partite:
        mapping = {n: rng.randint(0, 1) for n in range(n_nodes) if rng.random() < 0.8}

    ls = LinkStream(directed=directed, partite_mapping=mapping)
    seen = set()
    links = []
    for _ in range(rng.randint(2, 40)):
        a, b = rng.randrange(n_nodes), rng.randrange(n_nodes)
        t = rng.randrange(n_times)
        key = (a, b, t) if directed else (min(a, b), max(a, b), t)
        if key in seen:
            continue
        seen.add(key)
        if weighted:
            links.append((a, b, t, round(rng.uniform(0.2, 3.0), 3)))
        else:
            links.append((a, b, t))
    if not links:
        return None
    ls.add_links(links)

    n_comms = rng.randint(1, 4)
    communities: dict[int, set] = {}
    for node, t in ls.leaves_dict:
        communities.setdefault(rng.randrange(n_comms), set()).add((node, t))
    return ls, communities


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--ref-name", default="lago_ref")
    ap.add_argument("-n", "--cases", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, args.ref_dir)
    ref = importlib.import_module(args.ref_name)
    ref_lm = ref.longitudinal_modularity
    ref_lex = ref.LexType

    rng = random.Random(args.seed)
    lex_pairs = [
        (LexType.MM, ref_lex.MM),
        (LexType.JM, ref_lex.JM),
        (LexType.CM, ref_lex.CM),
    ]
    params = [(1.0, 2.0, 5), (0.0, 2.0, 5), (0.5, 0.0, 5), (2.0, 1.0, 12)]

    counts = {lex.name: [0, 0] for lex, _ in lex_pairs}  # [checked, mismatched]
    worst = 0.0
    examples = []
    t0 = time.perf_counter()

    built = 0
    while built < args.cases:
        case = random_case(rng)
        if case is None:
            continue
        built += 1
        ls, communities = case
        for lex, rlex in lex_pairs:
            for gamma, omega, ndigits in params:
                new = longitudinal_modularity(
                    ls, communities, lex=lex, gamma=gamma, omega=omega, ndigits=ndigits
                )
                old = ref_lm(
                    ls, communities, lex=rlex, gamma=gamma, omega=omega, ndigits=ndigits
                )
                counts[lex.name][0] += 1
                if new.value != old.value or new.time_penalty != old.time_penalty:
                    counts[lex.name][1] += 1
                    worst = max(worst, abs(new.value - old.value))
                    if len(examples) < 8:
                        examples.append(
                            f"    lex={lex.name} gamma={gamma} omega={omega} nd={ndigits}: "
                            f"ref={old.value!r} new={new.value!r}"
                        )
        if built % 500 == 0:
            done = sum(c[0] for c in counts.values())
            bad = sum(c[1] for c in counts.values())
            print(f"  {built} streams, {done} comparisons, {bad} mismatches", flush=True)

    print(f"\n{args.cases} streams in {time.perf_counter() - t0:.1f}s")
    total_checked = total_bad = 0
    for name, (checked, bad) in counts.items():
        total_checked += checked
        total_bad += bad
        rate = 100 * bad / checked if checked else 0
        print(f"  {name}: {bad}/{checked} mismatches ({rate:.3f}%)")
    print(f"  TOTAL: {total_bad}/{total_checked}, max |delta| = {worst:.3e}")
    for line in examples:
        print(line)
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
