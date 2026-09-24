"""Timing comparison across ``lago.accel`` backends.

Reports two numbers per case, because they answer different questions:

* **cold** -- one call on a freshly built stream. This is what a caller scoring
  a single partition sees. Flattening the topology is O(time-edges) in Python,
  the same order as the loop it replaces, which is why the compiled path only
  starts at the second scoring of a stream: a cold call runs the Python loops
  on every backend and should measure 1.0x.
* **warm** -- repeated calls on the same stream, the topology built once. This
  is what a parameter sweep, a lex comparison, or any evaluation loop sees.

Sizes vary along both axes the metric scales in -- number of nodes and number of
time steps -- since a link stream can grow either way.

    python3.11 benchmarks/bench_backends.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

from _common import REPO_ROOT

from lago import LexType, LinkStream, longitudinal_modularity

# (label, nodes, timesteps, edge probability)
CASES = (
    ("tiny      ", 20, 5, 0.30),
    ("wide      ", 300, 5, 0.06),
    ("wider     ", 800, 5, 0.02),
    ("deep      ", 20, 400, 0.30),
    ("deeper    ", 20, 1600, 0.30),
    ("square    ", 120, 120, 0.05),
)


def make_case(n_nodes: int, n_times: int, p: float, seed: int = 0):
    """A planted two-community stream plus its planted partition."""
    rng = random.Random(seed)
    comm = {n: n % 2 for n in range(n_nodes)}
    links = [
        (i, j, t)
        for t in range(n_times)
        for i in range(n_nodes)
        for j in range(i + 1, n_nodes)
        if rng.random() < (p if comm[i] == comm[j] else p / 6)
    ]
    stream = LinkStream()
    stream.add_links(links)
    communities: dict[int, set] = {}
    for node, time_step in stream.leaves_dict:
        communities.setdefault(comm[node], set()).add((node, time_step))
    return stream, communities


def run_worker(repeats: int) -> dict:
    """Time every case under whichever backend this process loaded."""
    from lago import accel

    rows = []
    for label, n_nodes, n_times, p in CASES:
        stream, communities = make_case(n_nodes, n_times, p)
        edges = int(stream.nb_edges)

        # Cold: rebuild the stream each time so the topology cache is empty.
        cold = []
        for _ in range(3):
            fresh, fresh_communities = make_case(n_nodes, n_times, p)
            start = time.perf_counter()
            longitudinal_modularity(fresh, fresh_communities, lex=LexType.MM)
            cold.append(time.perf_counter() - start)

        # Warm the cache: the compiled path starts at the second scoring of a
        # stream and builds the topology then.
        longitudinal_modularity(stream, communities, lex=LexType.MM)
        longitudinal_modularity(stream, communities, lex=LexType.MM)
        start = time.perf_counter()
        for _ in range(repeats):
            longitudinal_modularity(stream, communities, lex=LexType.MM)
        warm = (time.perf_counter() - start) / repeats

        rows.append(
            {
                "label": label,
                "nodes": n_nodes,
                "times": n_times,
                "edges": edges,
                "leaves": len(stream.leaves_dict),
                "cold": min(cold),
                "warm": warm,
            }
        )
    return {"backend": accel.backend_name, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--repeats", type=int, default=5)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        json.dump(run_worker(args.repeats), sys.stdout)
        return 0

    from lago import accel

    backends = accel.available_backends()
    results = {}
    for backend in backends:
        env = {**os.environ, "LAGO_ACCEL": backend}
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", "-r", str(args.repeats)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT / "benchmarks"),
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            print(f"{backend}: worker failed\n{proc.stderr}")
            return 1
        results[backend] = json.loads(proc.stdout)["rows"]

    others = [b for b in backends if b != "python"]
    header = f"{'case':<11}{'nodes':>7}{'times':>7}{'edges':>9}{'leaves':>8}{'py cold':>10}{'py warm':>10}"
    for backend in others:
        header += f"{backend + ' cold':>13}{backend + ' warm':>13}"
    print(header)
    print("-" * len(header))

    for index, row in enumerate(results["python"]):
        line = (
            f"{row['label']:<11}{row['nodes']:>7}{row['times']:>7}{row['edges']:>9}"
            f"{row['leaves']:>8}{row['cold'] * 1e3:>9.2f}m{row['warm'] * 1e3:>9.2f}m"
        )
        for backend in others:
            other = results[backend][index]
            line += f"{row['cold'] / other['cold']:>12.2f}x{row['warm'] / other['warm']:>12.2f}x"
        print(line)
    print("\ncold/warm columns for python are milliseconds; backend columns are speedups.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
