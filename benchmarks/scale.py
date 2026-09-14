"""How LAGO scales, on both axes, at sizes the correctness corpus does not reach.

`streams.py` feeds the equality harness and is deliberately small and fast. This
is the other thing: a handful of streams large enough to say whether the cost
model holds, and a runner that reports the two numbers that matter.

The measured model is **~150 microseconds and ~350-700 bytes per time-edge**,
roughly independent of shape -- which is why the two axes have to be varied
separately. Doubling the nodes of a dense generator quadruples the edges, so a
node sweep alone makes the cost look quadratic when it is linear in edges. A
`deep` stream and a `wide` stream with the same edge count stress quite
different parts of the representation: long runs and time-neighbour chains on
one side, large modules and high degree on the other.

    python3.11 benchmarks/scale.py                          # quick, under a minute
    python3.11 benchmarks/scale.py --max-edges 1_000_000    # the real question
    python3.11 benchmarks/scale.py --build-only             # representation cost only

Sizes above the default print their predicted time and memory first, because the
top of the range is minutes and gigabytes.
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
import time
import tracemalloc

from _common import REPO_ROOT  # noqa: F401  (fixes sys.path)

from lago import LinkStream, lago_modules

# Measured on this machine; used only to warn before a long run.
MICROSECONDS_PER_TIME_EDGE = 150.0
BYTES_PER_TIME_EDGE = 500.0

DEFAULT_MAX_EDGES = 200_000


def sparse_planted(
    n_nodes: int,
    n_times: int,
    edges_per_time: int,
    n_comms: int = 8,
    p_within: float = 0.85,
    seed: int = 0,
    directed: bool = False,
) -> list[tuple[int, ...]]:
    """Planted-partition links, sampled rather than enumerated.

    The generator in ``streams.py`` walks every node pair at every timestep,
    which is O(n**2 * T) and unusable past a few hundred nodes. This samples the
    requested number of edges directly, so building a 10**6-edge stream costs
    O(10**6) and not O(10**12).

    Args:
        n_nodes: Number of nodes.
        n_times: Number of timesteps.
        edges_per_time: Edges to sample at each timestep.
        n_comms: Number of planted communities.
        p_within: Fraction of edges drawn inside a community.
        seed: Sampling seed.
        directed: Whether to emit directed links.

    Returns:
        A list of (source, target, time) tuples, deduplicated per timestep.
    """
    rng = random.Random(seed)
    members: dict[int, list[int]] = {}
    for node in range(n_nodes):
        members.setdefault(node % n_comms, []).append(node)

    links: list[tuple[int, ...]] = []
    for time_step in range(n_times):
        seen: set[tuple[int, int]] = set()
        for _ in range(edges_per_time):
            if rng.random() < p_within:
                group = members[rng.randrange(n_comms)]
                if len(group) < 2:
                    continue
                source, target = rng.sample(group, 2)
            else:
                source, target = rng.randrange(n_nodes), rng.randrange(n_nodes)
                if source == target:
                    continue
            key = (source, target) if directed else (min(source, target), max(source, target))
            if key in seen:
                continue
            seen.add(key)
            links.append((key[0], key[1], time_step))
    return links


def shape(name: str, target_edges: int, seed: int = 0) -> tuple[int, int, int]:
    """(nodes, timesteps, edges_per_time) for a shape at a target edge count.

    Three shapes at equal edge counts, because equal size is not equal work:

    * ``wide`` -- many nodes, few timesteps: big modules, high degree.
    * ``deep`` -- few nodes, many timesteps: long runs, long time-neighbour
      chains, and the duration bookkeeping that walks them.
    * ``balanced`` -- both grown together.
    """
    if name == "wide":
        n_times = 10
        edges_per_time = target_edges // n_times
        return max(edges_per_time // 4, 16), n_times, edges_per_time
    if name == "deep":
        n_nodes = 60
        edges_per_time = 40
        return n_nodes, max(target_edges // edges_per_time, 1), edges_per_time
    if name == "balanced":
        side = max(int(target_edges**0.5), 4)
        return max(side // 2, 16), side, side
    msg = f"unknown shape {name!r}; expected wide, deep or balanced"
    raise ValueError(msg)


def measure(links, run_search: bool, lex: str, refinement: str | None) -> dict:
    """Build the stream (and optionally search it), timing and sizing both."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    stream = LinkStream()
    stream.add_links(links)
    build_seconds = time.perf_counter() - start
    build_bytes, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "leaves": len(stream.leaves_dict),
        "time_edges": int(stream.nb_edges),
        "nodes": stream.nb_nodes,
        "build_seconds": build_seconds,
        "build_bytes": build_bytes,
        "search_seconds": None,
        "modules": None,
    }
    if run_search:
        gc.collect()
        start = time.perf_counter()
        modules = lago_modules(stream, lex=lex, refinement=refinement)
        result["search_seconds"] = time.perf_counter() - start
        result["modules"] = modules.nb_modules
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="wide,deep,balanced")
    ap.add_argument(
        "--max-edges",
        type=int,
        default=DEFAULT_MAX_EDGES,
        help=f"largest target time-edge count (default {DEFAULT_MAX_EDGES:,})",
    )
    ap.add_argument("--build-only", action="store_true", help="skip the search")
    ap.add_argument("--lex", default="MM")
    ap.add_argument("--refinement", default="STEM")
    ap.add_argument("--yes", action="store_true", help="do not pause on large cases")
    args = ap.parse_args()

    targets = [n for n in (2_000, 20_000, 200_000, 1_000_000, 10_000_000) if n <= args.max_edges]
    if not targets:
        targets = [args.max_edges]
    refinement = None if args.refinement in ("None", "none", "") else args.refinement

    print(
        f"{'shape':10s} {'target':>10s} {'nodes':>7s} {'times':>7s} {'leaves':>9s} "
        f"{'edges':>10s} {'build(s)':>9s} {'MB':>8s} {'search(s)':>10s} "
        f"{'us/edge':>9s} {'B/edge':>8s}"
    )
    for shape_name in args.shapes.split(","):
        for target in targets:
            n_nodes, n_times, edges_per_time = shape(shape_name, target)

            if target > DEFAULT_MAX_EDGES and not args.yes:
                seconds = target * MICROSECONDS_PER_TIME_EDGE / 1e6
                gigabytes = target * BYTES_PER_TIME_EDGE / 1e9
                print(
                    f"  [{shape_name} {target:,}] predicted ~{seconds:.0f}s and "
                    f"~{gigabytes:.1f}GB -- running; Ctrl-C to skip",
                    flush=True,
                )

            links = sparse_planted(n_nodes, n_times, edges_per_time)
            if not links:
                continue
            stats = measure(links, not args.build_only, args.lex, refinement)

            edges = stats["time_edges"]
            search = stats["search_seconds"]
            per_edge = f"{search / edges * 1e6:9.1f}" if search else f"{'-':>9s}"
            print(
                f"{shape_name:10s} {target:10,} {stats['nodes']:7d} {n_times:7d} "
                f"{stats['leaves']:9,} {edges:10,} {stats['build_seconds']:9.3f} "
                f"{stats['build_bytes'] / 1e6:8.1f} "
                f"{(f'{search:10.3f}' if search else f'{chr(45):>10s}')} "
                f"{per_edge} {stats['build_bytes'] / edges:8.0f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
