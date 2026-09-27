# Performance

What to expect, how it scales, how to measure it, what is left. How the implementation
achieves it is in [`ARCHITECTURE.md`](ARCHITECTURE.md); the measurements behind every
statement here are in [`history/`](history/README.md) and
[`BENCHMARK_1.1.0_VS_1.2.0.md`](BENCHMARK_1.1.0_VS_1.2.0.md).

## 1. What to expect

Against release 1.1.0, same inputs and parameters, each side in its own process
(`benchmarks/bench_pypi.py`, Apple Silicon, Python 3.11):

| | pure Python | compiled |
|---|---:|---:|
| `lago_modules`, 15 small streams × all 24 parameter combinations (geometric mean) | **7.8×** | **12.4×** |
| `lago_modules`, planted streams of 4k–96k interactions, default parameters | 6–72× | 9–109× |
| `longitudinal_modularity`, one scoring | 1.5–2.5× | 1.5–2.5× |
| `longitudinal_modularity`, repeated scoring of one stream | 1.5–2.5× | 10–70× |

JM runs at the speed of MM (it was 8–14× slower). Wall time at 80k wide interactions: 60 s
before this work, 24.5 s pure, 17.8 s compiled. The tree's partition scores at least as
well as 1.1.0's by the 1.2.0 metric in 80 % of the 380 compared runs; the differences
either way are those of a greedy search taking another path, plus the bug fixes of round 1.

The full report, per stream and per parameter combination, is
[`BENCHMARK_1.1.0_VS_1.2.0.md`](BENCHMARK_1.1.0_VS_1.2.0.md) — a fixed comparison between the two
releases, not a moving target.

"Compiled" is what a binary wheel gives, or a source install on a machine with a C compiler;
the pure-Python path is the same source and gives identical results.

## 2. How it scales

Cost is linear in the number of **time-edges** on both axes — many nodes over few instants
("wide") and few nodes over many instants ("deep") stress different parts of the
representation and both stay flat per edge:

| shape | 20k | 80k | 200k time-edges |
|---|---:|---:|---:|
| wide, µs per time-edge, compiled | 174 | 223 | ≈ 230 |
| deep | ≈ 170 | ≈ 170 | ≈ 170 |

The term that used to grow with module size on wide streams (recomputing a module's
durations after every accepted move) is gone; the largest remaining one is the
per-module MM aggregate, O(nodes in the module) per accepted move, ~12 % at 80k wide.

**Memory** is what caps stream size: roughly 350–700 bytes per time-edge — every leaf holds
its edges as sets of `TimeEdge` objects — so 10⁶ time-edges is about half a gigabyte, and
`n_jobs` multiplies it. `Leaf` has `__slots__`; the objects are the next thing to shrink.

Small streams (hundreds of interactions) run in tens of milliseconds and are dominated by
per-call overheads; do not expect the large-stream ratios there.

## 3. How to measure

All in [`benchmarks/`](../benchmarks/README.md); each script's docstring is its manual.

| script | question it answers |
|---|---|
| `profile_lago.py` | where the wall time of `lago_modules` goes, per function, at two sizes per shape — without cProfile's distortion, so that what *grows* stands out |
| `scale.py` | how cost grows with size on both axes, up to 10⁶ time-edges (predicts time and memory before a long run) |
| `bench_backends.py` | the metric, cold (one scoring) and warm (repeated), per backend |
| `bench_pypi.py` | one release against another — the PyPI wheel as the baseline, this tree pure and compiled — every parameter combination, all stream features; renders `BENCHMARK_<baseline>_VS_<new>.md` |

Before trusting a timing: `lago.accel.core_name()` and `lago.accel.backend_name` say which
implementation ran; `LAGO_CORE=python` and `LAGO_ACCEL=python` force the pure paths.

## 4. What is left

In the order I would take them:

1. **Memory / the array representation.** Store each leaf's edges as flat arrays (target
   rows, weights, durations) instead of sets of objects — the representation the metric
   kernel already builds once per stream. It is also the next speed step: `evaluate_candidates`
   is ~70 % of a run and is dict/set work per edge of M0.
2. **`add_links` called repeatedly is quadratic.** `_compute_time_neighbors` rescans every
   leaf on every call; building a stream one link at a time costs O(N²). Nobody has hit it
   because inputs arrive in one batch.
3. **`_mm_sums` per accepted move.** Cache the per-node terms `degree·√duration` and
   recompute only the nodes whose duration changed; the sorted O(nodes) sum remains.
4. **STNM on the batch path**, which needs the stale-set behaviour of
   `ARCHITECTURE.md` §5.1 corrected — a decision about outputs, not about speed.
5. **Compiled wheels for every platform** (`cibuildwheel` in CI); until then Linux,
   Windows and Intel-Mac users get the sdist, which compiles when a compiler is present.
