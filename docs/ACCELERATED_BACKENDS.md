# A compiled backend for `longitudinal_modularity`

## Scope

The request was two alternative compiled backends, one in Cython or mypyc and
one in Rust + PyO3, each in its own folder, with results re-verified for every
stream option. It was then narrowed to **Cython only; Rust deferred**. This
document covers the Cython backend that was built and verified. Section 7 says
what the Rust one would be, so that it can be added later without redesign; it
was not written, and the Rust toolchain was not available in the environment
this work was done in anyway.

Everything below holds under the same rule as the rest of the optimisation
work: **the same value from `longitudinal_modularity`, the same sets of
`(node, time)` from `lago_modules`.** For the metric this is checked at the bit
level, not with a tolerance.

## 1. What is accelerated, and why only that

[`FASTER_LANGUAGE_ANALYSIS.md`](FASTER_LANGUAGE_ANALYSIS.md) §2 made the case
that LAGO's greedy loop cannot be "wrapped": its cost is spread across the
`Leaf` / `_LagoModule` object graph, and a compiled kernel that has to cross the
Python boundary once per candidate evaluation gains nothing. The one exception
is `longitudinal_modularity`, a pure function of (stream, partition). Inside it,
two loops are both O(time-edges) and pure accumulation:

* `_count_intra_community_interactions` — for each labelled time-node, sum the
  weight of its edges to time-nodes with the same label;
* `_count_community_switches` — for each labelled time-node, count its left and
  right temporal neighbours that carry a different label.

Everything else in the metric is O(nodes) closed form or dictionary
bookkeeping. So the boundary is drawn around those two loops, fused into one
pass. `lago_modules` is untouched: it scores iterations from its own incremental
deltas and never calls the metric, so its output cannot be affected. The
equivalence run confirms it anyway (11/11 partitions identical).

### Why Cython rather than mypyc

mypyc compiles typed Python modules more or less as written. The cost here is not
interpretation but object traversal: `Leaf.__hash__` is a Python method, so
every Leaf-keyed dictionary lookup is a Python call, and the loops perform one
per edge. mypyc would compile the traversal and still make those calls. Cython
lets the data be handed over as typed buffers — `array.array` exposes the buffer
protocol — and the kernel is then C integer indexing and double addition, with
no Python object touched inside the loop.

## 2. Design

```
lago/accel.py                 flat topology + reference kernel + backend dispatch
accel_cython/                 the compiled backend (setup.py, src/*.pyx, README)
benchmarks/compare_backends.py   bit-exactness across backends
benchmarks/bench_backends.py     cold / warm timings across backends
tests/metrics/test_accel.py      in-process contract tests
```

**Flat topology, built once per stream and cached on it.** Rows are time-nodes
in `leaves_dict` order; per row, a CSR slice of `(target row, weight)` in
`leaf.topo_neighbors` iteration order; per row, the rows of the left and right
temporal neighbours or −1. Also cached: a per-node sorted time index, so a
segment `[start, end]` becomes a slice of rows by binary search.

**Per call, only the labels.** `Topology.labels_from_segments` writes a
community number into an `int32` array from the partition's segments —
O(time-nodes) integer writes on top of a `memcpy` of a −1 template. Communities
that own no time-node in the stream get a row of zeros, which is what the Python
path produces through `dict.get(community, 0)`.

**One fused pass.** The kernel sums per-community weights and counts switches in
a single loop over rows, and returns `(list[float], int)`.

Two behaviours of the Python loops are folded into the flat form rather than
reproduced in the kernel:

* an undirected self-loop is counted twice — its target is the leaf itself,
  hence always in the leaf's own community, so the doubling is unconditional and
  is applied to the stored weight when the topology is built;
* the directed `× 2` applies to every community equally and stays in the caller.

**Bit-identity**, not closeness. Two things make the compiled result the same
double as the Python one:

1. the rows are in `leaves_dict` order and the neighbours in set-iteration
   order, i.e. exactly the order the Python loops visit them, so each
   community's partial sums are the same sequence of additions;
2. the kernel is compiled with `-O3` and no `-ffast-math`, so the compiler may
   not reassociate.

The one arithmetic liberty: the Python accumulator starts as `int 0` and stays an
exact integer while every weight is an integer; the kernel's is a C `double`.
They agree exactly while a community's total stays below 2⁵³ ≈ 9 × 10¹⁵.

**Dispatch.** `lago.accel` imports `lago_accel_cython` if it can, else falls
back to `_reference_kernel`, a pure-Python transcription that is the contract.
`LAGO_ACCEL=cython|python` forces a backend; an explicit choice that cannot be
imported raises rather than silently running something else. Without a compiled
backend, `longitudinal_modularity` runs the original Python loops, unchanged.

**Cache soundness.** The topology is stale if the stream is mutated. The two
mutations `LinkStream` performs after construction are `add_links` (changes
`nb_edges`) and `_split_continuous_linkstream` (replaces `leaves_dict`). The
cache holds a reference to the `leaves_dict` object it was built from and the
`nb_edges` it saw; either changing invalidates it. Holding the reference is what
makes the identity check sound — a live object cannot have its address recycled
under a replacement, which is the trap behind bug B6 in the earlier diagnosis.

## 3. The regression the first version had, and its fix

The first version built the flat topology with a `{leaf: row}` dictionary and
mapped the existing `leaf_labels` dictionary onto it. Correct — but a **single
cold call was 1.4–1.7× slower** than pure Python:

| case | py cold | cython cold (v1) | cython warm (v1) |
|---|---|---|---|
| wide (300 × 5) | 2.9 ms | 0.58× | 2.5× |
| deeper (20 × 1600) | 45 ms | 0.71× | 1.6× |

Flattening is O(time-edges), the same order as the loop it replaces, and each
of those steps was a Leaf-keyed dictionary lookup — a Python `__hash__` call —
so flattening cost more than the loop.

The fix removes every Leaf-keyed lookup from both the build and the per-call
path:

* the row index is stamped on the `Leaf` (`leaf._accel_row`), so an edge's
  target row is an attribute read;
* the per-node sorted time index that `_build_leaf_labels` rebuilds on every
  call is built once and cached on the topology;
* the accelerated path builds the label array straight from the segments and
  never constructs the Leaf-keyed `leaf_labels` dictionary at all.

## 4. Measurements

`benchmarks/bench_backends.py`, Apple Silicon, Python 3.11, Cython 3.3.0,
`-O3`. Planted two-community streams; both size axes varied. Speedups are
relative to the pure-Python path in the same process shape. *Cold* is one call
on a freshly built stream (topology built inside the measurement); *warm* is
repeated calls on the same stream.

| case | nodes | time steps | time-edges | time-nodes | py cold | py warm | cython cold | cython warm |
|---|---|---|---|---|---|---|---|---|
| tiny | 20 | 5 | 177 | 97 | 0.13 ms | 0.12 ms | 0.95× | 2.96× |
| wide | 300 | 5 | 7 814 | 1 500 | 2.9 ms | 3.1 ms | 1.09× | 6.09× |
| wider | 800 | 5 | 18 571 | 4 000 | 9.1 ms | 13.2 ms | 1.09× | 9.04× |
| deep | 20 | 400 | 12 856 | 7 799 | 9.3 ms | 9.2 ms | 1.20× | 6.04× |
| deeper | 20 | 1 600 | 51 258 | 31 216 | 45.5 ms | 41.9 ms | 1.42× | 5.63× |
| square | 120 | 120 | 24 833 | 13 981 | 19.6 ms | 20.7 ms | 1.34× | 7.97× |

Reading it honestly:

* **A single scoring of a fresh stream gains little** (0.95–1.42×). The
  topology build is Python work of the same order as the loop it replaces; the
  compiled kernel then makes that loop nearly free, and the net is a modest
  gain, or noise on a 0.1 ms call.
* **Repeated scoring of one stream gains 3–9×.** That is the shape of a lex
  comparison, a γ/ω sweep, an evaluation loop over candidate partitions, or a
  benchmark. The topology is paid once.
* The residual warm cost is Python outside the kernel: segment-to-label
  bisection, the closed-form expectations, and — when the partition is handed
  in as a raw `dict` of members rather than a `TimeModules` — its compression
  to segments on every call.

## 5. Verification — every option

`benchmarks/compare_backends.py -n 4000 --seed 1`, one subprocess per backend,
values compared by `float.hex()`:

```
python vs cython
  metric values:      120660/120660 bit-identical
  lago partitions:    11/11 identical
```

What those 120 660 values cover:

| option | where it is exercised |
|---|---|
| undirected / **directed** (the `× 2`) | 4 000 random streams, 40 % directed; `directed_medium`, `bipartite_directed_medium` |
| unweighted / **weighted** (float sums) | 35 % of random streams, weights in [0.2, 3.0]; `weighted_medium` |
| **bipartite / k-partite** (`partite_mapping`, partial mapping) | 15 % of random streams; `bipartite_medium`, `bipartite_directed_medium` |
| **continuous** (links with duration, split edges, overlapping module ends) | `continuous_medium` |
| **delayed** (different source / target times) | `delayed_medium` |
| **self-loops** (the folded doubling) | random streams draw `a == b` |
| **unlabelled time-nodes** (partial partitions) | every LAGO partition; random partitions |
| **empty community** (label with no time-node in the stream) | singletons partition on every named stream |
| **γ = 0**, **ω = 0** (a term skipped) | parameter sets `(0.0, 2.0)`, `(0.5, 0.0)` |
| **ndigits** 5, 10, 12 | all parameter sets |
| MM, JM, CM | every case |
| partitions **produced by LAGO**, not only random ones | every named stream, `seed=7, nb_iter=2` |
| large streams (cache and flat form under load) | `planted_xlarge`, fixture (2 096 links) |

Also:

* the full suite, **547 passed** (396 existing + 151 new), with `LAGO_ACCEL=cython`,
  and 546 passed + 1 skipped without the extension on the path;
* `tests/metrics/test_accel.py` checks the reference kernel against the Python
  loops, the segment-to-label array against `_build_leaf_labels`, the wired
  metric against the unwired one, cache invalidation on `add_links`, and — when
  importable — the compiled kernel against the reference kernel on random flat
  inputs with float weights.

## 6. Limits, stated plainly

* Only `longitudinal_modularity` is faster. `lago_modules` is not, by design of
  the boundary; making it faster is the port discussed in
  `FASTER_LANGUAGE_ANALYSIS.md` §5.
* The gain is for repeated scoring. A caller who scores each stream exactly
  once should not expect more than ~1.1–1.4×.
* Integer-weight totals above 2⁵³ would differ in the last bit; no realistic
  stream reaches that.
* A compiled artefact is per Python version and platform; the fallback is
  silent by default, so a deployment that expects the speedup should set
  `LAGO_ACCEL=cython` to make a missing build an error.
* `Leaf` gained one attribute (`_accel_row`, −1 until a topology is built). It
  is written only by `Topology.__init__`.

## 6a. Since then: the compiled path starts at the second scoring

The "cold" column of §4 was the honest weak spot: building the flat topology costs about one
Python scoring, so a stream scored a single time paid for nothing. `lago.accel.should_accelerate`
now takes the compiled path from the **second** scoring of a stream on (or whenever a valid
topology is already cached): a single scoring runs the Python loops and is exactly as fast as
without the kernel, the second breaks even, every later one is several times faster. The two
paths return the same double, so which one served a given call is invisible in the result.

## 6b. Since then: the compiled core

Round 2 (`PERFORMANCE_ROUND2.md`) added `accel_cython/setup_core.py`, which compiles the
package's own hot modules in place in Cython pure-Python mode — including `lago/accel.py`,
so the topology build of this kernel's cold path is compiled too. It is a separate, optional
build from the kernel above; `compare_backends.py` now checks the four combinations.

## 7. The Rust + PyO3 alternative, deferred

Not built, at the user's request. If it is picked up, nothing above needs to
change:

* the contract is `lago.accel._reference_kernel`, taking `array.array` buffers
  (`int32` labels / rows, `int64` row pointers, `float64` weights) and
  returning `(list[float], int)`; PyO3 receives these as `PyBuffer<i32>` /
  `PyBuffer<i64>` / `PyBuffer<f64>` or via `numpy` views if that dependency is
  acceptable;
* it is one more entry in `_BACKENDS` in `lago/accel.py`, before `cython` if it
  is to be preferred;
* `benchmarks/compare_backends.py` and `bench_backends.py` pick it up from
  `available_backends()` with no change;
* the same three prohibitions apply — no fast-math, no vectorised
  accumulation, no threads — for the same reason: the summation order is the
  result.

Expected speed is the same as the Cython kernel's: both are a tight C-level
loop over the same arrays, and the remaining cost is on the Python side of the
boundary either way. The reasons to prefer it would be packaging (`maturin`
wheels without a C compiler on the user's machine) and memory safety of the
kernel, not throughput.
