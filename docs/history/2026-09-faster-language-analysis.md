> **Historical.** Written during the question of porting the core to a faster language, between rounds 1 and 2; its recommendations were carried out in round 2, September 2026, as a working note; kept
> verbatim as the record and not maintained. The current state of the code is described in
> [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) and [`docs/PERFORMANCE.md`](../PERFORMANCE.md);
> the verification harness in [`benchmarks/README.md`](../../benchmarks/README.md).

# Should LAGO's core be wrapped in a faster language?

## Context

`lago_modules` is now 5–32× faster than it was (80.6 s → 1.61 s on the 120-node × 20-step
stream) after the algorithmic work: per-module caches, closed-form expectations, and an
O(|M0|) incremental candidate evaluation. The question is whether the next step is to move the
hot path into a compiled language.

**Answer: yes eventually, but not as a "wrapper", and not first.** The profile is now
interpreter-bound, so a compiled core would genuinely pay — but the change that unlocks it is a
data-model change that is worth doing in Python first, on its own merits, and which also fixes a
memory ceiling that will bite before the speed one does.

---

## 1. Where the time goes now

`planted_large`, MM + STEM, 1.61 s wall. The profile is **flat** — no function above ~15 % —
which is itself the finding:

| | calls | tottime |
|---|---:|---:|
| `set.__contains__` | 4 842 331 | 0.653 s |
| `_get_weight_diff` | 214 793 | 0.447 s |
| `Leaf.__hash__` | 6 890 457 | 0.361 s |
| `_duration_changes_adding` | 165 057 | 0.266 s |
| `duration_delta_on_add` | 486 281 | 0.252 s |
| everything else | | < 0.25 s each |

Aggregate: **23.1 M function calls in 1.6 s — about 69 ns per call, 54 % of them Python-level.**

That is the signature of a workload bound by interpreter dispatch, not by any one algorithm.
It is the profile where a compiled port genuinely wins, and equally the profile where further
micro-optimisation of Python yields very little.

## 1b. How it scales — on both axes

You were right that size is not just node count. Measured separately (planted streams, MM+STEM):

| vary **timesteps**, 40 nodes | leaves | time-edges | time | growth |
|---|---:|---:|---:|---:|
| 5 | 191 | 279 | 0.058 s | |
| 20 | 771 | 1 211 | 0.215 s | ×2.45 per doubling |
| 80 | 3 106 | 4 784 | 1.051 s | ×2.37 per doubling |

| vary **nodes**, 10 timesteps | leaves | time-edges | time | growth |
|---|---:|---:|---:|---:|
| 20 | 159 | 138 | 0.020 s | |
| 80 | 800 | 2 536 | 0.384 s | ×4.41 per doubling |
| 160 | 1 600 | 10 330 | 1.583 s | ×4.12 per doubling |

The node axis looks quadratic and the timestep axis linear, but that is an artefact of the
generator: at fixed edge probability, doubling nodes quadruples the edges. Normalised:

**≈ 150 µs per time-edge, in every configuration** — 145, 155, 151, 153 µs/edge across the node
sweep; 157–220 µs/edge across the timestep sweep.

> **Correction — this does not hold at scale.** `benchmarks/scale.py` was built to test exactly
> this claim and refuted half of it. Up to ~14 k time-edges the cost looks shape-independent;
> past that the two axes diverge sharply. See §1d.

## 1c. Memory will bite before time does

Memory of the `LinkStream` representation alone, before the search allocates anything:

| stream | leaves | time-edges | MB | bytes/leaf | bytes/time-edge |
|---|---:|---:|---:|---:|---:|
| 40 × 10 | 381 | 562 | 0.4 | 1 064 | 721 |
| 80 × 20 | 1 600 | 5 075 | 2.5 | 1 541 | 486 |
| 120 × 40 | 4 800 | 23 211 | 8.5 | 1 766 | 365 |

Extrapolating at ~350–500 bytes per time-edge (time now depends on shape — see §1d):

| time-edges | memory (representation only) |
|---:|---:|
| 10⁵ | ~0.05 GB |
| 10⁶ | ~0.4 GB |
| 10⁷ | ~4 GB |

Measured search times at 2×10⁵ time-edges: **74 s** (`deep`), **69 s** (`balanced`),
**137 s** (`wide`).

A `Leaf` is a full Python object with two sets and four attributes; `_LagoModule.leaves` are
hash sets of those objects. At 10⁷ time-edges you are out of memory on a laptop before the
runtime becomes the complaint. **Any plan for "extra big" has to address the representation, not
just the language.**

---

## 1d. At scale the two axes diverge — one is linear, one is not

Measured with `benchmarks/scale.py`, MM + STEM, at matched time-edge counts:

| shape | target | time-edges | µs/time-edge |
|---|---:|---:|---:|
| `wide` (many nodes, 10 steps) | 2 k / 20 k / 200 k | 1 216 / 19 091 / 199 056 | 155 → 271 → **687** |
| `deep` (60 nodes, many steps) | 2 k / 20 k / 200 k | 1 850 / 18 498 / 185 195 | 395 → 359 → **400** |
| `balanced` | 2 k / 20 k / 200 k | 993 / 16 549 / 188 871 | 205 → 429 → 366 |

**`deep` is linear. `wide` is not** — 4.4× dearer per edge over a 160× size increase.

Decomposing the wide growth into "how much work" and "how dear each unit of work is":

| shape | target | evaluations/edge | µs/evaluation |
|---|---:|---:|---:|
| `wide` | 5 k / 20 k / 80 k | 25.1 → 29.9 → **36.5** | 8.02 → 9.51 → **13.47** |
| `deep` | 5 k / 20 k / 80 k | 39.4 → 39.7 → 39.6 | 10.40 → 9.44 → 10.25 |

Both factors grow in `wide`, and they compound. The second has an identified cause — mean
parent-module size, measured per `find_best_module_for_submodule` call:

| shape | 5 k | 20 k | 80 k |
|---|---:|---:|---:|
| `wide` mean \|parent\| | 82.3 | 330.5 | **982.6** |
| `deep` mean \|parent\| | 139.3 | 63.9 | 62.8 |

\|M0\| stays at ~2 throughout. So `M1_leaves = submodule.parent.leaves - M0_leaves`
(`find_best_move.py`) is an **O(\|parent\|) set difference per candidate evaluation**, and in
wide streams \|parent\| grows roughly linearly with the node count. That is the last remaining
term that scales with module size rather than with the move.

**Fixed** (`leaf_set.LeafDifference`): `M1_leaves` is only ever used for membership tests and
one comparison, so it is now a view over `(parent.leaves, M0_leaves)` — O(1) to construct, two
set lookups per membership test, materialised only if something iterates it. Measured:

| shape | 2 k | 20 k | 200 k |
|---|---:|---:|---:|
| `wide` µs/time-edge **before** | 155 | 271 | **687** |
| `wide` µs/time-edge **after** | 171 | 276 | **437** |

200 k `wide` goes 136.7 s → 87.1 s, **1.57×**, and the benefit grows with parent size (13 % at
80 k, 36 % at 200 k). Per-evaluation cost growth across the range drops from 1.68× to 1.39×.

`wide` is still mildly superlinear. The remaining part is the other factor — more evaluations
per edge as the node count grows (25.1 → 29.9 → 36.5) — which is the search genuinely doing more
merging work, plus a smaller residue of per-module recomputation on accepted moves. No change of
language addresses the first of those.

## 2. "Wrapping" is not the available shape

A wrapper implies a self-contained numeric function you hand data to. The hot path is not that.
It is a search that walks a graph of `Leaf` objects through `topo_neighbors` and
`left/right_time_active_neighbor`, testing membership against live `_LagoModule.leaves`, mutating
module contents as it goes. Per candidate it touches a handful of objects and returns a float.

* A compiled kernel cannot use that representation. It needs indices and arrays, so there is a
  conversion layer at the boundary.
* The boundary cannot be per candidate: 214 793 candidate evaluations in a 1.6 s run, so
  per-call FFI overhead would eat the gain. The compiled side must own the **whole search loop** —
  `find_best_move`, `delta_lm`, and the `tmm`/`stem`/`stnm` exploration loops.
* That is a port of the algorithm core, not a wrapper around it.

`longitudinal_modularity` is the exception: it is a pure function of (stream, partition) and
*is* genuinely wrappable on its own.

---

## 3. What is available without leaving Python

Membership testing over 1000 probes:

| representation | time | vs current |
|---|---:|---:|
| `Leaf in set[Leaf]` — current, value-based `__hash__` | 47.8 µs | 1.00× |
| identity-hashed object in set | 14.7 µs | 3.26× faster |
| `int in set[int]` | 19.5 µs | 2.45× faster |
| `bytearray[index]` | 14.8 µs | 3.22× faster |

**The determinism mechanism costs ~69 % of every membership test.** Freezing the exploration
order was done by giving `Leaf` a value-based `__hash__` — those are the 6.89 M `__hash__` calls,
and they make each of the 4.84 M `set.__contains__` calls ~3× dearer than necessary.

Set operations on leaves are ~1.2 s of profiled time. Moving to **integer leaf indices with
array/bitset membership** would:

* recover most of that 3× — estimated **~1.5–1.8× end-to-end, still in pure Python**;
* remove the determinism tax entirely (index order *is* deterministic — no custom hash needed);
* cut memory by roughly an order of magnitude (§1c), which is the actual blocker at 10⁷;
* make the exploration order an explicit permutation of integers, which is what §4 needs;
* produce exactly the representation a compiled port requires.

That convergence is the argument for sequencing. It is not a detour on the way to a port — it is
most of the port's design work, done in a language where the existing test harness still applies.

---

## 4. A correctness issue — since fixed

A live regression from the determinism work:

```
seed 1 / 2 / 3 / 99    -> 1 distinct result   (seed is a no-op)
nb_iter=1 vs nb_iter=5 -> identical           (the 5 runs are the same search)
```

Before the freeze, run-to-run variation came from memory-address randomness, so `nb_iter`
accidentally explored different trajectories. Now **`nb_iter=5` runs the same search five times
at five times the cost**, and `seed` changes nothing — while the docstring still promises
"Number of LAGO runs. Best result is returned. Higher values reduce sensitivity to the greedy
optimization's starting point."

**Design that keeps reproducibility and avoids any regression:**

* Derive the exploration order from a seeded RNG — a permutation applied to the exploration sets
  in `tmm._exploration_loop`, `stem._build_stem_iterator`, `stnm._stnm_iterator`.
* Iteration *i* of `nb_iter` uses a seed derived from `(seed, i)`.
* **Define iteration 0 with the default seed as the identity permutation**, so every result
  produced today is reproduced bit-for-bit. `nb_iter=1` and `seed=None` change nothing;
  `nb_iter>1` and explicit seeds gain the exploration they were supposed to have.

Reproducible for a given `(input, seed)`, genuinely varied across seeds. This must land before
any port — porting a multi-restart search whose restarts do nothing would bake the bug into two
languages.

---

## 5. If you port: scope, tool, cost

**Scope.** The search loop only: `delta_lm.py`, `find_best_move.py`, and the inner loops of
`tmm.py` / `stem.py` / `stnm.py`, over an index/array representation built once from the
`LinkStream`. Keep in Python: the public API, `LinkStream` construction, `TimeModules`, I/O and
`viz`. `longitudinal_modularity` can be wrapped separately.

**Tool, ranked for this project:**

1. **Cython or mypyc.** Keeps the source recognisably Python, which matters for a published
   method that researchers are expected to read and modify. `cython`, `gcc` and `clang` are all
   already available here.
2. **Rust + PyO3.** More headroom and better memory safety, but introduces a second language into
   a research codebase; no Rust toolchain installed here.
3. **PyPy** is not a port but is worth one afternoon of measurement first — this workload
   (millions of cheap Python-level calls, no numpy in the core) is close to its best case, and it
   is zero code change. Not installed here, so I could not measure it.

**Distribution cost, honestly.** You said compiled is acceptable, so this is a note rather than
an objection: the repo currently has **no CI at all** — no `.github`, no workflows, no `tox`, no
`Makefile` — and ships pure Python. A compiled core adds `cibuildwheel` across ~15
platform/Python combinations, a release process, and an sdist fallback that needs a toolchain on
the user's machine. Given you want to serve *small* streams too, an **optional accelerated
backend with silent fallback to the Python implementation** is worth considering: small users
never need the wheel, and it keeps the Python core as the living reference the tests already
compare against.

(Aside: the core's only dependency is vestigial — `numpy` is imported in
`lago/algorithm/lago.py` solely for `np.random.seed(seed)` at line 111, which the algorithm never
consults. Dropping it makes the algorithm core dependency-free.)

**Expected gain.** 23 M operations at ~69 ns ≈ 1.6 s; the same logical work on arrays in compiled
code is plausibly 0.05–0.15 s, so **10–20× over current** is a realistic target — bringing 10⁶
time-edges from ~2.5 min to ~10 s, and making 10⁷ feasible once the memory is also fixed.

---

## 6. Recommendation

Given "all sizes, small to extra big" and "compiled is acceptable":

1. ~~**Fix `seed` / `nb_iter`**~~ — **done.** The exploration order is now seedable; the default
   is unchanged (51/51 identical partitions against the pre-change snapshot), and `nb_iter=5`
   gains +0.017 L-modularity on `planted_large`.
2. **Remove the last O(\|parent\|) term** (§1d). `M1_leaves` is only used for membership tests;
   replacing the set difference with a view is a small, contained change that removes about half
   the wide-stream superlinearity. Do this before deciding anything about a port — otherwise a
   compiled core inherits the same asymptotics.
3. **Move to integer leaf indices and array membership.** ~1.5–1.8× in pure Python, ~10× less
   memory, removes the determinism tax, and is most of the design work for a port. Re-profile
   after — it will change the shape again.
4. **Measure PyPy.** One afternoon, zero code change, possibly decisive for mid-size streams.
5. **Then port the search loop**, Cython/mypyc first, ideally as an optional backend with the
   Python implementation retained as the reference.

Steps 2–3 are worth doing whatever you decide about step 5, and step 3 is a prerequisite for it.
I would not start at step 5 — not because the idea is wrong, but because porting the *current*
object-graph representation would mean designing the array model under compilation pressure,
without the Python harness to check each step against; and because §1d shows there is still
asymptotic work to do, which no language change fixes.

---

## 7. Verification

Every step is checkable with the harness that already exists, which is what makes a port
tractable at all:

```bash
python3.11 -m pytest -q                                              # 380 tests
python3.11 benchmarks/compare_matrix.py  --ref-dir R --ref-name B    # identical partitions
python3.11 benchmarks/compare_metric.py  --ref-dir R --ref-name B -n 5000
python3.11 benchmarks/compare_delta.py                               # per-call numerics
python3.11 benchmarks/check_determinism.py --reps 2                  # 54/54
```

For a port: keep the Python implementation in the tree as the oracle and run the compiled core
against it call-for-call, the way `compare_delta.py` does.

**Two cautions from the work just completed.** A port is a rewrite of the numerics, and that is
precisely where this code has bitten twice — a factor-2 error in the directed closed form that
partition fingerprints did *not* catch, and a k-partite mask whose omission passed 321 tests.
Anything that re-derives the formulas must be checked **per call against an independent
reference**, never by comparing final outputs.
`tests/algorithm/test_delta_expectations.py` is that reference and should be the acceptance test
for any reimplementation.

Also add scale benchmarks to `benchmarks/streams.py` before starting: the current generators top
out at 200 nodes / 25 steps, and nothing there exercises the 10⁵–10⁷ time-edge range these
decisions are about. Both axes need covering independently — many nodes × few steps, and few
nodes × many steps — since they stress different parts of the representation even at equal edge
counts.
