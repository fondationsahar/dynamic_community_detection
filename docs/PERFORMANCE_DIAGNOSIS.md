# LAGO performance: diagnosis, changes and verification

**Scope:** speed up `longitudinal_modularity` (`lago/metrics/modularity.py`) and
`lago_modules` (`lago/algorithm/lago.py`) without changing their outputs.

**Status: implemented and verified.** See §0 for what shipped and how it was checked; §1–§5
are the diagnosis that led there; §6–§8 the decisions and the first round of work; §9 the
second round, which targeted scaling to large streams; §9b restores `seed` / `nb_iter`.

---

## 0. Result

| | speedup | verification |
|---|---|---|
| `longitudinal_modularity` | **1.2× – 12×** (grows with community size) | 60 000 comparisons against the pre-change reference, **0 differing values** |
| `lago_modules` | **5.3× – 32×** (32× on 120 nodes × 20 steps: 51.9 s → 1.6 s) | 96 configurations, **96/96 byte-identical partitions** |

Measured against a snapshot of the package taken immediately before each change, imported
side by side in the same process, so both arms run the same trajectory. End to end against the
*original* code the figures are larger still — **4.4× – 50×**, with a 200-node stream going from
218 s to 4.4 s — but that comparison also crosses the determinism fix, so the table above is the
defensible one.

What made the large cases move was removing the dependence on module size: after the first
round, evaluating one candidate move still cost O(size of the module being evaluated). It is now
O(|M0|), the number of time-nodes actually moving — 1.87 on average. See §9.

Also delivered:

* **`lago_modules` is now reproducible.** 48/48 configurations return the identical partition
  across repeats *and across separate processes*; before, 13 of 48 varied within a single
  process. This is what makes "outputs must be the same" checkable at all.
* **Eight bugs fixed** (§3), five of which were invisible before: the durations function
  returned different answers for the same input, the k-partite null model was identically
  zero, the oscillation caches were keyed on recycled memory addresses, delayed duplicate
  detection matched the wrong pairs, and the JM k-partite sum was missing a factor of 2.
* **A benchmark and equivalence harness** under [`benchmarks/`](../benchmarks/), so this stays
  measurable: `check_determinism.py`, `compare_matrix.py` (partition equality),
  `compare_metric.py` (metric equality at scale), `compare_delta.py` (per-call numerical
  equality), `bench_metric.py`, `compare_ref.py`, `make_ref.py`, and `scale.py` (growth on
  both size axes).
* Test suite grew from **246 to 396** passing; lint is below the starting baseline.
* The last term scaling with module size is gone: `M1_leaves` is a view, not a copy
  (`leaf_set.LeafDifference`) — 1.57× on a 200 k-time-edge many-node stream.

**One methodological finding worth keeping.** Partition fingerprints are *not* sufficient to
validate a change of formula. An early version of the closed-form MM delta was wrong by exactly
2× on every directed call, and the end-to-end check still reported 24/24 identical partitions —
the error did not happen to flip any greedy decision on that corpus. `compare_delta.py` exists
because of that: it compares the two implementations per call, and it caught it immediately.

---

## 1. Context & method

`lago_modules` runs the greedy optimisation; `longitudinal_modularity` scores a partition.
Both were profiled on **ten link streams of different types** — instantaneous, weighted,
directed, k-partite, continuous, delayed, and the 2096-link real fixture
(`tests/fixtures/linkstream.txt`) — from 24 to 200 nodes, across `lex ∈ {MM, JM, CM}`,
`refinement ∈ {STEM, STNM, None}`, `fast_exploration ∈ {True, False}` and
`refinement_in ∈ {True, False}`.

All measurements use `/opt/homebrew/bin/python3.11` (Python 3.11.11, numpy 2.2.1). The
default `python3` on this machine is 3.7 and cannot run the package.

Test baseline, recorded before any change:
`PYTHONDONTWRITEBYTECODE=1 python3.11 -m pytest -p no:cacheprovider -q` → **246 passed,
17 warnings, 45.0 s** (all 17 are `DeprecationWarning` from the deprecated `add_*_links`
aliases).

Findings were then re-derived independently by a 47-agent adversarial audit (4 investigators
× per-finding refutation panels, 3.9 M tokens). 38 findings survived verification, 5 were
refuted; the refutations are recorded in §10 because two of them corrected *my* claims.

Scripts live in the session scratchpad: `gen.py` (stream generators), `prof.py`/`prof2.py`,
`optlib.py`, `lm_full.py`, `lago_full.py`, `proto_lago.py` (shadow harness), `protoB.py`
(the O8/O9/O10 prototype), `freeze.py`/`freeze_check.py` (determinism), `bip_check.py`,
`debug_durations2.py`, `check_bugs.py`, `determinism.py`, `lm_scaling.py`.

---

## 2. Diagnosis

### 2.1 `lago_modules` — where the time goes

`cProfile`, planted network 120 nodes × 20 timesteps, `lex="MM"`, `refinement="STEM"`,
total **80.6 s**:

| function | calls | tottime | share |
|---|---:|---:|---:|
| `delta_lm._partial_expectation_mm_diff` | 42 434 752 | 29.0 s | 36 % |
| `core/utils.get_nodes_durations` | 427 038 | 15.1 s (17.1 cum) | 21 % |
| `delta_lm._get_weight_diff` | 213 519 | 10.2 s (22.1 cum) | 27 % |
| `delta_lm._get_expectation_mm_part` (own) | 213 519 | 7.5 s | 9 % |

≈ 97 % of the run is these four. Everything else is below 1 % each.

One structural cause: **evaluating one candidate move costs O(|Mx|), the size of the candidate
module**, although a move touches one or two time-nodes. Instrumented over 282 k candidate
evaluations on five stream types: mean |M0| = **1.87 leaves**, mean |Mx| = 46–75, mean
|parent| = 86–153.

* `_get_weight_diff` (`delta_lm.py:81`) walks **every leaf of Mx** and all its neighbours,
  then keeps the ones pointing into M0.
* `_get_expectation_mm_part` (`delta_lm.py:248`) iterates `n(n+1)/2` node pairs, although
  every pair with no node in M0 contributes *exactly* `0.0`. It also makes ~6 passes over Mx
  per candidate: it builds `M0_leaves | Mx_leaves` twice (`:270`, `:276`), calls
  `get_nodes_durations` twice (O(|Mx|) each), and rebuilds `{leaf.node for leaf in union}`.
* `get_nodes_durations` (`core/utils.py:95`) rebuilds everything from scratch each time,
  although only M0's own nodes can differ between `Mx` and `Mx ∪ M0`.

### 2.2 `longitudinal_modularity` — where the time goes

| part | share | cost model |
|---|---:|---|
| `_count_intra_community_interactions` (`modularity.py:309`) | ~41 % | O(time-edges) |
| expectation loop (`_compute_{mean,joint,coexistence}_expectations`) | ~41 % | **O(Σ_c n_c²)** |
| `_count_community_switches` (`modularity.py:483`) | ~13 % | O(leaves) |

The quadratic term dominates as soon as communities are large:

| network | time-edges | Σ n_c² | MM | JM | CM |
|---|---:|---:|---:|---:|---:|
| 60 nodes, 6 comms | 962 | 600 | 0.76 ms | 0.67 ms | 0.77 ms |
| 120 nodes, 4 comms | 2 894 | 3 600 | 2.52 ms | 2.26 ms | 2.64 ms |
| 240 nodes, 4 comms | 6 026 | 14 400 | 7.61 ms | 6.71 ms | 7.95 ms |
| 400 nodes, 2 comms | 16 223 | 80 000 | 32.6 ms | 25.8 ms | 33.5 ms |
| 600 nodes, 2 comms | 13 729 | 180 000 | **58.7 ms** | 51.2 ms | 61.7 ms |

The last row has *fewer* edges than the one above but is nearly twice as slow — Σ n_c², not
network size, drives the cost.

The counting loops are O(E) and cannot improve asymptotically, but they rebuild a
`(node, time)` tuple for **every neighbour of every leaf** just to index `labels`
(`:332`, `:509`, `:517`). That allocation is most of their cost. Separately,
`communities_leaves` (`:241-245`) is **dead code** — confirmed by bytecode disassembly, it is
stored and never read — and it costs a full pass over all members on every call.

### 2.3 Reproducibility

`lago_modules` is not reproducible, even with `seed=42`, even inside a single process:

```
run B  planted_medium     L-modularity 0.5377 / 0.5766 / 0.5041   (3 calls, same input)
       continuous_medium  L-modularity -0.446 / -0.516 / -0.397
       fixture            L-modularity  0.716 /  0.722 /  0.717
```

Over the full 48-configuration matrix (8 streams × lex × refinement), **13 of 48 configurations
returned different partitions across 3 repeats in one process.** An independent check on the
real fixture got 25 distinct partitions from 25 calls.

`Leaf`, `TimeEdge` and `_LagoModule` use identity hashing, so every `set` iteration order
depends on memory addresses. `seed` only seeds `random` and `numpy`, which the algorithm never
uses — the docstring's reproducibility claim (`lago.py:73-81`) does not hold, and its
`PYTHONHASHSEED` advice is wrong (int/tuple hashing is not randomised; the cause is `id()`).

That variability is expected of a greedy search. The problem is that the exploration order is
the *allocator's*, so no experiment can be repeated — which is why freezing it is Step 1 of
the plan (§8) rather than an optional extra.

---

## 3. Bugs found

Pre-existing and independent of the optimisation. **Nothing has been fixed.**
B1, B2, B4, B5 were found by profiling; B6 was found twice independently (by me and by the
audit); B7 and B8 come from the audit.

### B1 — `get_nodes_durations` is not a function of its input (`core/utils.py:95`)

`right_duration` (`:121`) is initialised to `1` and only updated *inside* the right-extension
loop (`:127-131`), so the result depends on which leaf `set.pop()` yields first. Repro
(`debug_durations2.py`; continuous stream, node 14 active at t=30,31, then t=32 with a
duration-6 edge; input = the two leaves at t=30 and t=31):

```
pop leftmost  first -> {14: 7}
pop rightmost first -> {14: 2}
```

Same input, two different answers. Only bites when consecutive leaves of a node carry edges of
different durations — i.e. **continuous link streams**. Consequence: on continuous streams the
objective LAGO optimises is address-dependent, and *"do not change the output" is undefined
there until this is fixed*. An independent check found the current function disagrees with a
deterministic recompute in 217 of 400 sampled states.

`get_expanded_module` (`core/utils.py:217`) extends a segment by the duration of an edge at the
segment's **last leaf**; `get_nodes_durations` uses one at the leaf **after** the segment. The
`get_expanded_module` convention looks like the intended one.

### B2 — k-partite expectation is always zero (`modularity.py:100`)

```python
if ls.partite_mapping.get(source, -1) == ls.partite_mapping.get(source, -2):
    return 0
```

`source` twice; the right-hand side should be `target`. Any node present in `partite_mapping`
makes the contribution `0`, so **every** degree contribution is zero and the null model
disappears from `longitudinal_modularity` on k-partite networks. Verified: `contribution(0,31)`
between two *different* partites returns `0`. `delta_lm.py:218,234,304` do the same comparison
correctly, so the algorithm and the metric disagree. Measured effect of the fix on the
bipartite generator: **−19 %** on the reported modularity.

### B3 — `lago_modules` is not reproducible

See §2.3.

### B4 — `LinkStream._compute_time_neighbors` is quadratic (`linkstream.py:646`)

```python
for node in self.nodes:
    times = sorted({time for tmp_node, time in self.leaves_dict if tmp_node == node})
```

Full rescan of `leaves_dict` per node ⇒ O(nodes × leaves). Measured 0.026 / 0.102 / 0.425 s for
200 / 400 / 800 nodes (8k / 16k / 32k leaves) — a clean 4× per doubling. Called from `:341`,
`:455`, `:580`, `:644`, i.e. on every `add_links`, and twice more for continuous streams inside
`lago_modules`. A single grouping pass is 2.4×–6.7× faster on `add_links` and removes the
quadratic term; verified to produce an identical neighbour structure.

### B5 — off-by-one in `_split_continuous_linkstream` (`linkstream.py:608` vs `:629`)

`duration = time_end - time_start + 1` for `topo_neighbors` but `time_end - time_start` for
`topo_neighbors_from`. **This is a live correctness bug, not a cosmetic inconsistency:** on
directed continuous streams every mirrored in-edge gets a duration one smaller than its
out-edge twin, which already makes `_get_weight_diff` direction-dependent today. Restoring the
`+ 1` took mirroring violations from 1478 → 0 and value mismatches from 20823 → 0 over 120
random directed-continuous streams. It is also the sole blocker for optimisation O1 (§4.2).

### B6 — the oscillation caches key on `id()` of throw-away objects (`stem.py:128`, `tmm.py:143`)

```python
move_key    = (id(child_module), old_parent_id, id(best_module))
reverse_key = (id(child_module), id(best_module), old_parent_id)
if reverse_key in move_cache: ...   # block the move
```

In STEM, `create_module_from_leaves` (`lago_tools.py:111`) allocates a brand-new `_LagoModule`
for **every edge evaluation** and nothing retains it, so CPython recycles the addresses:
**96–99 % of the cache keys are recycled addresses.** A lookup can therefore hit an entry
belonging to a completely unrelated earlier move and block a legitimate one.

Proof (`cont_probe.py`): keeping every `_LagoModule` alive for the duration of a run — the only
change — turns `continuous_medium MM STEM` from unstable (2 distinct partitions in 5 runs) into
perfectly stable (5/5 identical). It was also the **last** remaining source of nondeterminism
after value-based hashing was applied (§5.4).

### B7 — delayed mode misses time-swapped duplicates (`linkstream.py:533`)

For undirected delayed links the duplicate key is
`(min(source,target), max(source,target), source_time, target_time)`. The link `(a, b, t1, t2)`
and its time-swapped form `(b, a, t2, t1)` describe the same undirected temporal edge but
produce different keys, so both are accepted — double-counting `ls.weight` and both degrees.

### B8 — JM k-partite omits the `2^(n1≠n2)` factor (`delta_lm.py:225,239`)

`_get_expectation_jm_kpartite_part` sums `degree1 * degree2 * (...)` over
`combinations_with_replacement`, while the MM path (`:313`, `:316`) multiplies by
`2 ** (node1 != node2)`. The upper-triangular sum therefore under-counts off-diagonal pairs by
2× relative to MM. Same function also builds `in_Cx` with **list** membership (`:209-211`),
making it O(n²) — a one-word fix (`{*...}` instead of `[*...]`).

---

## 4. Proposed optimisations

### 4.1 Stage 1 — bit-exact

| # | change | file | effect |
|---|---|---|---|
| **O2** | `_get_expectation_mm_part`: skip node pairs with **no** node in M0 — such a node has an identical duration in `Mx` and `Mx ∪ M0`, so `√(U₁U₂) − √(R₁R₂)` is exactly `0.0`. Relative order of surviving terms preserved ⇒ same float sum. | `delta_lm.py:248` | 3.0× fewer pairs (28.1 M → 9.3 M across the corpus). **Proven bit-exact** for instantaneous, weighted, directed, k-partite and delayed streams (structural proof + 0 violations in ~50 k comparisons + 0 in 634 k real calls). Continuous streams: differs, and B1 is provably the sole cause. |
| **O4** | Build the union once, hoist `self.linkstream.*` to locals, inline `_partial_expectation_mm_diff` | `delta_lm.py:248,284` | removes 42 M Python-level calls |
| **O5** | `get_nodes_durations` micro-optimisations | `core/utils.py:95` | 1.33–1.54× on that function; 0 mismatches over 1.47 M real calls |
| **O6** | `longitudinal_modularity`: build a `Leaf → community` dict once (in `leaves_dict` order, so summation order is untouched), share it between both counting loops | `modularity.py:309,483` | **1.9×** on the two loops combined; bit-exact on all 9 streams incl. weighted |
| **O7** | `_compute_time_neighbors`: one grouping pass (fixes B4) | `linkstream.py:646` | O(N·L) → O(L); 2.4–6.7× on `add_links` |
| **O11** | Delete the dead `communities_leaves` block | `modularity.py:241-245` | removes a full pass over all members per call |
| **O12** | Drop the provably no-op `module.leaves - M0_leaves` copy | `find_best_move.py:97` | removes an O(\|Mx\|) set copy per candidate |
| **O13** | `gamma == 0` early exit in the LAGO delta path (the metric already has one) | `delta_lm.py:54-77` | **8.1×** when `gamma=0` |
| **O14** | `get_module_duration`: stop building range-sets and calling `np.max`/`np.min` per invocation | `core/utils.py:42-69` | 42 % of a JM run; **1.26–1.34×** end-to-end, bit-exact |
| **O15** | `in_Cx` via set instead of list membership (B8's second half) | `delta_lm.py:209-211` | removes an O(n²) scan in the JM k-partite path |

### 4.2 Stage 2 — algebraically identical, float-reassociated

**O1 — `_get_weight_diff` from the M0 side** (`delta_lm.py:81`). Iterate M0's ~1.9 leaves and
test `neighb.target in Mx_leaves` instead of iterating all of Mx.

The audit established the exact condition: the multiset of `weight × duration` terms is
preserved **iff** the mirroring invariant `N(u)|v == N(v)|u` holds, and that holds *by
construction* because `_add_edge_undirected` (`linkstream.py:175-180`) and `_add_edge_directed`
(`:197-202`) each add one `TimeEdge` with the **same** `(weight, duration)` to both endpoints.
Exhaustively enumerated over every (M0, Mx) subset pair of 16–19 hand-built streams covering
self-loops, overlap, reciprocal directed edges whose `TimeEdge`s collapse in the union, delayed
edges, and duplicate/parallel continuous links: **zero violations everywhere except
directed + continuous + split** — and that exception is caused entirely by **B5**. Fix B5 and
O1 is exact in every mode, with no guard needed.

Residual: identical multiset, different summation order ⇒ ~1 ulp (6.1e-16) drift with float
weights (8 254 of 1 286 000 calls). `find_best_move.py:115,131` tie-breaks on exact float
equality, so in principle a tie could flip; in practice this is 10¹² times smaller than the
existing run-to-run spread.

Measured: 22.1 s → 0.36 s in the profile; ~15× fewer neighbour touches.

**O3 — closed-form expectations in `longitudinal_modularity`** (`modularity.py:347,389,432`),
replacing the O(n²) pair loop:

* MM: undirected `(Σᵢ dᵢ√Dᵢ)²`, directed `(Σᵢ outᵢ√Dᵢ)(Σᵢ inᵢ√Dᵢ)`
* JM: the same, times the constant community duration
* CM: `Σ_t (Σ_{i active at t} dᵢ)²` — turns O(n²·|T|) into O(#members)

The `2^[i≠j]` multiplier is exactly what turns the upper-triangular sum into the full ordered
sum, which factorises. Verified algebraically exact (no `partite_mapping`) including the
diagonal, self-loops, directed asymmetry, zero-degree nodes and empty communities.

Implementation constraints the audit pinned down:

* degrees **must** use `.get(v, 0)`; durations/times carry no `KeyError` risk.
* the CM closed form must iterate `community_nodes`, not `members`.
* guard the empty-community case **before** touching the denominator, and preserve the existing
  `ZeroDivisionError` / `OverflowError` behaviour.
* CM iterates `sorted(nodes)` while MM/JM iterate the raw set, so after B2 is fixed the partite
  mask is min-based for CM even when directed — the masked closed forms must reproduce that.
* **directed + *partial* partite coverage** (some nodes mapped, some not) makes the existing
  loop not a function of its inputs; there is no closed form that can match it, so that one
  regime must fall back to the pair loop verbatim.
* MM's diagonal term is 1 ulp off in the closed form; correcting it is not worth it.

Measured: **1.2× – 108×** on the expectation term.

### 4.3 The rounding-tie effect — the one real obstacle to "identical values"

The loop divides each of the O(n²) pair terms by `(2W)²·ND` before summing; the closed form
sums first and divides once. On integer-weight networks the exact modularity is a rational that
lands **exactly** on a half-way tie at the 6th decimal often enough to matter, and the two land
on opposite sides.

Measured over 4 000–8 000 random streams × 3 lex at the default `ndigits=5`:

| lex | rate of changed 5-digit value |
|---|---|
| JM | 5–9 / 4000 (**0.13–0.23 %**) |
| CM | 4–5 / 4000 (**0.10–0.13 %**) |
| MM | 2 / 50 000 (**0.004 %**) — not immune, `√(dₐd_b)` is exact when the duration product is a perfect square |

Every mismatch is an exact half-way tie (verified with `fractions.Fraction`: −21/64, −97/64,
−63827/40000, …, denominators always `2^a·5^b`) and the change is always exactly `1e-5`.

Two things matter for the decision:

1. **The closed form is the more accurate one.** Across 24 exactly-verified mismatches the
   closed form's pre-round double was nearer the exact rational in 22/24 and was the
   correctly-rounded double in 18/24 (the loop: 2/24). Sum-then-divide rounds once;
   divide-then-sum rounds O(n²) times. The 5th decimal is already arbitrary in the current code.
2. **Nothing in the repository breaks.** With all three closed forms patched in, the suite is
   246/246; there are no doctests with expected numeric output.

**Mitigation, verified:** compute with the closed form; if the unrounded **final**
`lm_modularity` sits within `tol` of a half-way boundary at `10^ndigits`, recompute the
expectations with the original pair loop. Over 18 000 runs this gave **0 residual mismatches**
with the guard firing 0.23–0.30 % of the time. It must be applied to the final value, not per
community — a per-community guard missed 1 of 9 real mismatches. Recommended
`tol ≈ 1e-4` in scaled units. Caveats: the guarantee holds for `ndigits ≲ 10` (the proxy
`v·10^ndigits` has its own ULP that grows with `ndigits`; at `ndigits ≥ 16` the guard can never
fire), and `ModularityResult.modularity_without_penalty` needs no extra guard since it derives
from the already-rounded fields.

### 4.4 Stage 3 — the main prize: make candidate evaluation O(|M0|)

**O8 + O9 + O10, prototyped together as `protoB.py`.** After O1/O2 the per-candidate cost is
still linear in |Mx| — and that is 72 % of the remaining hot path (`fast_get_nodes_durations`,
481 814 calls, 18.95 s tottime of 39.1 s inside `find_best`). Three changes remove it:

* **O8** cache `get_nodes_durations(module.leaves)` and the aggregate `S_r = Σ_n degₙ·√durₙ` on
  the `_LagoModule`, keyed by a version counter. The invalidation surface was established
  exhaustively by grep — **exactly four sites mutate `_LagoModule.leaves`**:
  `lago_tools.py:72`, `lago_tools.py:74`, `stem.py:273`, `stem.py:275`. Every construction site
  allocates a fresh set and needs no bump. Measured hit rate: essentially 100 %.
* **O9** derive the union's durations from the cached ones with an O(1)-per-leaf incremental
  update.
* **O10** replace the pair loop with the exact identity
  `Σ_{ordered pairs} d₁d₂(√(u₁u₂) − √(r₁r₂)) = S_u² − S_r²` (directed `A_u·B_u − A_r·B_r`;
  k-partite: subtract `Σ_p (Σ_{n∈p})²` per mapped partite). Both the *leaving* term (cached
  parent) and the *joining* term become O(|M0|), and |M0| averages 1.87.

Verification already done on the prototype: the closed-form identity shadow-verified on 260 k+
real calls across all 8 stream types (0 mismatches, max relative error 1.4e-12); the cache
shadow-verified against the original `find_best` on 8 streams × {STEM, STNM, None} — 0 `None`
mismatches, 0 delta mismatches above 1e-9, 0–2 tie-break differences per config.

**Known limitation:** the O(1) incremental duration update as prototyped hard-codes edge
duration 1, so it is **wrong on continuous streams** (12 985 of 32 734 MM calls wrong on
`continuous_medium`). The fix is to carry the added/removed leaf's actual edge duration — still
O(1) per leaf — and it requires B1 to be fixed first, because on continuous streams the current
`get_nodes_durations` has no well-defined value to match.

---

## 5. Measured results

### 5.1 `longitudinal_modularity`, end-to-end (O6 + O3)

| case | lex | current | optimised | speedup | same value @5 |
|---|---|---:|---:|---:|:--:|
| lago:planted_small | MM | 0.246 ms | 0.099 ms | **2.48×** | yes |
| lago:planted_medium | MM | 1.265 ms | 0.542 ms | **2.33×** | yes |
| lago:weighted_medium | MM | 0.979 ms | 0.370 ms | **2.65×** | yes |
| lago:directed_medium | MM | 0.749 ms | 0.288 ms | **2.60×** | yes |
| lago:bipartite_medium | MM | 1.134 ms | 0.843 ms | 1.34× | yes |
| lago:continuous_medium | MM | 1.567 ms | 0.560 ms | **2.80×** | yes |
| lago:delayed_medium | MM | 0.687 ms | 0.274 ms | **2.51×** | yes |
| lago:fixture | MM | 16.29 ms | 9.27 ms | 1.76× | yes |
| truth:n240_c4 | MM | 7.51 ms | 1.97 ms | **3.82×** | yes |
| truth:n600_c2 | MM | 59.24 ms | 3.78 ms | **15.7×** | yes |

JM 1.6–15.1×, CM 1.3–17.0×. All 30 (case, lex) combinations returned an identical value —
but see §4.3: at a 0.1–0.3 % rate a 30-case corpus is expected to show nothing. The bipartite
row is the weakest gain because it currently falls back to the pair loop (B2); fixing B2
removes that fallback.

### 5.2 `lago_modules`, O1 + O2 + O7 (median of 5 runs, MM/STEM)

| stream | current | optimised | speedup |
|---|---:|---:|---:|
| planted_small | 0.154 s | 0.077 s | **1.99×** |
| planted_medium | 5.596 s | 1.630 s | **3.43×** |
| weighted_medium | 3.037 s | 0.916 s | **3.31×** |
| directed_medium | 1.607 s | 0.519 s | **3.10×** |
| continuous_medium | 1.576 s | 0.747 s | **2.11×** |
| delayed_medium | 1.096 s | 0.426 s | **2.57×** |
| fixture | 2.754 s | 1.440 s | **1.91×** |
| bipartite_medium | 2.053 s | 0.704 s | **2.92×** |

Where a run happens to be deterministic (planted_small, weighted, directed) the optimised
version returns **exactly** the baseline L-modularity.

The bipartite row initially looked alarming (baseline sd 0, optimised sd 3.5e-2). Re-run with
15 repeats × 4 configurations (`bip_check.py`): the **baseline itself** produces three distinct
answers, and all four configurations have the same mean and spread. The `sd = 0` was five lucky
runs. This is B3, not a regression.

```
_compute_time_neighbors structure identical: True
  base   mean=0.901724 sd=2.92e-02  values [0.827586, 0.905172, 0.913793]
  +O7    mean=0.902299 sd=2.93e-02  values [0.825431, 0.829741, 0.913793]
  +O1O2  mean=0.902299 sd=2.93e-02  values [0.827586, 0.913793]
  all    mean=0.908046 sd=2.15e-02  values [0.827586, 0.913793]
```

### 5.3 `lago_modules`, with Stage 3 (`protoB.py`)

base / O1+O2 / full, best of 1–3 reps, `seed=1`, MM/STEM:

| stream | base | O1+O2 | Stage 3 | total | on top of O1+O2 |
|---|---:|---:|---:|---:|---:|
| planted_medium | 4.75 s | 1.94 s | 0.220 s | **22×** | 8.8× |
| weighted_medium | 3.02 s | 1.09 s | 0.197 s | **15×** | 5.5× |
| directed_medium | 1.82 s | 0.58 s | 0.071 s | **26×** | 8.2× |
| bipartite_medium | 2.16 s | 0.79 s | 0.158 s | **14×** | 5.0× |
| delayed_medium | 0.96 s | 0.41 s | 0.104 s | 9.2× | 3.9× |
| fixture | 2.96 s | 1.66 s | 0.535 s | 5.5× | 3.1× |
| planted_large (120n×20t) | 60.2 s | 13.9 s | 1.008 s | **60×** | 13.8× |
| planted_xlarge (200n×25t) | 218.2 s | — | 2.95 s | **74×** | — |

`cProfile` after Stage 3 shows a flat profile with no |Mx|-linear term left. Partition quality
is preserved or slightly better (L-modularity base vs Stage 3: 0.576559 / 0.576559 on
planted_medium, 0.827586 / 0.827586 on bipartite, 0.612968 / 0.624675 on planted_large — better,
0.7265 / 0.7175 on fixture — −1.2 %, within the baseline's own spread).

### 5.4 Freezing the randomness — it works, and it is cheap

Shim (`freeze.py`): precomputed value hash on `Leaf`, value hash on `TimeEdge`, creation-order
hash on `_LagoModule` **reset per run**, value sort keys instead of `key=id` at `stem.py:219`
and `:260`, and no `id()` recycling (B6). With deterministic hashes, `set.pop()` order becomes a
function of the inputs, so the five `pop()` sites need no change.

| | baseline | frozen |
|---|---|---|
| configurations stable over 3 repeats (of 48) | **35** | **48** |
| identical across two separate processes | no | **yes, 48/48** |
| total wall time over 30 shared configurations | 30.23 s | 32.43 s (**1.07×**) |

Per-configuration cost ranges 0.79× to 1.24× (the spread is mostly trajectory-length variation,
not hashing). Two iterations were needed to get there, and both failures are worth recording:

* a **global** module counter leaves the second run in a process with different hashes → 19/48.
  Resetting it per run → 47/48.
* the last failure (`continuous_medium MM STEM`) was **B6**: keeping modules alive so `id()` is
  never recycled → 48/48.

---

## 6. Decisions taken

| # | question | decision |
|---|---|---|
| 1 | tolerance | **Stage 1 + Stage 2.** Accept the ≤1e-14 float reassociation. |
| 2 | bug B1 (`get_nodes_durations` pop-order) | **Fix.** Take `right_duration` from the segment's own last leaf, matching `get_expanded_module`. Makes continuous-stream results well-defined and O2 bit-exact everywhere. |
| 3 | bug B2 (k-partite null model) | **Fix.** `.get(source, -2)` → `.get(target, -2)`. Changes every k-partite metric value (−19 % on the bipartite generator) and removes the closed-form fallback. |
| 4 | bug B3 (reproducibility) | **Freeze the exploration order, opt-in via `seed`.** Purpose is the experiment: without it "nothing is broken" cannot be checked. Measured cost 1.07×. |
| 5 | rounding ties (§4.3) | **Ship the tie guard.** Bit-identical `ModularityResult` at a ~0.3 % fallback cost, valid for `ndigits ≲ 10`. Applied to the final value, not per community. |
| 6 | new bugs B5, B6, B7 | **Fix all three.** B5 and B6 are prerequisites anyway (B5 for O1, B6 for freezing the order); B7 is independent and small. Each lands as its own measured step. |
| 7 | Stage 3 (O8/O9/O10) | **Include it, last.** It is ~90 % of the total speedup on large networks; doing it after the cheap wins and bug fixes keeps each layer separately measurable. |

Consequences:

* B1, B2 and B5 are **deliberate behaviour changes**. They must land as separate, individually
  measured steps, before the performance work, or verification cannot tell a fix from a
  regression.
* B5 must be fixed before O1, otherwise O1 is mathematically wrong on directed continuous
  streams.
* B1 must be fixed before O2 and before Stage 3's incremental durations.

---

## 7. Verification plan

### 7.0 Success criterion

* `longitudinal_modularity` → the same `ModularityResult.value` and `time_penalty`;
* `lago_modules` → the same **sets of `(node, time)` members** in the returned `TimeModules`,
  up to module relabelling.

Comparison is a canonical label-invariant fingerprint: sort each module's member set, sort the
list of modules, hash.

Two caveats established above: `lago_modules` does not meet this against itself today (§2.3),
which is why Step 1 freezes the order; and `longitudinal_modularity` is **not** bit-exactly
deterministic across processes when community labels are strings and weights are inexact — the
audit refuted my original claim (§9). For int labels and exact weights it is.

### 7.1 Steps

1. **Baseline snapshot.** Copy `lago/` to the scratchpad as `lago_ref`; old and new run side by
   side in one process. No git.
2. **Freeze both arms** (Step 1 of §8) so every later comparison is exact.
3. **Shadow equivalence.** Run current and optimised implementations side by side on every real
   call, over 10 generators × lex × refinement × `fast_exploration` × `refinement_in`.
   *The audit's warning:* a naive two-arm replay has an **8.75 % false-positive rate** because
   the recorded tape is itself nondeterministic — so the harness needs a **control arm**
   (reference vs reference) to calibrate, or it must run on the frozen build.
4. **`longitudinal_modularity` corpus test.** All stream types × {MM, JM, CM} × gamma
   {0, 0.5, 1, 2} × omega {0, 1, 2} × ndigits {5, 12}, against `lago_ref`. Sized for the
   0.1–0.3 % tie rate: **≥ 5 000 randomised streams per lex**, not 30 cases.
5. **End-to-end partition equality for `lago_modules`**, frozen build, all configurations ×
   several seeds — must be **identical**.
6. **Existing suite** stays at 246 passed.
7. **Regression benchmark.** Keep `gen.py`, `lago_full.py`, `lm_full.py` as a committed
   benchmark.

---

## 8. Implementation plan

### Step 0 — harness

`lago_ref` copy; promote the scratchpad scripts into a `benchmarks/` set plus an equivalence
runner using the §7.0 fingerprint; record the 246-passed baseline.

### Step 1 — freeze the exploration order

Applied **identically to `lago_ref` and the working copy**. Mechanism validated in §5.4:
value hashes on `Leaf`/`TimeEdge`, per-run creation-order hash on `_LagoModule`, value sort keys
at `stem.py:219,260`, and stable keys for the `id()`-based caches (this *is* the B6 fix).
Gate on `seed is not None` so the default path keeps today's speed.

Acceptance: two consecutive runs, and two separate processes, return identical fingerprints on
all 48 configurations.

### Step 2 — behaviour fixes, each measured separately

1. **B4** `_compute_time_neighbors` → grouping pass. No behaviour change.
2. **B6** re-key the oscillation caches on stable identifiers (already covered by Step 1's
   mechanism; make it unconditional, since it is a correctness bug).
3. **B5** align the two `duration` expressions (`linkstream.py:608` / `:629`). Directed
   continuous only. Prerequisite for O1.
4. **B1** `right_duration` from the segment's own last leaf. Continuous only. Prerequisite for
   O2 and Stage 3. Add a test that `get_nodes_durations` is invariant under input ordering.
5. **B2** `.get(target, -2)`. Add a test asserting a non-zero null model between partites.
6. **B7** include the time-swapped form in the delayed duplicate key.
7. **B8** add the missing `2 ** (node1 != node2)` factor in the JM k-partite path.

### Step 3 — `longitudinal_modularity`

O11 (dead code) → O6 (leaf-keyed labels) → O3 (closed forms, with the partite regimes and
degenerate-input guards of §4.2) → the §4.3 tie guard if we take that option.

### Step 4 — `lago_modules` hot path

O12, O13, O15, O5, O14 (all bit-exact) → O2 → O4 → O1.

### Step 5 — Stage 3

O8 (version-stamped cache, 4 invalidation sites) → O9 (incremental durations, carrying the real
edge duration) → O10 (closed-form delta). This is where 60–74× comes from; it is no longer
"optional".

### What was actually delivered

All five steps were implemented. Measured per step, each against a snapshot taken just before
it, with partition/value equality verified at every stage:

| step | change | effect |
|---|---|---|
| 1 | freeze the exploration order | 35/48 → **48/48** stable, identical across processes; cost ~1.07× |
| 2 | bug fixes B1, B2, B4, B5, B6, B7, B8 | behaviour changes, each measured separately |
| 3 | metric: dead code, leaf-keyed labels, closed forms + tie guard | **1.4× – 11.8×**, 0/60 000 differing values |
| 4 | hot path: O1, O2, O4, O5, O12, O13, O14, O15 | part of the 3.0–7.7× below |
| 5 | per-module duration cache (O8) + closed-form delta (O10) | the rest; 96/96 identical partitions |

`longitudinal_modularity`, against the pre-metric-change snapshot:

| case | MM | JM | CM |
|---|---:|---:|---:|
| LAGO output, medium streams | 1.5–2.4× | 1.5–2.3× | 1.4–2.0× |
| LAGO output, real fixture | 1.7× | 1.5× | 1.2× |
| ground truth, 240 nodes / 4 communities | 2.6× | 2.7× | 2.9× |
| ground truth, 600 nodes / 2 communities | **11.3×** | **10.0×** | **11.8×** |

`lago_modules`, MM + STEM, against the pre-hot-path snapshot (identical trajectories):

| stream | before | after | speedup |
|---|---:|---:|---:|
| planted_small (24 nodes) | 0.183 s | 0.050 s | **3.64×** |
| planted_medium (60 nodes) | 5.411 s | 0.916 s | **5.91×** |
| planted_large (120 nodes × 20 steps) | 51.86 s | 6.699 s | **7.74×** |
| weighted_medium | 3.290 s | 0.540 s | **6.09×** |
| directed_medium | 1.073 s | 0.175 s | **6.14×** |
| bipartite_medium | 1.843 s | 0.468 s | **3.94×** |
| continuous_medium | 1.701 s | 0.473 s | **3.60×** |
| delayed_medium | 1.089 s | 0.274 s | **3.98×** |
| fixture (real, 2096 links) | 3.213 s | 1.076 s | **2.99×** |

Every row: identical partition, identical L-modularity.

### Not done

**O9, the incremental duration update.** After O8 the remaining `get_nodes_durations` calls are
the ones whose leaf set is not a whole module (`parent − M0` when leaving, `module ∪ M0` when
joining); deriving those incrementally from the cached ones would remove the last O(|Mx|) term.
It is the largest remaining win — `get_nodes_durations` is still ~6 s of the 6.7 s
`planted_large` run — but it is also the most intricate change, and the audit found the
published prototype of it silently wrong on continuous streams. Left as a follow-up with the
harness in place to gate it.

---

---

## 9. Second round: scaling to large streams

The first round left three places whose cost grew with something other than the work being
done. All three are pure-speed changes; none alters behaviour.

### A — candidate evaluation is now O(|M0|), not O(|Mx|)

Evaluating "should this submodule move into that module?" still walked the whole target module
to get its per-node durations. Now:

* a module memoises its durations **and** the degree aggregates the closed form needs
  (`Σ degree · √duration`, overall and per partite), recomputed from scratch whenever the module
  changes — moves are ~300× rarer than evaluations, so recomputing on change is both cheap and
  free of accumulated drift;
* the other side of the comparison differs from the cached one only on **M0's own nodes**, and
  the duration change for one time-node is O(1).

That last part is the key, and it needs no new data structure. A node's duration is the total
length of its maximal runs of consecutive active time-nodes, and runs are already delimited by
`left/right_time_active_neighbor`. So adding a time-node either extends a run, bridges two, or
starts a new one — decided by two membership tests — and in every case the run's far endpoints
cancel out of the difference:

| left neighbour in set | right neighbour in set | change in duration |
|---|---|---|
| no | no | `dur(leaf)` |
| yes | no | `leaf.t − left.t − dur(left) + dur(leaf)` |
| no | yes | `right.t − leaf.t` |
| yes | yes | `right.t − left.t − dur(left)` |

Integer arithmetic, so it is exact rather than merely close.
`tests/core/test_duration_deltas.py` checks it against full recomputation for **every subset**
of four small streams (instantaneous, directed, continuous, delayed) plus randomised batch
add/remove — 58 tests.

### B — continuous splitting no longer scales with the time span

`_split_continuous_linkstream` built `set(range(t, t + duration + 1))` per edge and intersected
it with the global instants: O(duration) per edge, i.e. proportional to the *time span* rather
than to the splits produced. Both bounds of an edge were registered as instants when the link
was added, so they are now plain lookups into the sorted instants.

| granularity | before | after | |
|---|---:|---:|---:|
| coarse: durations ≤ 20, span 300 | 0.050 s | 0.047 s | 1.07× |
| durations ≤ 400, span 1 200 | 0.907 s | 0.858 s | 1.06× |
| fine: durations ≤ 10⁴, span 3·10⁴ | 0.417 s | 0.300 s | 1.39× |
| fine: durations ≤ 10⁵, span 3·10⁵ | 1.299 s | 0.351 s | 3.70× |
| fine: durations ≤ 10⁶, span 3·10⁶ | 6.549 s | 0.135 s | **48.6×** |

No regression at coarse granularity, and the old behaviour was heading for unusable at real
timestamp resolution.

### C — `TimeModules` stores segments, not every instant

A module spans an interval; enumerating it as `(node, time)` pairs inflates it badly — **41 607
members for 166 segments** on the real fixture, 8–12× on the others. Segments are now the
canonical storage (`TimeModules.segments`, `TimeModules.from_segments`), `lago_modules` hands
them over directly without ever expanding, and every dense view — `_raw_modules` and the four
inverted indices — is built on first use, so a caller that never asks for one never pays for it.
`longitudinal_modularity` reads segments throughout: durations by summing run lengths, the
module duration by merging runs, and coexistence by sweeping segment boundaries instead of
visiting every instant.

Your observation that segments and dense members map onto each other is what made this safe to
do, and it holds here: `sum of run lengths` and `len(expanded members)` agree on **680/680**
node-durations across all seven stream types. (That invariant only holds *since* the B1 fix —
before it, `get_nodes_durations` was pop-order dependent.)

### A regression I introduced, and caught

Reviewing the k-partite coverage afterwards turned up a mistake of my own. **B8 was half
right.** The undirected k-partite JM branch genuinely lacked the `2 ** (n1 != n2)` factor and
returned half the convention the plain branch uses — fixing that was correct. But the directed
branch was *already* right: its `in1*out2 + in2*out1` already covers both ordered directions,
so multiplying by 2 made it **2× too large**. Measured against an ordered-pair reference:

| | undirected | directed |
|---|---|---|
| original | 0.50× (the bug) | 1.00× |
| after B8 | 1.00× | **2.00× (my regression)** |
| now | 1.00× | 1.00× |

Two further things fell out of chasing it:

* **`lago_modules(..., lex="JM")` has never worked on a directed k-partite stream.**
  `_get_expectation_jm_kpartite_part` read `degrees_in[node]` instead of `.get(node, 0)`, and in
  a directed k-partite network a node normally has only in-edges or only out-edges. It raises
  `KeyError` on the first candidate move — in the original, in the intermediate snapshots, and
  until now. The MM path had always used `.get`.
* **The 2× error was invisible to every check I had.** No benchmark stream was both directed
  *and* k-partite, so the path was never executed; `compare_matrix` reported 96/96 identical
  throughout. `benchmarks/streams.py` now includes `bipartite_directed_medium` (25 nodes with no
  in-degree, 25 with no out-degree), and it is in the determinism, matrix and delta checks.

The lesson is the same one as the factor-2 incident in §0, one level up: a behaviour *change*
cannot be validated by "identical to the reference", because it is supposed to differ. B8 was
argued from the code rather than measured against an independent reference, and the argument was
half wrong. `tests/algorithm/test_delta_expectations.py` now pins both expectation deltas
against a reference written out independently, over every stream mode including this one.

### One thing C uncovered

On continuous streams the module expansions **overlap**: 96 `(node, time)` pairs on
`continuous_medium` are claimed by two modules, 84 of them real time-nodes. The split gives each
edge a duration one instant longer than its gap, so a module's last instant coincides with the
next module's first — meaning `lago_modules` does not return a strict partition there. The dense
code resolved it silently, by "last module in iteration order wins"; `_build_leaf_labels`
reproduces that rule exactly, so the metric is unchanged. Worth fixing on its own terms, as a
deliberate behaviour change, rather than inside performance work.

### Delivered

| stream | before round 2 | after | |
|---|---:|---:|---:|
| planted_small (24 nodes) | 0.180 s | 0.031 s | **5.9×** |
| planted_medium (60 nodes) | 5.413 s | 0.330 s | **16.4×** |
| planted_large (120 nodes × 20 steps) | 51.94 s | 1.615 s | **32.2×** |
| weighted_medium | 3.273 s | 0.220 s | **14.9×** |
| directed_medium | 1.067 s | 0.067 s | **15.9×** |
| bipartite_medium | 1.868 s | 0.174 s | **10.8×** |
| continuous_medium | 1.699 s | 0.274 s | **6.2×** |
| delayed_medium | 1.093 s | 0.130 s | **8.4×** |
| fixture (real, 2096 links) | 3.241 s | 0.616 s | **5.3×** |

96/96 identical partitions; 0/60 000 differing metric values; 0/254 721 delta calls outside
1e-9; 48/48 configurations still reproducible; 321 tests passing.


## 9b. `seed` and `nb_iter` restored

Freezing the exploration order made results reproducible, but it also removed the only source of
variation between runs — so `seed` did nothing and `nb_iter=5` ran the identical search five
times at five times the cost, while the docstring still promised it "reduce[s] sensitivity to
the greedy optimization's starting point".

The order comes from two independent places, both sets consumed by `.pop()` and re-grown during
iteration: TMM's queue (ordered by `_LagoModule.index`) and STEM's (ordered by `Leaf.__hash__`).
Permuting either changes the outcome, and they are genuinely different levers — `planted_medium`
is invariant to both, `planted_large` responds to STEM's. So the seed had to reach the queue
itself, not the hashing.

`lago/algorithm/_internal/exploration.py` now provides an `ExplorationQueue` used by both loops:

* **without a generator it *is* a `set` consumed by `set.pop()`** — the previous code path
  verbatim, so default runs reproduce previously published results;
* **with one** it is a shuffled list plus a membership set.

`lago_modules` derives a per-iteration generator: unseeded iteration 0 uses the canonical order,
so `nb_iter=1` with no seed is byte-identical to before; every other combination gets an order
that is a function of `(seed, iteration)` alone.

| | before | after |
|---|---|---|
| distinct results over 4 seeds (`fixture`) | 1 | 4 |
| a given seed reproducible | n/a | yes, across processes |
| `nb_iter=5` vs `nb_iter=1` (`planted_large`) | identical | **+0.017383 L-modularity** |
| `nb_iter=5` vs `nb_iter=1` (`fixture`) | identical | +0.001502 |

Verified by `compare_matrix` at **51/51 identical partitions** on the canonical order. It also
reports 3 configurations the pre-change snapshot cannot run at all — directed k-partite JM, which
raised `KeyError` until the `.get(node, 0)` fix — rather than hiding them.

Folded in at the same time: each iteration's winner is now recorded as segments immediately
rather than as live `_LagoModule` objects (the next iteration rebuilds every module, so holding
the objects kept a partition that no longer described its leaves), and the vestigial `numpy`
import went — it existed only for `np.random.seed`, which the algorithm never consulted. The
algorithm core is now dependency-free.

---

## 10. What the audit refuted

Recorded because two of these corrected my own claims.

1. **"`longitudinal_modularity` is bit-exactly deterministic."** Refuted. It is, for
   exactly-representable weights (the default unweighted case) and integer community labels.
   With string labels, `PYTHONHASHSEED` changes dict iteration order and hence the float
   summation order; with inexact weights that changes the value across processes. Golden-value
   regression testing is still viable, but must pin the label type and weights.
2. **"Value-based `Leaf` hashing is the wrong fix — 3.2×–6.5× slower on the hot path."**
   Microbenchmarks do show set construction 5× slower with a cached int hash and ~10× with
   `hash((node, time))` computed per call. But end-to-end over 30 configurations the measured
   cost is **1.07×** (§5.4) — the hot path is arithmetic-bound, not set-bound. The "insufficient"
   half of that finding was right: hashing alone gave 47/48; B6 was the missing piece.
3. **"Shaving constants off the O(|Mx|) work is worth nothing."** Refuted in part — O5 and O14
   are real, if small. The finding's useful core is that only *removing* the |Mx| dependence
   (Stage 3) pays big, which §5.3 confirms.
4. **A complete enumeration of "18 identity-ordering sites"** — refuted as inaccurate; the
   working list is the one in §5.4 plus B6.
5. **A power analysis claiming a 1e-3 equivalence margin needs n=100 per arm** — refuted; with
   the frozen build the statistical route is not needed at all.

---

## 11. Open items for implementation time

Not blockers, but decisions to take with the code in front of us:

* **`ndigits` is unvalidated** (`modularity.py:183`). The tie guard's proxy `v·10^ndigits` has
  its own ULP that grows with `ndigits`; at `ndigits ≥ 16` `round()` is a no-op on a double and
  the guard can never fire. Either derive `tol` from `ndigits`
  (`tol = max(1e-4, 64·ulp(v·10**ndigits))`) or clamp/validate `ndigits`.
* **Directed + partial partite coverage** is the one regime with no closed form; it must fall
  back to the pair loop verbatim (§4.2).
* **Freezing gated on `seed`** means two code paths through the ordering-sensitive sites. If the
  measured 1.07 % … 7 % cost is acceptable in general, making it unconditional would be simpler
  and would make the published method reproducible by default — worth revisiting once the
  benchmark suite exists.
