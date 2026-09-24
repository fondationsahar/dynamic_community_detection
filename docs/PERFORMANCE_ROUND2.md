# LAGO speed, round 2

Round 1 (`docs/PERFORMANCE_DIAGNOSIS.md`, branch `perf/lago-core-optimisation`) made
`lago_modules` 5–32× faster and `longitudinal_modularity` 1.2–12× (3–9× more with the Cython
metric kernel, `docs/ACCELERATED_BACKENDS.md`). This round re-profiled what was left, on
both size axes, and went after it. The rule is the same throughout: **the same L-modularity
value, the same sets of `(node, time)` from `lago_modules`** — checked against a snapshot of
the code taken before any change (`benchmarks/make_ref.py`), on every stream mode, in every
configuration of the corpus. Two items change the order of a floating-point summation and
are validated per call within 1e-9 rather than bit for bit; both are named below, and both
still give identical partitions on the whole corpus.

## 1. Diagnosis

`benchmarks/profile_lago.py` (wall time per function, no cProfile — cProfile triples the
run and inflates the share of tiny functions). MM + STEM, planted streams:

| where the time went | wide 30k edges | wide 80k edges |
|---|---:|---:|
| total, unprofiled | 12.4 s | 60 s |
| `find_best_module_for_submodule` | 87 % | 89 % |
| `_get_expectation_mm_part` | 46 % | 56 % |
| **`get_nodes_durations`, recomputed on accepted moves** | 8 % | **25 %** (20 → 87 µs/call) |
| `_duration_changes_adding` | 17 % | 15 % |
| `_get_weight_diff` (one pass over M0's edges *per candidate*) | 14 % | 11 % |
| `_get_csc_diff` | 6 % | 5 % |
| `LeafDifference.__eq__` behind the `M1 == M2` test | ~7 % (profiled) | |
| `Leaf.__hash__` (a Python method, 48 M calls) + `set.__contains__` (27 M) | ~20 % (profiled) | |

Three findings decided the plan:

1. **JM was 8–14× slower than MM, and superlinear** — 2490 → 5755 µs/edge from 10k to 30k
   edges, 172 s against 12 s at 30k. Its delta was still O(|Mx|) per candidate
   (`Mx_leaves | M0_leaves`, `get_module_duration`, `_sum_degrees` over the whole module);
   round 1 had only given MM the O(|M0|) treatment.
2. **The per-module durations recompute was the remaining superlinear term on wide streams**:
   O(|parent|) per accepted move, and parent size grows with the node count.
3. **`Leaf.__hash__` being a Python method** looked like 17–19 % — measured by patching
   alternative classes in at runtime — but see §4: by the time the algorithmic changes had
   landed, the set operations that paid for those calls were gone.

## 2. What changed

### Step 1 — one pass per move instead of one per candidate (bit-identical)

`DeltaLongitudinalModularityComputer.evaluate_candidates` walks M0 once and credits every
candidate at the same time. The parents partition the leaves, so a neighbour belongs to a
candidate exactly when that candidate *owns* it (`leaf.module.parent` during a TMM/STNM
level, `leaf.module` during STEM); that replaces every set-membership test with an
attribute read, and the per-candidate passes over M0's edges with a single one. Each
candidate's weight is the same subsequence of additions as before, the switch count is an
integer and the duration changes are exact integers, so the results are identical to the
bit. Also: `Leaf.__slots__`; `edge_duration` cached on the leaf by the `LinkStream`
instead of read off an edge each time; the O(|M0|) `M1 == M2` test dropped (by the
partition invariant it can only hold for two empty sets).

The per-candidate formulation stays in the code as the **reference**
(`M0_to_Mx`, `_find_best_reference`), selected by `Computer.use_batch = False`.
`benchmarks/compare_candidates.py` runs both on every call of real runs and demands the same
candidate set, the same total per candidate and the same winner: 0 differences over
≈300 000 calls.

Three behaviours of the original had to be reproduced rather than "corrected" to keep the
output identical. They are documented in the code and listed in §3.

### Step 2 — durations maintained, not recomputed (validated per call)

The evaluation of the winning move already produced the exact per-node duration changes of
both modules; `_LagoModule.apply_duration_changes` applies them (a node reaching 0 is
dropped, as a recompute would leave it out). The MM aggregates derived from them are
recomputed from the durations in **sorted node order** on each change — never shifted
incrementally, because accumulated `+=` rounding could turn an exact tie between candidates
into a non-tie on symmetric inputs — so the memoised value is a function of the module's
content alone, identical whether the durations were maintained or recomputed
(`tests/algorithm/test_incremental_durations.py` checks memo == recompute after every move,
and that the whole search gives the same partition either way). The sorted order is the
numerics change: on the corpus, still 54/54 identical partitions.

### Step 3 — JM in O(|M0|) per candidate (validated per call)

`_JMAggregate`, maintained per module across moves: time-nodes per node, time-nodes per
interaction start and end instant with the distinct instants kept sorted, and cached degree
sums (total, in/out, per partite) in sorted node order. A candidate's union or a parent's
remainder then costs O(|M0|): the added or removed nodes shift the sums, the span comes from
the two extremes (walking inwards over exhausted instants for a removal — one step in
practice). The k-partite JM pair loop got the closed form its MM counterpart already had:
`(S² − Σ_p S_p²)·D`, `(Σin)(Σout)` when directed, with unmapped nodes never masked exactly as
the pair loop's `.get(n1, -1) == .get(n2, -2)` leaves them. `_get_expectation_jm_part` /
`_get_expectation_jm_kpartite_part` stay as the reference and the independent ordered-pair
reference in `tests/algorithm/test_delta_expectations.py` is the acceptance test.

On integer-weight streams the JM numerator is an exact integer, so the new path is
bit-identical to the old one; on the float-weighted stream the canonical summation order
shows as ≤ 1.2e-12 per call. 54/54 identical partitions including every JM configuration.

### Step 4 — `n_jobs` (opt-in; default unchanged)

`lago_modules(..., n_jobs=k)` spreads the `nb_iter` runs over `k` **forked** processes and
returns the very partition the sequential loop returns. Why fork and not a rebuild in a fresh
process: a worker that rebuilds the stream from `get_time_links()` inserts links in a
different order, and the resulting `leaves_dict` and edge-set layouts steer the exploration
order (`set.pop()`, the stem iterator, the float summation order) — measured: different
partitions on 3 of 5 test streams. A fork inherits the parent's objects as they are. Costs
and limits, all in the docstring: `n_jobs=1` is the sequential loop byte for byte, logs
included; workers are silent and the parent logs one line per run in run order; memory is
about `n_jobs ×` a single run (copy-on-write does not survive a search that touches every
object) — capped to `nb_iter` and the CPU count, reduced with a `ResourceWarning` when the
estimate exceeds available memory; no fork (Windows, or a process already running threads)
means sequential with a `RuntimeWarning`.

### Step 5 — the compiled core (identical by construction)

`accel_cython/setup_core.py` compiles a list of the package's own `.py` files in place with
Cython (pure-Python mode): `_leaf.py` and `_time_edge.py` get a `.pxd` beside them and become
extension types (C attributes, C `__hash__`); `delta_lm`, `find_best_move`, `lago_tools`,
`leaf_set`, `stem`, `tmm`, `_lago_module`, `core/utils` and `accel` are compiled as they are,
with Python object semantics (`annotation_typing=False`). Python's import system prefers the
`.so` beside the `.py`; without the build, or with `LAGO_CORE=python`, the same `.py` runs.
`_LagoModule` deliberately stays a Python class: extension types are immutable and the tests
monkeypatch its methods. `lago.accel.core_name()` reports which core is in use;
`benchmarks/compare_backends.py` now runs four variants (metric kernel × core) against pure
Python: 12 660 × 3 metric values bit-identical, 11/11 partitions identical.

## 3. Behaviours of the original that were preserved, not fixed

Each of these would change results if corrected. They are reproduced exactly and flagged in
the code; whether to fix them is a separate decision.

1. **STNM works from a stale module set from its second round on.**
   `SingleTimeNodeMover._update_modules_after_stnm` *rebinds* `self.modules` to a new set
   instead of mutating the set it shares with the TMM mover, so the two movers then disagree
   about which modules are current and `leaf.module.parent` no longer names the module that
   holds the leaf. The batch evaluation relies on that invariant, so STNM runs stay on the
   per-candidate reference path (`runner.py`) and keep their results.
2. **A TMM level after a STEM pass uses neighbour sets computed before the pass.**
   Candidates come from `compute_neighbors`, run when the level starts; STEM then moves leaves
   between modules and nothing recomputes. Deriving candidates from the live adjacency
   changed 6 of 54 corpus partitions, so TMM/STNM levels keep using the recorded sets
   (`find_best_move.py`); STEM, which recomputes per edge, derives them live.
3. **On directed streams a reciprocal pair of edges of equal weight and duration counts once
   in the weight delta.** `_get_weight_diff` iterates `topo_neighbors | topo_neighbors_from`,
   a set union that merges the two `TimeEdge`s (equal target, weight, duration). The batch
   path therefore still builds the union on directed streams and skips it only where there
   is nothing to unite with.

## 4. Negative results (measured, then not done)

* **Tuple-subclass `Leaf`** for a C-level hash: −5 % only, because every attribute read gets
  slower; rejected before implementation (runtime-patched measurement).
* **Compiling `Leaf`/`TimeEdge` alone** (Step 5's first slice): no measurable change
  (wide 30k 7.5 → 7.6 s). The −17–19 % measured at the start was against the *old* code;
  Step 1 had already removed most of the leaf-set operations that paid for the hash calls.
  The gain of Step 5 came from compiling the functions, not the data model.
* **Typed locals** (`@cython.locals(leaf=Leaf, edge=TimeEdge, ...)` on the hot loops, via a
  no-op shim for pure runs): nothing (5.4 → 6.1 s at 30k, noise elsewhere). Attribute reads
  are not where these loops spend their time; dict and set work is. Reverted.

## 5. Results

`benchmarks/profile_lago.py` numbers (wall time including its wrappers; the *baseline* row for
`deep` and for JM is unprofiled, i.e. slightly flattering to the baseline). Apple Silicon,
Python 3.11.

| stream | baseline | after Step 1 | after Step 2 | after Step 3 | pure Python, final | **compiled core** |
|---|---:|---:|---:|---:|---:|---:|
| wide 30k, MM | 15.3 s | 8.1 s | 7.5 s | — | 7.4 s | **5.2 s (2.9×)** |
| wide 80k, MM | 60.2 s | 31.0 s | 24.8 s | — | 24.5 s | **17.8 s (3.4×)** |
| deep 30k, MM | 8.4 s | 6.1 s | 6.1 s | — | 6.1 s | **4.2 s (2.0×)** |
| wide 10k, JM | 24.6 s | | | 2.7 s | 2.4 s | **1.4 s (18×)** |
| deep 10k, JM | 9.5 s | | | 2.7 s | 2.5 s | **1.5 s (6×)** |
| wide 30k, JM | 171.9 s | | | 10.0 s | 8.9 s | **6.1 s (28×)** |

(The "pure Python, final" column includes the last pure-Python item, landed after Step 3: on
unit-weight streams the batch evaluation walks a leaf's edge set directly instead of building
`topo_neighbors | topo_neighbors_from` per leaf per move — integer sums do not depend on the
order. Directed streams keep the union; see §3, item 3.)

JM now runs at MM speed and linearly in edges. The `get_nodes_durations` term that was 25 %
at 80k and growing is gone (0.6 %); the largest remaining term that scales with module size is
`_mm_sums` — O(nodes) per accepted move, 12 % at 80k wide, ~5× cheaper than what it replaced.

On the equivalence corpus (`compare_matrix.py`, small streams): 2.8× against the snapshot
with the compiled core, 1.1–1.2× pure — small streams are dominated by per-call overheads.

`longitudinal_modularity`: the compiled `Leaf` makes its pure-Python path ~1.3× faster
(Leaf-keyed dict lookups now hash in C). That exposed the kernel's cold-call cost, which the
topology build did not amortise on a stream scored once; the kernel now starts at the second
scoring of a stream (`ACCELERATED_BACKENDS.md` §6a), so a single scoring is exactly the
Python path (measured 0.9–1.15×, i.e. noise around parity) and repeated scoring keeps its
4–8× (`bench_backends.py`).

## 6. Verification

```bash
python3.11 -m pytest -q                                          # 628 passed, compiled and LAGO_CORE=python
python3.11 benchmarks/compare_matrix.py --ref-dir R --ref-name round2_before   # 54/54 identical
python3.11 benchmarks/compare_candidates.py --refinement STEM,None            # 0 differences, exact
python3.11 benchmarks/compare_candidates.py --lex JM --refinement STEM,None --tol 1e-9
python3.11 benchmarks/compare_delta.py                           # MM closed form vs pair loop, per call
python3.11 benchmarks/check_determinism.py --reps 2              # 54/54
PYTHONPATH=$PWD/accel_cython/src python3.11 benchmarks/compare_backends.py -n 1500   # 4 variants
python3.11 benchmarks/profile_lago.py --sizes wide:30k wide:80k deep:30k
```

## 7. Left on the table

* `_mm_sums`: cache per-node terms `d·√u` and recompute only the changed nodes' terms; the
  O(nodes) sum in sorted order remains, but the `sqrt` work per move goes away.
* `evaluate_candidates` is now ~70 % of the run and is dict/set work per M0 edge; the next
  level is an array representation of the stream (integer leaf ids, CSR adjacency), which is
  the port discussed in `FASTER_LANGUAGE_ANALYSIS.md` §5.
* The three preserved behaviours of §3, if the decision is to fix them.
