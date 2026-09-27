# How the implementation works

This is the map for anyone about to read or change the code. It describes the library as it
is; the reasoning and measurements behind each choice are in [`history/`](history/README.md),
the numbers in [`PERFORMANCE.md`](PERFORMANCE.md), the checks to run before changing anything
in [`benchmarks/README.md`](../benchmarks/README.md). The user-facing API is in
[`API_REFERENCE.md`](API_REFERENCE.md).

One rule shapes everything below: **a change to the implementation must not change the
output** — the same L-modularity value, the same sets of `(node, time)` from `lago_modules`.
Where a float is summed, the order of the summation is therefore part of the design.

## 1. The objects

A **`LinkStream`** (`lago/core/linkstream.py`) stores interactions as a graph of time-nodes.
A **`Leaf`** (`lago/algorithm/_internal/_leaf.py`) is one `(node, time)`: it holds the
node's edges at that instant (`topo_neighbors`, and `topo_neighbors_from` for the incoming
side of a directed stream) as `TimeEdge`s, its previous and next active instants for the
same node (`left_time_active_neighbor`, `right_time_active_neighbor`), the duration of its
interactions (`edge_duration` — all edges of a leaf share it), and the module it currently
belongs to during a run. `leaves_dict` maps `(node, time)` to its leaf in insertion order.

Modes: instantaneous links `(u, v, t[, w])`; continuous links `(u, v, t, duration[, w])`,
which `lago_modules` splits on the global set of instants into a fresh stream before
searching; delayed links `(u, v, t_source, t_target[, w])`. Any of them directed, weighted,
or k-partite (`partite_mapping`).

**Hashing is by value, not by address.** `Leaf.__hash__` is `hash((node, time))`,
`TimeEdge.__hash__` covers target, weight and duration, and a `_LagoModule` hashes by its
creation index. Sets of these objects therefore iterate in an order that depends only on
the data, and the search — which pops candidates from sets — is reproducible in any
process. It also means *how many* modules a run creates matters (each takes an index), see
invariant I4.

## 2. The search

`lago_modules` (`lago/algorithm/lago.py`) validates, prepares the stream, runs
`nb_iter` iterations of `lago_run`, and keeps the best by its accumulated L-modularity;
each iteration's partition is recorded as segments and turned into a `TimeModules` at the
end. One iteration (`_internal/runner.py`, `rir.py`) is a loop of two phases until neither
improves L-modularity by more than the stopping criterion:

1. **A TMM level** (`tmm.py`, *Time Module Movements*). Every current module gets a fresh
   parent holding the same leaves. Modules are then taken from an `ExplorationQueue`
   (`exploration.py` — a plain `set.pop()` order without a seed, a seeded shuffle with one)
   and each is moved to the neighbouring parent that increases L-modularity most, if any.
   With `fast_exploration` a moved module's neighbours are re-queued; without it the whole
   queue is replayed while any move happened. When the queue is empty the parents become
   the modules of the next level.
2. **A refinement** (`refinement_in=True`, the default): **STEM** (`stem.py`, *Single Time
   Edge Movements*) tries to move every edge — a pair of leaves inside one module, and every
   single leaf — to a neighbouring module; **STNM** (`stnm.py`) does the same with single
   leaves as the submodules of a TMM-style level. `refinement=None` skips this phase;
   `refinement_in=False` (`roor.py`, `rtmm.py`) runs all TMM levels first and refines after.

Oscillation guards in both movers remember the gain of each move by `(submodule index,
from, to)` and refuse the reverse move unless it gains strictly more. The stopping criterion
is scaled by the stream's total weight.

## 3. Scoring a move

`DeltaLongitudinalModularityComputer` (`_internal/delta_lm.py`) computes the change in
L-modularity of taking a submodule M0 out of its parent and into a candidate. The change has
three terms: the weight of M0's edges into the candidate, the change in the null-model
expectation, and the change in the number of community switches of M0's nodes.

**One pass per move, not per candidate.** `evaluate_candidates` walks M0's edges and
segment endpoints once and credits every candidate at the same time. It can do that because
the parents partition the leaves: a neighbour belongs to a candidate exactly when the
candidate *owns* it — `leaf.module.parent` during a TMM/STNM level, `leaf.module` during
STEM — so membership is an attribute read, not a set lookup. Each candidate's weight is the
same subsequence of additions as a per-candidate walk would perform, the switch count is an
integer, and the duration changes are exact integers, so the results are identical to the
bit. Candidates come from the neighbour sets the movers have always used (see §5.2), not
from the pass itself. The original one-candidate-at-a-time formulation, `M0_to_Mx`, is kept
as the reference; `Computer.use_batch = False` selects it, and STNM runs on it (§5.1).

**Memoised per-module aggregates.** A module keeps its per-node durations
(`get_durations`) and, for MM, the aggregates `Σ degree·√duration` overall and per partite
(`_mm_sums`); for JM, node counts, interaction start/end multisets with sorted instants and
degree sums (`_JMAggregate`). A candidate's score then costs O(|M0|): M0 only changes these
through its own nodes and its own extreme instants. After an accepted move the winner's
already-evaluated duration changes are applied to both modules
(`_LagoModule.apply_duration_changes`, `jm_add`/`jm_remove`) instead of recomputing over
the whole module. Derived float sums are recomputed from the maintained integers **in sorted
node order**, never shifted incrementally — accumulated `+=` rounding could turn an exact tie
between candidates into a non-tie on symmetric inputs.

**Expectations.** MM and JM have closed forms (`_mm_from_sums`, `_jm_from_sums`),
including the k-partite mask as "minus the same product per partite". The pair loops they
replace remain as `_expectation_mm_closed`'s fallback and in `_get_expectation_jm_part`;
they are used when the partite mapping carries the sentinel values `-1`/`-2` that the mask
compares against, and as the independent reference in
`tests/algorithm/test_delta_expectations.py`.

## 4. Invariants a change must respect

| | rule | relied on by |
|---|---|---|
| I1 | Parent modules partition the leaves; every candidate is non-empty; `M1 ∩ M2 = ∅` | `find_best_move.py` (the `M1 == M2` test is gone because it can never hold) |
| I2 | `leaf.module` is static during a TMM/STNM level and `leaf.module.parent` is live; in STEM `leaf.module` is live | `evaluate_candidates` (ownership = membership) |
| I3 | Candidate *eligibility* uses out-edges (`topo_neighbors`) and time neighbours only; *weights* use both edge sets | `compute_neighbors`, `get_neighbors_modules_parents`, `evaluate_candidates` |
| I4 | The number of `_LagoModule`s created in a run must not change: the creation index is the hash, hence the set order, hence the exploration order | everything that iterates a set of modules |
| I5 | Per-candidate float accumulations keep their order; the only canonical order is sorted node ids in the memoised aggregates | `evaluate_candidates`, `_mm_sums`, `_jm_sums` |
| I6 | `duration_delta_on_add` deltas are exact integers that telescope to `dur(Mx ∪ M0) − dur(Mx)` whatever the order; M0 is still processed in iteration order so the `changes` dict keeps its insertion order | `_changes_from_table`, `_shift_mm_sums` |

## 5. Three behaviours of the original that decide the output — kept, by decision

Each was found while optimising, reproduced exactly, and flagged in the code. Correcting any
of them changes the partition on the inputs named, so **they stay as they are**.

### 5.1 STNM works from a stale module set from its second round on

The TMM and STNM movers start out sharing one `modules` set. `_update_modules_after_stnm`
ends every STNM pass with `self.modules = set(...)` — a new set on the STNM object only —
so from then on STNM iterates the modules of the *previous* level, re-points every
`leaf.module` at throw-away modules whose parents are those old modules, and moves leaves
between them; the TMM level that follows finds `leaf.module.parent` naming modules that are
no longer current. Nothing crashes because the original evaluation tests membership on the
leaf sets. Effect: with `refinement="STNM"`, the refinement keeps refining the level-0
partition. A fix is one line (mutate the shared set); it would change STNM results only
(probably upward) and let STNM use the batch path — today `runner.py` pins it to the
reference path because I2 does not hold for it.

### 5.2 A TMM level after a STEM pass uses neighbour sets computed before the pass

Candidates come from `compute_neighbors()`, run at the end of the previous level; a STEM
pass then moves leaves between modules and nothing recomputes. A module can therefore miss
a candidate it now touches. Deriving candidates from the live adjacency changed 6 of 54
corpus partitions, in the **default configuration** — so the batch path takes the recorded
sets on TMM/STNM levels (`find_best_move.py`); STEM, which has always recomputed per edge,
derives them live.

### 5.3 On directed streams a reciprocal pair of equal edges counts once in the weight delta

`_get_weight_diff` sums a leaf's edges over `topo_neighbors | topo_neighbors_from`, a set
union; `TimeEdge` equality is (target, weight, duration), so u→v and v→u at the same
instant with equal weight and duration are one element and count once. The metric counts
them twice, so on such streams the search optimises a quantity slightly different from the
L-modularity it reports. The batch path keeps the union on directed streams and skips it
only on undirected ones, where there is nothing to unite with.

## 6. `seed`, `nb_iter`, `n_jobs`

Every run is a function of the input and the parameters. `seed` chooses the exploration
order among equally valid greedy trajectories (`None` is the historical `set.pop()` order);
different seeds reach different local optima, and `nb_iter` runs several and keeps the best
— iteration `i` of an unseeded call uses `random.Random(i)`, of a seeded one
`random.Random(seed * 1_000_003 + i)`. `n_jobs` spreads the iterations over **forked**
processes and returns the very same partition: a fork inherits the stream as it is, down to
the memory layout of every set that steers the search, which a rebuild in a fresh process
does not reproduce (measured: different partitions). Workers are silent, the parent logs in
run order, peak memory is about `n_jobs ×` one run, and platforms without fork run
sequentially with a warning.

## 7. The metric

`longitudinal_modularity` (`lago/metrics/modularity.py`) is a pure function of a stream and
a partition. Partitions are held as **segments** — per module, per node, inclusive runs of
instants (`TimeModules`, `lago/core/time_modules.py`) — because a module covering a long
interval has far more `(node, time)` members than runs. The intra-community interaction
count and the switch count are one pass over the labelled time-nodes; the expectations use
the same closed forms as the search, guarded by `_near_rounding_tie`: the closed form and the
pair loop can land on opposite sides of an exact rounding tie (~0.2 % of integer-weight
runs), and the loop is used there so the reported value is unchanged.

`lago/accel.py` holds an optional compiled path for the two counting loops: the stream is
flattened once into CSR arrays cached on it (`Topology`), the labels become an `int32`
array per call, and `lago/_accel_kernel.pyx` sums per community **in the same order the
Python loops visit the edges**, so the double is bit-identical. Flattening costs about one
scoring, so the kernel is taken from the **second** scoring of a stream on
(`should_accelerate`); a single scoring runs the Python loops. `LAGO_ACCEL=python|cython`
forces a backend.

## 8. The compiled build

`setup.py` compiles eleven of the package's own `.py` files with Cython in pure-Python mode
— the same sources, unchanged; `_leaf.py` and `_time_edge.py` have a `.pxd` beside them and
become extension types with C attributes and a C `__hash__`, the others are compiled with
Python object semantics (`annotation_typing=False`, no fast-math) — plus the metric kernel.
Every extension is *optional*: a failed compile is a warning and the install proceeds as
pure Python. The `.so` files sit beside the `.py` and the import system prefers them;
`LAGO_CORE=python` forces the sources, which is how `benchmarks/compare_backends.py` holds
the two to each other (bit-identical metric values, identical partitions).

`_LagoModule` stays a Python class on purpose: extension types are immutable and the tests
monkeypatch its methods. Two things were measured and did not pay: compiling the data model
alone (the hash calls it made cheap had already been removed by the batch evaluation) and
typed locals in the hot loops (their cost is dict and set work, not attribute reads).

## 9. Map of the code

| module | responsibility |
|---|---|
| `lago/core/linkstream.py` | `LinkStream`: modes, `add_links`, time neighbours, the continuous split |
| `lago/core/time_modules.py` | `TimeModules` / `TimeModule`: segments, members, queries, I/O |
| `lago/core/utils.py` | durations of runs (`get_nodes_durations`, `duration_delta_on_add`), segments, logging |
| `lago/algorithm/lago.py` | `lago_modules`: validation, iterations, `seed`, `n_jobs` |
| `_internal/runner.py`, `rir.py`, `roor.py`, `rtmm.py` | one iteration: movers, the TMM/refinement loop |
| `_internal/tmm.py`, `stem.py`, `stnm.py` | the three movers |
| `_internal/find_best_move.py` | best candidate for a submodule; the reference formulation |
| `_internal/delta_lm.py` | the delta computer: batch evaluation, aggregates, closed forms, reference paths |
| `_internal/_leaf.py`, `_time_edge.py`, `_lago_module.py` | the objects; `_JMAggregate` |
| `_internal/exploration.py`, `leaf_set.py`, `lago_tools.py` | exploration queue, lazy set difference, segment/move helpers |
| `lago/metrics/modularity.py` | `longitudinal_modularity` |
| `lago/accel.py`, `lago/_accel_kernel.pyx` | flat topology, kernel dispatch, the Cython kernel |
| `lago/viz/` | plotting (optional `viz` extra) |
| `setup.py` | the optional compiled build |
| `benchmarks/` | the equivalence harness and the timing scripts |
