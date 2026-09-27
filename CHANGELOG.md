# Changelog

## 1.2.0 — 2026-09-27

Same algorithm, same quality function; a great deal faster, reproducible, and
with fewer things to install. The verification behind every item below is in
`docs/` and `benchmarks/`.

### Speed

- `lago_modules`: 8× (pure Python) to 12× (compiled) faster than 1.1.0 on small
  streams over every parameter combination, 6–109× on streams of 4k–96k
  interactions, and no longer superlinear on wide streams. JM now runs at the
  speed of MM (it was 8–14× slower). Details: `docs/BENCHMARK_PYPI_VS_CURRENT.md`.
- `longitudinal_modularity`: 1.5–2.5× faster in pure Python, 10–70× on repeated
  scoring of the same stream with the compiled kernel.

### Reproducible results

- `lago_modules` is deterministic: the same input and parameters give the same
  modules, in any process, with or without a seed. In 1.1.0 the exploration order
  followed memory addresses and results varied from run to run.
- `seed` now does what it says: it selects the exploration order, so different
  seeds reach different local optima and `nb_iter` keeps the best of several.
  (In 1.1.0 it seeded numpy, which the algorithm never consulted.)
- New `n_jobs`: runs the `nb_iter` iterations in parallel processes (fork; falls
  back to sequential with a warning where fork is unavailable). Default 1, which
  is the sequential loop exactly. Each process holds its own copy of the
  stream; see the docstring for memory and platform notes.

### Bug fixes that can change results on the inputs they concern

1.1.0 users may see a different partition on some inputs — these are corrections,
and 1.1.0's own results were not reproducible in the first place:

- k-partite streams: the null model was zero for nodes absent from the mapping;
  the JM k-partite expectation used an inconsistent pair convention; directed
  k-partite JM raised `KeyError` (it never worked).
- Duplicate delayed links were merged by an order-dependent key.
- Durations of a module's nodes depended on set iteration order.
- Oscillation guards were keyed on `id()` values that were recycled, letting
  moves be undone and redone (1.1.0 could run for 15 minutes on a
  500-interaction continuous stream).
- The continuous-link split had an off-by-one at segment ends.

Three behaviours of the original that also decide the output were kept exactly
as they were, deliberately, and are documented with what a fix would change
(`docs/PERFORMANCE_ROUND2.md`, section 3).

### Packaging

- No runtime dependencies. `numpy` was only used by `lago.viz` and moved to the
  `viz` extra, which now also lists `pandas` (it was imported but not declared).
- Optional compiled build: the hot modules and the metric kernel are compiled
  with Cython when a C compiler is available (binary wheels, or at install from
  the sdist) and run as the same pure-Python sources otherwise. Nothing to
  configure; `lago.accel.core_name()` and `lago.accel.backend_name` say what is
  running; `LAGO_CORE=python` forces the sources. The two paths are verified to
  return identical results.
- `lago.__version__`.
- Python 3.11+ (unchanged).

## 1.1.0

Previous release; see the git history.
