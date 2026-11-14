# Changelog

## 1.2.1 — 2026-11-14

Same code paths and the same results as 1.2.0, bit for bit; what changes is what you
can install, and where.

### Packaging

- **No C compiler needed anywhere.** Compiled wheels for CPython 3.11–3.15 on Linux
  (x86_64 and aarch64, glibc and musl), macOS (Apple Silicon and Intel, no longer
  macOS 15 only) and Windows (x64 and ARM64). 1.2.0 had wheels for Apple Silicon on
  macOS 15+ with Python 3.11 and 3.13 only; everyone else built from source, and
  without a compiler silently got pure Python.
- **The pure-Python `py3-none-any` wheel is back** (1.2.0 dropped it). pip takes it
  wherever no binary wheel matches — PyPy, Pyodide, other architectures — so an
  install never runs a build, and `--only-binary :all:`, offline wheelhouses and
  Pyodide's micropip work again. To compile on such a platform instead:
  `pip install --no-binary dcd-lago dcd-lago`.
- Every distribution is built and tested in CI (`.github/workflows/wheels.yml`) and
  published through PyPI Trusted Publishing, with PEP 740 attestations linking each
  file to the commit and workflow that built it. Each binary wheel is tested as
  installed: every module asserted compiled, and the compiled paths held bit for bit
  to the pure-Python ones on every platform.
- Build requirement `Cython>=3.1,<4` (was `>=3.0`, unbounded).

### Safety

- The compiled metric kernel validates its indices: malformed input to the public
  `lago.accel.count_intra_and_switches` (a label beyond the community count, a short
  array, an edge or a neighbour outside the arrays) raises `IndexError`, as the
  pure-Python kernel does, instead of reading or writing out of bounds. About 10 % of
  the kernel's own time; not measurable on a whole `longitudinal_modularity` call.
- Windows, compiled build: a continuous link lasting more than 2³¹ − 1 time units
  (25 days in milliseconds) raised `OverflowError`, because the compiled time-node
  stored its duration in a C `long`, which is 32 bits on Windows. It is 64 bits on
  every platform now, as it already was on Linux and macOS.

## 1.2.0 — 2026-09-27

Same algorithm, same quality function; a great deal faster, reproducible, and
with fewer things to install. The verification behind every item below is in
`docs/` and `benchmarks/`.

### Speed

- `lago_modules`: 8× (pure Python) to 12× (compiled) faster than 1.1.0 on small
  streams over every parameter combination, 6–109× on streams of 4k–96k
  interactions, and no longer superlinear on wide streams. JM now runs at the
  speed of MM (it was 8–14× slower). Details: `docs/BENCHMARK_1.1.0_VS_1.2.0.md`.
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
(`docs/ARCHITECTURE.md`, section 5).

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
