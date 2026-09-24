# `lago_accel_cython` — compiled counting kernel for `longitudinal_modularity`

A Cython transcription of the two counting loops of the metric (intra-community
interactions and community switches). It is optional: `lago` runs identically
without it, only slower on repeated scoring. Design and measurements are in
[`docs/ACCELERATED_BACKENDS.md`](../docs/ACCELERATED_BACKENDS.md).

## Build

Requires a C compiler and Cython ≥ 3.0 for the Python that will import `lago`
(≥ 3.10; the project's own environment is 3.11).

```bash
python3.11 -m pip install Cython
python3.11 -m pip install ./accel_cython
```

For development, build in place and put `src/` on `PYTHONPATH` instead:

```bash
cd accel_cython && python3.11 setup.py build_ext --inplace && cd ..
export PYTHONPATH=$PWD/accel_cython/src
```

`lago.accel` picks the module up at import if it can be imported; nothing else
changes. `LAGO_ACCEL=cython` forces it (and raises if it is missing);
`LAGO_ACCEL=python` forces the pure-Python loops.

```python
from lago import accel
accel.backend_name        # 'cython' or 'python'
accel.available_backends()
```

## Verify

Results must be **bit-identical** to the Python loops, not merely close:

```bash
PYTHONPATH=$PWD/accel_cython/src python3.11 benchmarks/compare_backends.py -n 4000
PYTHONPATH=$PWD/accel_cython/src LAGO_ACCEL=cython python3.11 -m pytest -q
PYTHONPATH=$PWD/accel_cython/src python3.11 benchmarks/bench_backends.py
```

`tests/metrics/test_accel.py` checks the compiled kernel against the reference
kernel in-process whenever it is importable, and skips that check otherwise.

## The compiled core (`setup_core.py`)

Separate from the metric kernel: the package's own hot modules compiled in place, from their
own `.py` sources (Cython pure-Python mode).

```bash
python3.11 accel_cython/setup_core.py build_ext --inplace
```

Drops a `.so` beside each listed `.py` (gitignored); the import system prefers it, nothing
else changes. `LAGO_CORE=python` forces the `.py` sources; `lago.accel.core_name()` says which
is running. `_leaf.py` / `_time_edge.py` have a `.pxd` and become extension types; the other
modules are compiled with plain Python object semantics. Measured 1.35–1.6× on `lago_modules`
on top of the pure-Python round-2 work, identical results (four-way `compare_backends.py`).
See [`docs/PERFORMANCE_ROUND2.md`](../docs/PERFORMANCE_ROUND2.md) §2 and §4 for what was
tried and did not pay.

## What the kernel must not do

The contract is `lago.accel._reference_kernel`. Bit-identity rests on summing
each community's weights in the order the arrays are laid out, so:

* no `-ffast-math` or `-Ofast`, which allow the compiler to reassociate sums;
* no vectorisation of the accumulation, for the same reason;
* no threads — a per-community accumulator updated from several threads would
  make the summation order depend on scheduling.

`-O3` is fine: it preserves IEEE semantics.
