# History

The working notes of the performance work on this library, kept verbatim as the record of what
was measured, tried, kept and rejected — and why. They are not maintained: the current state
of the code is in [`ARCHITECTURE.md`](../ARCHITECTURE.md) and [`PERFORMANCE.md`](../PERFORMANCE.md),
the verification harness in [`benchmarks/README.md`](../../benchmarks/README.md).

| note | what it covers |
|---|---|
| [`2026-09-round1-diagnosis.md`](2026-09-round1-diagnosis.md) | Round 1: where the time went, nine latent bugs, the first optimisations, `seed` and `nb_iter` restored, the equivalence harness |
| [`2026-09-faster-language-analysis.md`](2026-09-faster-language-analysis.md) | Whether to port the core to a compiled language; scaling on both axes; the memory model |
| [`2026-09-metric-kernel.md`](2026-09-metric-kernel.md) | The Cython kernel of `longitudinal_modularity`: design, its first regression, measurements |
| [`2026-09-round2.md`](2026-09-round2.md) | Round 2: batch candidate evaluation, maintained aggregates, O(\|M0\|) JM, `n_jobs`, the compiled core, the three preserved behaviours, negative results |
