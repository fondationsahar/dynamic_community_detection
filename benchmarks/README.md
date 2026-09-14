# Benchmarks and equivalence harness

Tools for changing LAGO's hot paths without changing its results. See
[`docs/PERFORMANCE_DIAGNOSIS.md`](../docs/PERFORMANCE_DIAGNOSIS.md) for the analysis these
came out of.

Everything runs with the checkout's own `lago` package: `_common.py` pins `sys.path` and
**fails loudly** if an editable install of `dcd-lago` shadows it, which is easy to hit because
the shadowing only happens when the working directory does not contain `lago/`.

Requires Python ≥ 3.11.

## Streams

`streams.py` provides ten link streams covering every mode: instantaneous (plain, weighted,
directed, k-partite), continuous, delayed, and the real 2096-link fixture from
`tests/fixtures/`, from 24 to 200 nodes.

```python
from streams import GENERATORS
ls = GENERATORS["planted_medium"]()
```

## Checking that results did not change

The comparison is against a **snapshot** of the package taken before the change, imported
alongside the working copy in the same process:

```bash
python3.11 benchmarks/make_ref.py --out-dir /tmp/refs --name before
# ... make the change ...
python3.11 benchmarks/compare_matrix.py --ref-dir /tmp/refs --ref-name before
```

| script | what it checks |
|---|---|
| `compare_matrix.py` | `lago_modules` returns the **identical partition** (canonical, label-invariant) over streams × lex × refinement × flags × seeds |
| `compare_ref.py` | the same, per stream, with timings |
| `compare_metric.py` | `longitudinal_modularity` returns the identical value over thousands of random streams × lex × gamma × omega × ndigits |
| `compare_delta.py` | the MM expectation delta matches the pair loop **per call**, numerically |
| `check_determinism.py` | `lago_modules` returns the same partition across repeats (run twice and `diff` to cover separate processes) |
| `bench_metric.py` | timings for `longitudinal_modularity` |
| `scale.py` | how cost grows with size, on both axes, at sizes the correctness corpus does not reach |

Two suites in `tests/` cover the same ground fast enough for CI:

* `tests/core/test_duration_deltas.py` — the O(1) incremental duration update against full
  recomputation, exhaustively over every subset of four small streams.
* `tests/algorithm/test_exploration_order.py` — that the canonical order is stable, a seed is
  reproducible, different seeds reach different optima, and `nb_iter` never scores worse.
* `tests/algorithm/test_delta_expectations.py` — the MM and JM expectation deltas against an
  **independently written** O(n²) reference, on every stream mode. The reference is spelled out
  in the test file rather than reused from `delta_lm`, so that a rewrite replacing the whole
  function is still checked against something. Verified to fail on: the uncached draft, a
  variant dropping the k-partite mask, a directed form missing its factor 2, and one that
  returns float noise where the true delta is 0.

### Scale

`scale.py` answers a different question from the rest: not "did anything change?" but "how does
this grow?". It samples links rather than enumerating node pairs, so a 10⁶-edge stream costs
O(10⁶) to build, and it varies the two axes separately — `wide` (many nodes, few timesteps) and
`deep` (few nodes, many timesteps) at matched edge counts stress quite different parts of the
representation.

It earned its keep immediately: it refuted the shape-independent cost model measured on the
small corpus. `deep` is linear; `wide` is not. See §1d of
[`docs/FASTER_LANGUAGE_ANALYSIS.md`](../docs/FASTER_LANGUAGE_ANALYSIS.md).

Default ceiling is 2×10⁵ time-edges (under a minute); `--max-edges` opts into more, with a
predicted time and memory printed first.

### Two traps these exist to avoid

**A partition fingerprint does not validate a formula.** An error can leave every greedy
choice unchanged on a given corpus. An early closed-form delta was wrong by exactly 2× on all
9 684 directed calls and `compare_matrix.py` still reported 24/24 identical partitions;
`compare_delta.py` caught it on the first run. Use it for anything that changes arithmetic.

**Sample size matters for the metric.** Closed-form expectations can land on the other side of
an exact rounding tie in ~0.1–0.3 % of integer-weight runs. Thirty cases prove nothing — run
`compare_metric.py -n 5000` or more.
