# lago.algorithm

The LAGO search (Longitudinal Agglomerative Greedy Optimization) for temporal community
detection. Its one public entry point is `lago_modules`:

```python
from lago import LinkStream, lago_modules, LexType

ls = LinkStream()
ls.add_links([(0, 1, 0), (1, 2, 1)])
modules = lago_modules(ls, lex=LexType.MM)
```

- Parameters, return value and examples: [`docs/API_REFERENCE.md`](../../docs/API_REFERENCE.md).
- How the search works, the objects it manipulates, the invariants a change must respect
  and a map of the modules below: [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).
- What to run before changing anything here: [`benchmarks/README.md`](../../benchmarks/README.md).

## Layout

| | |
|---|---|
| `lago.py` | `lago_modules()`: validation, iterations, `seed`, `n_jobs` |
| `_internal/` | the search itself: movers (`tmm.py`, `stem.py`, `stnm.py`), the delta computer (`delta_lm.py`), candidate selection (`find_best_move.py`), the objects (`_leaf.py`, `_time_edge.py`, `_lago_module.py`) and helpers |

⚠️ `_internal/` is private: import from `lago`, not from here. Its API may change without notice.
