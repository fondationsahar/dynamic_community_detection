# lago.algorithm

LAGO (Longitudinal Agglomerative Greedy Optimization) algorithm for temporal community detection.

## Contents

| Module | Description |
|--------|-------------|
| `lago.py` | `lago_modules()` - Main algorithm entry point |
| `_internal/` | Internal algorithm implementation (private) |

## Usage

```python
# Via main package (recommended)
from lago import LinkStream, lago_modules, LexType

ls = LinkStream()
ls.add_links([(0, 1, 0), (1, 2, 1)])
modules = lago_modules(ls, lex_type=LexType.MM)

# Direct import
from lago.algorithm import lago_modules
```

## Function: lago_modules

```python
def lago_modules(
    linkstream: LinkStream,
    lex_type: LexType | str = LexType.MM,
    nb_iter: int = 1,
    gamma: float = 1,
    omega: float = 2,
    refinement: str | None = "STEM",
    fast_exploration: bool = True,
    refinement_in: bool = True,
    verbose: bool | int = 0,
    stopping_criterion: float = 1e-8,
    ndigits_logs: int = 8,
) -> TimeModules
```

### Parameters

- **linkstream**: Link Stream on which to find temporal modules
- **lex_type**: Longitudinal Expectation type (`LexType.JM` or `LexType.MM`)
- **nb_iter**: Number of LAGO runs (best results returned)
- **gamma**: Topological resolution (higher = smaller modules)
- **omega**: Temporal resolution (higher = smoother transitions)
- **refinement**: Strategy (`None`, `"STEM"`, or `"STNM"`)
- **fast_exploration**: Enable fast exploration heuristic
- **verbose**: Verbosity level (`0`=silent, `1`=progress info, `2`=debug). Also accepts `bool` for backward compatibility (`True`=1, `False`=0)

### Returns

`TimeModules` object containing detected temporal modules.

## Internal Modules

The `_internal/` directory contains implementation details:
- `runner.py`: Algorithm orchestration
- `delta_lm.py`: Modularity delta computation
- `rtmm.py`, `rir.py`, `roor.py`: Core algorithm steps
- `stem.py`, `stnm.py`: Refinement strategies
- `_leaf.py`, `_time_edge.py`, `_lago_module.py`: Internal data structures

⚠️ **Warning**: The `_internal/` module is private. Do not import from it directly as the API may change without notice.
