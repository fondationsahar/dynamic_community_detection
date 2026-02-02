# Logging Strategy Recommendations for LAGO Library

## Proposed Approach: Integer Verbosity Parameter

Keep the existing `verbose` parameter but extend it to accept integer levels (like scikit-learn):

```python
# Current API (keep working for backward compatibility)
tm = lago_modules(ls, verbose=False)  # Silent
tm = lago_modules(ls, verbose=True)   # Basic info (treated as verbose=1)

# New API (integer levels)
tm = lago_modules(ls, verbose=0)  # Silent (default)
tm = lago_modules(ls, verbose=1)  # Progress info
tm = lago_modules(ls, verbose=2)  # Detailed debug info
```

---

## Implementation

### 1. Type Definition

```python
# Add to lago/core/types.py or use inline
from typing import Literal

VerboseLevel = bool | Literal[0, 1, 2]
```

### 2. Function Signatures

Apply to all main entry points:

```python
# lago/algorithm/lago.py
def lago_modules(
    linkstream: LinkStream,
    ...,
    verbose: bool | int = 0,  # Changed from bool = False
) -> TimeModules:
    ...

# lago/metrics/modularity.py  
def longitudinal_modularity(
    linkstream: LinkStream,
    communities: TimeModules | dict,
    ...,
    verbose: bool | int = 0,
) -> ModularityResult:
    ...

# lago/core/linkstream.py
class LinkStream:
    def add_links(self, links: ..., verbose: bool | int = 0) -> None:
        ...
    
    def read_txt(self, path: str, ..., verbose: bool | int = 0) -> None:
        ...

# lago/core/time_modules.py
class TimeModules:
    def summary(self, verbose: bool | int = 0) -> dict:
        ...
```

### 3. Internal Helper Function

```python
# Add to lago/core/utils.py
def log(message: str, level: int, verbose: int | bool) -> None:
    """Print message if verbosity level is sufficient.
    
    Args:
        message: Message to print.
        level: Required verbosity level (1 or 2).
        verbose: Current verbosity setting.
    """
    # Convert bool to int for comparison
    v = int(verbose) if isinstance(verbose, bool) else verbose
    
    if v >= level:
        print(message)


def log_info(message: str, verbose: int | bool) -> None:
    """Log info message (verbose >= 1)."""
    log(message, level=1, verbose=verbose)


def log_debug(message: str, verbose: int | bool) -> None:
    """Log debug message (verbose >= 2)."""
    log(message, level=2, verbose=verbose)
```

### 4. Usage in Code

```python
# lago/algorithm/lago.py
from lago.core.utils import log_info, log_debug

def lago_modules(linkstream, ..., verbose: bool | int = 0):
    log_info(f"Starting LAGO: {linkstream.nb_nodes} nodes, {linkstream.nb_edges} edges", verbose)
    
    for iteration in range(nb_iter):
        log_debug(f"  Iteration {iteration + 1}: exploring moves...", verbose)
        
        # ... algorithm ...
        
        log_debug(f"  Time Module Movements: Δ = {delta:.6f}", verbose)
        log_debug(f"  Refinement [{refiner}]: Δ = {delta:.6f}", verbose)
        
        log_info(f"Iteration {iteration + 1} complete: modularity = {mod:.4f}", verbose)
    
    log_info(f"Found {nb_modules} modules", verbose)
    
    return result
```

---

## Output Examples

### `verbose=0` (Silent - Default)

No output.

### `verbose=1` (Progress)

```
Starting LAGO: 10 nodes, 121 edges
Iteration 1 complete: modularity = 0.8234
Iteration 2 complete: modularity = 0.8531
Found 2 modules
```

### `verbose=2` (Debug)

```
Starting LAGO: 10 nodes, 121 edges
  Iteration 1: exploring moves...
  Time Module Movements: Δ = 0.001234
  Refinement [STEM]: Δ = 0.000456
Iteration 1 complete: modularity = 0.8234
  Iteration 2: exploring moves...
  Time Module Movements: Δ = 0.000123
Iteration 2 complete: modularity = 0.8531
Found 2 modules
```

---

## Where to Add Verbose

| Class/Function | Verbosity Messages |
|----------------|-------------------|
| `lago_modules()` | Iteration progress, module counts |
| `longitudinal_modularity()` | Computation steps |
| `LinkStream.add_links()` | Link count, node count |
| `LinkStream.read_txt()` | File loading progress |
| `TimeModules.summary()` | Additional statistics |

---

## Implementation Plan

### Phase 1: Add Helper Functions
- [ ] Add `log_info()` and `log_debug()` to `lago/core/utils.py`
- [ ] Update type hints to accept `bool | int`

### Phase 2: Update Functions
- [ ] Update `lago_modules()` verbose handling
- [ ] Update `longitudinal_modularity()` verbose handling
- [ ] Add verbose to `LinkStream` methods
- [ ] Add verbose to `TimeModules.summary()`

### Phase 3: Documentation
- [ ] Document verbose levels in docstrings
- [ ] Update examples in README

---

## Benefits

| Aspect | Current | Proposed |
|--------|---------|----------|
| Verbosity levels | On/Off | 0, 1, 2 |
| Backward compat | - | `True` = 1, `False` = 0 |
| Complexity | Low | Low (no logging module) |
| Flexibility | Limited | Good balance |
| User learning curve | None | Minimal (same pattern as sklearn) |
