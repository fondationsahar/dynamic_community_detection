# Code Review — dcd-lago v1.0.4

**Library:** dcd-lago (LAGO — dynamic community detection on temporal networks)  
**Version reviewed:** 1.0.4  
**Python target:** 3.11+  
**Date:** April 2026  

---

## 1. Bugs — FIXED

### 1.1 Set modified during iteration — FIXED

**File:** `lago/algorithm/_internal/_lago_module.py`

Previously iterated over `self.neighbors` while removing `None` inside the loop. Now uses `discard(None)` after construction and filters `None` in the set comprehension itself.

### 1.2 Undirected duplicate-edge detection — FIXED

**File:** `lago/core/linkstream.py`

Edge keys for undirected graphs are now normalized as `(min(source, target), max(source, target), time)` in all three link modes (instantaneous, continuous, delayed). This correctly rejects `(A, B, t)` and `(B, A, t)` as duplicates.

### 1.3 Potential `IndexError` in `get_module_duration` — FIXED

**File:** `lago/core/utils.py`

Added a guard for empty `topo_neighbors | topo_neighbors_from` before accessing the first element. Uses `next(iter(...))` instead of building a full list.

---

## 2. Usability for Newcomers — FIXED

### 2.1 Documentation uses wrong parameter name — FIXED

**File:** `examples/01_getting_started.md`

Changed `alpha` to `gamma` throughout (section title, code examples, description).

### 2.2 Missing input validation — FIXED

Added validation in `lago/algorithm/lago.py`:
- `gamma >= 0` (mirrors existing `omega` check)
- `nb_iter >= 1`
- Empty `LinkStream` check before running the algorithm

Added validation in `lago/core/linkstream.py` (all three link modes):
- Node IDs must be integers (`TypeError` with clear message)
- Weights must be >= 0 (`ValueError` with clear message)
- Duration must be > 0 for continuous mode (`ValueError`)

### 2.3 Parameter explanations improved — FIXED

**File:** `lago/algorithm/lago.py`

Improved docstrings for `gamma`, `omega`, `nb_iter`, and `lex` with concrete descriptions of their effect (e.g., "omega=0 ignores temporal continuity entirely").

### 2.4 `__len__` vs `.size` clarified — FIXED

**File:** `lago/core/time_modules.py`

Added a note in the `__len__` docstring explaining it returns (node, time) pairs and pointing users to `.size` for unique node count.

### 2.5 Time normalization note added to README — FIXED

**File:** `README.md`

Added a note after the quick start example about normalizing timestamps so the smallest time gap is 1.

### 2.6 Deprecated typo method removed — FIXED

**File:** `lago/core/linkstream.py`

Removed `add_continous_links()` (the misspelled variant). Updated all test references to use the correctly-spelled `add_continuous_links()`.

---

## 3. Code Quality — FIXED

### 3.1 Set comprehensions — FIXED

Replaced all `set([...])` with `{...}` across `_lago_module.py`, `_leaf.py`, `delta_lm.py`, `find_best_move.py`.

### 3.2 Dead code removed — FIXED

Removed commented-out code blocks in `lago/viz/data_preparation.py`, `lago/viz/core.py`, `lago/viz/drawing.py`.

### 3.3 `import warnings` moved to module level — FIXED

**File:** `lago/viz/plot.py`

Moved to top-level imports; removed the two local `import warnings` statements.

### 3.4 Unused typing imports removed — FIXED

**File:** `lago/viz/utils.py`

Removed unused `Dict`, `List`, `Optional` imports.

### 3.5 `print()` replaced with `logging` — PARTIALLY FIXED

**File:** `lago/viz/utils.py` — replaced with `logging.getLogger(__name__).warning(...)`.

**File:** `lago/viz/plot.py` (lines 424–430) — kept as-is. These `print()` calls are gated behind a `print_mapping=True` parameter that the user explicitly opts into. This is intentional user-facing output, not debug logging.

---

## 4. Design Suggestions

These are longer-term improvements — address at your discretion.

### 4.1 Add a termination guard for convergence loops

**Files:** `lago/algorithm/_internal/tmm.py` (line 53), `rir.py` (line 28), `rtmm.py` (line 24)

The main optimization loops warn at 1,000–10,000 iterations but never terminate. For a published library, consider adding a `max_iterations` parameter that either raises an exception or returns the best-so-far result when hit.

### 4.2 Extract magic numbers into named constants

| File | Line | Value | Meaning |
|---|---|---|---|
| `tmm.py` | 53 | 10000 | Iteration warning threshold |
| `rir.py` | 28 | 1000 | Iteration warning threshold |
| `rtmm.py` | 24 | 1000 | Iteration warning threshold |
| `configuration.py` | 32–33 | 0.75 | Height layout factor |
| `configuration.py` | 100–101 | 8 | Hardcoded tick condition |

Named constants (e.g., `DEFAULT_MAX_ITERATIONS = 10_000`) make the code self-documenting.

### 4.3 Resolve or document open TODOs

| File | Line | Note |
|---|---|---|
| `lago/metrics/modularity.py` | 228 | Multipartite case not implemented |
| `lago/core/linkstream.py` | 21, 23, 340, 592 | Various TODOs |

For a v1.0 release, each should be resolved, converted to a GitHub issue, or documented as a known limitation.

### 4.4 Consider removing remaining deprecated methods

Three deprecated methods remain: `add_instantaneous_links()`, `add_continuous_links()`, `add_delayed_links()`. They have proper deprecation warnings. If this is the first public release alongside a paper, there is no installed user base to break — shipping without the deprecated methods results in a cleaner API from day one.

### 4.5 Pre-existing test issues

Two tests were already failing before this review:
- `test_viz/test_data_preparation.py` — `ImportError` for `create_time_node_community_mapping`
- `TestFileIO::test_roundtrip` — `IndexError` in `read_txt` weight parsing

These should be investigated separately.

---

## 5. What the Library Already Does Well

- **Clean package structure** — core / algorithm / metrics / viz separation with proper `__init__.py` exports.
- **Optional visualization dependency** — `[viz]` extra with lazy imports and helpful error messages when matplotlib is missing.
- **Comprehensive test fixtures** — 25+ parameterized pytest fixtures covering edge cases (empty links, self-loops, large timestamps, directed/undirected, weighted, k-partite).
- **Type hints throughout** the codebase.
- **Proper deprecation warnings** on legacy methods.
- **Good documentation breadth** — README, API reference, 7 progressive example notebooks.
- **Minimal core dependencies** — only numpy.
- **Standard packaging** — MIT license, `pyproject.toml`, ruff configuration, wheel distribution.

---

## Summary

| Category | Count | Status |
|---|---|---|
| Bugs (Section 1) | 3 | All fixed |
| Usability (Section 2) | 6 | All fixed |
| Code quality (Section 3) | 5 | All fixed (1 partially) |
| Design suggestions (Section 4) | 5 | Open — at maintainer's discretion |
