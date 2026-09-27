# LAGO Test Suite

This directory contains the test suite for the LAGO library. Tests are organized to mirror the package structure.

## Structure

```
tests/
├── conftest.py                  # Shared pytest fixtures
├── README.md                    # This file
├── fixtures/                    # Test data files
│   └── linkstream.txt          # Sample linkstream (2096 links) for integration tests
├── core/                        # Tests for lago.core
│   ├── test_linkstream.py
│   ├── test_time_modules.py
│   ├── test_time_modules_network.py
│   ├── test_utils.py
│   └── test_duration_deltas.py  # O(1) duration updates vs full recomputation, exhaustively
├── algorithm/                   # Tests for the search itself
│   ├── test_delta_expectations.py    # MM/JM deltas vs an independent pair-loop reference
│   ├── test_candidate_evaluation.py  # batch evaluation vs the per-candidate reference, per call
│   ├── test_incremental_durations.py # maintained aggregates vs recomputation, after every move
│   ├── test_exploration_order.py     # canonical order, seeds, nb_iter
│   └── test_parallel_iters.py        # n_jobs == sequential
├── metrics/                     # Tests for lago.metrics
│   ├── test_l_modularity.py
│   └── test_accel.py            # the compiled kernel vs the Python loops
├── integration/                 # End-to-end integration tests
│   └── test_lago_integration.py
└── test_viz/                    # Tests for lago.viz (needs dcd-lago[viz])
    ├── test_data_preparation.py
    └── test_utils.py
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run tests with verbose output
pytest tests/ -v

# Run specific test category
pytest tests/core/              # Core data structure tests
pytest tests/metrics/           # Modularity metric tests
pytest tests/test_viz/          # Visualization tests

# Run a specific test file
pytest tests/core/test_linkstream.py

# Run tests with coverage
pytest tests/ --cov=lago --cov-report=term-missing
```

## Test Categories

### Core Tests (`tests/core/`)
Tests for core data structures:
- **test_linkstream.py**: `LinkStream` class - temporal network representation
  - Discrete, continuous, and delayed modes
  - Directed and undirected edges
  - Weighted edges
  - File I/O operations
- **test_time_modules.py**: `TimeModules`, `TimeModule`, `TimeSegment`
  - Module creation and manipulation
  - Iteration and access patterns
  - Time segment operations
- **test_time_modules_network.py**: `TimeModulesNetwork`
  - Module relationship graphs
  - Edge extraction

### Metrics Tests (`tests/metrics/`)
Tests for quality metrics:
- **test_l_modularity.py**: `longitudinal_modularity()`
  - JM, MM, CM longitudinal expectations
  - Different resolution parameters

### Algorithm and equivalence tests (`tests/algorithm/`, `test_duration_deltas.py`, `test_accel.py`)
The search has two implementations of most of its arithmetic -- the fast path and a
reference formulation kept in the code -- and these tests hold them together on every call
of real runs, over every stream mode (see `docs/ARCHITECTURE.md`). They are what makes an
optimisation safe to land: exact equality where the operations are the same, a 1e-9 tolerance
only where a summation order was deliberately changed. `benchmarks/` extends the same checks
to larger corpora and to a snapshot of the code before a change.

### Visualization Tests (`tests/test_viz/`)
Tests for visualization components (requires `dcd-lago[viz]`):
- **test_data_preparation.py**: Data transformation for plotting
- **test_utils.py**: Visualization utilities

### Integration Tests (`tests/integration/`)
End-to-end tests for the full LAGO pipeline:
- **test_lago_integration.py**: Full workflow tests
  - Various parameter combinations (lex, omega, refinement)
  - STEM and STNM refinement strategies
  - Multiple iterations
  - Fast vs exhaustive exploration

## Test Fixtures

Common fixtures are defined in `conftest.py`:
- Sample link streams
- Pre-computed module results
- Standard test configurations

## Adding New Tests

1. Place tests in the appropriate subdirectory matching `lago/` structure
2. Name test files with `test_` prefix
3. Name test functions with `test_` prefix
4. Use fixtures from `conftest.py` when appropriate

Example:
```python
# tests/core/test_new_feature.py
import pytest
from lago import LinkStream

def test_new_feature():
    ls = LinkStream()
    ls.add_links([(0, 1, 0)])
    assert ls.nb_nodes == 2
```

## Coverage Goals

- Core data structures: >90%
- Algorithm: >80%
- Metrics: >90%
- Visualization: >70%
