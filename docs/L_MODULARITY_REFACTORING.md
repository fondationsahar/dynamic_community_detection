# Longitudinal Modularity Function - Refactoring Status

> **Created:** January 30, 2026  
> **Last Updated:** January 30, 2026  
> **File:** `lago/l_modularity_function.py`

---

## 1. Overview

The `longitudinal_modularity` function computes a modularity score for temporal community detection. It supports three expectation types:
- **CM** (Coexistence Modularity) - based on actual co-presence times
- **JM** (Joint Modularity) - based on community duration
- **MM** (Mean Modularity) - based on geometric mean of node durations

---

## 2. Summary of Completed Work

### ✅ Completed

| Issue | Description | Status |
|-------|-------------|--------|
| Code duplication | Extracted `_compute_degrees_contribution()` and `_compute_expected_value()` helpers | ✅ Done |
| Side effects | Replaced `leaf.module` modification with separate `labels` dict | ✅ Done |
| Magic strings | Added `LexType` enum with backward compatibility | ✅ Done |
| Missing type hints | Full type annotations added | ✅ Done |
| Test coverage | 25 comprehensive tests created | ✅ Done |
| Set comprehension | Replaced `set([...])` with `{...}` | ✅ Done |
| Unnecessary deep copy | Removed `copy.deepcopy()` | ✅ Done |
| Empty community bug | Fixed using `.get(community, 0)` | ✅ Done |
| `ModularityResult` dataclass | Created but not yet used as default return | ✅ Done |

### 🔄 Remaining / Future Work

| Issue | Description | Priority |
|-------|-------------|----------|
| Default return type | Change default to return `ModularityResult` instead of `float` | Medium |
| Remove string support | Remove deprecated string `lex_type` in major version | Low |
| Performance optimization | Consider caching or vectorization for large networks | Low |

---

## 3. Current API

```python
from lago import longitudinal_modularity, LexType, ModularityResult

# Modern way (recommended)
result = longitudinal_modularity(
    linkstream,
    communities,
    lex_type=LexType.MM,  # or LexType.CM, LexType.JM
)

# Backward compatible (deprecated, shows warning)
result = longitudinal_modularity(
    linkstream,
    communities,
    lex_type="MM",  # Still works, but emits DeprecationWarning
)

# To get time penalty separately (deprecated pattern)
value, penalty = longitudinal_modularity(
    linkstream,
    communities,
    return_time_penalty=True,
)
```

---

## 4. Naming Recommendations

### 4.1 Current Naming Issues

| Current | Problem | Recommendation |
|---------|---------|----------------|
| `lex_type` | Unclear abbreviation, "lex" could mean "lexical" | Consider `expectation_type` or `modularity_type` |
| `LexType` | Same issue as above | Consider `ExpectationType` or `ModularityExpectation` |
| `return_time_penalty` | Boolean flag that changes return type | Deprecated in favor of `ModularityResult` |
| `ndigits` | Abbreviation, inconsistent with other params | Consider `precision` or `decimal_places` |
| `alpha` / `omega` | Greek letters, not self-documenting | Keep as-is (standard in literature) but add docstring |
| `cscs` | Cryptic abbreviation in `get_community_switch_counts` | `switch_count` internally is fine |
| `communities_leaves` | Plural inconsistency | Consider `community_to_leaves` or `leaves_by_community` |

### 4.2 Recommended Naming Scheme (Future Major Version)

```python
# Rename LexType -> ExpectationType
class ExpectationType(Enum):
    COEXISTENCE = auto()  # CM - More descriptive than 2-letter codes
    JOINT = auto()        # JM
    MEAN = auto()         # MM

# Or keep short names but in a well-documented enum
class LexType(Enum):
    """Longitudinal Expectation Type for modularity computation."""
    CM = auto()  # Coexistence Modularity
    JM = auto()  # Joint Modularity
    MM = auto()  # Mean Modularity
```

### 4.3 Parameter Naming Consistency

Consider renaming in future major version:

```python
def longitudinal_modularity(
    linkstream: LinkStream,
    communities: dict[str | int, set[tuple[int, int]]],
    expectation_type: ExpectationType = ExpectationType.MEAN,  # Clearer
    expectation_weight: float = 1.0,   # Was: alpha
    switch_penalty_weight: float = 2.0,  # Was: omega
    precision: int = 5,  # Was: ndigits
) -> ModularityResult:
```

### 4.4 Internal Function Naming

Current names are good and follow conventions:
- `_compute_degrees_contribution` ✅ Clear
- `_compute_expected_value` ✅ Clear  
- `_get_communities_cmes` ⚠️ Could be `_compute_coexistence_expectations`
- `_get_communities_jmes` ⚠️ Could be `_compute_joint_expectations`
- `_get_communities_mmes` ⚠️ Could be `_compute_mean_expectations`
- `_get_communities_nb_interactions` ⚠️ Could be `_count_intra_community_interactions`

### 4.5 Decision: Keep or Change?

**Recommendation:** Keep `LexType` and the CM/JM/MM abbreviations since:
1. They are established in the literature
2. Users familiar with longitudinal modularity will recognize them
3. Breaking changes should be minimized

But **do rename** internal functions for clarity:
- `_get_communities_cmes` → `_compute_coexistence_expectations`
- `_get_communities_jmes` → `_compute_joint_expectations`
- `_get_communities_mmes` → `_compute_mean_expectations`
- `_get_communities_nb_interactions` → `_count_intra_community_interactions`

---

## 5. Architecture After Refactoring

```
lago/l_modularity_function.py
│
├── ModularityResult (dataclass)
│   ├── value: float
│   ├── time_penalty: float
│   ├── lex_type: LexType
│   └── total: property
│
├── Helper Functions (extracted to reduce duplication)
│   ├── _compute_degrees_contribution()
│   └── _compute_expected_value()
│
├── Main Function
│   └── longitudinal_modularity()
│       ├── Builds labels dict (no side effects!)
│       ├── Dispatches to expectation function
│       ├── Computes time penalty
│       └── Returns result
│
└── Internal Functions
    ├── _get_communities_nb_interactions(linkstream, labels)
    ├── _get_communities_cmes(linkstream, communities_leaves)
    ├── _get_communities_jmes(linkstream, communities_leaves)
    ├── _get_communities_mmes(linkstream, communities_leaves)
    └── get_community_switch_counts(linkstream, labels)
```

---

## 6. Test Coverage

**File:** `tests/test_l_modularity.py`

| Section | Tests | Status |
|---------|-------|--------|
| Basic Functionality | 4 tests | ✅ |
| Expectation Types | 5 tests | ✅ |
| Parameters | 5 tests | ✅ |
| Directed LinkStream | 2 tests | ✅ |
| Temporal Behavior | 2 tests | ✅ |
| Edge Cases | 4 tests | ✅ |
| Consistency | 2 tests | ✅ |
| Integration | 1 test | ✅ |
| **Total** | **25 tests** | ✅ |

---

## 7. Breaking Changes Avoided

The refactoring was done with **full backward compatibility**:

1. ✅ String `lex_type` still works (with deprecation warning)
2. ✅ `return_time_penalty=True` still returns tuple
3. ✅ Same numerical results as before
4. ✅ Same parameter names and defaults
5. ✅ No modification to linkstream (was a bug-prone side effect)

---

## 8. Files Modified

| File | Changes |
|------|---------|
| `lago/enums.py` | Added `LexType` enum |
| `lago/l_modularity_function.py` | Full refactor, added `ModularityResult` |
| `lago/__init__.py` | Export `LexType`, `ModularityResult` |
| `tests/test_l_modularity.py` | New file with 25 tests |

---

## 9. Future Migration Path

### Phase 1 (Next Minor) - Optional
- Rename internal functions for clarity
- Add `return_result: bool` parameter to return `ModularityResult`

### Phase 2 (Next Major) - Breaking
- Default `return_result=True`
- Remove `return_time_penalty` parameter
- Consider renaming `lex_type` to `expectation_type`
- Remove string support for `lex_type`

---

## 10. Related Files to Review

| File | Why | Priority |
|------|-----|----------|
| `lago/lago_src/delta_lm.py` | Similar logic for delta modularity | High |
| `lago/tools.py` | Used functions may need similar refactoring | Medium |
| `lago/leaf.py` | `module` attribute no longer used by l_modularity | Low |
