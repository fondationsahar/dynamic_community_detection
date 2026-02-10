# lago.metrics

Quality metrics for evaluating temporal community detection results.

## Contents

| Module | Description |
|--------|-------------|
| `modularity.py` | `longitudinal_modularity()` - L-Modularity computation |

## Usage

```python
# Via main package (recommended)
from lago import LinkStream, TimeModules, longitudinal_modularity, LexType

ls = LinkStream()
ls.add_links([(0, 1, 0), (1, 2, 1)])
modules = ...  # From lago_modules() or manual construction

result = longitudinal_modularity(ls, modules, lex=LexType.MM)
print(f"Modularity: {result.value}")
print(f"Time penalty: {result.time_penalty}")

# Direct import
from lago.metrics import longitudinal_modularity, ModularityResult
```

## Function: longitudinal_modularity

```python
def longitudinal_modularity(
    linkstream: LinkStream,
    time_modules: TimeModules,
    lex: LexType | str = LexType.MM,
    gamma: float = 1,
    omega: float = 2,
) -> ModularityResult
```

### Parameters

- **linkstream**: The temporal network
- **time_modules**: Detected temporal modules
- **lex**: Longitudinal Expectation type:
  - `LexType.JM`: Joint-Membership
  - `LexType.MM`: Mean-Membership
  - `LexType.CM`: Coexistence (only for modularity, not for detection)
- **gamma**: Topological resolution parameter
- **omega**: Temporal resolution parameter

### Returns

`ModularityResult` with:
- `value`: The L-Modularity value
- `time_penalty`: Temporal penalty component

## ModularityResult

```python
@dataclass
class ModularityResult:
    value: float        # L-Modularity value
    time_penalty: float # Temporal penalty component
```

## Longitudinal Expectation Types

| Type | Description |
|------|-------------|
| **JM** | Joint-Membership: Expects consistent module duration |
| **MM** | Mean-Membership: Allows temporal flexibility |
| **CM** | Coexistence: Based on co-presence (evaluation only) |
