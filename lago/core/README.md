# lago.core

Core data structures for LAGO temporal community detection.

## Contents

| Module | Description |
|--------|-------------|
| `linkstream.py` | `LinkStream` - Temporal network data structure |
| `time_modules.py` | `TimeModules`, `TimeModule`, `TimeSegment` - Detection results |
| `time_modules_network.py` | `TimeModulesNetwork` - Metagraph of module relationships |
| `enums.py` | `LexType`, `LinkStreamMode` - Configuration enums |
| `utils.py` | Utility functions for working with modules |

## Usage

```python
# Via main package (recommended)
from lago import LinkStream, TimeModules, LexType

# Direct import
from lago.core import LinkStream, TimeModules, LexType
```

## Classes

### TimeModules

Container for detected temporal modules. Supports multiple initialization formats:

```python
from lago import TimeModules

# 1. Standard format: {module_label: {(node, time), ...}}
raw_modules = {
    0: {(0, 0), (0, 1), (1, 0), (1, 1)},
    1: {(2, 0), (2, 1), (3, 0)},
}
tm = TimeModules(raw_modules)

# 2. Segment format: {module_label: {"nodes": {node: [[start, end], ...]}}}
# More compact for contiguous time ranges
segment_data = {
    0: {"nodes": {"0": [[0, 1]], "1": [[0, 1]]}},
    1: {"nodes": {"2": [[0, 1]], "3": [[0, 0]]}},
}
tm = TimeModules(segment_data)

# 3. From file (auto-detects format)
tm = TimeModules(path="communities.json")
tm = TimeModules(path="communities.csv")
tm = TimeModules(path="communities.txt")
```

**Segment Format**: Each segment is `[start, end]` (inclusive). Useful for contiguous time ranges:

```python
{
    0: {"nodes": {
        "0": [[0, 5], [10, 15]],  # Node 0 at times 0-5 and 10-15
        "1": [[2, 8]],             # Node 1 at times 2-8
    }}
}
```

### LinkStream
Temporal network representation supporting:
- Discrete and continuous time
- Directed and undirected edges
- Delayed (asynchronous) interactions
- Weighted edges

### TimeModule
Single temporal module with:
- Node membership
- Time segments
- Module statistics

### LexType
Enum for Longitudinal Expectation type:
- `JM`: Joint-Membership
- `MM`: Mean-Membership
- `CM`: Coexistence (for modularity computation only)

### LinkStreamMode
Enum for link stream configuration:
- `DISCRETE`: Instantaneous interactions
- `CONTINUOUS`: Interval-based interactions
- `DELAYED`: Asynchronous interactions
