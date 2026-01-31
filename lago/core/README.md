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

### LinkStream
Temporal network representation supporting:
- Discrete and continuous time
- Directed and undirected edges
- Delayed (asynchronous) interactions
- Weighted edges

### TimeModules
Container for detected temporal modules. Provides iteration and access to individual modules.

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
