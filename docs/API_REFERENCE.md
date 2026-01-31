# LAGO API Reference

Complete documentation for the LAGO library's main functions and classes.

---

## Table of Contents

1. [lago_modules](#lago_modules) - Detect temporal communities
2. [longitudinal_modularity](#longitudinal_modularity) - Compute quality metric
3. [TimeModules](#timemodules) - Navigate and explore results
4. [LongitudinalModulesPlot](#longitudinalplot) - Visualization

---

## lago_modules

Detect temporal communities in link streams using the LAGO algorithm.

### Signature

```python
lago_modules(
    linkstream: LinkStream,
    lex: LexType | str = LexType.MM,
    nb_iter: int = 1,
    alpha: float = 1,
    omega: float = 2,
    refinement: str | None = "STEM",
    fast_exploration: bool = True,
    refinement_in: bool = True,
    verbose: bool = False,
    stopping_criterion: float = 1e-8,
    ndigits_logs: int = 8,
) -> TimeModules
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `linkstream` | `LinkStream` | *required* | The temporal network to analyze |
| `lex` | `LexType \| str` | `"MM"` | Longitudinal expectation type: `"MM"`, `"JM"`, or `LexType.MM`, `LexType.JM` |
| `nb_iter` | `int` | `1` | Number of algorithm runs (best result kept) |
| `alpha` | `float` | `1` | Resolution parameter (higher = smaller communities) |
| `omega` | `float` | `2` | Temporal smoothness (higher = less community switching) |
| `refinement` | `str \| None` | `"STEM"` | Refinement strategy: `None`, `"STNM"`, or `"STEM"` |
| `fast_exploration` | `bool` | `True` | Use fast exploration (faster but may find lower quality) |
| `refinement_in` | `bool` | `True` | Apply refinement within main loop |
| `verbose` | `bool` | `False` | Print progress information |
| `stopping_criterion` | `float` | `1e-8` | Convergence threshold |
| `ndigits_logs` | `int` | `8` | Decimal places for logging |

### Returns

`TimeModules` - Container with detected temporal communities.

### Expectation Types (`lex`)

| Value | Name | Description |
|-------|------|-------------|
| `"MM"` / `LexType.MM` | Mean-Membership | Most flexible, recommended for general use |
| `"JM"` / `LexType.JM` | Joint-Membership | Favors stable, long-lasting communities |

### Example

```python
from lago import LinkStream, lago_modules

# Create linkstream
ls = LinkStream()
ls.add_links([
    (0, 1, 0), (1, 2, 0), (0, 2, 0),  # Dense group at time 0
    (0, 1, 1), (1, 2, 1),              # Same group at time 1
    (3, 4, 0), (3, 4, 1),              # Another group
])

# Basic detection
communities = lago_modules(ls)

# With custom parameters
communities = lago_modules(
    ls,
    lex="MM",
    alpha=1.5,    # Smaller communities
    omega=3,      # More temporal stability
    nb_iter=3,    # Run 3 times, keep best
)

# Access results
print(f"Found {communities.nb_modules} communities")
for module in communities.iter_modules():
    print(f"  {module.label}: {module.nodes}")
```

---

## longitudinal_modularity

Compute the longitudinal modularity quality metric for a community partition.

### Signature

```python
longitudinal_modularity(
    linkstream: LinkStream,
    communities: dict[CommunityLabel, set[tuple[int, int]]] | TimeModules,
    lex: LexType | str = LexType.MM,
    alpha: float = 1.0,
    omega: float = 2.0,
    ndigits: int = 5,
) -> ModularityResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `linkstream` | `LinkStream` | *required* | The temporal network |
| `communities` | `dict \| TimeModules` | *required* | Community partition to evaluate |
| `lex` | `LexType \| str` | `"MM"` | Expectation type: `"CM"`, `"JM"`, `"MM"` |
| `alpha` | `float` | `1.0` | Weight for expectation term |
| `omega` | `float` | `2.0` | Weight for time penalty |
| `ndigits` | `int` | `5` | Decimal places for rounding |

### Returns

`ModularityResult` with attributes:
- `value`: Total modularity value
- `time_penalty`: The time penalty component
- `modularity_without_penalty`: Value without time penalty
- `lex_type`: The expectation type used

### Expectation Types (`lex`)

| Value | Description |
|-------|-------------|
| `"CM"` / `LexType.CM` | Coexistence - based on node co-occurrence |
| `"JM"` / `LexType.JM` | Joint-Membership - uses community duration |
| `"MM"` / `LexType.MM` | Mean-Membership - uses geometric mean of node durations |

### Example

```python
from lago import LinkStream, lago_modules, longitudinal_modularity, LexType

ls = LinkStream()
ls.add_links([(0, 1, 0), (1, 2, 0), (0, 1, 1)])

# Evaluate LAGO's result
communities = lago_modules(ls)
result = longitudinal_modularity(ls, communities)

print(f"Modularity: {result.value}")
print(f"Time penalty: {result.time_penalty}")

# Compare expectation types
for lex in [LexType.MM, LexType.JM, LexType.CM]:
    result = longitudinal_modularity(ls, communities, lex=lex)
    print(f"  {lex.name}: {result.value}")

# Evaluate manual partition
manual = {
    0: {(0, 0), (1, 0), (0, 1), (1, 1)},
    1: {(2, 0)},
}
result = longitudinal_modularity(ls, manual)
```

---

## TimeModules

Container for navigating temporal community detection results.

### Constructor

```python
TimeModules(
    raw_modules: dict[int, set[tuple[int, int]]] | None = None,
    *,
    path: str | Path | None = None,
)
```

### Initialization Options

```python
# From raw modules dict
tm = TimeModules({
    0: {(0, 0), (0, 1), (1, 0), (1, 1)},  # Module 0
    1: {(2, 0), (2, 1)},                   # Module 1
})

# Empty (populate later)
tm = TimeModules()

# From file (auto-detect format)
tm = TimeModules(path="communities.json")
tm = TimeModules(path="communities.csv")
tm = TimeModules(path="communities.txt")
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `nodes` | `frozenset[int]` | All unique nodes |
| `times` | `frozenset[int]` | All unique time points |
| `modules` | `frozenset[int]` | All module labels |
| `nb_nodes` | `int` | Number of unique nodes |
| `nb_times` | `int` | Number of unique times |
| `nb_modules` | `int` | Number of modules |
| `time_range` | `tuple[int, int]` | (min_time, max_time) |

### Node-Centric Methods

```python
# Get all modules a node belongs to
memberships = tm.get_modules_of_node(node_id)
for m in memberships:
    print(f"Module {m.module}, times: {m.times}")

# Get node's module at specific time
module = tm.get_module_of_node_at_time(node_id, time)

# Get node's trajectory over time
trajectory = tm.get_node_trajectory(node_id)  # {time: module}

# Get module switches
switches = tm.get_node_switches(node_id)  # [(time, from, to), ...]

# Stability score (1.0 = never switches)
stability = tm.get_stability_score(node_id)
```

### Time-Centric Methods

```python
# Get all nodes and their modules at a time
membership = tm.get_nodes_modules_membership_at_time(time)  # {node: module}

# Get active nodes at a time
nodes = tm.get_active_nodes_at_time(time)

# Get modules present at a time
modules = tm.get_modules_at_time(time)
```

### Module-Centric Methods

```python
# Get a specific module
module = tm.get_module(label)
print(module.nodes)      # Nodes in module
print(module.times)      # Times module is active
print(module.size)       # Number of nodes
print(module.duration)   # Number of time points
print(module.cohesion)   # Cohesion score (0-1)

# Iterate over all modules
for module in tm.iter_modules():
    print(f"{module.label}: {module.nodes}")

# Get module nodes
nodes = tm.get_module_nodes(label)

# Get module size/duration
size = tm.get_module_size(label)
duration = tm.get_module_duration(label)

# Get module members at specific time
nodes_at_t = tm.get_module_members_at_time(label, time)
```

### Save/Load Methods

```python
# Save to files
tm.to_json("communities.json")
tm.to_json("communities.json", use_segments=True)  # Compact format
tm.to_csv("communities.csv")
tm.to_txt("communities.txt")

# Load from files (class methods)
tm = TimeModules.from_json("communities.json")
tm = TimeModules.from_csv("communities.csv")
tm = TimeModules.from_txt("communities.txt")

# Or use constructor
tm = TimeModules(path="communities.json")
```

### Conversion Methods

```python
# For longitudinal_modularity
communities_dict = tm.to_communities_dict()

# For custom processing
labels = tm.to_flat_labels()  # {(node, time): module}
```

### Example

```python
from lago import LinkStream, lago_modules

ls = LinkStream()
ls.add_links([(0, 1, 0), (1, 2, 0), (0, 1, 1), (2, 3, 1)])

tm = lago_modules(ls)

# Explore results
print(f"Summary: {tm.summary()}")

# Track a node
trajectory = tm.get_node_trajectory(0)
print(f"Node 0 trajectory: {trajectory}")

# Examine a module
module = tm.get_module(0)
print(f"Module 0: {module.size} nodes, {module.duration} time points")
print(f"Cohesion: {module.cohesion:.2f}")

# Get time segment breakdown
segments = module.get_node_segments()
for node, segs in segments.items():
    print(f"  Node {node}: {[(s.start, s.end) for s in segs]}")

# Save results
tm.to_json("results.json")
```

---

## LongitudinalModulesPlot

Visualize temporal communities as a longitudinal plot.

### Constructor

```python
LongitudinalModulesPlot(
    time_modules: TimeModules,
    linkstream: LinkStream | None = None,
    width: int = 800,
    height: int = 600,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_modules` | `TimeModules` | *required* | Communities to visualize |
| `linkstream` | `LinkStream` | `None` | Required for edge visualization |
| `width` | `int` | `800` | Figure width in pixels |
| `height` | `int` | `600` | Figure height in pixels |

### Configuration Methods

#### configure_nodes()

```python
plot.configure_nodes(
    nodes: list[int] | None = None,      # Node order (default: auto)
    labels: dict[int, str] | None = None, # Node labels
    focus: list[int] | None = None,       # Nodes to highlight
    auto_ordering: bool = False,          # Optimize node order
    fontsize: int = 10,
    label_padding: float = 0,
    linewidth: float = 0.5,
    linewidth_focus: float = 1.0,
)
```

#### configure_edges()

```python
plot.configure_edges(
    show_edges: bool = True,              # Show arc edges
    show_activity: bool = False,          # Show activity markers
    activity_width: float = 0.8,
    activity_height: float = 0.5,
    activity_alpha: float = 0.3,
)
```

#### configure_communities()

```python
plot.configure_communities(
    max_shown: int | None = None,         # Limit communities shown
    color_palette: str = "tab10",         # Matplotlib colormap
    height: float = 0.8,                  # Rectangle height
    focus_communities: dict | None = None, # {label: color} to highlight
    show_unfocused: bool = True,
    unfocused_style: str = "lighter",
)
```

#### configure_display()

```python
plot.configure_display(
    padding_bottom: float = 0.5,
    padding_top: float = 0.5,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_yticks: bool = True,
)
```

### Drawing Methods

```python
# Draw the plot
plot.draw()

# Draw and get matplotlib axis for further customization
fig, ax = plot.draw(return_ax=True)

# Save to file
plot.save("output.png", dpi=150)
```

### Complete Example

```python
from lago import LinkStream, lago_modules
from lago.viz import LongitudinalModulesPlot

# Create data
ls = LinkStream()
ls.add_links([
    (0, 1, 0), (1, 2, 0), (0, 2, 0),
    (0, 1, 1), (1, 2, 1),
    (3, 4, 0), (3, 4, 1),
])

communities = lago_modules(ls)

# Create plot
plot = LongitudinalModulesPlot(communities, linkstream=ls, width=1200, height=600)

# Configure
plot.configure_nodes(
    labels={0: "Alice", 1: "Bob", 2: "Carol", 3: "Dave", 4: "Eve"},
    auto_ordering=True,
    fontsize=10,
)

plot.configure_edges(
    show_activity=True,
    activity_alpha=0.4,
)

plot.configure_communities(
    color_palette="tab10",
    height=0.8,
)

plot.configure_display(
    show_xlabel=True,
    show_ylabel=True,
)

# Draw and save
plot.draw()
plot.save("communities.png", dpi=150)
```

### Advanced: Matplotlib Customization

```python
fig, ax = plot.draw(return_ax=True)

# Add annotations
ax.annotate("Merger", xy=(2, 1), fontsize=9, color="red")

# Add title
ax.set_title("Community Evolution", fontsize=14)

# Save with custom settings
fig.savefig("custom.png", dpi=300, bbox_inches="tight")
```

---

## Quick Reference

```python
from lago import (
    LinkStream,
    lago_modules,
    longitudinal_modularity,
    TimeModules,
    LexType,
)
from lago.viz import LongitudinalModulesPlot

# 1. Create linkstream
ls = LinkStream()
ls.add_links([(0, 1, 0), (1, 2, 0), (0, 1, 1)])

# 2. Detect communities
tm = lago_modules(ls, lex="MM", omega=2)

# 3. Evaluate quality
result = longitudinal_modularity(ls, tm)
print(f"Modularity: {result.value}")

# 4. Explore results
for module in tm.iter_modules():
    print(f"Module {module.label}: {module.nodes}")

# 5. Visualize
plot = LongitudinalModulesPlot(tm, linkstream=ls)
plot.configure_nodes(auto_ordering=True)
plot.configure_edges(show_activity=True)
plot.draw()
plot.save("result.png")

# 6. Save results
tm.to_json("communities.json")
```
