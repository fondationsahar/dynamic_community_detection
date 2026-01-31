# Getting Started with LAGO

This guide introduces the basic concepts of temporal networks and how to use LAGO for community detection.

## What You'll Learn

1. What is a LinkStream (temporal network)
2. How to create and populate a LinkStream
3. How to detect communities
4. How to explore results

---

## Part 1: Understanding Link Streams

A **Link Stream** represents interactions between entities over time. Each interaction has:
- A **source** node (who initiates)
- A **target** node (who receives)
- A **time** when it happens
- Optionally, a **weight** (importance/strength)

### Real-World Examples

| Domain | Nodes | Interactions |
|--------|-------|--------------|
| Social Media | Users | Messages, likes, follows |
| Email | People | Emails sent |
| Scientific Collaboration | Researchers | Co-authored papers |
| Physical Proximity | People | Face-to-face contacts |

---

## Part 2: Creating Your First LinkStream

### From Code

```python
from lago import LinkStream

# Create an empty link stream
ls = LinkStream()

# Add interactions as (source, target, time) tuples
ls.add_links([
    (0, 1, 0),  # Node 0 interacts with Node 1 at time 0
    (1, 2, 0),  # Node 1 interacts with Node 2 at time 0
    (0, 1, 1),  # Node 0 interacts with Node 1 at time 1
    (2, 3, 1),  # Node 2 interacts with Node 3 at time 1
    (0, 2, 2),  # Node 0 interacts with Node 2 at time 2
])

# Check basic properties
print(f"Number of nodes: {ls.nb_nodes}")
print(f"Number of edges: {ls.nb_edges}")
print(f"Time range: {ls.min_time} to {ls.max_time}")
```

### From a File

If you have a text file with interactions:

```
# interactions.txt
# source target time
0 1 0
1 2 0
0 1 1
2 3 1
0 2 2
```

Load it like this:

```python
from lago import LinkStream

ls = LinkStream()
ls.read_txt(
    path="interactions.txt",
    columns_order=["source", "target", "time"]
)
```

---

## Part 3: Detecting Communities

Use `lago_modules` to find temporal communities:

```python
from lago import LinkStream, lago_modules

# Create link stream
ls = LinkStream()
ls.add_links([
    # Group 1: nodes 0, 1, 2 interact frequently
    (0, 1, 0), (1, 2, 0), (0, 2, 0),
    (0, 1, 1), (1, 2, 1), (0, 2, 1),
    # Group 2: nodes 3, 4, 5 interact frequently
    (3, 4, 0), (4, 5, 0), (3, 5, 0),
    (3, 4, 1), (4, 5, 1), (3, 5, 1),
    # Weak connection between groups
    (2, 3, 1),
])

# Detect communities
communities = lago_modules(ls)

# Show results
print(f"Found {communities.nb_modules} communities")
```

---

## Part 4: Exploring Results

The `TimeModules` object provides many ways to explore communities:

### List All Communities

```python
for module in communities.iter_modules():
    print(f"Community {module.label}:")
    print(f"  Nodes: {module.nodes}")
    print(f"  Size: {module.size}")
    print(f"  Duration: {module.duration}")
```

### Track a Specific Node

```python
# Which communities does node 0 belong to?
memberships = communities.get_modules_of_node(0)
for membership in memberships:
    print(f"Node 0 is in community {membership.module}")
    print(f"  During times: {membership.times}")
```

### Get Snapshot at a Time

```python
# Which nodes are in which community at time 0?
snapshot = communities.get_nodes_modules_membership_at_time(0)
for node, community in snapshot.items():
    print(f"Node {node} is in community {community}")
```

---

## Part 5: Parameters to Know

### The `lex` Parameter

Controls how LAGO measures community quality:

| Value | Name | Best For |
|-------|------|----------|
| `"MM"` | Mean-Membership | General use (default) |
| `"JM"` | Joint-Membership | Stable, long-lasting communities |

```python
# Default (recommended)
communities = lago_modules(ls, lex="MM")

# For more stable communities
communities = lago_modules(ls, lex="JM")
```

### The `omega` Parameter

Controls how smooth community changes should be:

```python
# Default: moderate smoothness
communities = lago_modules(ls, omega=2)

# Allow more community changes
communities = lago_modules(ls, omega=1)

# Enforce stable communities
communities = lago_modules(ls, omega=5)
```

### The `alpha` Parameter

Controls community size:

```python
# Default: balanced communities
communities = lago_modules(ls, alpha=1)

# Smaller, more focused communities
communities = lago_modules(ls, alpha=2)

# Larger, more inclusive communities
communities = lago_modules(ls, alpha=0.5)
```

---

## Next Steps

- `02_linkstream_types.py` - Learn about directed, continuous, and delayed networks
- `03_community_detection.py` - Advanced community detection options
- `05_visualization.py` - Visualize your results
