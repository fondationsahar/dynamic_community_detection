# LAGO Examples

This folder contains examples to help you get started with LAGO for temporal community detection.

## 📚 Table of Contents

**Core Learning Path** (follow in order):

| File | Description | Difficulty |
|------|-------------|------------|
| `01_getting_started.md` | Introduction to LinkStream and basic concepts | 🟢 Beginner |
| `02_linkstream_types.ipynb` | Different types of temporal networks | 🟢 Beginner |
| `03_community_detection.ipynb` | Detecting communities with `lago_modules` | 🟡 Intermediate |
| `04_modularity.ipynb` | Computing longitudinal modularity | 🟡 Intermediate |
| `05_visualization.ipynb` | Creating visualizations | 🟡 Intermediate |

**Practical Guides** (use when needed):

| File | Description | When to Use |
|------|-------------|-------------|
| `real_world_preprocessing.ipynb` | Working with names and date strings | 💡 You have real data with labels/dates |
| `viz_example.ipynb` | Advanced visualization API reference | 🎨 Need custom plot configurations |

## 🚀 Quick Start

Start with [01_getting_started.md](01_getting_started.md) for a gentle introduction, then explore the interactive notebooks:

```python
from lago import LinkStream, lago_modules

# 1. Create a temporal network
ls = LinkStream()
ls.add_links([
    (0, 1, 0),  # Alice talks to Bob at time 0
    (1, 2, 0),  # Bob talks to Carol at time 0
    (0, 1, 1),  # Alice talks to Bob at time 1
    (2, 3, 1),  # Carol talks to Dave at time 1
])

# 2. Detect communities
communities = lago_modules(ls)

# 3. Explore results
for module in communities.iter_modules():
    print(f"Community {module.label}: {module.nodes}")
```

**Try it yourself:** Open [02_linkstream_types.ipynb](02_linkstream_types.ipynb) to run this code interactively!

## 📖 Concepts

### What is a Link Stream?
A **link stream** is a temporal network where edges (interactions) occur at specific times. Think of it like a sequence of events:
- Social media interactions over time
- Email exchanges with timestamps
- Physical contacts between people

### What are Temporal Communities?
**Temporal communities** are groups of nodes that interact closely during certain time periods. Unlike static communities, these can:
- Change over time
- Merge or split
- Have nodes that join or leave

### How does LAGO work?
LAGO (Longitudinal Agglomerative Greedy Optimization) finds communities by optimizing a quality function called **L-Modularity**. It balances:
- **Topological structure**: Are nodes densely connected?
- **Temporal smoothness**: Do communities change gradually over time?

## 🔧 Installation

```bash
pip install dcd-lago
```

Or from source:
```bash
git clone https://github.com/fondationsahar/dynamic_community_detection.git
cd dynamic_community_detection
pip install -e .
```

## 💡 How to Use These Examples

All examples are provided as **Jupyter notebooks** for interactive exploration:

1. **Start Jupyter:** `jupyter notebook` or `jupyter lab`
2. **Navigate** to the `examples/` folder
3. **Open** any `.ipynb` file to get started
4. **Run cells** with Shift+Enter to execute code and see results inline

For a text-based introduction, start with [01_getting_started.md](01_getting_started.md).
