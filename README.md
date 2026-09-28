# LAGO

**Dynamic Community Detection for Temporal Networks**

[![PyPI version](https://badge.fury.io/py/dcd-lago.svg)](https://pypi.org/project/dcd-lago/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/fondationsahar/dynamic_community_detection/blob/main/LICENSE.txt)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Community detection reveals the hidden structure of a network — groups that interact more among themselves than with the rest — and turns a network too large to read into a small set of groups that can be. LAGO does this for temporal networks (link streams): it finds communities that form, merge, split and dissolve as interactions happen.

**No time window needed.** Traditional approaches require aggregating interactions into snapshots (e.g., daily or hourly networks), losing temporal precision and forcing you to choose an arbitrary window size. LAGO works directly on the raw timestamped data—no aggregation, no information loss.

<p align="center">
<img src="https://raw.githubusercontent.com/fondationsahar/dynamic_community_detection/main/img/lmodules_ex.png" alt="Temporal communities example" width="600"/>
<br>
<em>A link stream with 5 nodes showing two dynamic communities (blue and green) that evolve over time.</em>
</p>

## Installation

```bash
pip install dcd-lago           # Core library (no dependencies)
pip install dcd-lago[viz]      # With visualization (numpy, pandas, matplotlib, scikit-learn)
```

## Quick Start

```python
from lago import LinkStream, lago_modules

# Create a temporal network
ls = LinkStream()
ls.add_links([
    (0, 1, 0), (1, 2, 0), (0, 2, 0),  # Triangle at t=0
    (0, 1, 1), (1, 2, 1),              # Path at t=1
    (3, 4, 0), (3, 4, 1), (3, 4, 2),   # Pair at t=0,1,2
    (2, 3, 2),                          # Bridge at t=2
])

# Detect temporal communities
communities = lago_modules(ls)

# Explore results
for module in communities.iter_modules():
    print(f"Community {module.label}: nodes {module.nodes}, duration {module.duration}")
```

> **Note:** LAGO works on discrete integer timestamps. For best results, normalize your
> timestamps so that the smallest time gap is 1 (e.g., divide all timestamps by their GCD).

## Features

| Feature | Description |
|---------|-------------|
| **Temporal precision** | Handles exact timestamps, not time windows |
| **Quality metric** | Built-in Longitudinal Modularity scoring |
| **Flexible input** | Weighted, directed, delayed, continuous, and k-partite networks |
| **Rich output** | Track node trajectories, community evolution, and transitions |
| **Visualization** | Publication-ready longitudinal plots |

## Usage Examples

### Evaluate Community Quality

```python
from lago import longitudinal_modularity

result = longitudinal_modularity(ls, communities, lex="MM")
print(f"Quality score: {result.value:.4f}")
```

### Visualize Results

```python
from lago.viz import LongitudinalModulesPlot

plot = LongitudinalModulesPlot(communities, linkstream=ls, width=1200, height=600)
plot.configure_nodes(auto_ordering=True)
plot.configure_edges(show_activity=True)
plot.configure_modules(color_palette="tab10")
plot.draw()
plot.save("communities.png", dpi=150)
```

### Explore Results

```python
# Track a specific node
trajectory = communities.get_node_trajectory(node=0)
print(f"Node 0 was in communities: {trajectory}")

# Get community membership at a specific time
membership = communities.get_nodes_modules_membership_at_time(time=1)
print(f"At t=1: {membership}")

# Save/load results
communities.to_json("results.json")
```

## Key Parameters

### `lago_modules()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lex` | `"MM"` | Expectation type: `"MM"` (flexible) or `"JM"` (stable communities) |
| `omega` | `2` | Temporal smoothness (higher = fewer community switches) |
| `gamma` | `1` | Resolution (higher = smaller communities) |
| `refinement` | `"STEM"` | Refinement strategy: `None`, `"STNM"`, or `"STEM"` |
| `nb_iter` | `1` | Number of runs (keeps best result) |
| `seed` | `None` | Exploration order to follow. Results are reproducible with or without it; different seeds reach different optima, which `nb_iter` exploits |
| `n_jobs` | `1` | Processes over which to spread the `nb_iter` runs (same result as `1`; memory grows with it) |

### `longitudinal_modularity()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lex` | `"MM"` | Expectation type: `"MM"`, `"JM"`, or `"CM"` |
| `omega` | `2.0` | Weight for temporal penalty |
| `gamma` | `1.0` | Weight for expectation term |
| `ndigits` | `5` | Decimal places of the returned value |

## Performance

Nothing to configure: `pip install dcd-lago` gives the fastest implementation your platform supports.

- From a binary wheel, the hot modules of the search and the metric's counting kernel come compiled (Cython). From a source install they are compiled when a C compiler is available and run as pure Python otherwise. Both give **the same results, bit for bit** — the pure-Python sources are the reference and ship alongside the compiled modules.
- `lago_modules` is deterministic: the same input and parameters give the same modules, in any process. `seed` selects the exploration order, `nb_iter` tries several orders and keeps the best, and `n_jobs` runs those in parallel processes (memory and platform notes in the docstring).
- To see what is running: `lago.accel.core_name()` is `"compiled"` or `"python"`, `lago.accel.backend_name` is `"cython"` or `"python"` for the metric kernel. `LAGO_CORE=python` forces the pure sources. In a checkout, `python setup.py build_ext --inplace` builds the compiled modules next to the sources.

Against release 1.1.0: 8× (pure Python) to 12× (compiled) on small streams over every parameter combination, 6–109× on streams of 4k–96k interactions, JM at the speed of MM — see [docs/BENCHMARK_1.1.0_VS_1.2.0.md](https://github.com/fondationsahar/dynamic_community_detection/blob/main/docs/BENCHMARK_1.1.0_VS_1.2.0.md).

## Documentation

📖 **Guides**
- [Getting Started](https://github.com/fondationsahar/dynamic_community_detection/blob/main/examples/01_getting_started.md) — Concepts and first steps
- [API Reference](https://github.com/fondationsahar/dynamic_community_detection/blob/main/docs/API_REFERENCE.md) — Complete function documentation
- [Changelog](https://github.com/fondationsahar/dynamic_community_detection/blob/main/CHANGELOG.md) — What changed in each release
- [How it works](https://github.com/fondationsahar/dynamic_community_detection/blob/main/docs/ARCHITECTURE.md) — The implementation, its invariants, the compiled build
- [Performance](https://github.com/fondationsahar/dynamic_community_detection/blob/main/docs/PERFORMANCE.md) — What to expect, how it scales, how to measure

📁 **Examples** ([examples/](https://github.com/fondationsahar/dynamic_community_detection/tree/main/examples/))
- [LinkStream Types](https://github.com/fondationsahar/dynamic_community_detection/blob/main/examples/02_linkstream_types.ipynb) — Weighted, directed, continuous, delayed, k-partite networks
- [Community Detection](https://github.com/fondationsahar/dynamic_community_detection/blob/main/examples/03_community_detection.ipynb) — Using `lago_modules` and exploring results
- [Modularity](https://github.com/fondationsahar/dynamic_community_detection/blob/main/examples/04_modularity.ipynb) — Computing and understanding quality scores
- [Visualization](https://github.com/fondationsahar/dynamic_community_detection/blob/main/examples/05_visualization.ipynb) — Creating publication-ready plots

💡 **Practical Guides**
- [Real-World Preprocessing](https://github.com/fondationsahar/dynamic_community_detection/blob/main/examples/real_world_preprocessing.ipynb) — Working with names and date strings
- [Advanced Visualization](https://github.com/fondationsahar/dynamic_community_detection/blob/main/examples/viz_example.ipynb) — Custom plot configurations

The examples are Jupyter notebooks (Getting Started is a page); run `jupyter notebook` in the examples folder.

📄 **Papers**
- [LAGO Method (ICDM)](https://ieeexplore.ieee.org/document/11391928) — Algorithm details and experiments
- [Longitudinal Modularity (EPJ Data Science)](https://rdcu.be/eC5fA) — Quality function theory

## Citation

If you use LAGO in your research, please cite:

**LAGO Method:**
```bibtex
@INPROCEEDINGS{11391928,
  author={Brabant, Victor and Bonifati, Angela and Cazabet, Rémy},
  booktitle={2025 IEEE International Conference on Data Mining (ICDM)}, 
  title={Discovering Communities in Continuous-Time Temporal Networks by Optimizing L-Modularity}, 
  year={2025},
  volume={},
  number={},
  pages={1065-1074},
  keywords={Accuracy;Network analyzers;Benchmark testing;Market research;Data mining;Optimization;Guidelines;temporal networks;community detection;dynamic communities;link stream;modularity},
  doi={10.1109/ICDM65498.2025.00115}}
```

**Longitudinal Modularity:**
```bibtex
@article{Brabant2025lmod,
    title={Longitudinal modularity, a modularity for link streams},
    author={Brabant, Victor and Asgari, Yasaman and Borgnat, Pierre and Bonifati, Angela and Cazabet, Rémy},
    journal={EPJ Data Science},
    volume={14},
    number={1},
    year={2025},
    doi={10.1140/epjds/s13688-025-00529-x},
}
```

## Contributing

Questions, suggestions, or issues? Please open a [GitHub issue](https://github.com/fondationsahar/dynamic_community_detection/issues).

## Acknowledgements

We thank [Jean-Loup Guillaume](http://jlguillaume.free.fr/www/) for the discussions that helped improve this project.

## License

MIT License — see [LICENSE.txt](https://github.com/fondationsahar/dynamic_community_detection/blob/main/LICENSE.txt)
