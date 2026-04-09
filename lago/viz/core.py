"""
Core utility functions for longitudinal module plotting.

This module contains helper functions for node ordering and weight normalization
that are used by the LongitudinalPlot class.

Key Functions:
    - rank_from_similarities: Rank nodes based on similarity matrix using MDS
    - _auto_node_ordering_given_time_modules: Auto-order nodes to minimize edge crossings
    - normalize_weights: Normalize edge weights using hybrid approach
"""

from itertools import combinations
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.manifold import Isomap


def rank_from_similarities(
    similarity_list: list,
) -> Tuple[Dict[int, int], np.ndarray, list]:
    """
    Rank nodes based on similarity using Multi-Dimensional Scaling (MDS).

    This function takes a list of similarity values between pairs of nodes and
    produces a 1D ranking that preserves the similarity relationships.

    Args:
        similarity_list: List of [id1, id2, similarity] tuples where similarity
            is a value between 0 and 1.

    Returns:
        Tuple containing:
            - rank_mapping: Dictionary mapping node IDs to their rank (0 = lowest position)
            - positions: Array of 1D positions from MDS
            - ids: List of unique node IDs
    """
    # Collect unique ids
    ids = sorted({i for pair in similarity_list for i in pair[:2]})
    id_to_idx = {id_: i for i, id_ in enumerate(ids)}
    idx_to_id = {i: id_ for id_, i in id_to_idx.items()}
    n = len(ids)

    # Build similarity matrix
    similarity = np.zeros((n, n))
    np.fill_diagonal(similarity, 1.0)

    for id1, id2, sim in similarity_list:
        i, j = id_to_idx[id1], id_to_idx[id2]
        similarity[i, j] = sim
        similarity[j, i] = sim

    # Convert similarity to distance
    eps = 1e-8
    safe_similarity = np.clip(similarity, eps, None)
    distance = 1 / safe_similarity - 1

    n_neighbors = min(
        max(5, int(np.sqrt(distance.shape[0] - 1))), distance.shape[0] - 1
    )

    isomap = Isomap(n_components=1, n_neighbors=n_neighbors, metric="precomputed")

    transformed = isomap.fit_transform(distance)
    positions = transformed.flatten() if transformed is not None else np.array([])

    # Rank elements
    sorted_indices = np.argsort(positions)
    rank_mapping = {idx_to_id[idx]: rank for rank, idx in enumerate(sorted_indices)}

    return rank_mapping, positions, ids


def _auto_node_ordering_given_time_modules(
    modules: Dict,
    time_links: Optional[list] = None,
    factor: float = 0.9,
    subsets: list = [],
) -> Dict[int, int]:
    """
    Automatically order nodes to minimize visual clutter in the plot.

    This function computes an optimal ordering of nodes based on:
    1. Module co-membership (nodes in the same module should be close)
    2. Edge connectivity (nodes connected by edges should be close)

    Args:
        modules: Dictionary mapping module labels to list of (node, time) tuples
        time_links: List of temporal links (for connectivity-based ordering)
        nodes_to_display: Set of nodes to include in the ordering
        factor: Weight for module-based ordering vs edge-based ordering (0-1)
            Higher values prioritize module structure over edge structure

    Returns:
        Dictionary mapping node IDs to their display position (0-indexed)
    """
    nodes_mapping = {}
    for subset in subsets:
        tmp_nodes_mapping = _auto_node_ordering_given_time_modules_forsubset(
            modules, time_links, subset, factor
        )
        tmp_nodes_mapping = {
            key: val + len(nodes_mapping) for key, val in tmp_nodes_mapping.items()
        }
        nodes_mapping.update(tmp_nodes_mapping)
    return nodes_mapping


def _auto_node_ordering_given_time_modules_forsubset(
    modules: Dict,
    time_links: Optional[list] = None,
    nodes_to_display: Optional[Set] = None,
    factor: float = 0.9,
) -> Dict[int, int]:
    """
    Automatically order nodes to minimize visual clutter in the plot.

    This function computes an optimal ordering of nodes based on:
    1. Module co-membership (nodes in the same module should be close)
    2. Edge connectivity (nodes connected by edges should be close)

    Args:
        modules: Dictionary mapping module labels to list of (node, time) tuples
        time_links: List of temporal links (for connectivity-based ordering)
        nodes_to_display: Set of nodes to include in the ordering
        factor: Weight for module-based ordering vs edge-based ordering (0-1)
            Higher values prioritize module structure over edge structure

    Returns:
        Dictionary mapping node IDs to their display position (0-indexed)
    """
    # Handle default mutable arguments
    if time_links is None:
        time_links = []
    if nodes_to_display is None:
        nodes_to_display = set()

    # Flatten time links and compute edge weights
    time_links = [sorted(list(elem)[:2]) + list(elem)[-1:] for elem in time_links]
    df_time_links: pd.DataFrame = pd.DataFrame(
        time_links, columns=["source", "target", "weight"]
    )
    if nodes_to_display:
        source_col: pd.Series = df_time_links["source"]  # type: ignore[assignment]
        target_col: pd.Series = df_time_links["target"]  # type: ignore[assignment]
        df_time_links = df_time_links[
            source_col.isin(nodes_to_display) & target_col.isin(nodes_to_display)
        ]
    df_sum: pd.DataFrame = df_time_links.groupby(  # type: ignore[assignment]
        ["source", "target"], as_index=False
    )["weight"].sum()
    df_sum["weight"] = df_sum["weight"] / df_sum["weight"].max()
    links_weights = dict(zip(zip(df_sum["source"], df_sum["target"]), df_sum["weight"]))
    del df_sum

    # Build node-to-module mapping
    nodes_modules_times: Dict[int, Dict] = {}
    for label, module in modules.items():
        for node, time in module:
            if node not in nodes_to_display:
                continue
            if node not in nodes_modules_times:
                nodes_modules_times[node] = {}
            if label not in nodes_modules_times[node]:
                nodes_modules_times[node][label] = set()
            nodes_modules_times[node][label].add(time)

    # Compute pairwise similarities
    nodes = list(nodes_modules_times.keys())
    similarities = []
    for node1, node2 in combinations(nodes, 2):
        all_commus, all_times = set(), set()
        for node in [node1, node2]:
            for commu, times in nodes_modules_times[node].items():
                all_commus.add(commu)
                all_times |= times

        dur1 = 0
        dur2 = 0
        for commu in all_commus:
            if (
                commu in nodes_modules_times[node1]
                and commu in nodes_modules_times[node2]
            ):
                dur1 += len(nodes_modules_times[node1][commu])
                dur2 += len(nodes_modules_times[node2][commu])

        tnode1, tnode2 = sorted([node1, node2])
        links_sim = links_weights.get((tnode1, tnode2), 0)
        commu_sim = (dur1 * dur2) ** 0.5 / len(all_times) if all_times else 0
        fin_sim = commu_sim * factor + (1 - factor) * links_sim
        similarities.append(
            [
                tnode1,
                tnode2,
                round(fin_sim, ndigits=5),
            ]
        )

    nodes_mapping, _, _ = rank_from_similarities(similarities)
    return nodes_mapping


def normalize_weights(
    source_min: float,
    source_max: float,
    target_min: float,
    target_max: float,
    values: list,
    alpha: float = 0.2,
) -> Dict:
    """
    Normalize weight values using a hybrid rank-based and linear approach.

    This function maps values from a source range to a target range using a
    weighted combination of rank-based and linear normalization.

    Args:
        source_min: Minimum value in the source range
        source_max: Maximum value in the source range
        target_min: Minimum value in the target range
        target_max: Maximum value in the target range
        values: Collection of values to normalize
        alpha: Weight for rank-based normalization (0-1)
            - alpha=0: Pure linear normalization
            - alpha=1: Pure rank-based normalization
            - Default 0.2 means 20% rank-based, 80% linear

    Returns:
        Dictionary mapping original values to normalized values

    Note:
        The hybrid approach helps handle outliers while still preserving
        relative magnitudes between values.
    """
    values_bysort = {
        val: idx / len(values) for idx, val in enumerate(sorted(list(values)))
    }
    values_bynorm = {
        val: (val - source_min) / (source_max - source_min) for val in values
    }

    final_values = {
        val: values_bysort[val] * alpha / 2 + values_bynorm[val] * (1 - alpha) / 2
        for val in values
    }

    final_values = {
        val: target_min + (target_max - target_min) * value
        for val, value in final_values.items()
    }

    return final_values
