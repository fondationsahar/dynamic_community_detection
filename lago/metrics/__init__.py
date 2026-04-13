"""Metrics for temporal community detection.

This module contains quality metrics:
- longitudinal_modularity: Compute modularity score for temporal modules
- ModularityResult: Result container with modularity value and time penalty
"""

from .modularity import ModularityResult, longitudinal_modularity

__all__ = [
    "ModularityResult",
    "longitudinal_modularity",
]
