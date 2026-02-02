"""Core data structures for LAGO.

This module contains the fundamental data structures:
- LinkStream: Temporal network representation
- TimeModules, TimeModule: Module detection results
- TimeModulesNetwork: Metagraph of module relationships
- LexType, LinkStreamMode: Configuration enums
- Utility functions for working with modules
"""

from .enums import LexType, LinkStreamMode
from .linkstream import LinkStream
from .time_modules import TimeModule, TimeModules, TimeSegment
from .time_modules_network import TimeModulesNetwork
from .utils import (
    get_expanded_module,
    get_module_duration,
    get_nodes_durations,
    get_nodes_times,
    log_debug,
    log_info,
)

__all__ = [
    "LexType",
    "LinkStream",
    "LinkStreamMode",
    "TimeModule",
    "TimeModules",
    "TimeModulesNetwork",
    "TimeSegment",
    # Utils
    "get_expanded_module",
    "get_module_duration",
    "get_nodes_durations",
    "get_nodes_times",
    # Logging
    "log_debug",
    "log_info",
]
