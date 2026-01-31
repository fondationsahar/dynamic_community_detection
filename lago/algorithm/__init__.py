"""LAGO algorithm for temporal community detection.

This module contains the main detection algorithm:
- lago_modules: Main entry point for detecting temporal modules
"""

from .lago import lago_modules

__all__ = [
    "lago_modules",
]
