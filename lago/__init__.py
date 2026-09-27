"""LAGO: Longitudinal Agglomerative Greedy Optimization for temporal community detection.

Core Exports (always available):
    - LinkStream: Temporal network data structure
    - TimeModules, TimeModule: Module detection results container
    - TimeModulesNetwork: Metagraph of module relationships
    - lago_modules: Main detection algorithm
    - longitudinal_modularity: Modularity computation
    - LexType, LinkStreamMode: Enums for configuration

Visualization Exports (requires dcd-lago[viz]):
    - LongitudinalModulesPlot: Temporal module visualization

Example:
    >>> from lago import LinkStream, lago_modules, LexType
    >>> ls = LinkStream()
    >>> ls.add_links([(0, 1, 0), (1, 2, 1)])
    >>> modules = lago_modules(ls, lex=LexType.MM)

Subpackage Access:
    >>> from lago.core import LinkStream, TimeModules
    >>> from lago.algorithm import lago_modules
    >>> from lago.metrics import longitudinal_modularity
"""

import os as _os
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("dcd-lago")
except _PackageNotFoundError:  # a checkout that was never installed
    __version__ = "0+unknown"

if _os.environ.get("LAGO_CORE", "").strip().lower() == "python":
    # Import lago's own modules from their .py sources even where a compiled
    # extension built by setup.py sits beside them. The equivalence harness
    # uses this to hold the compiled modules to the Python ones; it has no
    # other purpose.
    import importlib.abc as _abc
    import importlib.machinery as _machinery
    import sys as _sys

    class _SourceOnlyFinder(_abc.MetaPathFinder):
        _loaders = ((_machinery.SourceFileLoader, _machinery.SOURCE_SUFFIXES),)

        def find_spec(self, fullname, path, target=None):
            if not fullname.startswith("lago."):
                return None
            for entry in path or []:
                spec = _machinery.FileFinder(entry, *self._loaders).find_spec(fullname)
                if spec is not None:
                    return spec
            return None

    _sys.meta_path.insert(0, _SourceOnlyFinder())

# Core exports - re-exported from subpackages
# Algorithm exports
from .algorithm import lago_modules
from .core import (
    LexType,
    LinkStream,
    LinkStreamMode,
    TimeModule,
    TimeModules,
    TimeModulesNetwork,
    TimeSegment,
)

# Metrics exports
from .metrics import ModularityResult, longitudinal_modularity

# Track what's available
__all__ = [
    # Core
    "LexType",
    "LinkStream",
    "LinkStreamMode",
    "ModularityResult",
    "TimeModule",
    "TimeModules",
    "TimeModulesNetwork",
    "TimeSegment",
    # Algorithm
    "lago_modules",
    # Metrics
    "longitudinal_modularity",
    # Viz (conditional)
    "LongitudinalModulesPlot",
]

# Optional visualization export - requires matplotlib/sklearn
# Use lazy loading to provide helpful error message
_viz_import_error: ImportError | None = None


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "LongitudinalModulesPlot":
        global _viz_import_error
        try:
            from .viz import LongitudinalModulesPlot

            return LongitudinalModulesPlot
        except ImportError as e:
            _viz_import_error = e
            raise ImportError(
                f"'{name}' requires visualization dependencies.\n"
                "Install with: pip install dcd-lago[viz]\n"
                f"Original error: {e}"
            ) from None

    raise AttributeError(f"module 'lago' has no attribute '{name}'")
