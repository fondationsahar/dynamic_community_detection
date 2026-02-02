"""
Time Modules Plotting Library (lago.viz)

A modular library for visualizing time modules in temporal networks.

This subpackage requires optional dependencies:
    pip install dcd-lago[viz]

This library provides an object-oriented API (LongitudinalModulesPlot) for creating
temporal module visualizations with matplotlib.

Example:
    >>> from lago import LongitudinalModulesPlot
    >>>
    >>> plot = LongitudinalModulesPlot(linkstream, width=1600, height=1200)
    >>> plot.configure_nodes(nodes=node_list, labels=labels, auto_ordering=True)
    >>> plot.configure_edges(show_edges=False, show_activity=True)
    >>> plot.configure_modules(modules=time_modules, color_palette="tab20")
    >>> plot.draw()
    >>> plot.save("output.png", dpi=750)
"""

__version__ = "1.1.0"


# Check for required dependencies before importing anything
def _check_viz_dependencies():
    """Check that visualization dependencies are installed."""
    import importlib.util

    missing = []

    if importlib.util.find_spec("matplotlib") is None:
        missing.append("matplotlib")

    if importlib.util.find_spec("sklearn") is None:
        missing.append("scikit-learn")

    if missing:
        raise ImportError(
            f"lago.viz requires additional dependencies: {', '.join(missing)}\n"
            "Install with: pip install dcd-lago[viz]"
        )


# Check dependencies on import
_check_viz_dependencies()

# Now safe to import all modules that depend on matplotlib/sklearn
# Type definitions (for type hints in user code)
# Configuration functions
from .configuration import (
    configure_axes,
    finalize_plot,
    setup_figure_and_axes,
)

# Constants (for users who want to use defaults)
from .constants import (
    DEFAULT_COMMUNITY_HEIGHT,
    DEFAULT_DPI,
    DEFAULT_EDGE_ALPHA,
    DEFAULT_EDGE_FLATTEN_FACTOR,
    DEFAULT_FIGURE_HEIGHT,
    DEFAULT_FIGURE_WIDTH,
    DEFAULT_LABEL_FONT_SIZE,
    DEFAULT_NODE_ALPHA,
    DEFAULT_NODE_LABEL_FONT_SIZE,
    DEFAULT_NODE_LABEL_PADDING,
    DEFAULT_WEIGHT_SCALE,
    MONOCHROME_FALLBACK_COLOR,
)

# Data preparation functions
from .data_preparation import (
    calculate_node_time_ranges,
    create_time_node_module_mapping,
    filter_and_sort_modules,
    prepare_modules_for_display,
)

# Drawing functions
from .drawing import (
    draw_module_periods,
    draw_edge_activity,
    draw_edges,
    draw_edges_delayed,
    draw_focus_highlights,
    draw_night_highlights,
    draw_nodes,
)

# Geometry functions
from .geometry import (
    create_center_cross,
    create_edge_arc,
    create_highlight_rectangle,
    draw_arrow_head,
)
from .plot import LongitudinalModulesPlot

# Main API
from .types import (
    ColorMapping,
    Modules,
    ModuleLabel,
    ModuleMember,
    ModuleNodesSegments,
    NodeFocus,
    NodeId,
    NodesMapping,
    TimeLinks,
    TimeNodeModuleMapping,
    TimePoint,
)

# Utility functions
from .utils import (
    generate_color_mapping,
    generate_monochrome_colors,
    generate_pastel_colors,
    to_segments,
)

__all__ = [
    # Main API
    "LongitudinalModulesPlot",
    # Type definitions
    "ColorMapping",
    "Modules",
    "ModuleLabel",
    "ModuleMember",
    "ModuleNodesSegments",
    "NodeFocus",
    "NodeId",
    "NodesMapping",
    "TimeLinks",
    "TimeNodeModuleMapping",
    "TimePoint",
    # Constants
    "DEFAULT_COMMUNITY_HEIGHT",
    "DEFAULT_DPI",
    "DEFAULT_EDGE_ALPHA",
    "DEFAULT_EDGE_FLATTEN_FACTOR",
    "DEFAULT_FIGURE_HEIGHT",
    "DEFAULT_FIGURE_WIDTH",
    "DEFAULT_LABEL_FONT_SIZE",
    "DEFAULT_NODE_ALPHA",
    "DEFAULT_NODE_LABEL_FONT_SIZE",
    "DEFAULT_NODE_LABEL_PADDING",
    "DEFAULT_WEIGHT_SCALE",
    "MONOCHROME_FALLBACK_COLOR",
    # Utility functions
    "generate_color_mapping",
    "generate_monochrome_colors",
    "generate_pastel_colors",
    "to_segments",
    # Configuration
    "configure_axes",
    "finalize_plot",
    "setup_figure_and_axes",
    # Data preparation
    "calculate_node_time_ranges",
    "create_time_node_module_mapping",
    "filter_and_sort_modules",
    "prepare_modules_for_display",
    # Drawing
    "draw_module_periods",
    "draw_edge_activity",
    "draw_edges",
    "draw_edges_delayed",
    "draw_focus_highlights",
    "draw_night_highlights",
    "draw_nodes",
    # Geometry
    "create_center_cross",
    "create_edge_arc",
    "create_highlight_rectangle",
    "draw_arrow_head",
]
