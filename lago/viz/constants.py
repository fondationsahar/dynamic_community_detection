"""
Constants and default values for the longitudinal plotting library.

This module defines default configuration values used throughout the longitudinal
plotting library. These constants provide consistent defaults and make it easy
to adjust settings in one place.
"""

# Figure defaults
DEFAULT_FIGURE_WIDTH: int = 800
"""Default figure width in pixels."""

DEFAULT_FIGURE_HEIGHT: int = 600
"""Default figure height in pixels."""

DEFAULT_DPI: int = 100
"""Default dots per inch for figure rendering."""

# Edge styling defaults
DEFAULT_EDGE_ALPHA: float = 1.0
"""Default transparency for edges."""

DEFAULT_EDGE_FLATTEN_FACTOR: float = 0.7
"""Default flattening factor for edge arcs."""

DEFAULT_WEIGHT_SCALE: float = 2.0
"""Default scale factor for edge weights."""

# Node styling defaults
DEFAULT_NODE_ALPHA: float = 0.4
"""Default transparency for node lines."""

DEFAULT_NODE_LINE_WIDTH: float = 0.25
"""Default line width for node lines."""

DEFAULT_FOCUSED_NODE_LINE_WIDTH: float = 0.5
"""Default line width for focused node lines."""

# Module styling defaults
DEFAULT_MODULE_HEIGHT: float = 0.7
"""Default height of module rectangles."""

# Backward compat alias
DEFAULT_COMMUNITY_HEIGHT = DEFAULT_MODULE_HEIGHT

MONOCHROME_FALLBACK_COLOR: str = "gainsboro"
"""Default color for monochrome/unfocused modules."""

DEFAULT_BACKGROUND_MODULE_COLOR: str = "gainsboro"
"""Default color for background modules. Not to be confused with plot canvas background."""

# Backward compat alias
DEFAULT_BACKGROUND_COMMUNITY_COLOR = DEFAULT_BACKGROUND_MODULE_COLOR

# Label styling defaults
DEFAULT_LABEL_FONT_SIZE: int = 18
"""Default font size for axis labels."""

DEFAULT_NODE_LABEL_FONT_SIZE: int = 8
"""Default font size for node labels."""

DEFAULT_NODE_LABEL_PADDING: int = 4
"""Default padding for node labels."""

# Highlight styling
DEFAULT_HIGHLIGHT_ALPHA: float = 0.5
"""Default transparency for highlight rectangles."""

DEFAULT_HIGHLIGHT_WIDTH: float = 0.25
"""Default width for highlight rectangles."""

DEFAULT_HIGHLIGHT_HEIGHT: float = 1.0
"""Default height for highlight rectangles."""

# Arrow styling (for directed edges)
DEFAULT_ARROW_ANGLE_DEGREES: float = 25.0
"""Default angle for arrow heads in degrees."""

# Cross marker styling (for edge activity)
DEFAULT_CROSS_SIZE: float = 0.2
"""Default size for cross markers."""

DEFAULT_CROSS_LINE_WIDTH: float = 0.5
"""Default line width for cross markers."""

# Night highlight styling
DEFAULT_NIGHT_LINE_WIDTH: float = 2.0
"""Default line width for night highlight lines."""

# Color generation
DEFAULT_PASTEL_SATURATION: float = 0.5
"""Default saturation for pastel colors."""

DEFAULT_PASTEL_VALUE: float = 0.9
"""Default value (brightness) for pastel colors."""
