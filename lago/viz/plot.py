"""
LongitudinalModulesPlot class for creating and customizing time modules visualizations.

This module provides an object-oriented interface for creating longitudinal module plots
with matplotlib-like customization capabilities.

Example:
    >>> from lago.viz import LongitudinalModulesPlot
    >>> plot = LongitudinalModulesPlot(linkstream, width=1600, height=1200)
    >>> plot.set_modules(modules)
    >>> plot.set_edge_style(alpha=0.5)
    >>> plot.draw()
    >>> plot.save("output.png")
"""

import warnings
from collections.abc import Sequence
from typing import (
    Any,
    Literal,
    overload,
)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .configuration import configure_axes, finalize_plot, setup_figure_and_axes
from .constants import (
    DEFAULT_BACKGROUND_COMMUNITY_COLOR,
    DEFAULT_COMMUNITY_HEIGHT,
    DEFAULT_EDGE_ALPHA,
    DEFAULT_EDGE_FLATTEN_FACTOR,
    DEFAULT_FIGURE_HEIGHT,
    DEFAULT_FIGURE_WIDTH,
    DEFAULT_LABEL_FONT_SIZE,
    DEFAULT_NODE_ALPHA,
    DEFAULT_NODE_LABEL_FONT_SIZE,
    DEFAULT_NODE_LABEL_PADDING,
    DEFAULT_WEIGHT_SCALE,
)
from .data_preparation import (
    calculate_node_time_ranges,
    create_time_node_module_mapping,
    prepare_modules_for_display,
)
from .drawing import (
    draw_continuous_duration_lines,
    draw_edge_activity,
    draw_edge_activity_delayed,
    draw_edges,
    draw_edges_delayed,
    draw_focus_highlights,
    draw_module_periods,
    draw_night_highlights,
    draw_nodes,
)
from .types import (
    ColorMapping,
    ModuleNodesSegments,
    Modules,
    NodeFocus,
    NodeId,
    NodesMapping,
    TimeLinks,
    TimeNodeModuleMapping,
    TimePoint,
)


def _convert_time_modules_to_modules(time_modules: Any) -> Modules:
    """
    Convert a TimeModules object to the Modules dict format.

    Args:
        time_modules: A lago TimeModules object.

    Returns:
        Modules dict in the format {label: [(node, time), ...]}.

    Raises:
        TypeError: If input is not a TimeModules object.
    """
    if not hasattr(time_modules, "to_communities_dict"):
        raise TypeError(
            f"modules must be a TimeModules object, got {type(time_modules).__name__}. "
            "Raw dicts are no longer supported - use TimeModules instead."
        )

    raw_dict = time_modules.to_communities_dict()
    return {
        key: list(members) if isinstance(members, set) else members
        for key, members in raw_dict.items()
    }


from .utils import generate_color_mapping


class LongitudinalModulesPlot:
    """
    Main class for creating and customizing time modules visualizations.

    This class provides a matplotlib-like interface for creating longitudinal module
    plots with method chaining support for configuration.

    Attributes:
        linkstream: The linkstream object containing temporal network data.
        width: Figure width in pixels.
        height: Figure height in pixels.
        modules: Dictionary of modules to display.
        node_focus: List of nodes to focus/highlight.
        nodes_to_display: List of node indices to display.

    Example:
        >>> plot = LongitudinalModulesPlot(linkstream)
        >>> plot.set_nodes(nodes_to_display, labels=labels)
        >>> plot.set_modules(modules)
        >>> plot.toggle_edges(False)
        >>> plot.toggle_edge_activity(True)
        >>> fig, ax = plot.draw(return_ax=True)
        >>> plot.save("output.png", dpi=750)
    """

    def __init__(
        self,
        linkstream_or_modules: Any,
        linkstream: Any = None,
        width: int = DEFAULT_FIGURE_WIDTH,
        height: int = DEFAULT_FIGURE_HEIGHT,
    ) -> None:
        """
        Initialize a longitudinal plot.

        Accepts either a LinkStream as the first argument, or a TimeModules
        with the LinkStream passed via the ``linkstream`` keyword:

            LongitudinalModulesPlot(linkstream)
            LongitudinalModulesPlot(modules, linkstream=linkstream)

        Args:
            linkstream_or_modules: Either a LinkStream (temporal network data)
                or a TimeModules object (detected modules). When a TimeModules
                is passed, ``linkstream`` must also be provided.
            linkstream: The linkstream object. Only used when the first
                argument is a TimeModules; otherwise must be None.
            width: Figure width in pixels.
            height: Figure height in pixels.
        """
        # Resolve the polymorphic first argument.
        from lago.core.time_modules import TimeModules

        pending_modules: Any = None
        if isinstance(linkstream_or_modules, TimeModules):
            if linkstream is None:
                msg = (
                    "When passing TimeModules as the first argument, "
                    "the 'linkstream' keyword argument is required."
                )
                raise ValueError(msg)
            pending_modules = linkstream_or_modules
            self.linkstream = linkstream
        else:
            if linkstream is not None:
                msg = (
                    "'linkstream' keyword argument is only valid when the "
                    "first argument is a TimeModules object."
                )
                raise ValueError(msg)
            self.linkstream = linkstream_or_modules

        self.width = width
        self.height = height

        # Data configuration
        self.nodes: set[NodeId] | None = None
        self._nodes_ordered_list: list[NodeId] | None = None  # Preserves order when set via list
        self.node_labels: list[str] = []
        self._original_node_labels: list[str] | None = None  # Original labels before remapping
        self.modules: Modules | None = None
        self.node_focus: NodeFocus = []
        self.module_focus: list[Any] = []  # List of module labels to focus on
        self._module_focus_colors: dict[Any, Any] = {}  # Explicit colors for focused modules
        self.background_color: Any = DEFAULT_BACKGROUND_COMMUNITY_COLOR
        self.time_focus: TimePoint | None = None
        self.node_OR_time_focus: bool = False
        self.nodes_to_display: list[int] | None = None

        # Visualization settings
        self._show_edges: bool = True
        self._show_edge_activity: bool = False
        self._show_nodes: bool = True
        self.auto_nodes_ordering: bool = False
        self.monochrome: bool = False
        self.trim: bool = False
        self.max_shown_modules: int = -1
        self.hide_self_modules: bool = False
        self.nights: list[TimePoint] = []

        # Style settings
        self.edge_alpha: float = DEFAULT_EDGE_ALPHA
        self.node_alpha: float = DEFAULT_NODE_ALPHA
        self.longitudinal_nodes_margin: float = 0
        self.color_edges: bool = False
        self.edge_flatten_factor: float = DEFAULT_EDGE_FLATTEN_FACTOR
        self.height_module_color: float = DEFAULT_COMMUNITY_HEIGHT
        self.weight_scale: float = DEFAULT_WEIGHT_SCALE
        self.edge_linewidth: float = 1.0  # Edge line width in points
        self.edge_curve_intensity: float = 0.0  # Curve intensity (positive=right, negative=left)
        self.show_edge_orientation: bool = False
        self.show_continuous_duration: bool = False
        self.color_palette: str | Sequence[Any] = "tab10"

        # Label settings
        self.display_xticks_labels: bool = False
        self.display_yticks_labels: bool = False
        self.display_xlabel: bool = True
        self.display_ylabel: bool = True
        self.label_font_size: int = DEFAULT_LABEL_FONT_SIZE
        self.node_label_fontsize: int = DEFAULT_NODE_LABEL_FONT_SIZE
        self.node_labels_padding: int = DEFAULT_NODE_LABEL_PADDING
        self.bold_labels: list[str] = []
        self.highlight_node_focus: bool = True

        # Axis margin settings
        self.y_padding_bottom: float = 0.0  # Padding at bottom of y-axis (in node units)
        self.y_padding_top: float = 0.0  # Padding at top of y-axis (in node units)
        self.x_margin: float = 0.02  # X-axis margin
        self.y_margin: float = 0.05  # Y-axis margin

        # X-axis label customization
        self.xlabel_text: str = "Time"
        self.xlabel_fontsize: int | None = None  # None means use label_font_size
        self.xlabel_fontweight: str = "bold"
        self.xlabel_coords: tuple[float, float] = (0.0, 0.025)
        self.xlabel_rotation: float = 0.0
        self.xlabel_ha: str = "left"  # horizontal alignment

        # Y-axis label customization
        self.ylabel_text: str = "Nodes"
        self.ylabel_fontsize: int | None = None  # None means use label_font_size
        self.ylabel_fontweight: str = "bold"
        self.ylabel_coords: tuple[float, float] = (-0.05, 1.0)
        self.ylabel_rotation: float = 90.0
        self.ylabel_ha: str = "right"  # horizontal alignment

        # Node line style
        self.node_linewidth: float = 0.25  # Line width for regular nodes
        self.node_linewidth_focus: float = 0.5  # Line width for focused nodes

        # Edge activity marker style
        self.edge_activity_width: float = 0.8  # Width of edge activity rectangles
        self.edge_activity_height: float = 0.4  # Height of edge activity rectangles
        self.edge_activity_alpha: float = DEFAULT_EDGE_ALPHA  # Alpha for markers

        # Internal state
        self._fig: Figure | None = None
        self._ax: Axes | None = None
        self._drawn: bool = False

        # Computed data (populated during _prepare_data)
        self._nodes_mapping: NodesMapping | None = None
        self._remapped_modules: Modules | None = None
        self._remapped_node_focus: list[Any] = []
        self._time_links: TimeLinks | None = None
        self._focused_modules: Modules | None = None  # Primary: specified modules
        self._secondary_modules: Modules | None = None  # Secondary: big ones, less visible
        self._background_modules: Modules | None = None  # Other: gainsboro
        self._start_nodes: dict[int, int] | None = None
        self._end_nodes: dict[int, int] | None = None
        self._time_node_module_mapping: TimeNodeModuleMapping | None = None
        self._color_mapping: ColorMapping | None = None
        self._modules_nodes_segments: ModuleNodesSegments | None = None

        # Apply modules if provided via the polymorphic first argument
        if pending_modules is not None:
            self.set_modules(pending_modules)

    def set_nodes(
        self,
        nodes: set[NodeId] | list[NodeId] | None = None,
        labels: list[str] | None = None,
        focus: list[NodeId] | None = None,
        longitudinal_nodes_margin: float = 0,
    ) -> "LongitudinalModulesPlot":
        """
        Configure nodes to display.

        If a list is provided, the order is preserved (useful when reusing node ordering
        from another plot via `get_node_order()`). If a set is provided, order is arbitrary.

        Args:
            nodes: Set or list of nodes to display. If None, uses all nodes from linkstream.
                   If a list, the order is preserved for display.
            labels: Optional list of labels for nodes.
            focus: Optional list of nodes to focus/highlight.

        Returns:
            Self for method chaining.

        Example:
            >>> # Reuse node order from another plot
            >>> node_order = plot1.get_node_order()
            >>> plot2.set_nodes(node_order)  # Preserves order
            >>> plot2.auto_node_ordering(False)  # Don't recompute
        """
        self.longitudinal_nodes_margin = longitudinal_nodes_margin
        if nodes is None:
            self.nodes = self.linkstream.nodes
            self._nodes_ordered_list = None
        elif isinstance(nodes, list):
            # Preserve order when a list is provided
            self._nodes_ordered_list = list(nodes)
            self.nodes = set(nodes)
        else:
            self.nodes = set(nodes)
            self._nodes_ordered_list = None
        self.node_labels = labels or []
        if focus:
            self.set_focus_nodes(focus)
        return self

    def set_focus_nodes(
        self, nodes: list[NodeId], bold_labels: list[str] | None = None
    ) -> "LongitudinalModulesPlot":
        """
        Set nodes to focus on. Only modules containing these nodes will be colored.

        Note: Cannot be used together with set_focus_modules().

        Args:
            nodes: List of node indices to focus on.
            bold_labels: Optional list of labels to display in bold. If None, preserves
                existing bold_labels (useful when called from configure_modules).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If module_focus is already set.
        """
        if self.module_focus:
            raise ValueError(
                "Cannot use both node_focus and module_focus. "
                "Use set_focus_modules() OR set_focus_nodes(), not both."
            )
        self.node_focus = nodes
        # Only update bold_labels if explicitly provided (don't reset if None)
        if bold_labels is not None:
            self.bold_labels = bold_labels
        return self

    def set_focus_modules(
        self, module_labels: list[Any] | dict[Any, Any]
    ) -> "LongitudinalModulesPlot":
        """
        Set modules to focus on by their labels. Only these modules will be colored.

        Note: Cannot be used together with set_focus_nodes().

        Args:
            module_labels: Either a list of module labels to focus on,
                or a dict mapping module labels to colors.
                Labels should match the keys in the modules dictionary.
                Colors can be specified as:
                - Named colors: 'red', 'blue', 'green', etc.
                - Hex strings: '#FF5733', '#33FF57', etc.
                - RGB tuples: (0.9, 0.1, 0.1), (0.1, 0.9, 0.1), etc.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If node_focus is already set.

        Example:
            >>> # Using a list (colors will be auto-assigned)
            >>> plot.set_focus_modules(['module_1', 'module_5'])
            >>>
            >>> # Using a dict with explicit colors
            >>> plot.set_focus_modules({
            ...     'module_1': 'red',
            ...     'module_5': '#00FF00',
            ...     'ukraine_cluster': (0.1, 0.1, 0.9)
            ... })
        """
        if self.node_focus:
            raise ValueError(
                "Cannot use both node_focus and module_focus. "
                "Use set_focus_modules() OR set_focus_nodes(), not both."
            )
        if isinstance(module_labels, dict):
            self.module_focus = list(module_labels.keys())
            self._module_focus_colors = dict(module_labels)
        else:
            self.module_focus = list(module_labels)
            self._module_focus_colors = {}
        return self

    def get_color_mapping(
        self, print_mapping: bool = False, focused_only: bool = True
    ) -> dict[Any, Any]:
        """
        Get the color mapping used for modules.

        Returns the mapping of module labels to colors. If the plot has not
        been drawn yet, it will prepare the data first.

        Args:
            print_mapping: If True, prints the color mapping to stdout.
            focused_only: If True (default), only returns modules with actual colors
                (excludes monochrome/grey modules). Set to False to include all modules.

        Returns:
            Dictionary mapping module labels to their assigned colors.
            By default, only includes focused modules (not monochrome ones).

        Example:
            >>> plot.draw()
            >>> colors = plot.get_color_mapping()
            >>> print(colors)
            {'module_1': (0.12, 0.47, 0.71), 'module_2': (1.0, 0.5, 0.05), ...}
            >>>
            >>> # Include all modules (including grey/monochrome)
            >>> colors = plot.get_color_mapping(focused_only=False)
            >>>
            >>> # Print while getting
            >>> plot.get_color_mapping(print_mapping=True)
        """
        if self._color_mapping is None:
            if not self._drawn:
                self._validate_configuration()
                self._prepare_data()

        if self._color_mapping is None:
            return {}

        # Filter to only focused modules if requested
        if focused_only:
            result = {}
            for label, color in self._color_mapping.items():
                # Compare safely with strings (colors can be strings, tuples, or arrays)
                is_background = False
                if isinstance(color, str):
                    is_background = color == "gainsboro" or color == self.background_color
                elif self.background_color == "gainsboro":
                    is_background = False  # If color is not a string, it's not gainsboro
                else:
                    # Both are non-string, compare as-is
                    try:
                        is_background = color == self.background_color
                    except (ValueError, TypeError):
                        is_background = False

                if not is_background:
                    result[label] = color
        else:
            result = dict(self._color_mapping)

        if print_mapping:
            print("Module Color Mapping:")
            print("-" * 40)
            for label, color in result.items():
                if color == "gainsboro":
                    print(f"  {label}: {color} (monochrome)")
                else:
                    print(f"  {label}: {color}")

        return result

    def set_modules(
        self,
        modules: Any,
        max_shown: int = -1,
        hide_self: bool = False,
        margin_commu_segment: float = 0.25,
    ) -> "LongitudinalModulesPlot":
        """
        Configure modules to display.

        Args:
            modules: Modules data. Can be either:
                - Dictionary mapping module labels to list of (node, time) tuples.
                - TimeModules object (from lago library) - will be auto-converted.
            max_shown: Maximum number of modules to show (-1 for all).
            hide_self: Whether to hide single-node modules.

        Returns:
            Self for method chaining.

        Example:
            >>> # Using a dictionary
            >>> plot.set_modules({'com_1': [(0, 0), (0, 1), (1, 0)]})
            >>>
            >>> # Using a TimeModules object from lago
            >>> from lago import TimeModules
            >>> tm = TimeModules(path="modules.json")
            >>> plot.set_modules(tm)
        """
        # Store original TimeModules object for efficient access to node segments
        self._original_time_modules = modules if hasattr(modules, "get_time_modules_dict") else None
        self.modules = _convert_time_modules_to_modules(modules)
        self.max_shown_modules = max_shown
        self.hide_self_modules = hide_self
        self.margin_commu_segment = margin_commu_segment
        return self

    def set_time_focus(self, time: TimePoint) -> "LongitudinalModulesPlot":
        """
        Set time period to focus on.

        Args:
            time: Time point to focus on.

        Returns:
            Self for method chaining.
        """
        self.time_focus = time
        return self

    def set_night_highlights(self, nights: list[TimePoint]) -> "LongitudinalModulesPlot":
        """
        Set night periods to highlight.

        Args:
            nights: List of time points for night highlights.

        Returns:
            Self for method chaining.
        """
        self.nights = nights
        return self

    def toggle_edges(self, show: bool = True) -> "LongitudinalModulesPlot":
        """
        Control whether to display edges.

        Args:
            show: Whether to show edges.

        Returns:
            Self for method chaining.
        """
        self._show_edges = show
        return self

    def toggle_edge_activity(self, show: bool = True) -> "LongitudinalModulesPlot":
        """
        Control whether to display edge activity markers.

        Args:
            show: Whether to show edge activity markers.

        Returns:
            Self for method chaining.
        """
        self._show_edge_activity = show
        return self

    def set_edge_activity_style(
        self,
        width: float = 0.8,
        height: float = 0.4,
        alpha: float | None = None,
    ) -> "LongitudinalModulesPlot":
        """
        Configure edge activity marker visualization style.

        Edge activity markers are displayed as rectangles at node positions
        where edges occur.

        Args:
            width: Width of edge activity rectangles (>= 0.0). Default 0.8.
                At 1.0, rectangles fill the entire time unit width. Values > 1.0
                create wider rectangles that may overlap adjacent time units.
            height: Height of edge activity rectangles (0.0 to 1.0). Default 0.4.
                At 1.0, rectangles fill the entire node row height.
            alpha: Transparency of markers (0.0 to 1.0). Default uses edge_alpha.
                0.0 = fully transparent, 1.0 = fully opaque.

        Returns:
            Self for method chaining.

        Example:
            >>> # Larger, more visible markers
            >>> plot.set_edge_activity_style(width=0.9, height=0.6, alpha=0.5)
            >>>
            >>> # Smaller, subtle markers
            >>> plot.set_edge_activity_style(width=0.5, height=0.2, alpha=0.3)
            >>>
            >>> # Square markers
            >>> plot.set_edge_activity_style(width=0.5, height=0.5)
        """
        if width < 0.0:
            raise ValueError(f"width must be >= 0.0, got {width}")
        if not 0.0 <= height <= 1.0:
            raise ValueError(f"height must be between 0.0 and 1.0, got {height}")
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")

        self.edge_activity_width = width
        self.edge_activity_height = height
        if alpha is not None:
            self.edge_activity_alpha = alpha
        return self

    def toggle_nodes(self, show: bool = True) -> "LongitudinalModulesPlot":
        """
        Control whether to display nodes.

        Args:
            show: Whether to show nodes.

        Returns:
            Self for method chaining.
        """
        self._show_nodes = show
        return self

    def auto_node_ordering(
        self, enable: bool = True, subsets: list = []
    ) -> "LongitudinalModulesPlot":
        """
        Enable/disable automatic node ordering.

        When enabled, nodes are automatically reordered to minimize edge crossings.

        Args:
            enable: Whether to enable automatic node ordering.

        Returns:
            Self for method chaining.
        """
        self.auto_nodes_ordering = enable
        self.auto_nodes_ordering_subsets = subsets
        return self

    def set_monochrome(self, enable: bool = True) -> "LongitudinalModulesPlot":
        """
        Enable/disable monochrome color scheme.

        Args:
            enable: Whether to use monochrome colors.

        Returns:
            Self for method chaining.
        """
        self.monochrome = enable
        return self

    def trim_node_timelines(self, enable: bool = True) -> "LongitudinalModulesPlot":
        """
        Enable/disable trimming of node timelines.

        When enabled, node lines are trimmed to their active time range.

        Args:
            enable: Whether to trim node timelines.

        Returns:
            Self for method chaining.
        """
        self.trim = enable
        return self

    def set_color_palette(self, palette: str | Sequence[Any]) -> "LongitudinalModulesPlot":
        """
        Set the color palette for module visualization.

        Args:
            palette: Color palette specification. Can be:
                - str: Name of a matplotlib colormap (e.g., 'viridis', 'tab10', 'Set2',
                       'Pastel1', 'Paired', 'tab20', 'Dark2', 'Accent')
                - Sequence of colors: List/tuple of color specifications
                  (hex strings like '#FF5733', RGB tuples like (0.5, 0.5, 0.5),
                  or named colors like 'red', 'blue')

        Returns:
            Self for method chaining.

        Example:
            >>> # Using a matplotlib colormap name
            >>> plot.set_color_palette('tab10')
            >>> plot.set_color_palette('viridis')
            >>> plot.set_color_palette('Set2')
            >>>
            >>> # Using custom colors
            >>> plot.set_color_palette(['red', 'blue', 'green'])
            >>> plot.set_color_palette(['#FF5733', '#33FF57', '#3357FF'])
            >>> plot.set_color_palette([(0.9, 0.1, 0.1), (0.1, 0.9, 0.1)])
        """
        self.color_palette = palette
        return self

    def set_edge_style(
        self,
        alpha: float = DEFAULT_EDGE_ALPHA,
        flatten_factor: float = DEFAULT_EDGE_FLATTEN_FACTOR,
        color: bool = False,
        orientation: bool = False,
        linewidth: float | None = None,
        curve_intensity: float | None = None,
    ) -> "LongitudinalModulesPlot":
        """
        Configure edge visualization style.

        Args:
            alpha: Edge transparency (0.0 to 1.0).
            flatten_factor: Flattening factor for edge arcs (higher = flatter).
            color: Whether to color edges by module.
            orientation: Whether to show edge orientation arrows for directed graphs.
            linewidth: Edge line width in points. Default 1.0.
            curve_intensity: Controls how much edges curve.
                0.0 = straight lines, positive values curve right, negative values curve left.
                Typical values are between -0.5 and 0.5. Default is 0.0.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If alpha is not in range [0.0, 1.0].

        Example:
            >>> # Thicker edges
            >>> plot.set_edge_style(linewidth=2.0)
            >>>
            >>> # Curved edges
            >>> plot.set_edge_style(curve_intensity=0.3)
            >>>
            >>> # Curved to the left
            >>> plot.set_edge_style(curve_intensity=-0.2)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")
        self.edge_alpha = alpha
        self.edge_flatten_factor = flatten_factor
        self.color_edges = color
        self.show_edge_orientation = orientation
        if linewidth is not None:
            self.edge_linewidth = linewidth
        if curve_intensity is not None:
            self.edge_curve_intensity = curve_intensity
        return self

    def set_module_style(
        self,
        height: float | None = None,
        max_shown: int | None = None,
        hide_self: bool | None = None,
    ) -> "LongitudinalModulesPlot":
        """
        Configure module visualization style.

        Args:
            height: Height of module rectangles (0.0 to 1.0).
            max_shown: Maximum number of modules to show (-1 for all).
                When using focus with show_unfocused=True, this limits the
                total number of displayed modules (focused + unfocused).
            hide_self: Whether to hide single-node modules.

        Returns:
            Self for method chaining.
        """
        if height is not None:
            self.height_module_color = height
        if max_shown is not None:
            self.max_shown_modules = max_shown
        if hide_self is not None:
            self.hide_self_modules = hide_self
        return self

    def set_background_module_color(self, color: Any) -> "LongitudinalModulesPlot":
        """
        Set the color for background modules.

        Background modules are those that are neither focused nor
        secondary. By default, they are displayed in gainsboro (light grey).

        Note: This sets the color for background *modules*, not the plot canvas
        background. For plot background, use matplotlib's `ax.set_facecolor()`.

        Args:
            color: The color to use for background modules. Can be:
                - Named color: 'gainsboro', 'lightgrey', 'white', 'lavender', etc.
                - Hex string: '#DCDCDC', '#F0F0F0', etc.
                - RGB tuple: (0.86, 0.86, 0.86), (0.9, 0.9, 0.95), etc.

        Returns:
            Self for method chaining.

        Example:
            >>> # Use a lighter grey for background modules
            >>> plot.set_background_module_color('whitesmoke')
            >>>
            >>> # Use a custom grey with hex
            >>> plot.set_background_module_color('#E8E8E8')
            >>>
            >>> # Use an RGB tuple
            >>> plot.set_background_module_color((0.9, 0.9, 0.9))
            >>>
            >>> # Use a subtle color tint
            >>> plot.set_background_module_color('lavender')
        """
        self.background_color = color
        return self

    def set_node_style(
        self,
        alpha: float = DEFAULT_NODE_ALPHA,
        fontsize: int = DEFAULT_NODE_LABEL_FONT_SIZE,
        padding: int = DEFAULT_NODE_LABEL_PADDING,
        linewidth: float | None = None,
        linewidth_focus: float | None = None,
    ) -> "LongitudinalModulesPlot":
        """
        Configure node visualization style.

        Args:
            alpha: Node transparency (0.0 to 1.0).
            fontsize: Node label font size.
            padding: Node label padding.
            linewidth: Line width for regular (non-focused) node lines. Default 0.25.
            linewidth_focus: Line width for focused node lines. Default 0.5.

        Returns:
            Self for method chaining.

        Example:
            >>> # Thicker node lines
            >>> plot.set_node_style(linewidth=0.5, linewidth_focus=1.0)
            >>>
            >>> # Thinner node lines
            >>> plot.set_node_style(linewidth=0.1, linewidth_focus=0.3)
        """
        self.node_alpha = alpha
        self.node_label_fontsize = fontsize
        self.node_labels_padding = padding
        if linewidth is not None:
            self.node_linewidth = linewidth
        if linewidth_focus is not None:
            self.node_linewidth_focus = linewidth_focus
        return self

    def set_labels(
        self,
        display_xticks: bool = False,
        display_yticks: bool = False,
        display_xlabel: bool = True,
        display_ylabel: bool = True,
        font_size: int = DEFAULT_LABEL_FONT_SIZE,
        bold_labels: list[str] | None = None,
    ) -> "LongitudinalModulesPlot":
        """
        Configure label display.

        Args:
            display_xticks: Whether to display x-axis tick labels.
            display_yticks: Whether to display y-axis tick labels.
            display_xlabel: Whether to display x-axis label.
            display_ylabel: Whether to display y-axis label.
            font_size: Axis label font size.
            bold_labels: List of labels to display in bold.

        Returns:
            Self for method chaining.
        """
        self.display_xticks_labels = display_xticks
        self.display_yticks_labels = display_yticks
        self.display_xlabel = display_xlabel
        self.display_ylabel = display_ylabel
        self.label_font_size = font_size
        if bold_labels:
            self.bold_labels = bold_labels
        return self

    def set_xlabel_style(
        self,
        text: str = "Time",
        fontsize: int | None = None,
        fontweight: str = "bold",
        coords: tuple[float, float] = (0.0, 0.5),
        rotation: float = 0.0,
        ha: str = "center",
    ) -> "LongitudinalModulesPlot":
        """
        Customize the X-axis label appearance and position.

        Args:
            text: Label text. Default "Time".
            fontsize: Font size. If None, uses label_font_size.
            fontweight: Font weight ('normal', 'bold', 'light', 'semibold', etc.).
            coords: Label position as (x, y) in axes fraction coordinates.
                Default (0.0, 0.025) places it at the left, slightly below the axis.
            rotation: Label rotation angle in degrees. Default 0.0.
            ha: Horizontal alignment ('left', 'center', 'right'). Default 'left'.

        Returns:
            Self for method chaining.

        Example:
            >>> # Center the xlabel with larger font
            >>> plot.set_xlabel_style(text="Time (hours)", fontsize=14, coords=(0.5, -0.05), ha="center")
            >>>
            >>> # Position at bottom left
            >>> plot.set_xlabel_style(coords=(0.0, 0.025), ha="left")
        """
        self.xlabel_text = text
        self.xlabel_fontsize = fontsize
        self.xlabel_fontweight = fontweight
        self.xlabel_coords = coords
        self.xlabel_rotation = rotation
        self.xlabel_ha = ha
        return self

    def set_ylabel_style(
        self,
        text: str = "Nodes",
        fontsize: int | None = None,
        fontweight: str = "bold",
        coords: tuple[float, float] = (-0.01, 0.5),
        rotation: float = 90.0,
        ha: str = "center",
    ) -> "LongitudinalModulesPlot":
        """
        Customize the Y-axis label appearance and position.

        Args:
            text: Label text. Default "Nodes".
            fontsize: Font size. If None, uses label_font_size.
            fontweight: Font weight ('normal', 'bold', 'light', 'semibold', etc.).
            coords: Label position as (x, y) in axes fraction coordinates.
                Default (-0.05, 1.0) places it at the top left of the plot.
            rotation: Label rotation angle in degrees. Default 90.0.
            ha: Horizontal alignment ('left', 'center', 'right'). Default 'right'.

        Returns:
            Self for method chaining.

        Example:
            >>> # Center the ylabel vertically with normal weight
            >>> plot.set_ylabel_style(text="Accounts", fontweight="normal", coords=(-0.08, 0.5), ha="center")
            >>>
            >>> # Position at top with no rotation
            >>> plot.set_ylabel_style(coords=(-0.05, 1.0), rotation=0, ha="right")
        """
        self.ylabel_text = text
        self.ylabel_fontsize = fontsize
        self.ylabel_fontweight = fontweight
        self.ylabel_coords = coords
        self.ylabel_rotation = rotation
        self.ylabel_ha = ha
        return self

    def set_margins(
        self,
        x: float = 0.02,
        y: float = 0.05,
    ) -> "LongitudinalModulesPlot":
        """
        Set the axis margins.

        Margins add padding around the data. A value of 0.05 means 5% of the
        data range is added as padding on each side.

        Args:
            x: X-axis margin (fraction of data range). Default 0.02.
            y: Y-axis margin (fraction of data range). Default 0.05.

        Returns:
            Self for method chaining.

        Example:
            >>> # Tight margins
            >>> plot.set_margins(x=0.01, y=0.02)
            >>>
            >>> # More padding
            >>> plot.set_margins(x=0.05, y=0.1)
        """
        self.x_margin = x
        self.y_margin = y
        return self

    def set_axis_padding(
        self,
        bottom: float = 0.0,
        top: float | None = None,
    ) -> "LongitudinalModulesPlot":
        """
        Set padding (blank space) at top and bottom of the y-axis.

        Args:
            bottom: Padding at the bottom of y-axis in node units. Default 0.0.
                Use positive values to add blank space below the first node.
            top: Padding at the top of y-axis in node units. Default None.
                If None, uses the same value as bottom (symmetric padding).
                Use positive values to add blank space above the last node.

        Returns:
            Self for method chaining.

        Example:
            >>> # Symmetric padding at top/bottom
            >>> plot.set_axis_padding(0.5)  # 0.5 on both sides
            >>>
            >>> # Different padding for top and bottom
            >>> plot.set_axis_padding(bottom=0.5, top=1.0)
            >>>
            >>> # More padding at top only
            >>> plot.set_axis_padding(bottom=0.0, top=1.5)
            >>>
            >>> # No padding (tight)
            >>> plot.set_axis_padding(0.0)
        """
        self.y_padding_bottom = bottom
        self.y_padding_top = top if top is not None else bottom
        return self

    # =========================================================================
    # SIMPLIFIED API - Consolidated Configuration Methods
    # =========================================================================

    def configure_nodes(
        self,
        nodes: set[NodeId] | list[NodeId] | None = None,
        labels: list[str] | dict[NodeId, str] | None = None,
        focus: list[NodeId] | None = None,
        show_labels: bool = True,
        auto_ordering: bool = False,
        ordering_subsets: list[list[NodeId]] | None = None,
        longitudinal_margin: float = 0,
        alpha: float = DEFAULT_NODE_ALPHA,
        fontsize: int = DEFAULT_NODE_LABEL_FONT_SIZE,
        label_padding: int = DEFAULT_NODE_LABEL_PADDING,
        linewidth: float = 0.25,
        linewidth_focus: float = 0.5,
        highlight_focus: bool = True,
        bold_labels: list[str] | None = None,
    ) -> "LongitudinalModulesPlot":
        """
        Configure all node-related settings in one call.

        This is a simplified API that consolidates:
        - set_nodes()
        - auto_node_ordering()
        - set_node_style()

        Note: Node focus for module coloring should be set via configure_modules(focus_nodes=...).

        Args:
            nodes: Set or list of nodes to display. If None, uses all nodes from linkstream.
                If a list, the order is preserved for display.
            labels: Optional list of labels for nodes.
            show_labels: Whether to display node labels on the y-axis. Default True.
            auto_ordering: Whether to enable automatic node ordering to minimize edge crossings.
            ordering_subsets: Optional list of node subsets for constrained ordering.
            longitudinal_margin: Margin for longitudinal nodes display.
            alpha: Node transparency (0.0 to 1.0).
            fontsize: Node label font size.
            label_padding: Node label padding.
            linewidth: Line width for regular (non-focused) node lines.
            linewidth_focus: Line width for focused node lines.
            highlight_focus: Whether to highlight focused nodes.
            bold_labels: List of labels to display in bold.

        Returns:
            Self for method chaining.

        Example:
            >>> plot.configure_nodes(
            ...     nodes=node_list,
            ...     labels=label_list,
            ...     show_labels=True,
            ...     auto_ordering=True,
            ...     fontsize=8,
            ...     label_padding=-15,
            ...     linewidth=0.3,
            ... )
        """
        # Normalize labels: accept dict {node_id: label} by converting to list
        # ordered by the resolved node order
        labels_list: list[str] | None
        if isinstance(labels, dict):
            resolved_nodes: list[NodeId]
            if isinstance(nodes, list):
                resolved_nodes = list(nodes)
            elif nodes is not None:
                resolved_nodes = sorted(nodes)
            else:
                resolved_nodes = sorted(self.linkstream.nodes)
            labels_list = [str(labels.get(n, n)) for n in resolved_nodes]
        else:
            labels_list = labels

        # Set nodes
        self.set_nodes(
            nodes=nodes,
            labels=labels_list,
            longitudinal_nodes_margin=longitudinal_margin,
        )

        # Configure ordering
        self.auto_node_ordering(enable=auto_ordering, subsets=ordering_subsets or [])

        # Set node style
        self.set_node_style(
            alpha=alpha,
            fontsize=fontsize,
            padding=label_padding,
            linewidth=linewidth,
            linewidth_focus=linewidth_focus,
        )

        # Focus-related settings (visual highlighting when focus is set via configure_modules)
        self.highlight_node_focus = highlight_focus
        if bold_labels:
            self.bold_labels = bold_labels

        # Apply node focus if requested (alternative to set_focus_nodes)
        if focus:
            self.set_focus_nodes(focus)

        # Display node labels on y-axis
        self.display_yticks_labels = show_labels

        return self

    def configure_edges(
        self,
        show_edges: bool = False,
        show_activity: bool = True,
        # Edge style parameters
        edge_alpha: float = DEFAULT_EDGE_ALPHA,
        edge_flatten_factor: float = DEFAULT_EDGE_FLATTEN_FACTOR,
        edge_color: bool = False,
        edge_orientation: bool = False,
        edge_linewidth: float = 1.0,
        edge_curve_intensity: float = 0.0,
        # Activity marker parameters
        activity_width: float = 0.8,
        activity_height: float = 0.4,
        activity_alpha: float | None = None,
        continuous_duration: bool = False,
    ) -> "LongitudinalModulesPlot":
        """
        Configure all edge-related settings in one call.

        This is a simplified API that consolidates:
        - toggle_edges()
        - toggle_edge_activity()
        - set_edge_style()
        - set_edge_activity_style()

        Args:
            show_edges: Whether to show edges (arcs between nodes).
            show_activity: Whether to show edge activity markers (rectangles at node positions).

            Edge style parameters:
            edge_alpha: Edge transparency (0.0 to 1.0).
            edge_flatten_factor: Flattening factor for edge arcs.
            edge_color: Whether to color edges by module.
            edge_orientation: Whether to show edge orientation arrows for directed graphs.
            edge_weight_scale: Scale factor for edge weights.
            edge_width_scale: Scale for edge width (0.0 to 1.0).
            edge_curve_intensity: For delayed linkstreams, controls edge curvature.

            Activity marker parameters:
            activity_width: Width of edge activity rectangles (0.0 to 1.0).
            activity_height: Height of edge activity rectangles (0.0 to 1.0).
            activity_alpha: Alpha for activity markers. If None, uses edge_alpha.
            continuous_duration: Whether to show horizontal duration lines for
                continuous linkstreams. Only has effect when the linkstream is continuous.

        Returns:
            Self for method chaining.

        Example:
            >>> # Show only edge activity markers (common pattern)
            >>> plot.configure_edges(
            ...     show_edges=False,
            ...     show_activity=True,
            ...     activity_width=0.9,
            ...     activity_height=0.5,
            ...     activity_alpha=0.3,
            ... )
            >>>
            >>> # Show both edges and activity
            >>> plot.configure_edges(
            ...     show_edges=True,
            ...     show_activity=True,
            ...     edge_alpha=0.15,
            ...     edge_flatten_factor=5,
            ... )
        """
        # Toggle display
        self.toggle_edges(show_edges)
        self.toggle_edge_activity(show_activity)

        # Set edge style
        self.set_edge_style(
            alpha=edge_alpha,
            flatten_factor=edge_flatten_factor,
            color=edge_color,
            orientation=edge_orientation,
            linewidth=edge_linewidth,
            curve_intensity=edge_curve_intensity,
        )

        # Set activity style
        self.set_edge_activity_style(
            width=activity_width,
            height=activity_height,
            alpha=activity_alpha if activity_alpha is not None else edge_alpha,
        )

        # Continuous duration lines
        self.show_continuous_duration = continuous_duration

        return self

    def configure_modules(
        self,
        modules: Modules | None = None,
        max_shown: int = -1,
        hide_self: bool = False,
        margin_segment: float = 0.25,
        height: float = DEFAULT_COMMUNITY_HEIGHT,
        # Color configuration
        color_palette: str | Sequence[Any] | None = None,
        background_color: Any = DEFAULT_BACKGROUND_COMMUNITY_COLOR,
        monochrome: bool = False,
        # Focus configuration
        focus_modules: list[Any] | dict[Any, Any] | None = None,
        focus_nodes: list[NodeId] | None = None,
    ) -> "LongitudinalModulesPlot":
        """
        Configure all module-related settings in one call.

        When focus_modules is a dict with explicit colors AND max_shown is set,
        the remaining slots are automatically filled with the biggest unfocused
        modules using remaining palette colors.

        Note: You can specify either focus_modules OR focus_nodes, but not both.

        Args:
            modules: Dictionary mapping module labels to list of (node, time) tuples.
            max_shown: Maximum number of modules to show (-1 for all).
                When focus_modules is a dict, this fills remaining slots with
                the biggest unfocused modules.
            hide_self: Whether to hide single-node modules.
            margin_segment: Margin between module segments.
            height: Height of module rectangles (0.0 to 1.0).

            Color configuration:
            color_palette: Color palette (matplotlib colormap name or list of colors).
            background_color: Color for background modules.
            monochrome: Whether to use monochrome color scheme.

            Focus configuration:
            focus_modules: List of module labels to focus, or dict mapping labels to colors.
                When a dict, remaining modules up to max_shown get palette colors.
            focus_nodes: List of node indices to focus on (alternative to focus_modules).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If both focus_modules and focus_nodes are specified.

        Example:
            >>> # Basic usage with auto-assigned colors
            >>> plot.configure_modules(
            ...     modules=dynamic_modules,
            ...     max_shown=10,
            ...     color_palette="tab20",
            ... )
            >>>
            >>> # Explicit colors for focused + auto-fill remaining slots
            >>> plot.configure_modules(
            ...     modules=dynamic_modules,
            ...     focus_modules={0: 'blue', 1: 'red', 2: 'green'},  # 3 focused
            ...     color_palette='tab10',
            ...     max_shown=10,  # 7 more get remaining tab10 colors
            ... )
        """
        if focus_modules is not None and focus_nodes is not None:
            raise ValueError(
                "Cannot specify both focus_modules and focus_nodes. Use one or the other."
            )

        # Set modules (use already-configured modules if not explicitly provided)
        if modules is not None:
            self.set_modules(
                modules=modules,
                max_shown=max_shown,
                hide_self=hide_self,
                margin_commu_segment=margin_segment,
            )
        elif self.modules is None:
            raise ValueError(
                "No modules configured. Pass modules as first argument to "
                "configure_modules(), or set them via the constructor: "
                "LongitudinalModulesPlot(modules, linkstream=ls)."
            )
        else:
            # Update ancillary module settings when modules are already set
            self.max_shown_modules = max_shown
            self.hide_self_modules = hide_self
            self.margin_commu_segment = margin_segment

        # Set module style
        self.set_module_style(height=height)

        # Set color configuration
        if color_palette is not None:
            self.set_color_palette(color_palette)
        self.set_background_module_color(background_color)
        self.set_monochrome(monochrome)

        # Set focus (reset any previous focus first)
        self.node_focus = []
        self.module_focus = []
        self._module_focus_colors = {}

        if focus_modules is not None:
            self.set_focus_modules(focus_modules)
        elif focus_nodes is not None:
            self.set_focus_nodes(focus_nodes)

        return self

    def configure_display(
        self,
        # Axis padding
        padding_bottom: float = 0.0,
        padding_top: float | None = None,
        # Axis labels (not node labels - use configure_nodes(show_labels=...) for that)
        show_xlabel: bool = False,
        show_ylabel: bool = False,
        show_xticks: bool = False,
        label_fontsize: int = DEFAULT_LABEL_FONT_SIZE,
    ) -> "LongitudinalModulesPlot":
        """
        Configure global display settings in one call.

        This is a simplified API that consolidates:
        - set_axis_padding()
        - set_labels() (for axis labels only)

        Note: Node labels on y-axis are controlled by configure_nodes(show_labels=...).

        Args:
            Axis padding:
            padding_bottom: Padding at the bottom of y-axis in node units.
            padding_top: Padding at the top of y-axis. If None, uses padding_bottom.

            Axis labels:
            show_xlabel: Whether to display x-axis label ("Time").
            show_ylabel: Whether to display y-axis label ("Nodes").
            show_xticks: Whether to display x-axis tick labels (time values).
            label_fontsize: Axis label font size.

        Returns:
            Self for method chaining.

        Example:
            >>> plot.configure_display(
            ...     padding_bottom=0.5,
            ...     padding_top=1.0,
            ...     show_xlabel=True,  # Show "Time" label
            ...     show_ylabel=False,
            ... )
        """
        # Set axis padding
        self.set_axis_padding(bottom=padding_bottom, top=padding_top)

        # Set axis labels (not node labels - those are controlled by configure_nodes)
        self.display_xlabel = show_xlabel
        self.display_ylabel = show_ylabel
        self.display_xticks_labels = show_xticks
        self.label_font_size = label_fontsize

        return self

    @overload
    def draw(
        self,
        title: str = ...,
        *,
        return_ax: Literal[True],
        show_plot: bool = ...,
    ) -> tuple[Figure, Axes]: ...

    @overload
    def draw(
        self,
        title: str = ...,
        *,
        return_ax: Literal[False] = ...,
        show_plot: bool = ...,
    ) -> Figure: ...

    def draw(
        self,
        title: str = "",
        return_ax: bool = False,
        show_plot: bool = False,
    ) -> Figure | tuple[Figure, Axes]:
        """
        Draw the longitudinal module plot.

        Args:
            title: Plot title.
            return_ax: Whether to return the axes object along with figure.
            show_plot: Whether to display the plot immediately.

        Returns:
            Figure object, or tuple of (Figure, Axes) if return_ax is True.

        Raises:
            ValueError: If modules are not configured before drawing.
        """
        # Validate and prepare data
        self._validate_configuration()
        self._prepare_data()

        # Set up figure and axes
        self._fig, self._ax = setup_figure_and_axes(self.width, self.height)
        self._drawn = True

        # Apply all drawing operations
        self._draw_module_periods()
        self._draw_nodes()
        self._draw_focus_highlights()
        self._draw_edges()
        self._draw_edge_activity()
        self._draw_night_highlights()

        # Configure axes and finalize
        self._configure_axes()
        self._finalize_plot(title, show_plot)

        if return_ax:
            return self._fig, self._ax
        return self._fig

    def save(
        self, filename: str, dpi: int = 300, bbox_inches: str = "tight"
    ) -> "LongitudinalModulesPlot":
        """
        Save the plot to a file.

        Args:
            filename: Output filename (path).
            dpi: Dots per inch resolution.
            bbox_inches: Bounding box setting ('tight' for tight layout).

        Returns:
            Self for method chaining.
        """
        if not self._drawn:
            self.draw()
        if self._fig is not None:
            self._fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        return self

    def show(self) -> "LongitudinalModulesPlot":
        """
        Display the plot.

        Returns:
            Self for method chaining.
        """
        if not self._drawn:
            self.draw(show_plot=True)
        else:
            plt.show()
        return self

    def get_figure(self) -> Figure:
        """
        Get the matplotlib figure object.

        Returns:
            The matplotlib Figure object.

        Raises:
            RuntimeError: If the plot has not been drawn yet.
        """
        if not self._drawn:
            self.draw()
        if self._fig is None:
            raise RuntimeError("Figure not created")
        return self._fig

    def get_axes(self) -> Axes:
        """
        Get the matplotlib axes object.

        Returns:
            The matplotlib Axes object.

        Raises:
            RuntimeError: If the plot has not been drawn yet.
        """
        if not self._drawn:
            self.draw()
        if self._ax is None:
            raise RuntimeError("Axes not created")
        return self._ax

    def get_node_order(self) -> list[NodeId]:
        """
        Get the original node IDs in their display order.

        This returns the raw node IDs (as they appear in the original data)
        sorted by their display position. This is useful when you want to
        reuse the same node ordering computed by auto_node_ordering() in
        another LongitudinalPlot instance.

        Returns:
            List of original node IDs in display order (from top to bottom).

        Raises:
            RuntimeError: If no node mapping has been computed yet.

        Example:
            >>> # Compute ordering on first plot
            >>> plot1 = LongitudinalPlot(linkstream1)
            >>> plot1.set_modules(modules1)
            >>> plot1.auto_node_ordering(True)
            >>> plot1.draw()
            >>>
            >>> # Get the computed node order
            >>> node_order = plot1.get_node_order()
            >>>
            >>> # Apply same ordering to second plot
            >>> plot2 = LongitudinalPlot(linkstream2)
            >>> plot2.set_nodes(node_order)  # Uses this order directly
            >>> plot2.auto_node_ordering(False)  # Don't recompute
            >>> plot2.set_modules(modules2)
            >>> plot2.draw()
        """
        if self._nodes_mapping is None:
            if not self._drawn:
                self._validate_configuration()
                self._prepare_data()

        if self._nodes_mapping is None:
            raise RuntimeError("Node mapping not computed")

        # Invert the mapping: {original_id: display_index} -> {display_index: original_id}
        inverted = {
            display_idx: original_id for original_id, display_idx in self._nodes_mapping.items()
        }

        # Return original IDs sorted by display index
        return [inverted[i] for i in sorted(inverted.keys())]

    def _validate_configuration(self) -> None:
        """
        Validate that required data is configured.

        Raises:
            ValueError: If required configuration is missing.
        """
        if self.modules is None:
            raise ValueError("Modules must be set before drawing")
        if self.nodes is None:
            self.nodes = self.linkstream.nodes

    def _prepare_data(self) -> None:
        """Prepare data for visualization."""
        # Create nodes mapping like the original function does
        # If an ordered list was provided via set_nodes(), use that order
        # Otherwise use the set (arbitrary order)
        if self._nodes_ordered_list is not None:
            nodes_to_display_ordered = self._nodes_ordered_list
        else:
            nodes_to_display_ordered = list(self.nodes if self.nodes else self.linkstream.nodes)
        nodes_to_display_set = set(nodes_to_display_ordered)

        nodes_mapping: NodesMapping = {
            node: ite for ite, node in enumerate(nodes_to_display_ordered)
        }

        # Apply auto node ordering if enabled
        if self.auto_nodes_ordering and self.modules is not None:
            from .core import _auto_node_ordering_given_time_modules

            time_links = self.linkstream.get_time_links(include_weights=True)
            if self.auto_nodes_ordering_subsets == []:
                self.auto_nodes_ordering_subsets = [nodes_to_display_set]
            else:
                self.auto_nodes_ordering_subsets = [
                    set(subset) & nodes_to_display_set
                    for subset in self.auto_nodes_ordering_subsets
                ]
            nodes_mapping = _auto_node_ordering_given_time_modules(
                self.modules,
                [] if not self._show_edges else time_links,
                1 if not self._show_edges else 0.9,
                self.auto_nodes_ordering_subsets,
            )

            # Note: We do NOT extend the mapping to include nodes not in modules
            # because those nodes don't have module data and would show edge activity
            # without any module color coverage. This keeps modules and edge activity aligned.

        # ALWAYS remap node labels if they exist (not just when auto_nodes_ordering)
        # This ensures labels match the display indices for all cases
        # Use _original_node_labels to avoid corruption when _prepare_data is called multiple times
        if self.node_labels or self._original_node_labels:
            # Save original labels on first call, use them for all subsequent remapping
            if self._original_node_labels is None:
                self._original_node_labels = list(self.node_labels)

            original_labels = self._original_node_labels
            self.node_labels = [""] * len(nodes_mapping)
            # If we have an ordered list, use it to pair labels with node IDs
            # Otherwise fall back to sequential indices
            if self._nodes_ordered_list is not None and len(original_labels) == len(
                self._nodes_ordered_list
            ):
                missing_nodes = []
                for node_id, label in zip(self._nodes_ordered_list, original_labels):
                    if node_id in nodes_mapping:
                        self.node_labels[nodes_mapping[node_id]] = label
                    else:
                        missing_nodes.append((node_id, label))
                if missing_nodes:
                    warnings.warn(
                        f"Labels for {len(missing_nodes)} nodes were not assigned because "
                        f"those nodes are not in the nodes_mapping (likely filtered out by auto_node_ordering). "
                        f"Missing: {missing_nodes[:5]}{'...' if len(missing_nodes) > 5 else ''}",
                        UserWarning,
                    )
            else:
                # Fallback: assume labels are indexed sequentially 0, 1, 2, ...
                for node, label in enumerate(original_labels):
                    if node in nodes_mapping:
                        self.node_labels[nodes_mapping[node]] = label

        # ALWAYS remap node focus (not just when auto_nodes_ordering)
        # Use a local variable to avoid mutating self.node_focus, which would corrupt
        # repeated calls to _prepare_data() (e.g. get_node_order() then draw()).
        if self.node_focus:
            remapped_node_focus = [
                nodes_mapping[nfo] for nfo in self.node_focus if nfo in nodes_mapping
            ]
        else:
            remapped_node_focus = []
        self._remapped_node_focus = remapped_node_focus

        # Store the final nodes_to_display list with mapped indices
        # Only include nodes that are in the mapping
        self.nodes_to_display = [
            nodes_mapping[node] for node in nodes_to_display_set if node in nodes_mapping
        ]
        self._nodes_mapping = nodes_mapping

        # Remap modules
        modules_dict = self.modules if self.modules is not None else {}
        self._remapped_modules = {
            key: [(nodes_mapping.get(node, -1), time) for node, time in elems]
            for key, elems in modules_dict.items()
        }

        # COMMUNITY COLOR SYSTEM:
        # 1. Focused: specified modules with full-color visibility
        # 2. Secondary: biggest remaining modules until max_shown
        # 3. Background: all remaining modules in gainsboro

        # Step 1: Determine primary/focused modules
        if self.module_focus:
            # Use module labels directly to determine focused modules
            # Handle both string and int keys for robust matching
            focus_labels_set = set(self.module_focus)
            focus_labels_extended = set()
            for label in focus_labels_set:
                focus_labels_extended.add(label)
                # Add both string and int versions for robust matching
                if isinstance(label, str):
                    try:
                        focus_labels_extended.add(int(label))
                    except ValueError:
                        pass
                elif isinstance(label, int):
                    focus_labels_extended.add(str(label))

            self._focused_modules = {
                label: members
                for label, members in self._remapped_modules.items()
                if label in focus_labels_extended
            }
            remaining_modules = {
                label: members
                for label, members in self._remapped_modules.items()
                if label not in focus_labels_extended
            }
        else:
            # Use node_focus and time_focus to determine focused modules
            self._focused_modules, remaining_modules = prepare_modules_for_display(
                self._remapped_modules, remapped_node_focus, self.time_focus, False
            )

        # Step 2: From remaining, split into secondary (with colors) and background (gainsboro)
        self._secondary_modules = {}
        self._background_modules = {}

        # Auto-enable secondary tier if:
        # - explicit colors provided (focus_modules dict) AND
        # - max_shown is set AND
        # - there are remaining slots to fill
        auto_enable_secondary = (
            self._module_focus_colors
            and self.max_shown_modules > -1
            and self.max_shown_modules > len(self._focused_modules)
        )

        if auto_enable_secondary and remaining_modules:
            # Sort remaining by size (descending)
            remaining_sorted = sorted(
                remaining_modules.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )

            # Determine how many secondary slots we have
            if self.max_shown_modules > -1:
                # max_shown includes both primary and secondary
                secondary_slots = max(0, self.max_shown_modules - len(self._focused_modules))
            else:
                # No limit - all remaining become secondary (colored but de-emphasized)
                secondary_slots = len(remaining_sorted)

            # Split into secondary (colored, de-emphasized) and background (gainsboro)
            self._secondary_modules = dict(remaining_sorted[:secondary_slots])
            self._background_modules = dict(remaining_sorted[secondary_slots:])
        else:
            # Not showing unfocused - all remaining go to background
            self._background_modules = remaining_modules

        # If max_shown also limits primary modules, apply that
        if self.max_shown_modules > -1:
            focused_modules_sorted = sorted(
                self._focused_modules.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
            if len(focused_modules_sorted) > self.max_shown_modules:
                # Move excess focused to background
                self._focused_modules = dict(focused_modules_sorted[: self.max_shown_modules])
                excess = dict(focused_modules_sorted[self.max_shown_modules :])
                self._background_modules.update(excess)
                self._secondary_modules = {}

        # Apply hide_self_modules filter (modules with single member)
        if self.hide_self_modules:
            self._focused_modules = {
                label: members
                for label, members in self._focused_modules.items()
                if len(members) > 1
            }
            self._secondary_modules = {
                label: members
                for label, members in self._secondary_modules.items()
                if len(members) > 1
            }
            self._background_modules = {
                label: members
                for label, members in self._background_modules.items()
                if len(members) > 1
            }

        # Get and remap time links - ALWAYS remap using nodes_mapping
        # to ensure consistency with remapped modules
        time_links = self.linkstream.get_time_links()

        # Check for nodes in linkstream that are not in the display set
        # and emit a warning if found
        unmapped_nodes: set[NodeId] = set()
        for node1, node2, *rest in time_links:
            if node1 not in nodes_mapping:
                unmapped_nodes.add(node1)
            if node2 not in nodes_mapping:
                unmapped_nodes.add(node2)

        if unmapped_nodes:
            warnings.warn(
                f"The linkstream contains nodes that are not in the set_nodes() list: {sorted(unmapped_nodes)}. "
                f"Edges involving these nodes will be drawn at incorrect positions (y=-1). "
                f"Fix this by either: (1) adding these nodes to set_nodes(), or "
                f"(2) filtering the linkstream to only include edges between known nodes.",
                UserWarning,
            )

        time_links = [
            [nodes_mapping.get(node1, -1), nodes_mapping.get(node2, -1), *rest]
            for node1, node2, *rest in time_links
        ]
        self._time_links = time_links

        # Calculate node time ranges using mapped node indices
        self._start_nodes, self._end_nodes = calculate_node_time_ranges(
            list(nodes_mapping.values()),
            time_links,
            self.linkstream.network_duration,
            self.trim,
        )

        # Create time-node to module mapping using remapped modules
        self._time_node_module_mapping = create_time_node_module_mapping(self._remapped_modules)

        # Generate color mapping:
        # - Focused: full-color from palette (or explicit colors if provided)
        # - Secondary: remaining palette colors
        # - Background: gainsboro
        # Pass explicit colors so that palette excludes similar colors
        self._color_mapping = generate_color_mapping(
            self._focused_modules,
            self._secondary_modules,
            self.monochrome,
            self.color_palette,
            explicit_colors=self._module_focus_colors if self._module_focus_colors else None,
        )

        # Add background color for all background modules
        for label in self._background_modules:
            self._color_mapping[label] = self.background_color

    def _auto_node_ordering(self) -> NodesMapping:
        """
        Apply automatic node ordering.

        Returns:
            Dictionary mapping original node IDs to display indices.
        """
        from .core import _auto_node_ordering_given_time_modules

        time_links = self.linkstream.get_time_links()
        nodes_set: set[NodeId] = (
            set(self.nodes_to_display) if self.nodes_to_display else set(self.linkstream.nodes)
        )
        modules_dict = self.modules if self.modules is not None else {}

        if self.auto_nodes_ordering_subsets == []:
            self.auto_nodes_ordering_subsets = [nodes_set]
        else:
            self.auto_nodes_ordering_subsets = [
                set(subset) & nodes_set for subset in self.auto_nodes_ordering_subsets
            ]
        return _auto_node_ordering_given_time_modules(
            modules_dict,
            [] if not self._show_edges else time_links,
            1 if not self._show_edges else 0.9,
            self.auto_nodes_ordering_subsets,
        )

    def _create_modules_nodes_segments(self) -> ModuleNodesSegments:
        """
        Create modules nodes segments for optimized drawing.

        Uses TimeModules.get_time_modules_dict() for efficient segment extraction.

        Returns:
            Dictionary mapping module labels to node time segments.

        Raises:
            ValueError: If TimeModules object is not available.
        """
        if self._original_time_modules is None:
            raise ValueError(
                "TimeModules object required but not available. "
                "Ensure set_modules() was called with a TimeModules object."
            )

        # Determine which module labels to include (respects max_shown, hide_self, etc.)
        filtered_labels: set[Any] = set()
        if self._focused_modules:
            filtered_labels.update(self._focused_modules.keys())
        if self._secondary_modules:
            filtered_labels.update(self._secondary_modules.keys())
        if self._background_modules:
            filtered_labels.update(self._background_modules.keys())

        nodes_mapping = self._nodes_mapping or {}
        nodes_to_display_set = set(self.nodes_to_display) if self.nodes_to_display else set()

        modules_nodes_segments: ModuleNodesSegments = {}

        # Use TimeModules API directly for efficient segment extraction
        time_modules_dict = self._original_time_modules.get_time_modules_dict()

        for module_label, nodes_segments_raw in time_modules_dict.items():
            if module_label not in filtered_labels:
                continue

            nodes_segments: dict[int, list[list[int]]] = {}
            for node, time_segments in nodes_segments_raw.items():
                # Remap node to display index
                mapped_node = nodes_mapping.get(node, -1)
                if mapped_node == -1:
                    continue
                if nodes_to_display_set and mapped_node not in nodes_to_display_set:
                    continue

                # Convert TimeSegment objects to [start, end] lists
                nodes_segments[mapped_node] = [[seg.start, seg.end] for seg in time_segments]

            if nodes_segments:
                modules_nodes_segments[module_label] = nodes_segments

        return modules_nodes_segments

    def _draw_module_periods(self) -> None:
        """Draw module periods as colored rectangles."""
        if self._modules_nodes_segments is None:
            self._modules_nodes_segments = self._create_modules_nodes_segments()

        if self._ax is not None and self._color_mapping is not None:
            draw_module_periods(
                self._ax,
                self._modules_nodes_segments,
                self._color_mapping,
                self.height_module_color,
                self.margin_commu_segment,
            )

    def _draw_nodes(self) -> None:
        """Draw nodes as horizontal lines."""
        if self._show_nodes and self._ax is not None:
            if self._start_nodes is not None and self._end_nodes is not None:
                # Use the actual mapped node indices, not sequential 0..N-1
                # nodes_to_display contains the remapped node indices
                nodes_set = set(self.nodes_to_display) if self.nodes_to_display else set()
                draw_nodes(
                    self._ax,
                    nodes_set,
                    self._start_nodes,
                    self._end_nodes,
                    self.node_alpha,
                    self._remapped_node_focus,
                    self.highlight_node_focus,
                    self.node_linewidth,
                    self.node_linewidth_focus,
                    self.longitudinal_nodes_margin,
                )

    def _draw_edges(self) -> None:
        """Draw edges between nodes."""
        if self._show_edges and self._ax is not None and self._time_links is not None:
            if self._time_node_module_mapping is not None and self._color_mapping is not None:
                if self.linkstream.delayed:
                    draw_edges_delayed(
                        self._ax,
                        self._time_links,
                        self._time_node_module_mapping,
                        self._color_mapping,
                        self.edge_alpha,
                        self.color_edges,
                        self.edge_flatten_factor,
                        self.show_edge_orientation,
                        self.linkstream.directed,
                        self.edge_curve_intensity,
                        self.edge_linewidth,
                    )
                else:
                    draw_edges(
                        self._ax,
                        self._time_links,
                        self._time_node_module_mapping,
                        self._color_mapping,
                        self.edge_alpha,
                        self.color_edges,
                        self.edge_flatten_factor,
                        self.show_edge_orientation,
                        self.linkstream.directed,
                        self.linkstream.continuous,
                        self.linkstream,
                        self.edge_linewidth,
                    )
                    if self.show_continuous_duration and self.linkstream.continuous:
                        draw_continuous_duration_lines(
                            self._ax,
                            self._time_links,
                            self._time_node_module_mapping,
                            self._color_mapping,
                            self.edge_alpha,
                            self.color_edges,
                            self.edge_flatten_factor,
                            self.linkstream.directed,
                            self.edge_linewidth,
                        )

    def _draw_edge_activity(self) -> None:
        """Draw edge activity markers as rectangles."""
        if self._show_edge_activity and self._ax is not None and self._time_links is not None:
            if self._time_node_module_mapping is not None and self._color_mapping is not None:
                if self.linkstream.delayed:
                    draw_edge_activity_delayed(
                        self._ax,
                        self._time_links,
                        self._time_node_module_mapping,
                        self._color_mapping,
                        self.edge_activity_alpha,
                        self.color_edges,
                        self.edge_flatten_factor,
                        self.show_edge_orientation,
                        self.linkstream.directed,
                        self.edge_activity_width,
                        self.edge_activity_height,
                    )
                else:
                    draw_edge_activity(
                        self._ax,
                        self._time_links,
                        self._time_node_module_mapping,
                        self._color_mapping,
                        self.edge_activity_alpha,
                        self.color_edges,
                        self.edge_flatten_factor,
                        self.show_edge_orientation,
                        self.linkstream.directed,
                        self.linkstream.continuous,
                        self.edge_activity_width,
                        self.edge_activity_height,
                    )

    def _draw_focus_highlights(self) -> None:
        """Draw focus highlights for selected nodes/times."""
        if self._ax is not None:
            num_nodes = len(self.nodes_to_display) if self.nodes_to_display else 0
            draw_focus_highlights(
                self._ax,
                self.node_focus,
                self.time_focus,
                self.node_OR_time_focus,
                self.linkstream.network_duration,
                num_nodes,
            )

    def _draw_night_highlights(self) -> None:
        """Draw night highlights."""
        if self.nights and self._ax is not None and self._end_nodes is not None:
            draw_night_highlights(self._ax, self.nights, self._end_nodes)

    def _configure_axes(self) -> None:
        """Configure axes labels and ticks."""
        if self._ax is not None:
            num_nodes = len(self.nodes_to_display) if self.nodes_to_display else 0
            configure_axes(
                self._ax,
                self.linkstream.network_duration,
                self.display_xticks_labels,
                self.display_yticks_labels,
                self.label_font_size,
                self.display_xlabel,
                self.display_ylabel,
                self.node_labels,
                self.node_label_fontsize,
                self.node_labels_padding,
                self.bold_labels,
                num_nodes,
                self.y_padding_bottom,
                self.y_padding_top,
                # X-axis label customization
                xlabel_text=self.xlabel_text,
                xlabel_fontsize=self.xlabel_fontsize,
                xlabel_fontweight=self.xlabel_fontweight,
                xlabel_coords=self.xlabel_coords,
                xlabel_rotation=self.xlabel_rotation,
                xlabel_ha=self.xlabel_ha,
                # Y-axis label customization
                ylabel_text=self.ylabel_text,
                ylabel_fontsize=self.ylabel_fontsize,
                ylabel_fontweight=self.ylabel_fontweight,
                ylabel_coords=self.ylabel_coords,
                ylabel_rotation=self.ylabel_rotation,
                ylabel_ha=self.ylabel_ha,
                # Margins
                x_margin=self.x_margin,
                y_margin=self.y_margin,
            )

    def _finalize_plot(self, title: str, show_plot: bool) -> None:
        """
        Finalize plot configuration.

        Args:
            title: Plot title.
            show_plot: Whether to display the plot.
        """
        if self._ax is not None:
            finalize_plot(self._ax, title, show_plot)
