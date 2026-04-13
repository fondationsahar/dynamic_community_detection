"""
Drawing operations module for longitudinal module plotting.

This module provides functions for rendering various visual elements in the longitudinal
module visualization. It handles drawing of module periods, nodes, edges, highlights,
and other graphical components.

Key Functions:
    - draw_module_periods: Draw colored rectangles for module time periods
    - draw_nodes: Draw horizontal lines representing nodes
    - draw_focus_highlights: Draw highlight rectangles for focused nodes/times
    - draw_edges: Draw arcs between nodes for regular linkstreams
    - draw_edges_delayed: Draw arcs between nodes for delayed linkstreams
    - draw_edge_activity: Draw cross markers for edge activity
    - draw_night_highlights: Draw vertical lines for night periods
"""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from .drawing_utils import _draw_edge_common
from .geometry import (
    create_highlight_rectangle,
    draw_arrow_head,
)


def draw_module_periods(
    ax: plt.Axes,
    modules_nodes_segments: dict[Any, dict[Any, list[list[int]]]],
    color_mapping: dict[Any, Any],
    height_module_color: float,
    margin_commu_segment: float,
):
    """
    Draw module periods as colored rectangles using pre-computed time segments.

    This optimized function draws rectangles for entire time segments rather than
    individual time points, significantly improving rendering performance for
    modules with consecutive time periods.

    Args:
        ax: Matplotlib axes to draw on
        modules_nodes_segments: Dictionary mapping module labels to node time segments
            Structure: {module_label: {node: [[start1, end1], [start2, end2], ...]}}
        color_mapping: Dictionary mapping module labels to colors
        height_module_color: Height of module rectangles in plot coordinates

    Note:
        This function assumes time segments are pre-computed and represent
        continuous time ranges for each node within each module.
    """
    periods = []
    for module_label, nodes_segment in modules_nodes_segments.items():
        for node, segments in nodes_segment.items():
            for segment in segments:
                rect = Rectangle(
                    (
                        segment[0] + margin_commu_segment,
                        node + (0.5 - height_module_color / 2),
                    ),
                    width=segment[1] - segment[0] + 1 - 2 * margin_commu_segment,
                    height=height_module_color,
                    facecolor=color_mapping.get(module_label, "gainsboro"),
                    edgecolor=color_mapping.get(module_label, "gainsboro"),
                )

                periods.append(rect)

    pc = PatchCollection(periods, match_original=True)
    ax.add_collection(pc)


def draw_nodes(
    ax: Axes,
    nodes: set,
    start_nodes: dict,
    end_nodes: dict,
    node_alpha: float,
    node_focus: list = [],
    highlight_node_focus: bool = True,
    linewidth: float = 0.25,
    linewidth_focus: float = 0.5,
    longitudinal_margin: float = 0,
):
    """
    Draw nodes as horizontal lines.

    Args:
        ax: Matplotlib axes
        nodes: Set of nodes
        start_nodes: Start times for nodes
        end_nodes: End times for nodes
        node_alpha: Transparency for node lines
        node_focus: List of focused node indices
        highlight_node_focus: Whether to highlight focused nodes
        linewidth: Line width for regular (non-focused) nodes. Default 0.25.
        linewidth_focus: Line width for focused nodes. Default 0.5.
    """
    for node in nodes:
        if highlight_node_focus and node in node_focus:
            ax.hlines(
                y=node + 0.5,
                xmin=start_nodes[node] + longitudinal_margin,
                xmax=end_nodes[node] - longitudinal_margin,
                color="black",
                linestyle="-",
                linewidth=linewidth_focus,
                alpha=1,
            )
            continue
        ax.hlines(
            y=node + 0.5,
            xmin=start_nodes[node] + longitudinal_margin,
            xmax=end_nodes[node] - longitudinal_margin,
            color="black",
            linestyle="-",
            linewidth=linewidth,
            alpha=node_alpha,
        )


def draw_focus_highlights(
    ax: Axes,
    node_focus: list | None,
    time_focus: int | None,
    node_OR_time_focus: bool,
    network_duration: int,
    num_nodes: int,
):
    """
    Draw highlight rectangles for focused nodes and times.

    Args:
        ax: Matplotlib axes
        node_focus: Node to highlight
        time_focus: Time to highlight
        node_OR_time_focus: Whether to use OR logic for focus
        network_duration: Total network duration
        num_nodes: Number of nodes
    """
    # Ensure node_focus is a list (default to empty list if None)
    node_focus_list = node_focus if node_focus is not None else []

    if (node_OR_time_focus and len(node_focus_list) > 0) or (time_focus is not None):
        # Highlight node focus area
        for nfocus in node_focus_list:
            rec = create_highlight_rectangle(
                node=nfocus,
                time=0,
                x_decay=0,
                y_delay=0.33,
                width=network_duration + 1,
                height=0.33,
                alpha=1,
            )
            ax.add_patch(rec)
        rec = create_highlight_rectangle(
            node=0.0,
            time=0,
            x_decay=0,
            y_delay=0.33,
            width=network_duration + 1,
            height=0.33,
            alpha=1,
        )
        ax.add_patch(rec)

    if time_focus is not None:
        if node_OR_time_focus or len(node_focus_list) == 0:
            # Highlight entire time column
            rec = create_highlight_rectangle(node=-0.25, time=time_focus, height=num_nodes + 0.5)
            ax.add_patch(rec)
        else:
            # Highlight specific node-time intersection
            for nfocus in node_focus_list:
                rec = create_highlight_rectangle(
                    node=float(nfocus), time=time_focus, x_decay=0.1, width=0.8
                )
                ax.add_patch(rec)


def draw_edge_activity(
    ax: Axes,
    time_links: list,
    time_node_module_mapping: dict,
    color_mapping: dict,
    edge_alpha: float,
    color_edges: bool,
    edge_flatten_factor: float,
    show_edge_orientation: bool = False,
    is_directed: bool = False,
    is_continuous: bool = False,
    marker_width: float = 0.8,
    marker_height: float = 0.4,
):
    """
    Draw rectangle markers for edge activity (non-delayed linkstreams).

    Args:
        ax: Matplotlib axes
        time_links: List of (source, target, time[, weight]) tuples.
            Weight is optional and defaults to 1.0 if not provided.
        time_node_module_mapping: Mapping from (node, time) to module
        color_mapping: Color mapping for modules
        edge_alpha: Transparency for markers
        color_edges: Whether to color edges by module
        edge_flatten_factor: Flattening factor for edge arcs
        show_edge_orientation: Whether to display arrow heads showing edge orientation
        is_directed: Whether the linkstream is directed
        is_continuous: Whether the linkstream has continuous time intervals
        marker_width: Width of edge activity rectangle markers (0.0 to 1.0). Default 0.8.
        marker_height: Height of edge activity rectangle markers (0.0 to 1.0). Default 0.4.
    """
    rectangles = []

    for link in time_links:
        # Handle both 3-element (no weight) and 4-element (with weight) tuples
        source, target, time = link[0], link[1], link[2]
        for node in [source, target]:
            if node == -1:
                continue
            # Calculate rectangle position (centered on node+0.5, time+0.5)
            x = time + 0.5 - marker_width / 2
            y = node + 0.5 - marker_height / 2
            rect = Rectangle(
                (x, y),
                width=marker_width,
                height=marker_height,
                facecolor="black",
                edgecolor="none",
                alpha=edge_alpha,
            )
            rectangles.append(rect)

    if rectangles:
        pc = PatchCollection(rectangles, match_original=True)
        ax.add_collection(pc)


def draw_edge_activity_delayed(
    ax: Axes,
    time_links: list,
    time_node_module_mapping: dict,
    color_mapping: dict,
    edge_alpha: float,
    color_edges: bool,
    edge_flatten_factor: float,
    show_edge_orientation: bool = False,
    is_directed: bool = False,
    marker_width: float = 0.8,
    marker_height: float = 0.4,
):
    """
    Draw rectangle markers for edge activity (delayed linkstreams).

    For delayed linkstreams, each edge has a source time and target time.
    This draws rectangle markers at:
    - Source node at source_time
    - Target node at target_time

    Args:
        ax: Matplotlib axes
        time_links: List of (source, target, source_time, target_time[, weight]) tuples.
            Weight is optional and defaults to 1.0 if not provided.
        time_node_module_mapping: Mapping from (node, time) to module
        color_mapping: Color mapping for modules
        edge_alpha: Transparency for markers
        color_edges: Whether to color edges by module
        edge_flatten_factor: Flattening factor for edge arcs
        show_edge_orientation: Whether to display arrow heads showing edge orientation
        is_directed: Whether the linkstream is directed
        marker_width: Width of edge activity rectangle markers (0.0 to 1.0). Default 0.8.
        marker_height: Height of edge activity rectangle markers (0.0 to 1.0). Default 0.4.
    """
    # For undirected graphs, filter duplicate edges like in draw_edges_delayed
    seen_edges: set = set()
    rectangles = []

    for link in time_links:
        # Handle both 4-element (no weight) and 5-element (with weight) tuples
        source, target, source_time, target_time = link[0], link[1], link[2], link[3]
        # Skip if we already processed the reverse edge (for undirected graphs)
        if not is_directed:
            edge_key = tuple(sorted([(source, source_time), (target, target_time)]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

        # Draw rectangle at source node position (at source_time)
        if source != -1:
            x = source_time + 0.5 - marker_width / 2
            y = source + 0.5 - marker_height / 2
            rect = Rectangle(
                (x, y),
                width=marker_width,
                height=marker_height,
                facecolor="black",
                edgecolor="none",
                alpha=edge_alpha,
            )
            rectangles.append(rect)

        # Draw rectangle at target node position (at target_time)
        if target != -1:
            x = target_time + 0.5 - marker_width / 2
            y = target + 0.5 - marker_height / 2
            rect = Rectangle(
                (x, y),
                width=marker_width,
                height=marker_height,
                facecolor="black",
                edgecolor="none",
                alpha=edge_alpha,
            )
            rectangles.append(rect)

    if rectangles:
        pc = PatchCollection(rectangles, match_original=True)
        ax.add_collection(pc)


def draw_edges(
    ax: Axes,
    time_links: list,
    time_node_module_mapping: dict,
    color_mapping: dict,
    edge_alpha: float,
    color_edges: bool,
    edge_flatten_factor: float,
    show_edge_orientation: bool = False,
    is_directed: bool = False,
    is_continuous: bool = False,
    linkstream=None,
    linewidth: float = 1.0,
):
    """
    Draw edges between nodes.

    Args:
        ax: Matplotlib axes
        time_links: List of edge tuples. Format depends on linkstream type:
            - Regular: (source, target, time, weight)
            - Continuous: (source, target, start_time, duration, weight)
        time_node_module_mapping: Mapping from (node, time) to module
        color_mapping: Color mapping for modules
        edge_alpha: Transparency for edges
        color_edges: Whether to color edges by module
        edge_flatten_factor: Flattening factor for edge arcs
        show_edge_orientation: Whether to display arrow heads showing edge orientation
        is_directed: Whether the linkstream is directed
        is_continuous: Whether the linkstream has continuous time intervals
        linkstream: The linkstream object for accessing original edge directions
        linewidth: Edge line width in points. Default 1.0.
    """
    for link in time_links:
        if is_continuous:
            # Continuous linkstreams have: (source, target, start_time, duration[, weight])
            source, target, start_time = link[0], link[1], link[2]
            duration = link[3] if len(link) > 3 else 1
            time = start_time  # Draw arc at start time (like an instantaneous edge)
        else:
            # Regular linkstreams have: (source, target, time[, weight])
            source, target, time = link[0], link[1], link[2]

        node1, node2 = source, target
        if not is_directed:
            node1, node2 = sorted([node1, node2])
        center1 = (time + 0.5, node1 + 0.5)
        center2 = (time + 0.5, node2 + 0.5)

        # Use common edge drawing logic
        _draw_edge_common(
            ax=ax,
            center1=center1,
            center2=center2,
            time_node_module_mapping=time_node_module_mapping,
            color_mapping=color_mapping,
            edge_alpha=edge_alpha,
            color_edges=color_edges,
            source_node=source,
            target_node=target,
            source_time=time,
            target_time=time,
            edge_flatten_factor=edge_flatten_factor,
            linewidth=linewidth,
            straight=False,
        )

        # Draw arrow head for edge orientation if requested and linkstream is directed
        if show_edge_orientation and is_directed:
            draw_arrow_head(ax, center1, center2, "black", edge_alpha, linewidth)


def draw_continuous_duration_lines(
    ax: Axes,
    time_links: list,
    time_node_module_mapping: dict,
    color_mapping: dict,
    edge_alpha: float,
    color_edges: bool,
    edge_flatten_factor: float,
    is_directed: bool = False,
    linewidth: float = 1.0,
):
    """
    Draw horizontal lines showing the duration of continuous edges.

    For each continuous link, draws a horizontal line starting where the arc
    curve passes at the line's y-level and extending rightward for a length
    equal to the link's duration. The line is positioned slightly above the
    vertical midpoint between the two connected nodes, with a small random
    y-jitter to reduce overlap.

    Args:
        ax: Matplotlib axes
        time_links: List of (source, target, start_time, duration[, weight]) tuples.
        time_node_module_mapping: Mapping from (node, time) to module
        color_mapping: Color mapping for modules
        edge_alpha: Base transparency for edges
        color_edges: Whether to color lines by module
        edge_flatten_factor: Flattening factor for edge arcs (must match draw_edges).
        is_directed: Whether the linkstream is directed
        linewidth: Line width in points. Default 1.0.
    """
    import numpy as np
    from matplotlib.collections import LineCollection

    rng = np.random.default_rng(42)

    segments = []
    colors = []
    seen_edges: set = set()

    for link in time_links:
        source, target, start_time = link[0], link[1], link[2]
        duration = link[3] if len(link) > 3 else 1

        node1, node2 = source, target
        if not is_directed:
            node1, node2 = sorted([node1, node2])
            edge_key = (node1, node2, start_time, duration)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

        # Arc center x (arc drawn at start_time)
        time = start_time
        arc_x = time + 0.5

        # Arc geometry (must match create_edge_arc / draw_edges)
        node_dist = abs(node2 - node1)
        arc_semi_w = (node_dist ** 0.5 * 0.9 * edge_flatten_factor) / 2  # half-width in x
        arc_semi_h = node_dist / 2  # half-height in y

        # Vertical midpoint between the two node centers, slightly above
        # with random jitter to reduce overlap between co-located edges
        y_mid = (node1 + node2 + 1) / 2
        jitter = rng.uniform(-0.15, 0.15)
        y = y_mid + 0.08 + jitter

        # Compute where the arc curve intersects this y-level.
        # The arc is the left half of an ellipse centered at (arc_x, y_mid).
        # Ellipse: ((x - arc_x)/arc_semi_w)^2 + ((y - y_mid)/arc_semi_h)^2 = 1
        # Solving for x on the left half: x = arc_x - arc_semi_w * sqrt(1 - t^2)
        dy_ratio = (y - y_mid) / arc_semi_h if arc_semi_h > 0 else 0
        t2 = 1 - dy_ratio ** 2
        if t2 > 0:
            # Right side of the arc at this y-level
            x_start = arc_x + arc_semi_w * np.sqrt(t2)
        else:
            x_start = arc_x

        x_end = arc_x + duration

        # Resolve color (same logic as _draw_edge_common)
        color = "black"
        if color_edges:
            source_module = time_node_module_mapping.get((source, time))
            target_module = time_node_module_mapping.get((target, time))
            if (
                source_module is not None
                and source_module == target_module
            ):
                c = color_mapping.get(source_module)
                if c is not None and not (isinstance(c, str) and c == "gainsboro"):
                    color = c

        segments.append([(x_start, y), (x_end, y)])
        # Small vertical tick at the end
        tick_half = 0.06
        segments.append([(x_end, y - tick_half), (x_end, y + tick_half)])
        colors.append(color)
        colors.append(color)

    if segments:
        lc = LineCollection(
            segments,
            colors=colors,
            alpha=edge_alpha,
            linewidths=linewidth / 2,
        )
        ax.add_collection(lc)


def draw_edges_delayed(
    ax: Axes,
    time_links: list,
    time_node_module_mapping: dict,
    color_mapping: dict,
    edge_alpha: float,
    color_edges: bool,
    edge_flatten_factor: float,
    show_edge_orientation: bool = False,
    is_directed: bool = False,
    curve_intensity: float = 0.0,
    linewidth: float = 1.0,
):
    """
    Draw edges between nodes for delayed linkstreams.

    Args:
        ax: Matplotlib axes
        time_links: List of (source, target, source_time, target_time[, weight]) tuples.
            Weight is optional and defaults to 1.0 if not provided.
        time_node_module_mapping: Mapping from (node, time) to module
        color_mapping: Color mapping for modules
        edge_alpha: Transparency for edges
        color_edges: Whether to color edges by module
        edge_flatten_factor: Flattening factor for edge arcs
        show_edge_orientation: Whether to display arrow heads showing edge orientation
        is_directed: Whether the linkstream is directed
        curve_intensity: Controls curve intensity for delayed edges (0.0 = straight,
            positive = curve right, negative = curve left). Default 0.0.
        linewidth: Base edge line width in points. This is multiplied by edge weight.
            Default 1.0.
    """
    # For undirected graphs, lago LinkStream returns both A→B and B→A
    # We need to filter to only draw each unique edge once
    seen_edges: set = set()

    for link in time_links:
        # Handle both 4-element (no weight) and 5-element (with weight) tuples
        source, target, source_time, target_time = link[0], link[1], link[2], link[3]
        weight = link[4] if len(link) > 4 else 1.0
        # Skip if we already drew the reverse edge (for undirected graphs)
        if not is_directed:
            # Create a canonical key that's the same for both directions
            # Use sorted node pair + sorted time pair
            edge_key = tuple(sorted([(source, source_time), (target, target_time)]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

        center1 = (source_time + 0.5, source + 0.5)
        center2 = (target_time + 0.5, target + 0.5)

        # For consistent curve direction, always draw from lower y (node) to higher y
        # This ensures all curves bend the same way relative to the node axis
        if not is_directed and center1[1] > center2[1]:
            center1, center2 = center2, center1

        # Use common edge drawing logic
        _draw_edge_common(
            ax=ax,
            center1=center1,
            center2=center2,
            time_node_module_mapping=time_node_module_mapping,
            color_mapping=color_mapping,
            edge_alpha=edge_alpha,
            color_edges=color_edges,
            source_node=source,
            target_node=target,
            source_time=source_time,
            target_time=target_time,
            edge_flatten_factor=edge_flatten_factor,
            linewidth=linewidth * weight,
            straight=True,
            curve_intensity=curve_intensity,
        )

        # Draw arrow head for edge orientation if requested and linkstream is directed
        if show_edge_orientation and is_directed:
            draw_arrow_head(ax, center1, center2, "black", edge_alpha, linewidth * weight)


def draw_night_highlights(ax: Axes, nights: list, end_nodes: dict):
    """
    Draw vertical lines to highlight night periods.

    Args:
        ax: Matplotlib axes
        nights: List of time points for night highlights
        end_nodes: End times for nodes
    """
    for night in nights:
        ax.vlines(
            x=night + 0.5,
            ymin=0,
            ymax=len(end_nodes),
            color="black",
            linewidth=2,
        )
