"""
Drawing utility functions for longitudinal module plotting.

This module contains helper functions to reduce code duplication in the main drawing module.
"""

from typing import Dict, Tuple

from matplotlib.axes import Axes

from .geometry import create_edge_arc


def _draw_edge_common(
    ax: Axes,
    center1: Tuple[float, float],
    center2: Tuple[float, float],
    time_node_module_mapping: Dict,
    color_mapping: Dict,
    edge_alpha: float,
    color_edges: bool,
    source_node: int,
    target_node: int,
    source_time: int,
    target_time: int,
    edge_flatten_factor: float,
    linewidth: float,
    straight: bool = False,
    curve_intensity: float = 0.0,
) -> None:
    """
    Common edge drawing logic shared between draw_edges and draw_edges_delayed.

    Args:
        ax: Matplotlib axes
        center1: First center point (x, y)
        center2: Second center point (x, y)
        time_node_module_mapping: Mapping from (node, time) to module
        color_mapping: Color mapping for modules
        edge_alpha: Transparency for edges
        color_edges: Whether to color edges by module
        source_node: Source node
        target_node: Target node
        source_time: Source time
        target_time: Target time
        edge_flatten_factor: Flattening factor for edge arcs
        linewidth: Line width
        straight: Whether to draw straight edges (for delayed linkstreams)
        curve_intensity: For delayed edges, controls curve intensity (0.0 = straight,
            positive = curve right, negative = curve left). Default 0.0.
    """
    # For delayed/straight edges, use center1→center2 order (source→target)
    # For regular arcs, use center2→center1 for historical arc direction reasons
    start_center = center1 if straight else center2
    end_center = center2 if straight else center1

    if color_edges:
        source_module = time_node_module_mapping.get((source_node, source_time))
        target_module = time_node_module_mapping.get((target_node, target_time))

        if source_module is None or target_module is None:
            # No module info - draw default edge
            arc = create_edge_arc(
                start_center,
                end_center,
                alpha=edge_alpha,
                flatten_factor=edge_flatten_factor,
                linewidth=linewidth,
                straight=straight,
                curve_intensity=curve_intensity,
            )
            ax.add_patch(arc)
            return

        if source_module != target_module:
            # Different modules - draw default edge
            arc = create_edge_arc(
                start_center,
                end_center,
                alpha=edge_alpha,
                flatten_factor=edge_flatten_factor,
                linewidth=linewidth,
                straight=straight,
                curve_intensity=curve_intensity,
            )
            ax.add_patch(arc)
            return

        # Same module - draw colored edge
        color = color_mapping.get(source_module)

        if color is None or (isinstance(color, str) and color == "gainsboro"):
            arc = create_edge_arc(
                start_center,
                end_center,
                alpha=edge_alpha,
                flatten_factor=edge_flatten_factor,
                linewidth=linewidth,
                straight=straight,
                curve_intensity=curve_intensity,
            )
            ax.add_patch(arc)
            return

        arc = create_edge_arc(
            start_center,
            end_center,
            alpha=edge_alpha,
            color=color,
            linewidth=linewidth,
            flatten_factor=edge_flatten_factor,
            straight=straight,
            curve_intensity=curve_intensity,
        )
        ax.add_patch(arc)
    else:
        # Draw default edge
        arc = create_edge_arc(
            start_center,
            end_center,
            alpha=edge_alpha,
            flatten_factor=edge_flatten_factor,
            linewidth=linewidth,
            straight=straight,
            curve_intensity=curve_intensity,
        )
        ax.add_patch(arc)
