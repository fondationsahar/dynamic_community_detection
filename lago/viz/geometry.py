"""
Geometry module for longitudinal module plotting.

This module provides geometric utility functions for creating and manipulating
visual elements in the longitudinal module visualization. It includes functions
for creating arcs, rectangles, crosses, and arrow heads used in the plotting.

Key Functions:
    - create_center_cross: Create cross markers for edge activity
    - create_edge_arc: Create arc patches for node-to-node edges
    - create_highlight_rectangle: Create highlight rectangles for focus areas
    - draw_arrow_head: Draw arrow heads for directed edges
"""

import matplotlib.patches as patches
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


def create_center_cross(
    center: tuple[float, float],
    size: float = 0.05,
    color: str = "black",
    linewidth: float = 0.5,
    alpha: float = 1.0,
) -> list[Line2D]:
    """
    Create cross markers at two center points.

    Args:
        center1: First center point (x, y)
        center2: Second center point (x, y)
        size: Half-length of each cross arm
        color: Cross color
        linewidth: Line width
        alpha: Transparency

    Returns:
        List of Line2D objects representing the crosses
    """
    x, y = center

    return [
        Line2D(
            [x - size, x + size],
            [y, y],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        ),
        Line2D(
            [x, x],
            [y - size, y + size],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        ),
    ]


def create_edge_arc(
    center1: tuple[float, float],
    center2: tuple[float, float],
    alpha: float = 1,
    flatten_factor: float = 0.7,
    color: str = "black",
    linewidth: float = 0.5,
    straight: bool = False,
    curve_intensity: float = 0.0,
) -> patches.Arc | FancyArrowPatch:
    """
    Create an arc patch for edges.

    Args:
        center1: First center point (x, y)
        center2: Second center point (x, y)
        alpha: Transparency
        flatten_factor: Flattening factor for the arc
        color: Edge color
        linewidth: Line width
        straight: Whether this is a delayed/straight edge (uses FancyArrowPatch)
        curve_intensity: For delayed edges, controls curve intensity (0.0 = straight,
            positive = curve right, negative = curve left). Default 0.0.

    Returns:
        Arc patch or FancyArrowPatch
    """

    if straight:
        # For delayed linkstreams - use FancyArrowPatch with optional curve
        # Positive rad curves right, negative curves left
        # shrinkA=0, shrinkB=0 ensures edge starts/ends exactly at center points
        arc = FancyArrowPatch(
            center1,
            center2,
            connectionstyle=f"arc3,rad={curve_intensity}",
            color=color,
            linewidth=linewidth / 2,
            alpha=alpha,
            clip_on=True,  # Clip curves to axes bounds to prevent extra padding
            shrinkA=0,  # Don't shrink from start point
            shrinkB=0,  # Don't shrink from end point
        )
        return arc

    radius = np.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2) / 2
    angle1 = np.arctan2(center2[1] - center1[1], center2[0] - center1[0])
    angle2 = angle1 + np.pi

    # Convert to degrees
    angle1_deg = np.degrees(angle1)
    angle2_deg = np.degrees(angle2)

    arc_width = 2 * radius
    arc_height = (arc_width / abs(center1[1] - center2[1]) ** 0.5 * 0.9) * flatten_factor

    # Handle potential NaN values
    if np.isnan(arc_height):
        arc_height = 0
    arc = patches.Arc(
        ((center1[0] + center2[0]) / 2, (center1[1] + center2[1]) / 2),
        arc_height,
        arc_width,
        angle=0,
        theta1=min(angle1_deg, angle2_deg),
        theta2=max(angle1_deg, angle2_deg),
        facecolor="black",
        edgecolor=color,
        linewidth=linewidth / 2,
        alpha=alpha,
    )

    return arc


def create_highlight_rectangle(
    node: float,
    time: float,
    x_decay: float = 1 / 2,
    y_delay: float = 0,
    width: float = 1 / 4,
    height: float = 1,
    alpha: float = 0.5,
) -> patches.Rectangle:
    """
    Create a highlight rectangle patch.

    Args:
        node: Y-position for the rectangle
        time: X-position for the rectangle
        x_decay: X-offset
        y_delay: Y-offset
        width: Rectangle width
        height: Rectangle height
        alpha: Transparency

    Returns:
        Rectangle patch
    """
    rectangle = patches.Rectangle(
        (time + x_decay, node + y_delay),
        width,
        height,
        facecolor="black",
        alpha=alpha,
    )
    return rectangle


def draw_arrow_head(
    ax: Axes,
    center1: tuple[float, float],
    center2: tuple[float, float],
    color: str,
    alpha: float = 1,
    linewidth: float = 0.5,
) -> None:
    """
    Draw an arrow head for directed edges.

    Args:
        ax: Matplotlib axes
        center1: Start point of the edge (source)
        center2: End point of the edge (target)
        color: Arrow color
        alpha: Arrow transparency
    """
    # Arrow head parameters
    arrow_length = linewidth * 0.15  # Length of arrow head
    arrow_angle = np.radians(25)  # Angle of arrow head

    # Vector from center1 to center2 (direction of the edge)
    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]
    edge_length = np.sqrt(dx**2 + dy**2)

    if edge_length == 0:
        return

    # Normalize direction vector
    dx_norm = dx / edge_length
    dy_norm = dy / edge_length

    # Arrow tip points exactly at center2 (target node)
    arrow_tip = center2

    # Calculate arrow base points by moving back from the tip
    # Left arrow point (perpendicular to edge direction)
    left_x = arrow_tip[0] - arrow_length * (
        dx_norm * np.cos(arrow_angle) - dy_norm * np.sin(arrow_angle)
    )
    left_y = arrow_tip[1] - arrow_length * (
        dy_norm * np.cos(arrow_angle) + dx_norm * np.sin(arrow_angle)
    )
    arrow_left = (left_x, left_y)

    # Right arrow point (perpendicular to edge direction)
    right_x = arrow_tip[0] - arrow_length * (
        dx_norm * np.cos(arrow_angle) + dy_norm * np.sin(arrow_angle)
    )
    right_y = arrow_tip[1] - arrow_length * (
        dy_norm * np.cos(arrow_angle) - dx_norm * np.sin(arrow_angle)
    )
    arrow_right = (right_x, right_y)

    # Create arrow head polygon pointing exactly at center2
    arrow_polygon = patches.Polygon(
        [arrow_tip, arrow_left, arrow_right],
        closed=True,
        facecolor=color,
        edgecolor=color,
        linewidth=min(0.5, linewidth),
        alpha=alpha,
    )

    ax.add_patch(arrow_polygon)
