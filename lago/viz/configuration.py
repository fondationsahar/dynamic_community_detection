"""
Visual configuration module for longitudinal module plotting.

This module provides functions for setting up and configuring the visualization layout,
axes, and final plot appearance. It handles figure creation, axis configuration,
label placement, and final plot adjustments.

Key Functions:
    - setup_figure_and_axes: Create and configure matplotlib figure and axes
    - configure_axes: Configure axes labels, ticks, and appearance
    - finalize_plot: Apply final plot configuration and display options
"""

from typing import Tuple, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def setup_figure_and_axes(width: int, height: int) -> tuple[Figure, Axes]:
    """
    Set up the matplotlib figure and axes with appropriate dimensions.

    Args:
        width: Figure width in pixels
        height: Figure height in pixels

    Returns:
        Tuple of (figure, axes)
    """
    width_inches = width / 100
    height_inches = height / 100 * 0.75
    fig, ax = plt.subplots(1, figsize=(width_inches, height_inches), dpi=100)
    return fig, cast("Axes", ax)


def configure_axes(
    ax: Axes,
    network_duration: int,
    display_xticks_labels: bool,
    display_yticks_labels: bool,
    label_font_size: int,
    display_xlabel: bool,
    display_ylabel: bool,
    node_labels: list,
    node_label_fontsize: int,
    ylabel_padding: int,
    bold_labels: list,
    num_nodes: int = 0,
    y_padding_bottom: float = 0.0,
    y_padding_top: float = 0.0,
    # X-axis label customization
    xlabel_text: str = "Time",
    xlabel_fontsize: int | None = None,
    xlabel_fontweight: str = "bold",
    xlabel_coords: tuple[float, float] = (0.0, 0.025),
    xlabel_rotation: float = 0.0,
    xlabel_ha: str = "left",
    # Y-axis label customization
    ylabel_text: str = "Nodes",
    ylabel_fontsize: int | None = None,
    ylabel_fontweight: str = "bold",
    ylabel_coords: tuple[float, float] = (-0.05, 1.0),
    ylabel_rotation: float = 90.0,
    ylabel_ha: str = "right",
    # Margins
    x_margin: float = 0.02,
    y_margin: float = 0.05,
):
    """
    Configure axes labels, ticks, and appearance.

    Args:
        ax: Matplotlib axes
        network_duration: Total network duration
        nodes: Set of nodes
        display_xticks_labels: Whether to display x-axis tick labels
        display_yticks_labels: Whether to display y-axis tick labels
        label_font_size: Font size for axis labels
        display_xlabel: Whether to display x-axis label
        display_ylabel: Whether to display y-axis label
    """
    if display_xticks_labels:
        # Configure x-axis ticks and labels
        tmp_current_xticks = ax.get_xticks()
        current_xticks = []
        for x1, x2 in zip(tmp_current_xticks[:-1], tmp_current_xticks[1:]):
            current_xticks.append((x1 + x2) / 2)
        if len(tmp_current_xticks) > 0:
            current_xticks.append(tmp_current_xticks[-1])

        interval = current_xticks[1] - current_xticks[0] if len(current_xticks) > 1 else 1.0
        new_xticks = [x + interval for x in current_xticks]
        new_xticks = list(range(int(min(new_xticks)), int(max(new_xticks)) + 1))
        new_xticks = [tck - 0.5 for tck in new_xticks]
        ax.set_xticks(new_xticks)

        new_labels = [str(ts + 1) for ts in range(network_duration)]
        if network_duration == 8:
            new_labels = [""] + new_labels + ["", ""]
        else:
            new_labels = new_labels + [""] * (len(new_xticks) - len(new_labels))
        ax.set_xticklabels(new_labels, fontsize=14)
    else:
        ax.set_xticklabels([])

    if display_yticks_labels:
        # Configure y-axis ticks and labels
        # Ensure the number of ticks matches the number of node labels
        if node_labels:
            # Create y-ticks that match the node labels
            # Node positions are typically at 0.5, 1.5, 2.5, etc.
            new_yticks = [i + 0.5 for i in range(len(node_labels))]
            ax.set_yticks(new_yticks)
            ax.set_yticklabels(node_labels, fontsize=node_label_fontsize)
            ax.tick_params(axis="y", pad=ylabel_padding)

            for tick, label_text in zip(ax.get_yticklabels(), node_labels):
                if label_text in bold_labels:
                    tick.set_fontweight("bold")
                else:
                    tick.set_fontweight("normal")

        else:
            # Fallback to default behavior if no node labels provided
            current_yticks = ax.get_yticks()
            interval = current_yticks[1] - current_yticks[0] if len(current_yticks) > 1 else 1.0
            new_yticks = [x + interval / 2 for x in current_yticks]
            ax.set_yticks(new_yticks)
            ax.set_yticklabels([])
    else:
        ax.set_yticklabels([])

    # Apply margins
    ax.margins(x=x_margin)
    ax.margins(y=y_margin)

    # Set explicit y-limits to prevent FancyArrowPatch curves from expanding bounds
    # This is especially important for delayed linkstreams with curved edges
    # y_padding_bottom/top adds extra space at bottom/top (in node units)
    if num_nodes > 0:
        ax.set_ylim(-y_padding_bottom, num_nodes + y_padding_top)

    if display_xlabel:
        ax.xaxis.set_label_coords(*xlabel_coords)
        effective_xlabel_fontsize = (
            xlabel_fontsize if xlabel_fontsize is not None else label_font_size
        )
        ax.set_xlabel(
            xlabel_text,
            fontsize=effective_xlabel_fontsize,
            fontweight=xlabel_fontweight,
            rotation=xlabel_rotation,
            ha=xlabel_ha,
        )

    if display_ylabel:
        # Adjust y-axis label position to prevent overlap with tick labels
        ax.yaxis.set_label_coords(*ylabel_coords)
        effective_ylabel_fontsize = (
            ylabel_fontsize if ylabel_fontsize is not None else label_font_size
        )
        ax.set_ylabel(
            ylabel_text,
            fontsize=effective_ylabel_fontsize,
            fontweight=ylabel_fontweight,
            rotation=ylabel_rotation,
            ha=ylabel_ha,
        )
        # Add extra left margin when displaying y-tick labels to prevent overlap
        if display_yticks_labels and node_labels:
            ax.margins(x=0.01)

    # Remove axes spines
    for spine_key in ax.spines:
        ax.spines[spine_key].set_visible(False)

    plt.tick_params(left=False, bottom=False)


def finalize_plot(ax: Axes, title: str, show_plot: bool):
    """
    Finalize plot configuration.

    Args:
        ax: Matplotlib axes
        title: Plot title
        show_plot: Whether to display the plot
    """
    fig = ax.get_figure()
    if fig is not None and isinstance(fig, Figure):
        fig.tight_layout()

    if title:
        ax.set_title(title)

    if show_plot:
        plt.show()
