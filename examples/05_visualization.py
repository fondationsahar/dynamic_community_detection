"""
Visualization with LAGO

This example shows how to create visualizations of temporal communities
using LongitudinalModulesPlot. The visualization shows nodes on the y-axis,
time on the x-axis, and communities as colored rectangles.

Run this file: python 05_visualization.py
NOTE: Requires matplotlib to be installed
"""

from lago import LinkStream, lago_modules
from lago.viz import LongitudinalModulesPlot


def create_sample_data():
    """Create sample linkstream and communities."""
    ls = LinkStream()
    ls.add_links(
        [
            # Group 1: nodes 0, 1, 2
            (0, 1, 0),
            (1, 2, 0),
            (0, 2, 0),
            (0, 1, 1),
            (1, 2, 1),
            (0, 1, 2),
            (1, 2, 2),
            # Group 2: nodes 3, 4
            (3, 4, 0),
            (3, 4, 1),
            (3, 4, 2),
            # Bridge at time 3
            (2, 3, 3),
            # Evolution: groups merge
            (0, 3, 4),
            (1, 4, 4),
            (0, 1, 4),
            (3, 4, 4),
        ]
    )
    communities = lago_modules(ls)
    return ls, communities


def basic_plot():
    """Create a basic visualization."""
    print("\n" + "=" * 60)
    print("1. BASIC PLOT")
    print("=" * 60)

    ls, communities = create_sample_data()

    # Create plot from TimeModules (simplest approach)
    plot = LongitudinalModulesPlot(communities)

    # Draw
    plot.draw()

    # Save
    plot.save("basic_plot.png")
    print("\nSaved: basic_plot.png")


def plot_with_linkstream():
    """Plot with linkstream to show edge activity."""
    print("\n" + "=" * 60)
    print("2. PLOT WITH EDGE ACTIVITY")
    print("=" * 60)

    ls, communities = create_sample_data()

    # Create plot with linkstream (required for edge visualization)
    plot = LongitudinalModulesPlot(communities, linkstream=ls)

    # Configure edges to show activity
    plot.configure_edges(
        show_edges=False,  # Don't show arc connections
        show_activity=True,  # Show when edges are active
        activity_alpha=0.5,
    )

    plot.draw()
    plot.save("plot_with_activity.png")
    print("\nSaved: plot_with_activity.png")


def customized_plot():
    """Create a fully customized visualization."""
    print("\n" + "=" * 60)
    print("3. CUSTOMIZED PLOT")
    print("=" * 60)

    ls, communities = create_sample_data()

    # Create plot with custom size
    plot = LongitudinalModulesPlot(communities, linkstream=ls, width=1200, height=600)

    # Configure nodes
    plot.configure_nodes(
        labels={0: "Alice", 1: "Bob", 2: "Carol", 3: "Dave", 4: "Eve"},
        auto_ordering=True,  # Reorder nodes to minimize crossings
        fontsize=10,
    )

    # Configure edges
    plot.configure_edges(
        show_activity=True,
        activity_width=0.8,
        activity_height=0.6,
        activity_alpha=0.3,
    )

    # Configure communities
    plot.configure_modules(
        color_palette="tab10",
        height=0.8,
    )

    # Configure display
    plot.configure_display(
        padding_bottom=0.5,
        padding_top=0.5,
        show_xlabel=True,
        show_ylabel=True,
    )

    plot.draw()
    plot.save("customized_plot.png", dpi=150)
    print("\nSaved: customized_plot.png")


def focus_on_communities():
    """Focus on specific communities or nodes."""
    print("\n" + "=" * 60)
    print("4. FOCUSED VIEW")
    print("=" * 60)

    ls, communities = create_sample_data()

    plot = LongitudinalModulesPlot(communities, linkstream=ls)

    # Focus on specific nodes
    plot.configure_nodes(
        focus=[0, 1, 2],  # Highlight nodes 0, 1, 2
        linewidth=0.3,  # Thin lines for non-focused
        linewidth_focus=1.0,  # Thick lines for focused
    )

    # Optionally focus on specific communities
    # plot.configure_modules(
    #     focus_modules={0: 'red'},  # Highlight community 0 in red
    #     show_unfocused=True,
    #     unfocused_style='lighter',
    # )

    plot.draw()
    plot.save("focused_plot.png")
    print("\nSaved: focused_plot.png")


def plot_from_dict():
    """Create plot from manual community dict."""
    print("\n" + "=" * 60)
    print("5. PLOT FROM MANUAL PARTITION")
    print("=" * 60)

    ls = LinkStream()
    ls.add_links(
        [
            (0, 1, 0),
            (1, 2, 0),
            (0, 1, 1),
            (2, 3, 1),
        ]
    )

    # Define communities as dict
    from lago import TimeModules

    manual_communities = TimeModules(
        {
            "Group A": {(0, 0), (1, 0), (0, 1), (1, 1)},
            "Group B": {(2, 0), (2, 1), (3, 1)},
        }
    )

    plot = LongitudinalModulesPlot(manual_communities, linkstream=ls)
    plot.draw()
    plot.save("manual_partition_plot.png")
    print("\nSaved: manual_partition_plot.png")


def return_matplotlib_axis():
    """Get matplotlib axis for further customization."""
    print("\n" + "=" * 60)
    print("6. ADVANCED MATPLOTLIB CUSTOMIZATION")
    print("=" * 60)

    ls, communities = create_sample_data()

    plot = LongitudinalModulesPlot(communities)

    # Get the matplotlib figure and axis
    fig, ax = plot.draw(return_ax=True)

    # Add custom annotations
    ax.annotate(
        "Communities merge here",
        xy=(3.5, 2),
        xytext=(4, 3),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9,
        color="red",
    )

    # Add title
    ax.set_title("Temporal Community Evolution", fontsize=14)

    fig.savefig("annotated_plot.png", dpi=150, bbox_inches="tight")
    print("\nSaved: annotated_plot.png")


if __name__ == "__main__":
    print("=" * 60)
    print("LAGO: Visualization Examples")
    print("=" * 60)
    print("\nNOTE: This script creates PNG files in the current directory.")
    print("      Make sure matplotlib is installed: pip install matplotlib")

    try:
        basic_plot()
        plot_with_linkstream()
        customized_plot()
        focus_on_communities()
        plot_from_dict()
        return_matplotlib_axis()

        print("\n" + "=" * 60)
        print("Done! Check the generated PNG files.")
        print("=" * 60)
    except ImportError as e:
        print(f"\nError: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")
