"""
Advanced Visualization API Example

This example demonstrates the full visualization API for LongitudinalModulesPlot.
For simpler examples, see 05_visualization.py first.

The API consolidates configuration into 4 main methods:
1. configure_nodes()       - All node-related settings
2. configure_edges()       - All edge-related settings
3. configure_communities() - All community-related settings
4. configure_display()     - Global display settings (axis, labels, padding)

This simplifies usage compared to the original API which required
multiple separate method calls for each configuration aspect.
"""

from lago import LinkStream, lago_modules
from lago.viz import LongitudinalModulesPlot


def example_with_lago_modules():
    """
    Most common use case: detect communities and visualize them.

    This is the recommended approach for most users.
    """
    print("\n" + "=" * 60)
    print("1. VISUALIZING lago_modules RESULTS")
    print("=" * 60)

    # Create linkstream
    ls = LinkStream()
    ls.add_links(
        [
            # Dense group 1
            (0, 1, 0),
            (1, 2, 0),
            (0, 2, 0),
            (0, 1, 1),
            (1, 2, 1),
            # Dense group 2
            (3, 4, 0),
            (4, 5, 0),
            (3, 4, 1),
            (4, 5, 1),
            # Evolution
            (2, 3, 2),
            (0, 1, 2),
            (3, 4, 2),
        ]
    )

    # Detect communities
    communities = lago_modules(ls)

    # Create plot with TimeModules (simplest API)
    plot = LongitudinalModulesPlot(communities, linkstream=ls, width=1200, height=600)

    # Configure with simplified API
    plot.configure_nodes(
        labels={0: "Alice", 1: "Bob", 2: "Carol", 3: "Dave", 4: "Eve", 5: "Frank"},
        auto_ordering=True,
        fontsize=10,
    )

    plot.configure_edges(
        show_activity=True,
        activity_alpha=0.4,
    )

    plot.configure_communities(
        color_palette="tab10",
        height=0.75,
    )

    plot.configure_display(
        padding_bottom=0.5,
        padding_top=0.5,
        show_xlabel=True,
        show_ylabel=True,
    )

    plot.draw()
    plot.save("lago_result.png", dpi=150)
    print("\nSaved: lago_result.png")

    return plot


def example_simplified_api():
    """
    Full example using the simplified API.

    The simplified API follows a clear structure:
    1. Create plot with TimeModules (and optional linkstream)
    2. configure_nodes() - labels, focus, ordering, style
    3. configure_edges() - visibility, activity markers, style
    4. configure_communities() - colors, focus, style
    5. configure_display() - axis, padding, labels
    """
    print("\n" + "=" * 60)
    print("2. FULL SIMPLIFIED API EXAMPLE")
    print("=" * 60)

    # Create sample data
    ls = LinkStream()
    ls.add_links(
        [
            (0, 1, 0),
            (1, 2, 0),
            (0, 1, 1),
            (2, 3, 1),
            (0, 1, 2),
            (2, 3, 2),
        ]
    )
    communities = lago_modules(ls)

    # Create plot with custom size
    plot = LongitudinalModulesPlot(communities, linkstream=ls, width=1600, height=1200)

    # === Configure nodes ===
    plot.configure_nodes(
        nodes=[0, 1, 2, 3],  # Explicit node list (optional)
        labels={0: "A", 1: "B", 2: "C", 3: "D"},  # Node labels
        focus=None,  # Optional: list of node IDs to highlight
        auto_ordering=True,  # Reorder nodes to minimize edge crossings
        fontsize=8,
        label_padding=-15,
        linewidth=0.25,
        linewidth_focus=0.5,
    )

    # === Configure edges ===
    plot.configure_edges(
        show_edges=False,  # Don't show arc edges
        show_activity=True,  # Show edge activity markers
        activity_width=0.9,
        activity_height=0.5,
        activity_alpha=0.3,
    )

    # === Configure communities ===
    plot.configure_communities(
        max_shown=10,  # Maximum communities to display
        color_palette="tab20",  # Color palette
        height=0.75,  # Community rectangle height
        # Optional focus settings:
        # focus_communities={'com_1': 'red', 'com_2': 'blue'},
        # show_unfocused=True,
        # unfocused_style='lighter',
    )

    # === Configure display ===
    plot.configure_display(
        padding_bottom=0.5,
        padding_top=1.0,
        show_xlabel=False,
        show_ylabel=False,
        show_yticks=True,
    )

    # === Draw and save ===
    fig, ax = plot.draw(return_ax=True)
    plot.save("simplified_api.png", dpi=300)
    print("\nSaved: simplified_api.png")

    return plot


def example_original_api_equivalent():
    """
    The same result using the original (more verbose) API.

    This shows what the simplified API consolidates.
    Both APIs are compatible - you can mix them as needed.
    """
    print("\n" + "=" * 60)
    print("3. ORIGINAL API (for reference)")
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
    communities = lago_modules(ls)

    plot = LongitudinalModulesPlot(communities, linkstream=ls, width=1600, height=1200)

    # Configure nodes (scattered across multiple calls)
    plot.set_nodes([0, 1, 2, 3], labels={0: "A", 1: "B", 2: "C", 3: "D"})
    plot.auto_node_ordering(True)
    plot.set_node_style(fontsize=8, padding=-15)

    # Configure edges (scattered across multiple calls)
    plot.toggle_edges(False)
    plot.toggle_edge_activity(True)
    plot.set_edge_activity_style(width=0.9, height=0.5, alpha=0.3)

    # Configure communities (scattered across multiple calls)
    plot.set_color_palette("tab20")
    plot.set_community_style(height=0.75)

    # Configure display (scattered across multiple calls)
    plot.set_axis_padding(bottom=0.5, top=1.0)
    plot.set_labels(display_xlabel=False, display_ylabel=False, display_yticks=True)

    plot.draw()
    plot.save("original_api.png", dpi=300)
    print("\nSaved: original_api.png")

    return plot


def api_comparison_summary():
    """
    Summary of API consolidation.

    Original API (scattered):
    - set_nodes() + auto_node_ordering() + set_node_style() + set_focus_nodes()
    - toggle_edges() + toggle_edge_activity() + set_edge_style() + set_edge_activity_style()
    - set_communities() + set_color_palette() + set_community_style() + set_focus_communities()
      + set_background_community_color() + set_monochrome() + set_unfocused_style()
    - set_axis_padding() + set_labels()

    Simplified API (consolidated):
    - configure_nodes()       - All node settings in one call
    - configure_edges()       - All edge settings in one call
    - configure_communities() - All community settings in one call
    - configure_display()     - All display settings in one call

    Both APIs are fully compatible - you can mix them as needed.
    The original methods are still available for fine-grained control.
    """
    print("\n" + "=" * 60)
    print("4. API COMPARISON")
    print("=" * 60)
    print(api_comparison_summary.__doc__)


if __name__ == "__main__":
    print("=" * 60)
    print("LAGO: Advanced Visualization Examples")
    print("=" * 60)
    print("\nNOTE: This creates PNG files in the current directory.")
    print("      For simpler examples, see 05_visualization.py")

    try:
        example_with_lago_modules()
        example_simplified_api()
        example_original_api_equivalent()
        api_comparison_summary()

        print("\n" + "=" * 60)
        print("Done! Check the generated PNG files.")
        print("=" * 60)
    except ImportError as e:
        print(f"\nError: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")
