"""
Example demonstrating the simplified longitudinal_plotting API.

The new API consolidates related configuration into 4 main methods:
1. configure_nodes()      - All node-related settings
2. configure_edges()      - All edge-related settings
3. configure_communities() - All community-related settings
4. configure_display()    - Global display settings (axis, labels, padding)

This simplifies usage compared to the original API which required
multiple separate method calls for each configuration aspect.
"""

import json

from lago import LinkStream

from . import LongitudinalPlot


def example_simplified_api():
    """
    Example using the new simplified API.

    The simplified API follows a clear structure:
    1. Initiate plot with linkstream
    2. Declare nodes (labels, focus, ordering, style)
    3. Declare edges (activity, style)
    4. Declare communities (colors, focus, style)
    5. Declare global features (axis padding, labels)
    """

    # === STEP 1: Initialize with linkstream ===
    path = "path/to/linkstream.txt"
    my_linkstream = LinkStream(directed=True)
    my_linkstream.read_txt(
        path=path,
        columns_order=["source", "target", "time", "weight"],
    )

    # Load communities
    with open("path/to/communities.json") as file:
        dynamic_communities = json.load(file)

    dynamic_communities = {
        key: [(node, time) for node, time in elems] for key, elems in dynamic_communities.items()
    }

    # Node labels mapping
    node_labels = ["Node A", "Node B", "Node C"]  # etc.
    node_list = [0, 1, 2]  # Node IDs

    # Create plot
    plot = LongitudinalPlot(my_linkstream, width=1600, height=1200)

    # === STEP 2: Configure nodes ===
    plot.configure_nodes(
        nodes=node_list,
        labels=node_labels,
        focus=None,  # Optional: list of node IDs to focus
        auto_ordering=True,  # Automatically reorder nodes to minimize edge crossings
        fontsize=8,
        label_padding=-15,
        linewidth=0.25,
        linewidth_focus=0.5,
    )

    # === STEP 3: Configure edges ===
    plot.configure_edges(
        show_edges=False,  # Don't show arc edges
        show_activity=True,  # Show edge activity markers
        activity_width=0.9,
        activity_height=0.5,
        activity_alpha=0.3,
    )

    # === STEP 4: Configure communities ===
    plot.configure_communities(
        communities=dynamic_communities,
        max_shown=10,  # Maximum communities to display
        color_palette="tab20",  # Color palette
        height=0.75,  # Community rectangle height
        # Optional focus settings:
        # focus_communities={'com_1': 'red', 'com_2': 'blue'},
        # show_unfocused=True,
        # unfocused_style='lighter',
    )

    # === STEP 5: Configure display ===
    plot.configure_display(
        padding_bottom=0.5,
        padding_top=1.0,
        show_xlabel=False,
        show_ylabel=False,
        show_yticks=True,
    )

    # === Draw and save ===
    fig, ax = plot.draw(
        return_ax=True
    )  # Usefull for adding extra plotting stuff out of main function
    plot.save("output.png", dpi=750)

    return plot


def example_original_api_equivalent():
    """
    The same example using the original (more verbose) API.

    This shows what the simplified API is consolidating.
    """

    # === STEP 1: Initialize with linkstream ===
    path = "path/to/linkstream.txt"
    my_linkstream = LinkStream(directed=True)
    my_linkstream.read_txt(
        path=path,
        columns_order=["source", "target", "time", "weight"],
    )

    with open("path/to/communities.json") as file:
        dynamic_communities = json.load(file)

    dynamic_communities = {
        key: [(node, time) for node, time in elems] for key, elems in dynamic_communities.items()
    }

    node_labels = ["Node A", "Node B", "Node C"]
    node_list = [0, 1, 2]

    plot = LongitudinalPlot(my_linkstream, width=1600, height=1200)

    # === STEP 2: Configure nodes (scattered across multiple calls) ===
    plot.set_nodes(
        node_list,
        labels=node_labels,
        longitudinal_nodes_margin=0,
    )
    plot.auto_node_ordering(True)
    plot.set_node_style(fontsize=8, padding=-15)

    # === STEP 3: Configure edges (scattered across multiple calls) ===
    plot.toggle_edges(False)
    plot.toggle_edge_activity(True)
    plot.set_edge_activity_style(
        width=0.9,
        height=0.5,
        alpha=0.3,
    )

    # === STEP 4: Configure communities (scattered across multiple calls) ===
    plot.set_communities(
        communities=dynamic_communities,
        max_shown=10,
    )
    plot.set_color_palette("tab20")
    plot.set_community_style(height=0.75)
    # Optional: plot.set_focus_communities(...)
    # Optional: plot.set_background_community_color(...)
    # Optional: plot.set_unfocused_style(...)

    # === STEP 5: Configure display (scattered across multiple calls) ===
    plot.set_axis_padding(bottom=0.5, top=1.0)
    plot.set_labels(
        display_xlabel=False,
        display_ylabel=False,
        display_yticks=True,
    )

    # === Draw and save ===
    fig, ax = plot.draw(return_ax=True)
    plot.save("output.png", dpi=750)

    return plot


def comparison_summary():
    """
    Summary of API consolidation:

    Original API (scattered):
    - set_nodes() + auto_node_ordering() + set_node_style() + set_focus_nodes()
    - toggle_edges() + toggle_edge_activity() + set_edge_style() + set_edge_activity_style()
    - set_communities() + set_color_palette() + set_community_style() + set_focus_communities()
      + set_background_community_color() + set_monochrome() + set_unfocused_style()
    - set_axis_padding() + set_labels()

    Simplified API (consolidated):
    - configure_nodes()      - All node settings in one call
    - configure_edges()      - All edge settings in one call
    - configure_communities() - All community settings in one call
    - configure_display()    - All display settings in one call

    Both APIs are fully compatible - you can mix them as needed.
    The original methods are still available for fine-grained control.
    """
    pass


if __name__ == "__main__":
    # Note: This file is for documentation purposes.
    # It requires actual data files to run.
    print("See the docstrings and examples for usage patterns.")
