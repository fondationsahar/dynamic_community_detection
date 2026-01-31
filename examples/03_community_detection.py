"""
Community Detection with LAGO

This example shows how to detect temporal communities using lago_modules,
explore the results, and understand the different parameters.

Run this file: python 03_community_detection.py
"""

from lago import LinkStream, lago_modules


def create_sample_linkstream():
    """Create a sample linkstream with clear community structure."""
    ls = LinkStream()
    ls.add_links(
        [
            # Community A: nodes 0, 1, 2 (densely connected at times 0-2)
            (0, 1, 0),
            (1, 2, 0),
            (0, 2, 0),
            (0, 1, 1),
            (1, 2, 1),
            (0, 2, 1),
            (0, 1, 2),
            (1, 2, 2),
            # Community B: nodes 3, 4, 5 (densely connected at times 0-2)
            (3, 4, 0),
            (4, 5, 0),
            (3, 5, 0),
            (3, 4, 1),
            (4, 5, 1),
            (3, 5, 1),
            (3, 4, 2),
            (4, 5, 2),
            # Bridge between communities at time 2
            (2, 3, 2),
        ]
    )
    return ls


def basic_detection():
    """Basic community detection example."""
    print("\n" + "=" * 60)
    print("1. BASIC COMMUNITY DETECTION")
    print("=" * 60)

    ls = create_sample_linkstream()

    # Detect communities with default parameters
    communities = lago_modules(ls)

    print(f"\nFound {communities.nb_modules} communities")
    print(f"Over {communities.nb_times} time steps")
    print(f"Involving {communities.nb_nodes} nodes")

    # List all communities
    print("\nCommunity details:")
    for module in communities.iter_modules():
        print(f"  Community {module.label}:")
        print(f"    Nodes: {module.nodes}")
        print(f"    Duration: {module.duration} time steps")
        print(f"    Cohesion: {module.cohesion:.2f}")

    return communities


def explore_node_trajectory():
    """Track how a node's community membership changes over time."""
    print("\n" + "=" * 60)
    print("2. NODE TRAJECTORY")
    print("=" * 60)

    ls = create_sample_linkstream()
    communities = lago_modules(ls)

    # Track node 0
    node_id = 0
    print(f"\nTracking node {node_id}:")

    # Get all memberships
    memberships = communities.get_modules_of_node(node_id)
    for m in memberships:
        print(f"  In community {m.module} during: {sorted(m.times)}")

    # Get trajectory over time
    trajectory = communities.get_node_trajectory(node_id)
    print(f"\n  Time -> Community: {trajectory}")

    # Stability score (1.0 = never switches, lower = more switches)
    stability = communities.get_stability_score(node_id)
    print(f"  Stability score: {stability:.2f}")


def time_snapshots():
    """Get community structure at specific time points."""
    print("\n" + "=" * 60)
    print("3. TIME SNAPSHOTS")
    print("=" * 60)

    ls = create_sample_linkstream()
    communities = lago_modules(ls)

    for time in range(3):
        print(f"\nAt time {time}:")

        # Which nodes are in which community?
        membership = communities.get_nodes_modules_membership_at_time(time)
        for node, comm in sorted(membership.items()):
            print(f"  Node {node} -> Community {comm}")


def parameter_comparison():
    """Compare different parameter settings."""
    print("\n" + "=" * 60)
    print("4. PARAMETER COMPARISON")
    print("=" * 60)

    ls = create_sample_linkstream()

    # Different omega values (temporal smoothness)
    print("\nEffect of omega (temporal smoothness):")
    for omega in [0.5, 2, 5]:
        communities = lago_modules(ls, omega=omega)
        print(f"  omega={omega}: {communities.nb_modules} communities")

    # Different alpha values (resolution)
    print("\nEffect of alpha (resolution):")
    for alpha in [0.5, 1, 2]:
        communities = lago_modules(ls, alpha=alpha)
        print(f"  alpha={alpha}: {communities.nb_modules} communities")

    # Different lex types
    print("\nEffect of lex (expectation type):")
    for lex in ["MM", "JM"]:
        communities = lago_modules(ls, lex=lex)
        print(f"  lex='{lex}': {communities.nb_modules} communities")


def save_and_load():
    """Save and load communities to/from files."""
    print("\n" + "=" * 60)
    print("5. SAVE AND LOAD COMMUNITIES")
    print("=" * 60)

    ls = create_sample_linkstream()
    communities = lago_modules(ls)

    print("\nSaving communities to files:")
    print("  (Files not actually created in this example)")
    print()
    print("  # Save as JSON")
    print('  communities.to_json("communities.json")')
    print()
    print("  # Save as CSV")
    print('  communities.to_csv("communities.csv")')
    print()
    print("  # Save as text")
    print('  communities.to_txt("communities.txt")')
    print()
    print("Loading communities from files:")
    print()
    print("  from lago import TimeModules")
    print('  tm = TimeModules(path="communities.json")')


def module_analysis():
    """Analyze individual modules in detail."""
    print("\n" + "=" * 60)
    print("6. MODULE ANALYSIS")
    print("=" * 60)

    ls = create_sample_linkstream()
    communities = lago_modules(ls)

    # Get a specific module
    if communities.nb_modules > 0:
        module = communities.get_module(0)

        print(f"\nAnalyzing Module {module.label}:")
        print(f"  Nodes: {module.nodes}")
        print(f"  Times: {module.times}")
        print(f"  Size: {module.size} nodes")
        print(f"  Duration: {module.duration} time steps")
        print(f"  Time range: {module.time_range}")
        print(f"  Cohesion: {module.cohesion:.2f}")

        # Node segments within module
        segments = module.get_node_segments()
        print("\n  Node time segments:")
        for node, segs in segments.items():
            print(f"    Node {node}: {[(s.start, s.end) for s in segs]}")


if __name__ == "__main__":
    print("=" * 60)
    print("LAGO: Community Detection Examples")
    print("=" * 60)

    basic_detection()
    explore_node_trajectory()
    time_snapshots()
    parameter_comparison()
    save_and_load()
    module_analysis()

    print("\n" + "=" * 60)
    print("Done! See 04_modularity.py for quality metrics.")
    print("=" * 60)
