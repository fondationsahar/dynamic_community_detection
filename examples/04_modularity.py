"""
Computing Longitudinal Modularity

This example shows how to evaluate the quality of temporal community partitions
using longitudinal modularity. You can use this to:
- Compare different partitions
- Understand the quality metric
- Tune parameters for better results

Run this file: python 04_modularity.py
"""

from lago import LexType, LinkStream, lago_modules, longitudinal_modularity


def create_sample_linkstream():
    """Create a sample linkstream."""
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
            (3, 5, 0),
            (3, 4, 1),
            # Bridge
            (2, 3, 1),
        ]
    )
    return ls


def basic_modularity():
    """Compute modularity for detected communities."""
    print("\n" + "=" * 60)
    print("1. BASIC MODULARITY COMPUTATION")
    print("=" * 60)

    ls = create_sample_linkstream()
    communities = lago_modules(ls)

    # Compute modularity
    result = longitudinal_modularity(ls, communities)

    print(f"\nModularity: {result.value}")
    print(f"Time penalty: {result.time_penalty}")
    print(f"Modularity without penalty: {result.modularity_without_penalty}")
    print(f"Expectation type: {result.lex_type.name}")

    return result


def compare_lex_types():
    """Compare different longitudinal expectation types."""
    print("\n" + "=" * 60)
    print("2. COMPARING EXPECTATION TYPES")
    print("=" * 60)

    ls = create_sample_linkstream()
    communities = lago_modules(ls)

    print("\nModularity with different lex types:")
    print("-" * 40)

    for lex in [LexType.MM, LexType.JM, LexType.CM]:
        result = longitudinal_modularity(ls, communities, lex=lex)
        print(f"  {lex.name}: {result.value}")

    print()
    print("Interpretation:")
    print("  MM (Mean-Membership): Most flexible, general use")
    print("  JM (Joint-Membership): Favors stable communities")
    print("  CM (Coexistence): Based on node co-occurrence in time")


def manual_partition():
    """Evaluate a manually defined partition."""
    print("\n" + "=" * 60)
    print("3. EVALUATING MANUAL PARTITIONS")
    print("=" * 60)

    ls = create_sample_linkstream()

    # Define communities manually as dict of (node, time) tuples
    # Partition 1: All in one community
    all_together = {
        0: {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)}
    }

    # Partition 2: Each node in own community
    all_separate = {
        i: {(i, 0), (i, 1)} for i in range(6) if any((i, t) in ls.leaves_dict for t in [0, 1])
    }

    # Partition 3: Two groups
    two_groups = {
        0: {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)},
        1: {(3, 0), (4, 0), (5, 0), (3, 1), (4, 1)},
    }

    print("\nComparing manual partitions:")
    print("-" * 40)

    for name, partition in [
        ("All together", all_together),
        ("Two groups", two_groups),
    ]:
        result = longitudinal_modularity(ls, partition)
        print(f"  {name}: {result.value:.4f}")

    # LAGO's result
    communities = lago_modules(ls)
    result = longitudinal_modularity(ls, communities)
    print(f"  LAGO result: {result.value:.4f}")


def parameter_sensitivity():
    """Show how alpha and omega affect modularity."""
    print("\n" + "=" * 60)
    print("4. PARAMETER SENSITIVITY")
    print("=" * 60)

    ls = create_sample_linkstream()
    communities = lago_modules(ls)

    print("\nAlpha (resolution) effect:")
    print("-" * 40)
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        result = longitudinal_modularity(ls, communities, alpha=alpha)
        print(f"  alpha={alpha}: modularity={result.value:.4f}")

    print("\nOmega (time smoothness) effect:")
    print("-" * 40)
    for omega in [0.0, 1.0, 2.0, 5.0]:
        result = longitudinal_modularity(ls, communities, omega=omega)
        print(f"  omega={omega}: modularity={result.value:.4f}, penalty={result.time_penalty:.4f}")


def understanding_components():
    """Understand the components of modularity."""
    print("\n" + "=" * 60)
    print("5. UNDERSTANDING MODULARITY COMPONENTS")
    print("=" * 60)

    ls = create_sample_linkstream()
    communities = lago_modules(ls)

    # Full modularity
    full = longitudinal_modularity(ls, communities, alpha=1.0, omega=2.0)

    # Without expectation term
    no_expect = longitudinal_modularity(ls, communities, alpha=0.0, omega=2.0)

    # Without time penalty
    no_time = longitudinal_modularity(ls, communities, alpha=1.0, omega=0.0)

    print("\nModularity breakdown:")
    print("-" * 40)
    print(f"  Full modularity (α=1, ω=2): {full.value:.4f}")
    print(f"  Without expectation (α=0):  {no_expect.value:.4f}")
    print(f"  Without time penalty (ω=0): {no_time.value:.4f}")
    print()
    print("Components:")
    print(f"  Internal edges contribution: ~{no_expect.modularity_without_penalty:.4f}")
    print(f"  Expectation penalty: ~{no_expect.value - full.modularity_without_penalty:.4f}")
    print(f"  Time penalty: {full.time_penalty:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("LAGO: Longitudinal Modularity Examples")
    print("=" * 60)

    basic_modularity()
    compare_lex_types()
    manual_partition()
    parameter_sensitivity()
    understanding_components()

    print("\n" + "=" * 60)
    print("Done! See 05_visualization.py for plotting examples.")
    print("=" * 60)
