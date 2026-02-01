"""
Different Types of LinkStreams

This example shows the various types of temporal networks supported by LAGO:
0. Time Normalization (Important!) - Normalize timestamps by GCD
1. Instantaneous (default) - Interactions happen at discrete time points
2. Directed - Interactions have a direction (source -> target)
3. Continuous - Interactions have a duration
4. Delayed - Source and target can be active at different times
5. Weighted - Interactions have different strengths
6. K-partite - Nodes are divided into groups (bipartite, tripartite, etc.)

Run this file: python 02_linkstream_types.py
"""

from lago import LinkStream


def example_time_normalization():
    """
    TIME NORMALIZATION (Important!)

    LAGO expects timestamps to be normalized integers where the minimum
    time step is 1. If your timestamps have gaps, divide by their GCD.

    This is strongly recommended for:
    - Better performance (fewer empty time steps to process)
    - Correct interpretation of the omega parameter

    Example: Unix timestamps or dates should be converted to consecutive integers.
    """
    print("\n" + "=" * 60)
    print("0. TIME NORMALIZATION (Important!)")
    print("=" * 60)

    from functools import reduce
    from math import gcd

    # Original timestamps with gaps (e.g., events at hours 0, 2, 6, 8)
    original_times = [0, 2, 6, 8]
    print(f"\nOriginal times: {original_times}")

    # Compute GCD of all time differences
    time_diffs = [t2 - t1 for t1, t2 in zip(original_times[:-1], original_times[1:])]
    time_gcd = reduce(gcd, time_diffs)
    print(f"Time differences: {time_diffs}")
    print(f"GCD: {time_gcd}")

    # Normalize: divide all times by GCD
    normalized_times = [t // time_gcd for t in original_times]
    print(f"Normalized times: {normalized_times}")

    # Example with actual links
    print("\n--- Example: Normalizing a LinkStream ---")

    # Bad: Original timestamps with gaps
    raw_links = [
        (0, 1, 0),
        (1, 2, 2),
        (0, 1, 6),
        (2, 3, 8),
    ]
    print(f"Raw links: {raw_links}")

    # Good: Normalize by GCD
    times = [link[2] for link in raw_links]
    time_gcd = reduce(gcd, [t for t in times if t > 0]) if any(t > 0 for t in times) else 1
    normalized_links = [(s, t, time // time_gcd) for s, t, time in raw_links]
    print(f"Normalized links (gcd={time_gcd}): {normalized_links}")

    # Create LinkStream with normalized data
    ls = LinkStream()
    ls.add_links(normalized_links)
    print(f"\nLinkStream: {ls.nb_nodes} nodes, duration {ls.network_duration} time steps")

    return ls


def example_instantaneous():
    """
    INSTANTANEOUS LINKSTREAM (Default)

    Interactions happen at specific time points.
    Format: (source, target, time)

    Example: Text messages, emails, tweets
    """
    print("\n" + "=" * 60)
    print("1. INSTANTANEOUS LINKSTREAM")
    print("=" * 60)

    ls = LinkStream()
    ls.add_links(
        [
            (0, 1, 0),  # Alice (0) messages Bob (1) at time 0
            (1, 2, 0),  # Bob messages Carol at time 0
            (0, 1, 1),  # Alice messages Bob at time 1
            (2, 3, 2),  # Carol messages Dave at time 2
        ]
    )

    print(f"Nodes: {ls.nb_nodes}")
    print(f"Time edges: {ls.nb_time_edges}")
    print(f"Time range: {ls.min_time} to {ls.max_time}")

    return ls


def example_weighted():
    """
    WEIGHTED LINKSTREAM

    Interactions can have different weights (importance/strength).
    Format: (source, target, time, weight)

    Example: Number of messages exchanged, transaction amounts
    """
    print("\n" + "=" * 60)
    print("2. WEIGHTED LINKSTREAM")
    print("=" * 60)

    ls = LinkStream()
    ls.add_links(
        [
            (0, 1, 0, 5),  # Strong interaction (weight=5)
            (1, 2, 0, 1),  # Weak interaction (weight=1)
            (0, 1, 1, 3),  # Medium interaction (weight=3)
            (2, 3, 2, 10),  # Very strong interaction (weight=10)
        ]
    )

    print(f"Total weight: {ls.weight}")
    print(f"Degrees: {ls.degrees}")

    return ls


def example_directed():
    """
    DIRECTED LINKSTREAM

    Interactions have a direction: source -> target.
    The reverse interaction (target -> source) is different.

    Example: Twitter follows, bank transfers, citations
    """
    print("\n" + "=" * 60)
    print("3. DIRECTED LINKSTREAM")
    print("=" * 60)

    ls = LinkStream(directed=True)
    ls.add_links(
        [
            (0, 1, 0),  # Alice follows Bob (not the same as Bob follows Alice)
            (0, 2, 0),  # Alice follows Carol
            (1, 0, 1),  # Bob follows Alice back at time 1
            (2, 0, 1),  # Carol follows Alice back at time 1
        ]
    )

    print(f"Out-degrees: {ls.degrees_out}")
    print(f"In-degrees: {ls.degrees_in}")

    # Node 0 has high out-degree (follows 2 people)
    # Node 0 also has high in-degree (followed by 2 people)

    return ls


def example_continuous():
    """
    CONTINUOUS LINKSTREAM

    Interactions have a duration (they last over time).
    Format: (source, target, start_time, duration, weight)

    Example: Phone calls, meetings, co-location
    """
    print("\n" + "=" * 60)
    print("4. CONTINUOUS LINKSTREAM")
    print("=" * 60)

    ls = LinkStream(continuous=True)
    ls.add_links(
        [
            # (source, target, start_time, duration, weight)
            (0, 1, 0, 3),  # Alice and Bob meet from time 0 to 3
            (1, 2, 2, 2),  # Bob and Carol meet from time 2 to 4
            (0, 2, 5, 1),  # Alice and Carol meet briefly at time 5
        ]
    )

    print(f"Nodes: {ls.nb_nodes}")
    print(f"Network duration: {ls.network_duration}")

    # Note: Continuous linkstreams automatically handle overlapping intervals

    return ls


def example_delayed():
    """
    DELAYED LINKSTREAM

    Source and target can be active at different times.
    Format: (source, target, source_time, target_time)

    Example: Postal mail, asynchronous communication, causal chains
    """
    print("\n" + "=" * 60)
    print("5. DELAYED LINKSTREAM")
    print("=" * 60)

    ls = LinkStream(delayed=True)
    ls.add_links(
        [
            # (source, target, source_time, target_time)
            (0, 1, 0, 2),  # Alice sends at time 0, Bob receives at time 2
            (1, 2, 3, 5),  # Bob sends at time 3, Carol receives at time 5
            (0, 2, 1, 6),  # Alice sends at time 1, Carol receives at time 6
        ]
    )

    print(f"Nodes: {ls.nb_nodes}")
    print(f"Time range: {ls.min_time} to {ls.max_time}")

    return ls


def example_kpartite():
    """
    K-PARTITE LINKSTREAM

    Nodes are divided into k disjoint groups (partitions).
    Edges only occur between nodes of different partitions.

    - Bipartite (k=2): Users & Products, Authors & Papers
    - Tripartite (k=3): Users, Movies, Actors

    Use set_partite() with a mapping: {node_id: partition_id}
    """
    print("\n" + "=" * 60)
    print("6. K-PARTITE LINKSTREAM")
    print("=" * 60)

    # BIPARTITE EXAMPLE: Users (0,1,2) and Products (3,4,5)
    print("\n--- Bipartite: Users & Products ---")
    ls_bipartite = LinkStream()

    # Define which partition each node belongs to
    # Users (nodes 0,1,2) -> partition 0
    # Products (nodes 3,4,5) -> partition 1
    partite_mapping = {
        0: 0,
        1: 0,
        2: 0,  # Users
        3: 1,
        4: 1,
        5: 1,  # Products
    }
    ls_bipartite.set_partite(partite_mapping)

    # Edges go between users and products
    ls_bipartite.add_links(
        [
            (0, 3, 0),  # User 0 buys Product 3 at time 0
            (0, 4, 1),  # User 0 buys Product 4 at time 1
            (1, 3, 1),  # User 1 buys Product 3 at time 1
            (1, 5, 2),  # User 1 buys Product 5 at time 2
            (2, 4, 2),  # User 2 buys Product 4 at time 2
        ]
    )

    print(f"  Nodes: {ls_bipartite.nb_nodes}")
    print(f"  Partitions: {set(partite_mapping.values())}")
    print(f"  Users (partition 0): {[n for n, p in partite_mapping.items() if p == 0]}")
    print(f"  Products (partition 1): {[n for n, p in partite_mapping.items() if p == 1]}")

    # TRIPARTITE EXAMPLE: Authors (0,1), Papers (2,3), Venues (4,5)
    print("\n--- Tripartite: Authors, Papers, Venues ---")
    ls_tripartite = LinkStream()

    partite_mapping_tri = {
        0: 0,
        1: 0,  # Authors
        2: 1,
        3: 1,  # Papers
        4: 2,
        5: 2,  # Venues
    }
    ls_tripartite.set_partite(partite_mapping_tri)

    ls_tripartite.add_links(
        [
            (0, 2, 0),  # Author 0 writes Paper 2 at time 0
            (1, 2, 0),  # Author 1 co-writes Paper 2 at time 0
            (2, 4, 1),  # Paper 2 published at Venue 4 at time 1
            (0, 3, 2),  # Author 0 writes Paper 3 at time 2
            (3, 5, 3),  # Paper 3 published at Venue 5 at time 3
        ]
    )

    print(f"  Nodes: {ls_tripartite.nb_nodes}")
    print(f"  Authors (0): {[n for n, p in partite_mapping_tri.items() if p == 0]}")
    print(f"  Papers (1): {[n for n, p in partite_mapping_tri.items() if p == 1]}")
    print(f"  Venues (2): {[n for n, p in partite_mapping_tri.items() if p == 2]}")

    return ls_bipartite


def example_from_file():
    """
    LOADING FROM FILE

    You can load linkstreams from text files with various formats.
    """
    print("\n" + "=" * 60)
    print("7. LOADING FROM FILE")
    print("=" * 60)

    # Example file content (not actually created):
    example_content = """
    # Example file formats:
    
    # Simple: source target time
    # 0 1 0
    # 1 2 0
    # 0 1 1
    
    # With weight: source target time weight
    # 0 1 0 5
    # 1 2 0 3
    
    # Continuous: source target time_start duration [weight]
    # 0 1 0 3
    # 1 2 2 2
    
    # Delayed: source target source_time target_time [weight]
    # 0 1 0 2
    # 1 2 3 5
    """

    print("To load from file:")
    print()
    print("  # Simple format")
    print("  ls = LinkStream()")
    print('  ls.read_txt("data.txt", columns_order=["source", "target", "time"])')
    print()
    print("  # With weights")
    print('  ls.read_txt("data.txt", columns_order=["source", "target", "time", "weight"])')
    print()
    print("  # Directed")
    print("  ls = LinkStream(directed=True)")
    print('  ls.read_txt("data.txt", columns_order=["source", "target", "time"])')
    print()
    print("  # Continuous")
    print("  ls = LinkStream(continuous=True)")
    print('  ls.read_txt("data.txt", columns_order=["source", "target", "time_start", "duration"])')


def example_combining_options():
    """
    COMBINING OPTIONS

    You can combine directed + weighted, etc.
    """
    print("\n" + "=" * 60)
    print("8. COMBINING OPTIONS")
    print("=" * 60)

    # Directed + Weighted
    ls = LinkStream(directed=True)
    ls.add_links(
        [
            (0, 1, 0, 10),  # Alice sends $10 to Bob at time 0
            (1, 0, 1, 5),  # Bob sends $5 to Alice at time 1
            (0, 2, 1, 20),  # Alice sends $20 to Carol at time 1
        ]
    )

    print("Directed + Weighted LinkStream:")
    print(f"  Out-degrees (total sent): {ls.degrees_out}")
    print(f"  In-degrees (total received): {ls.degrees_in}")

    return ls


if __name__ == "__main__":
    print("=" * 60)
    print("LAGO: Different Types of LinkStreams")
    print("=" * 60)

    example_time_normalization()
    example_instantaneous()
    example_weighted()
    example_directed()
    example_continuous()
    example_delayed()
    example_kpartite()
    example_from_file()
    example_combining_options()

    print("\n" + "=" * 60)
    print("Done! See the code for more details.")
    print("=" * 60)
