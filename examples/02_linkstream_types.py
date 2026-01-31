"""
Different Types of LinkStreams

This example shows the various types of temporal networks supported by LAGO:
1. Instantaneous (default) - Interactions happen at discrete time points
2. Directed - Interactions have a direction (source -> target)
3. Continuous - Interactions have a duration
4. Delayed - Source and target can be active at different times
5. Weighted - Interactions have different strengths

Run this file: python 02_linkstream_types.py
"""

from lago import LinkStream


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


def example_from_file():
    """
    LOADING FROM FILE

    You can load linkstreams from text files with various formats.
    """
    print("\n" + "=" * 60)
    print("6. LOADING FROM FILE")
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
    print("7. COMBINING OPTIONS")
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

    example_instantaneous()
    example_weighted()
    example_directed()
    example_continuous()
    example_delayed()
    example_from_file()
    example_combining_options()

    print("\n" + "=" * 60)
    print("Done! See the code for more details.")
    print("=" * 60)
