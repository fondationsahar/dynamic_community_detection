"""LinkStream: A class for representing temporal networks."""

from __future__ import annotations

import sys
import warnings
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from lago.algorithm._internal._leaf import Leaf
from lago.algorithm._internal._time_edge import TimeEdge
from lago.core.enums import LinkStreamMode

# NOTE Add a function to preprocess time scale:
# min should be 0 and min step should be 1 (use pgcd etc.)

# TODO add an option for merging recurrent modules: same nodes at different times
# NOTE a priori LM will be the same for MM, not necessary for JM: check that.
# TODO when the concept of TimeModules will have its own class with output formats as functions


class LinkStream:
    """Represents a temporal network (link stream).

    A link stream is a collection of temporal edges between nodes, where each edge
    is associated with a time or time interval.

    Attributes:
        continuous: If True, links have duration (continuous time model).
        directed: If True, edges are directed.
        delayed: If True, source and target can have different timestamps.
        partite_mapping: Maps nodes to their partition (for k-partite networks).
    """

    def __init__(
        self,
        continuous: bool = False,
        directed: bool = False,
        delayed: bool = False,
        partite_mapping: dict[int, int] | None = None,
    ) -> None:
        """Initialize a LinkStream.

        Args:
            continuous: If True, links have duration.
            directed: If True, edges are directed.
            delayed: If True, source and target can have different timestamps.
            partite_mapping: Maps nodes to their partition. Defaults to empty dict.

        Raises:
            ValueError: If both continuous and delayed are True.
        """
        # Validate mode combinations
        if continuous and delayed:
            raise ValueError(
                "LinkStream cannot be both continuous and delayed. Choose one mode or the other."
            )

        self.continuous: bool = continuous
        self.directed: bool = directed
        self.delayed: bool = delayed
        self.partite_mapping: dict[int, int] = (
            partite_mapping if partite_mapping is not None else {}
        )

        self.nodes: set[int] = set()

        self.degrees: dict[int, float] = {}
        self.degrees_in: dict[int, float] = {}
        self.degrees_out: dict[int, float] = {}

        self.min_time: int = sys.maxsize
        self.max_time: int = -sys.maxsize

        self.leaves_dict: dict[tuple[int, int], Leaf] = {}
        self.nodes_durations: dict[int, float] = {}
        self.nb_edges: float = 0
        self.weight: float = 0
        self.network_duration: int = 0

        self.time_instants: set[int] = set()

    # =========================================================================
    # Helper Methods (extracted to reduce duplication)
    # =========================================================================

    def _validate_time(self, time: int | float, field_name: str = "time") -> int:
        """Validate that a time value is an integer.

        Args:
            time: The time value to validate.
            field_name: Name of the field for error messages.

        Returns:
            The validated time as an integer.

        Raises:
            TypeError: If time is not an integer type.
        """
        if not isinstance(time, int):
            if isinstance(time, float):
                if time.is_integer():
                    return int(time)
                raise TypeError(
                    f"{field_name} must be an integer, got float {time}. "
                    "Times must be discrete integer values."
                )
            raise TypeError(
                f"{field_name} must be an integer, got {type(time).__name__}. "
                "Times must be discrete integer values."
            )
        return time

    def _ensure_leaf_exists(self, node: int, time: int) -> Leaf:
        """Get or create a leaf for the given node-time pair.

        Args:
            node: The node identifier.
            time: The timestamp.

        Returns:
            The Leaf object for this node-time pair.
        """
        key = (node, time)
        if key not in self.leaves_dict:
            self.leaves_dict[key] = Leaf(node=node, time=time)
        return self.leaves_dict[key]

    def _update_degree_undirected(self, node: int, weight: float) -> None:
        """Update degree for undirected graph.

        Args:
            node: The node identifier.
            weight: The edge weight to add.
        """
        if node not in self.degrees:
            self.degrees[node] = 0
        self.degrees[node] += weight

    def _update_degree_directed(self, source: int, target: int, weight: float) -> None:
        """Update in/out degrees for directed graph.

        Args:
            source: The source node identifier.
            target: The target node identifier.
            weight: The edge weight to add.
        """
        if source not in self.degrees_out:
            self.degrees_out[source] = 0
        self.degrees_out[source] += weight

        if target not in self.degrees_in:
            self.degrees_in[target] = 0
        self.degrees_in[target] += weight

    def _add_edge_undirected(
        self,
        source_leaf: Leaf,
        target_leaf: Leaf,
        weight: float,
        duration: int = 1,
    ) -> None:
        """Add an undirected edge between two leaves.

        Args:
            source_leaf: The source leaf.
            target_leaf: The target leaf.
            weight: The edge weight.
            duration: The edge duration (for continuous links).
        """
        source_leaf.topo_neighbors.add(
            TimeEdge(target=target_leaf, weight=weight, duration=duration)
        )
        target_leaf.topo_neighbors.add(
            TimeEdge(target=source_leaf, weight=weight, duration=duration)
        )

    def _add_edge_directed(
        self,
        source_leaf: Leaf,
        target_leaf: Leaf,
        weight: float,
        duration: int = 1,
    ) -> None:
        """Add a directed edge between two leaves.

        Args:
            source_leaf: The source leaf.
            target_leaf: The target leaf.
            weight: The edge weight.
            duration: The edge duration (for continuous links).
        """
        source_leaf.topo_neighbors.add(
            TimeEdge(target=target_leaf, weight=weight, duration=duration)
        )
        target_leaf.topo_neighbors_from.add(
            TimeEdge(target=source_leaf, weight=weight, duration=duration)
        )

    # =========================================================================
    # Main Link Addition Methods
    # =========================================================================

    def add_links(self, links: Sequence[tuple[int, ...]]) -> None:
        """Add links to the stream (unified method).

        Automatically dispatches to the appropriate method based on the mode:
        - INSTANTANEOUS: (source, target, time, [weight])
        - CONTINUOUS: (source, target, time, duration, [weight])
        - DELAYED: (source, target, source_time, target_time, [weight])

        Args:
            links: Sequence of link tuples.

        Note:
            Times must be integers. For best results, normalize times so that
            the GCD of all time differences is 1.
        """
        if self.continuous:
            self._add_continuous_links_impl(links)
        elif self.delayed:
            self._add_delayed_links_impl(links)
        else:
            self._add_instantaneous_links_impl(links)

    def add_instantaneous_links(self, links: Sequence[tuple[int, ...]]) -> None:
        """Add instantaneous links to the stream.

        .. deprecated:: 1.0.0
            Use :meth:`add_links` instead. This method will be removed in a future version.

        Each link is a tuple of (source, target, time) or (source, target, time, weight).

        Args:
            links: Sequence of link tuples.
        """
        warnings.warn(
            "add_instantaneous_links is deprecated, use add_links instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._add_instantaneous_links_impl(links)

    def _add_instantaneous_links_impl(self, links: Sequence[tuple[int, ...]]) -> None:
        """Internal implementation for adding instantaneous links.

        Each link is a tuple of (source, target, time) or (source, target, time, weight).

        Args:
            links: Sequence of link tuples.

        Raises:
            TypeError: If time is not an integer.
            ValueError: If duplicate edges are detected.
        """
        # Track edges to detect duplicates
        seen_edges: set[tuple[int, int, int]] = set()

        for link in links:
            if len(link) > 3:
                source, target, time_raw, weight = link[0], link[1], link[2], link[3]
            else:
                source, target, time_raw = link[0], link[1], link[2]
                weight = 1

            # Validate node types
            if not isinstance(source, int) or not isinstance(target, int):
                raise TypeError(
                    f"Node IDs must be integers, got {type(source).__name__} and "
                    f"{type(target).__name__}. Use integer node IDs (e.g., 0, 1, 2)."
                )

            # Validate weight
            if weight < 0:
                raise ValueError(
                    f"Weight must be >= 0, got {weight} for edge ({source}, {target})."
                )

            # Validate time is an integer
            time = self._validate_time(time_raw, "time")

            # Check for duplicate edges
            if not self.directed:
                edge_key = (min(source, target), max(source, target), time)
            else:
                edge_key = (source, target, time)
            if edge_key in seen_edges:
                if self.directed:
                    raise ValueError(
                        f"Duplicate edge detected: ({source}, {target}, {time}). "
                        f"The same edge appears multiple times in your input. "
                        f"Please either:\n"
                        f"  1. Keep only one occurrence of this edge, or\n"
                        f"  2. Merge the duplicates by summing their weights manually before adding to LinkStream."
                    )
                else:
                    raise ValueError(
                        f"Duplicate edge detected: ({source}, {target}, {time}). "
                        f"The same edge appears multiple times in your input. "
                        f"Please either:\n"
                        f"  1. Keep only one occurrence of this edge, or\n"
                        f"  2. Merge the duplicates by summing their weights manually before adding to LinkStream.\n"
                        f"Note: For undirected graphs, (A, B, t) and (B, A, t) are considered the same edge."
                    )
            seen_edges.add(edge_key)

            self.nb_edges += 1
            self.weight += weight
            self.min_time = min(self.min_time, time)
            self.max_time = max(self.max_time, time)

            # Register nodes and create leaves
            for node in [source, target]:
                self.nodes.add(node)
                self._ensure_leaf_exists(node, time)

                if not self.directed:
                    self._update_degree_undirected(node, weight)

            # Update degrees and edges based on directedness
            if self.directed:
                self._update_degree_directed(source, target, weight)
                self._add_edge_directed(
                    self.leaves_dict[(source, time)],
                    self.leaves_dict[(target, time)],
                    weight,
                )
            else:
                self._add_edge_undirected(
                    self.leaves_dict[(source, time)],
                    self.leaves_dict[(target, time)],
                    weight,
                )

        self.nodes_durations = {}
        self.network_duration = self.max_time - self.min_time + 1
        self._compute_time_neighbors()

    def add_continuous_links(self, links: Sequence[tuple[int, ...]]) -> None:
        """Add continuous links (with duration) to the stream.

        .. deprecated:: 1.0.0
            Use :meth:`add_links` instead with `continuous=True` mode.
            This method will be removed in a future version.

        Each link is a tuple of (source, target, time, duration) or
        (source, target, time, duration, weight).

        Args:
            links: Sequence of link tuples.
        """
        warnings.warn(
            "add_continuous_links is deprecated, use add_links instead with continuous=True mode",
            DeprecationWarning,
            stacklevel=2,
        )
        self._add_continuous_links_impl(links)

    def _add_continuous_links_impl(self, links: Sequence[tuple[int, ...]]) -> None:
        """Internal implementation for adding continuous links.

        Each link is a tuple of (source, target, time, duration) or
        (source, target, time, duration, weight).

        Args:
            links: Sequence of link tuples.

        Raises:
            TypeError: If time, duration, or node IDs are not integers.
            ValueError: If duration <= 0 or weight < 0.

        Note:
            Times must be integers. The total weight of a continuous link
            is weight * duration. Duplicate continuous links are allowed
            (e.g., to represent repeated interactions over the same interval).
        """
        for link in links:
            if len(link) > 4:
                source, target, time_raw, duration_raw, initial_weight = (
                    link[0],
                    link[1],
                    link[2],
                    link[3],
                    link[4],
                )
            else:
                source, target, time_raw, duration_raw = link[0], link[1], link[2], link[3]
                initial_weight = 1

            # Validate node types
            if not isinstance(source, int) or not isinstance(target, int):
                raise TypeError(
                    f"Node IDs must be integers, got {type(source).__name__} and "
                    f"{type(target).__name__}. Use integer node IDs (e.g., 0, 1, 2)."
                )

            # Validate weight
            if initial_weight < 0:
                raise ValueError(
                    f"Weight must be >= 0, got {initial_weight} for edge ({source}, {target})."
                )

            # Validate time and duration are integers
            time = self._validate_time(time_raw, "time")
            duration = self._validate_time(duration_raw, "duration")

            # Validate duration is positive
            if duration <= 0:
                raise ValueError(
                    f"Duration must be > 0, got {duration} for edge ({source}, {target}, {time})."
                )

            # Weight is scaled by duration for continuous links
            weight = initial_weight * duration

            self.nb_edges += 1
            self.weight += weight
            self.min_time = min(self.min_time, time)
            self.max_time = max(self.max_time, time + duration)
            self.time_instants.add(time)
            self.time_instants.add(time + duration)

            # Register nodes and create leaves
            for node in [source, target]:
                self.nodes.add(node)
                self._ensure_leaf_exists(node, time)

                if not self.directed:
                    self._update_degree_undirected(node, weight)

            # Update degrees and edges based on directedness
            if self.directed:
                self._update_degree_directed(source, target, weight)
                self._add_edge_directed(
                    self.leaves_dict[(source, time)],
                    self.leaves_dict[(target, time)],
                    initial_weight,
                    duration,
                )
            else:
                self._add_edge_undirected(
                    self.leaves_dict[(source, time)],
                    self.leaves_dict[(target, time)],
                    initial_weight,
                    duration,
                )

        self.nodes_durations = {}
        self.network_duration = self.max_time - self.min_time + 1

        self._compute_time_neighbors()

    def add_delayed_links(self, links: Sequence[tuple[int, ...]]) -> None:
        """Add delayed links (different source/target times) to the stream.

        .. deprecated:: 1.0.0
            Use :meth:`add_links` instead with `delayed=True` mode.
            This method will be removed in a future version.

        Each link is a tuple of (source, target, source_time, target_time) or
        (source, target, source_time, target_time, weight).

        Args:
            links: Sequence of link tuples.
        """
        warnings.warn(
            "add_delayed_links is deprecated, use add_links instead with delayed=True mode",
            DeprecationWarning,
            stacklevel=2,
        )
        self._add_delayed_links_impl(links)

    def _add_delayed_links_impl(self, links: Sequence[tuple[int, ...]]) -> None:
        """Internal implementation for adding delayed links.

        Each link is a tuple of (source, target, source_time, target_time) or
        (source, target, source_time, target_time, weight).

        Args:
            links: Sequence of link tuples.

        Raises:
            TypeError: If times are not integers.
            ValueError: If duplicate edges are detected.

        Note:
            Times must be integers.
        """
        # Track edges to detect duplicates
        seen_edges: set[tuple[int, int, int, int]] = set()

        for link in links:
            if len(link) > 4:
                source, target, source_time_raw, target_time_raw, weight = (
                    link[0],
                    link[1],
                    link[2],
                    link[3],
                    link[4],
                )
            else:
                source, target, source_time_raw, target_time_raw = (
                    link[0],
                    link[1],
                    link[2],
                    link[3],
                )
                weight = 1

            # Validate node types
            if not isinstance(source, int) or not isinstance(target, int):
                raise TypeError(
                    f"Node IDs must be integers, got {type(source).__name__} and "
                    f"{type(target).__name__}. Use integer node IDs (e.g., 0, 1, 2)."
                )

            # Validate weight
            if weight < 0:
                raise ValueError(
                    f"Weight must be >= 0, got {weight} for edge ({source}, {target})."
                )

            # Validate times are integers
            source_time = self._validate_time(source_time_raw, "source_time")
            target_time = self._validate_time(target_time_raw, "target_time")

            # Check for duplicate edges
            if not self.directed:
                edge_key = (min(source, target), max(source, target), source_time, target_time)
            else:
                edge_key = (source, target, source_time, target_time)
            if edge_key in seen_edges:
                raise ValueError(
                    f"Duplicate edge detected: ({source}, {target}, {source_time}, {target_time}). "
                    f"The same edge appears multiple times in your input. "
                    f"Please either:\n"
                    f"  1. Keep only one occurrence of this edge, or\n"
                    f"  2. Merge the duplicates by summing their weights manually before adding to LinkStream.\n"
                    f"Note: For undirected graphs, (A, B, t1, t2) and (B, A, t1, t2) are considered the same edge."
                )
            seen_edges.add(edge_key)

            self.nb_edges += 1
            self.weight += weight

            # Update time bounds
            for time in [source_time, target_time]:
                self.min_time = min(self.min_time, time)
                self.max_time = max(self.max_time, time)

            # Register nodes and create leaves
            for node, time in [(source, source_time), (target, target_time)]:
                self.nodes.add(node)
                self._ensure_leaf_exists(node, time)

                if not self.directed:
                    self._update_degree_undirected(node, weight)

            # Update degrees and edges based on directedness
            if self.directed:
                self._update_degree_directed(source, target, weight)
                self._add_edge_directed(
                    self.leaves_dict[(source, source_time)],
                    self.leaves_dict[(target, target_time)],
                    weight,
                )
            else:
                self._add_edge_undirected(
                    self.leaves_dict[(source, source_time)],
                    self.leaves_dict[(target, target_time)],
                    weight,
                )

        self.nodes_durations = {}
        self.network_duration = self.max_time - self.min_time + 1
        self._compute_time_neighbors()

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _split_continuous_linkstream(self) -> None:
        """Split continuous links at time instant boundaries.

        This method restructures the leaves_dict to have separate entries
        for each time segment where continuous links exist.
        """
        # Store references to old leaves (no deep copy needed - we just iterate)
        # We can't use deepcopy due to circular references between Leaf objects
        old_leaves_items = list(self.leaves_dict.items())
        self.leaves_dict = {}

        for (node, time), leaf in old_leaves_items:
            # Split edges based on time_instants
            for time_edge in leaf.topo_neighbors:
                # NOTE: This may be a bottleneck for large networks
                # TODO: Refactor with segments instead of range
                tmp_time_instants = self.time_instants & set(
                    range(time, time + time_edge.duration + 1)
                )
                tmp_time_instants_sorted = sorted(tmp_time_instants)

                for time_start, time_end in pairwise(tmp_time_instants_sorted):
                    duration = time_end - time_start + 1
                    self._ensure_leaf_exists(node, time_start)

                    target_node = time_edge.target.node
                    self._ensure_leaf_exists(target_node, time_start)

                    self.leaves_dict[(node, time_start)].topo_neighbors.add(
                        TimeEdge(
                            target=self.leaves_dict[(target_node, time_start)],
                            weight=time_edge.weight,
                            duration=duration,
                        )
                    )

            for time_edge in leaf.topo_neighbors_from:
                tmp_time_instants = self.time_instants & set(
                    range(time, time + time_edge.duration + 1)
                )
                tmp_time_instants_sorted = sorted(tmp_time_instants)

                for time_start, time_end in pairwise(tmp_time_instants_sorted):
                    duration = time_end - time_start
                    self._ensure_leaf_exists(node, time_start)

                    target_node = time_edge.target.node
                    self._ensure_leaf_exists(target_node, time_start)

                    self.leaves_dict[(node, time_start)].topo_neighbors_from.add(
                        TimeEdge(
                            target=self.leaves_dict[(target_node, time_start)],
                            weight=time_edge.weight,
                            duration=duration,
                        )
                    )

        # Recompute time neighbors for the new leaves_dict
        self._compute_time_neighbors()

    def _compute_time_neighbors(self) -> None:
        """Compute temporal neighbors for each leaf.

        Links leaves of the same node across consecutive time steps.
        """
        for node in self.nodes:
            times = sorted({time for tmp_node, time in self.leaves_dict if tmp_node == node})
            for tm1, tm2 in pairwise(times):
                self.leaves_dict[(node, tm1)].right_time_active_neighbor = self.leaves_dict[
                    (node, tm2)
                ]
                self.leaves_dict[(node, tm2)].left_time_active_neighbor = self.leaves_dict[
                    (node, tm1)
                ]

    # =========================================================================
    # Configuration Methods
    # =========================================================================

    def set_partite(self, partite_mapping: dict[int, int]) -> None:
        """Set the partition mapping for k-partite networks.

        Args:
            partite_mapping: Maps node IDs to partition IDs.
        """
        self.partite_mapping = partite_mapping

    # =========================================================================
    # Query Methods
    # =========================================================================

    @property
    def is_weighted(self) -> bool:
        """Check if the network has non-trivial weights (any weight != 1).

        Returns:
            True if any edge has a weight different from 1, False otherwise.
        """
        for leaf in self.leaves_dict.values():
            for neighb in leaf.topo_neighbors:
                if neighb.weight != 1:
                    return True
        return False

    def get_time_links(self, include_weights: bool | None = None) -> set[tuple[int, ...]]:
        """Get all temporal links in the stream.

        Automatically returns links in the appropriate format based on the mode:
        - INSTANTANEOUS: (source, target, time[, weight])
        - CONTINUOUS: (source, target, time, duration[, weight])
        - DELAYED: (source, target, source_time, target_time[, weight])

        Args:
            include_weights: Whether to include weights in the output tuples.
                - None (default): Include weights only if network is weighted (any weight != 1)
                - True: Always include weights
                - False: Never include weights

        Returns:
            Set of link tuples in the mode-appropriate format.
        """
        # Determine whether to include weights
        if include_weights is None:
            include_weights = self.is_weighted

        if self.mode == LinkStreamMode.CONTINUOUS:
            return self._get_continuous_time_links(include_weights)
        if self.mode == LinkStreamMode.DELAYED:
            return self._get_delayed_time_links(include_weights)
        return self._get_instantaneous_time_links(include_weights)

    def _get_instantaneous_time_links(self, include_weights: bool = True) -> set[tuple[int, ...]]:
        """Get time links for instantaneous mode.

        Args:
            include_weights: Whether to include weights in the output.

        Returns:
            Set of (source, target, time[, weight]) tuples.
        """
        time_links: set[tuple[int, ...]] = set()
        for leaf in self.leaves_dict.values():
            source = leaf.node
            time = leaf.time
            for neighb in leaf.topo_neighbors:
                target = neighb.target.node
                if include_weights:
                    time_links.add((source, target, time, neighb.weight))
                else:
                    time_links.add((source, target, time))
        return time_links

    def _get_continuous_time_links(self, include_weights: bool = True) -> set[tuple[int, ...]]:
        """Get time links for continuous mode.

        Args:
            include_weights: Whether to include weights in the output.

        Returns:
            Set of (source, target, time, duration[, weight]) tuples.
        """
        time_links: set[tuple[int, ...]] = set()
        for leaf in self.leaves_dict.values():
            source = leaf.node
            time = leaf.time
            for neighb in leaf.topo_neighbors:
                target = neighb.target.node
                duration = neighb.duration
                if include_weights:
                    time_links.add((source, target, time, duration, neighb.weight))
                else:
                    time_links.add((source, target, time, duration))
        return time_links

    def _get_delayed_time_links(self, include_weights: bool = True) -> set[tuple[int, ...]]:
        """Get time links for delayed mode.

        Args:
            include_weights: Whether to include weights in the output.

        Returns:
            Set of (source, target, source_time, target_time[, weight]) tuples.
        """
        time_links: set[tuple[int, ...]] = set()
        for leaf in self.leaves_dict.values():
            source = leaf.node
            source_time = leaf.time
            for neighb in leaf.topo_neighbors:
                target = neighb.target.node
                target_time = neighb.target.time
                # Normalize order by time
                if source_time < target_time:
                    tsource, ttarget = source, target
                    tsource_time, ttarget_time = source_time, target_time
                else:
                    tsource, ttarget = target, source
                    tsource_time, ttarget_time = target_time, source_time

                if include_weights:
                    link = (tsource, ttarget, tsource_time, ttarget_time, neighb.weight)
                else:
                    link = (tsource, ttarget, tsource_time, ttarget_time)
                time_links.add(link)
        return time_links

    # =========================================================================
    # File I/O Methods
    # =========================================================================

    def _get_default_columns_for_mode(self) -> list[str]:
        """Get default column order based on the current mode.

        Returns:
            Default column names for the current mode.
        """
        if self.mode == LinkStreamMode.CONTINUOUS:
            return ["source", "target", "time_start", "duration"]
        if self.mode == LinkStreamMode.DELAYED:
            return ["source", "target", "source_time", "target_time"]
        return ["source", "target", "time"]

    def read_txt(
        self,
        path: str,
        columns_order: list[str] | None = None,
    ) -> None:
        """Read link stream from a text file.

        Args:
            path: Path to the input file.
            columns_order: List specifying column names in order.
                If None, defaults based on mode:
                - INSTANTANEOUS: ["source", "target", "time"]
                - CONTINUOUS: ["source", "target", "time_start", "duration"]
                - DELAYED: ["source", "target", "source_time", "target_time"]
                Add "weight" to include weights.

        Example:
            ```python
            # Auto-detect columns based on mode
            ls = LinkStream(continuous=True)
            ls.read_txt("links.txt")  # Expects: source target time_start duration

            # Custom columns with weight
            ls.read_txt("links.txt", columns_order=["source", "target", "time", "weight"])
            ```
        """
        if columns_order is None:
            columns_order = self._get_default_columns_for_mode()

        order_mapping = {col: idx for idx, col in enumerate(columns_order)}

        links: list[list[int | float]] = []
        with Path(path).open() as file:
            for rline in file:
                elements = rline.strip().split()
                weight: float = 1.0
                if "weight" in order_mapping:
                    weight = float(elements[order_mapping["weight"]])

                if self.mode == LinkStreamMode.CONTINUOUS:
                    nline: list[int | float] = [
                        int(elements[order_mapping["source"]]),
                        int(elements[order_mapping["target"]]),
                        int(elements[order_mapping["time_start"]]),
                        int(elements[order_mapping["duration"]]),
                        weight,
                    ]
                elif self.mode == LinkStreamMode.DELAYED:
                    nline = [
                        int(elements[order_mapping["source"]]),
                        int(elements[order_mapping["target"]]),
                        int(elements[order_mapping["source_time"]]),
                        int(elements[order_mapping["target_time"]]),
                        weight,
                    ]
                else:
                    nline = [
                        int(elements[order_mapping["source"]]),
                        int(elements[order_mapping["target"]]),
                        int(elements[order_mapping["time"]]),
                        weight,
                    ]

                links.append(nline)

        # Convert to tuples and call unified add_links (dispatches based on mode)
        link_tuples = [tuple(link) for link in links]
        self.add_links(link_tuples)

    def to_txt(self, path: str) -> None:
        """Write link stream to a text file.

        Args:
            path: Path to the output file.
        """
        with Path(path).open("w") as file:
            for triplet in self.get_time_links():
                line = " ".join(map(str, triplet))
                file.write(line + "\n")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def nb_timesteps(self) -> int:
        """Number of unique timesteps in the stream."""
        timesteps = {time for _, time in self.leaves_dict}
        return len(timesteps)

    @property
    def nb_time_edges(self) -> int:
        """Number of temporal edges in the stream."""
        return len(self.get_time_links())

    @property
    def nb_nodes(self) -> int:
        """Number of unique nodes in the stream."""
        return len(self.nodes)

    @property
    def mode(self) -> LinkStreamMode:
        """Get the current mode of the link stream as an enum.

        Returns:
            LinkStreamMode.CONTINUOUS if continuous,
            LinkStreamMode.DELAYED if delayed,
            LinkStreamMode.INSTANTANEOUS otherwise.
        """
        return LinkStreamMode.from_flags(
            continuous=self.continuous,
            delayed=self.delayed,
        )
