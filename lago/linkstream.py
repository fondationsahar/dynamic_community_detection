"""LinkStream: A class for representing temporal networks."""

from __future__ import annotations

import copy
import logging
import sys
import warnings
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from lago.enums import LinkStreamMode
from lago.leaf import Leaf
from lago.time_edge import TimeEdge

# Configure module logger
logger = logging.getLogger(__name__)

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
        """
        for link in links:
            if len(link) > 3:
                source, target, time, weight = link[0], link[1], link[2], link[3]
            else:
                source, target, time = link[0], link[1], link[2]
                weight = 1

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

        Note:
            Times must be integers. The total weight of a continuous link
            is weight * duration.
        """
        for link in links:
            if len(link) > 4:
                source, target, time, duration, initial_weight = (
                    link[0],
                    link[1],
                    link[2],
                    link[3],
                    link[4],
                )
            else:
                source, target, time, duration = link[0], link[1], link[2], link[3]
                initial_weight = 1

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

    def add_continous_links(self, links: Sequence[tuple[int, ...]]) -> None:
        """Add continuous links to the stream.

        .. deprecated::
            Use :meth:`add_continuous_links` instead. This method will be
            removed in a future version.

        Args:
            links: Sequence of link tuples.
        """
        warnings.warn(
            "add_continous_links is deprecated due to typo, use add_continuous_links instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.add_continuous_links(links)

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

        Note:
            Times must be integers.
        """
        for link in links:
            if len(link) > 4:
                source, target, source_time, target_time, weight = (
                    link[0],
                    link[1],
                    link[2],
                    link[3],
                    link[4],
                )
            else:
                source, target, source_time, target_time = (link[0], link[1], link[2], link[3])
                weight = 1

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
        old_leaves_dict = copy.deepcopy(self.leaves_dict)
        self.leaves_dict = {}

        for (node, time), leaf in old_leaves_dict.items():
            # Split edges based on time_instants
            for time_edge in leaf.topo_neighbors:
                # NOTE: This may be a bottleneck for large networks
                # TODO: Refactor with segments instead of range
                tmp_time_instants = self.time_instants & set(
                    range(time, time + time_edge.duration + 1)
                )
                tmp_time_instants_sorted = sorted(tmp_time_instants)

                for time_start, time_end in pairwise(tmp_time_instants_sorted):
                    duration = time_end - time_start
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

    def get_time_links(self) -> set[tuple[int, ...]]:
        """Get all temporal links in the stream.

        Automatically returns links in the appropriate format based on the mode:
        - INSTANTANEOUS: (source, target, time, weight)
        - CONTINUOUS: (source, target, time, duration, weight)
        - DELAYED: (source, target, source_time, target_time, weight)

        Returns:
            Set of link tuples in the mode-appropriate format.
        """
        if self.mode == LinkStreamMode.CONTINUOUS:
            return self._get_continuous_time_links()
        if self.mode == LinkStreamMode.DELAYED:
            return self._get_delayed_time_links()
        return self._get_instantaneous_time_links()

    def _get_instantaneous_time_links(self) -> set[tuple[int, ...]]:
        """Get time links for instantaneous mode.

        Returns:
            Set of (source, target, time, weight) tuples.
        """
        time_links: set[tuple[int, ...]] = set()
        for leaf in self.leaves_dict.values():
            source = leaf.node
            time = leaf.time
            for neighb in leaf.topo_neighbors:
                target = neighb.target.node
                weight = neighb.weight
                time_links.add((source, target, time, weight))
        return time_links

    def _get_continuous_time_links(self) -> set[tuple[int, ...]]:
        """Get time links for continuous mode.

        Returns:
            Set of (source, target, time, duration, weight) tuples.
        """
        time_links: set[tuple[int, ...]] = set()
        for leaf in self.leaves_dict.values():
            source = leaf.node
            time = leaf.time
            for neighb in leaf.topo_neighbors:
                target = neighb.target.node
                weight = neighb.weight
                duration = neighb.duration
                time_links.add((source, target, time, duration, weight))
        return time_links

    def _get_delayed_time_links(self) -> set[tuple[int, ...]]:
        """Get time links for delayed mode.

        Returns:
            Set of (source, target, source_time, target_time, weight) tuples.
        """
        time_links: set[tuple[int, ...]] = set()
        for leaf in self.leaves_dict.values():
            source = leaf.node
            source_time = leaf.time
            for neighb in leaf.topo_neighbors:
                target = neighb.target.node
                weight = neighb.weight
                target_time = neighb.target.time
                # Normalize order by time
                if source_time < target_time:
                    tsource, ttarget = source, target
                    tsource_time, ttarget_time = source_time, target_time
                else:
                    tsource, ttarget = target, source
                    tsource_time, ttarget_time = target_time, source_time
                link = (tsource, ttarget, tsource_time, ttarget_time, weight)
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
