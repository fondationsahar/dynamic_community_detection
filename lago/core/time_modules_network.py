"""TimeModulesNetwork: Meta-network of temporal modules with inter-module edges."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from lago.core.linkstream import LinkStream
    from lago.core.time_modules import TimeModules


@dataclass(frozen=True)
class ModuleTimeEdge:
    """A time edge between modules (preserves original edge info).

    Attributes:
        source_module: Module label of the source node.
        target_module: Module label of the target node.
        time: Time of the edge (source_time for delayed networks).
        target_time: Target time (only for delayed networks, None otherwise).
        weight: Edge weight.
        duration: Edge duration (only for continuous networks, None otherwise).
        source_node: Original source node ID.
        target_node: Original target node ID.
    """

    source_module: int
    target_module: int
    time: int
    target_time: int | None
    weight: float
    duration: int | None
    source_node: int
    target_node: int

    @property
    def is_intra_module(self) -> bool:
        """Whether this is an intra-module (self-loop) edge."""
        return self.source_module == self.target_module


class TimeModulesNetwork:
    """Meta-network of temporal modules with time-stamped inter-module edges.

    This class aggregates LinkStream edges into module-level interactions,
    allowing analysis of community structure at a higher level while preserving:
    - Temporal dimension (edges are time-stamped)
    - Raw edge information (easy to trace back to original linkstream)
    - All linkstream properties (directed, weighted, delayed, continuous)

    Example:
        ```python
        from lago import LinkStream, TimeModules, TimeModulesNetwork

        # Create linkstream and detect communities
        ls = LinkStream()
        ls.add_links([(0, 1, 0), (0, 2, 0), (2, 3, 0)])

        # Define modules
        tm = TimeModules({0: {(0, 0), (1, 0)}, 1: {(2, 0), (3, 0)}})

        # Build module network
        network = TimeModulesNetwork(tm, ls)

        # Analyze module interactions
        neighbors = network.get_neighbors_at_time(module=0, time=0)
        ```
    """

    def __init__(
        self,
        time_modules: TimeModules,
        linkstream: LinkStream,
        include_intra: bool = True,
    ) -> None:
        """Initialize TimeModulesNetwork.

        Args:
            time_modules: The temporal modules (community assignments).
            linkstream: The underlying link stream.
            include_intra: Whether to include intra-module edges (self-loops).

        Raises:
            ValueError: If time_modules is incompatible with linkstream.
        """
        self._time_modules = time_modules
        self._linkstream = linkstream
        self._include_intra = include_intra

        # Copy properties from linkstream
        self._directed = linkstream.directed
        self._delayed = linkstream.delayed
        self._continuous = linkstream.continuous

        # Validate compatibility
        self._validate_compatibility()

        # Build module edges
        self._edges: list[ModuleTimeEdge] = []
        self._edges_by_time: dict[int, list[ModuleTimeEdge]] = defaultdict(list)
        self._edges_by_module: dict[int, list[ModuleTimeEdge]] = defaultdict(list)
        self._build_module_edges()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def directed(self) -> bool:
        """Whether the network is directed."""
        return self._directed

    @property
    def delayed(self) -> bool:
        """Whether the network has delayed edges."""
        return self._delayed

    @property
    def continuous(self) -> bool:
        """Whether the network has continuous (duration) edges."""
        return self._continuous

    @property
    def modules(self) -> frozenset[int]:
        """All module labels."""
        return self._time_modules.modules

    @property
    def times(self) -> frozenset[int]:
        """All time points with edges."""
        return frozenset(self._edges_by_time.keys())

    @property
    def nb_module_edges(self) -> int:
        """Number of module-level time edges."""
        return len(self._edges)

    @property
    def nb_inter_module_edges(self) -> int:
        """Number of inter-module edges."""
        return sum(1 for e in self._edges if not e.is_intra_module)

    @property
    def nb_intra_module_edges(self) -> int:
        """Number of intra-module edges (self-loops)."""
        return sum(1 for e in self._edges if e.is_intra_module)

    # =========================================================================
    # Raw Edge Access
    # =========================================================================

    def get_all_edges(self) -> list[ModuleTimeEdge]:
        """Get all raw module time edges.

        Returns:
            List of all ModuleTimeEdge objects.
        """
        return list(self._edges)

    def get_edges_at_time(self, time: int) -> list[ModuleTimeEdge]:
        """Get module edges at a specific time.

        Args:
            time: Time point.

        Returns:
            List of ModuleTimeEdge objects at this time.
        """
        return list(self._edges_by_time.get(time, []))

    def get_edges_between(
        self,
        m1: int,
        m2: int,
        include_reverse: bool = True,
    ) -> list[ModuleTimeEdge]:
        """Get all edges between two modules.

        Args:
            m1: First module label.
            m2: Second module label.
            include_reverse: If True and undirected, include both directions.

        Returns:
            List of ModuleTimeEdge objects between m1 and m2.
        """
        edges = [e for e in self._edges if (e.source_module == m1 and e.target_module == m2)]

        if include_reverse and not self._directed and m1 != m2:
            edges.extend(
                e for e in self._edges if (e.source_module == m2 and e.target_module == m1)
            )

        return edges

    def get_edges_of_module(self, module: int) -> list[ModuleTimeEdge]:
        """Get all edges involving a module.

        Args:
            module: Module label.

        Returns:
            List of ModuleTimeEdge objects where module is source or target.
        """
        return list(self._edges_by_module.get(module, []))

    def iter_edges(self) -> Iterator[ModuleTimeEdge]:
        """Iterate over all module edges.

        Yields:
            ModuleTimeEdge objects.
        """
        yield from self._edges

    # =========================================================================
    # Temporal Module Neighbors
    # =========================================================================

    def get_neighbors_at_time(
        self,
        module: int,
        time: int,
        include_self: bool = False,
    ) -> dict[int, float]:
        """Get module neighbors at a specific time.

        Args:
            module: Module label.
            time: Time point.
            include_self: Whether to include intra-module weight.

        Returns:
            Dictionary mapping neighbor_module -> total_weight.
        """
        neighbors: dict[int, float] = defaultdict(float)

        for edge in self._edges_by_time.get(time, []):
            if edge.source_module == module and (include_self or edge.target_module != module):
                neighbors[edge.target_module] += edge.weight
            elif (
                not self._directed
                and edge.target_module == module
                and (include_self or edge.source_module != module)
            ):
                neighbors[edge.source_module] += edge.weight

        return dict(neighbors)

    def get_neighbors_over_time(
        self,
        module: int,
        include_self: bool = False,
    ) -> dict[int, dict[int, float]]:
        """Get module neighbors across all times.

        Args:
            module: Module label.
            include_self: Whether to include intra-module weight.

        Returns:
            Dictionary: {time: {neighbor_module: weight}}.
        """
        result: dict[int, dict[int, float]] = {}

        for time in sorted(self._edges_by_time.keys()):
            neighbors = self.get_neighbors_at_time(module, time, include_self)
            if neighbors:
                result[time] = neighbors

        return result

    def get_aggregated_neighbors(
        self,
        module: int,
        include_self: bool = False,
    ) -> dict[int, float]:
        """Get module neighbors aggregated across all times.

        Args:
            module: Module label.
            include_self: Whether to include intra-module weight.

        Returns:
            Dictionary mapping neighbor_module -> total_weight_across_all_times.
        """
        neighbors: dict[int, float] = defaultdict(float)

        for edge in self._edges_by_module.get(module, []):
            if edge.source_module == module and (include_self or edge.target_module != module):
                neighbors[edge.target_module] += edge.weight

        # For undirected, also count edges where module is target
        if not self._directed:
            for edge in self._edges:
                if edge.target_module == module and edge.source_module != module:
                    neighbors[edge.source_module] += edge.weight

        return dict(neighbors)

    def get_closest_neighbors_at_time(
        self,
        module: int,
        time: int,
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """Get k closest neighbors at a specific time by weight.

        Args:
            module: Module label.
            time: Time point.
            k: Number of neighbors to return.

        Returns:
            List of (neighbor_module, weight) tuples, sorted by weight descending.
        """
        neighbors = self.get_neighbors_at_time(module, time, include_self=False)
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        return sorted_neighbors[:k]

    def get_closest_neighbors_overall(
        self,
        module: int,
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """Get k closest neighbors by total weight across time.

        Args:
            module: Module label.
            k: Number of neighbors to return.

        Returns:
            List of (neighbor_module, total_weight) tuples, sorted by weight descending.
        """
        neighbors = self.get_aggregated_neighbors(module, include_self=False)
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        return sorted_neighbors[:k]

    # =========================================================================
    # Inter vs Intra Analysis
    # =========================================================================

    def get_inter_module_weight_at_time(self, time: int) -> float:
        """Get total weight of inter-module edges at a time.

        Args:
            time: Time point.

        Returns:
            Sum of weights of inter-module edges.
        """
        return sum(e.weight for e in self._edges_by_time.get(time, []) if not e.is_intra_module)

    def get_intra_module_weight_at_time(self, time: int) -> float:
        """Get total weight of intra-module edges at a time.

        Args:
            time: Time point.

        Returns:
            Sum of weights of intra-module edges.
        """
        return sum(e.weight for e in self._edges_by_time.get(time, []) if e.is_intra_module)

    def get_inter_intra_over_time(self) -> dict[int, tuple[float, float]]:
        """Get inter and intra module weights per time.

        Returns:
            Dictionary: {time: (inter_weight, intra_weight)}.
        """
        result: dict[int, tuple[float, float]] = {}
        for time in sorted(self._edges_by_time.keys()):
            inter = self.get_inter_module_weight_at_time(time)
            intra = self.get_intra_module_weight_at_time(time)
            result[time] = (inter, intra)
        return result

    # =========================================================================
    # Conversion & Export
    # =========================================================================

    def to_adjacency_at_time(self, time: int) -> dict[int, dict[int, float]]:
        """Get adjacency dict for a specific time.

        Args:
            time: Time point.

        Returns:
            Dictionary: {module: {neighbor_module: weight}}.
        """
        adj: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

        for edge in self._edges_by_time.get(time, []):
            adj[edge.source_module][edge.target_module] += edge.weight
            if not self._directed and edge.source_module != edge.target_module:
                adj[edge.target_module][edge.source_module] += edge.weight

        return {m: dict(neighbors) for m, neighbors in adj.items()}

    def to_edge_list(self) -> list[tuple[int, int, int, float]]:
        """Export as (source_module, target_module, time, weight) tuples.

        Returns:
            List of edge tuples.
        """
        return [(e.source_module, e.target_module, e.time, e.weight) for e in self._edges]

    def to_linkstream_format(
        self,
    ) -> list[tuple[int, int, int, float]] | list[tuple[int, int, int, int, float]]:
        """Export in linkstream format (can create a new LinkStream).

        Returns:
            List of tuples suitable for LinkStream.add_links().
            - For standard: (source, target, time, weight)
            - For delayed/continuous: (source, target, time, time2/duration, weight)
        """
        if self._delayed:
            return [
                (e.source_module, e.target_module, e.time, e.target_time or e.time, e.weight)
                for e in self._edges
            ]
        if self._continuous:
            return [
                (e.source_module, e.target_module, e.time, e.duration or 1, e.weight)
                for e in self._edges
            ]
        return [(e.source_module, e.target_module, e.time, e.weight) for e in self._edges]

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _validate_compatibility(self) -> None:
        """Validate TimeModules is compatible with LinkStream."""
        labels = self._time_modules.to_flat_labels()

        # Check all module members exist in linkstream
        for node, time in labels:
            if (node, time) not in self._linkstream.leaves_dict:
                raise ValueError(
                    f"TimeModules member (node={node}, time={time}) not found in LinkStream"
                )

        # Check coverage (warn if incomplete)
        ls_leaves = set(self._linkstream.leaves_dict.keys())
        tm_leaves = set(labels.keys())
        uncovered = ls_leaves - tm_leaves
        if uncovered:
            warnings.warn(
                f"{len(uncovered)} LinkStream (node, time) pairs have no module "
                f"assignment. First few: {list(uncovered)[:5]}",
                UserWarning,
                stacklevel=3,
            )

    def _build_module_edges(self) -> None:
        """Build module edges from linkstream edges."""
        labels = self._time_modules.to_flat_labels()
        seen_edges: set[tuple] = set()

        for (node, time), leaf in self._linkstream.leaves_dict.items():
            source_module = labels.get((node, time))
            if source_module is None:
                continue

            # Process outgoing edges
            for time_edge in leaf.topo_neighbors:
                target_node = time_edge.target.node
                target_time = time_edge.target.time

                target_module = labels.get((target_node, target_time))
                if target_module is None:
                    continue

                # Skip intra-module if not included
                if not self._include_intra and source_module == target_module:
                    continue

                # For undirected graphs, avoid duplicates
                if not self._directed:
                    edge_key = tuple(sorted([(node, time), (target_node, target_time)]))
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)

                # Create module edge
                module_edge = ModuleTimeEdge(
                    source_module=source_module,
                    target_module=target_module,
                    time=time,
                    target_time=target_time if self._delayed and time != target_time else None,
                    weight=time_edge.weight,
                    duration=time_edge.duration if self._continuous else None,
                    source_node=node,
                    target_node=target_node,
                )

                self._edges.append(module_edge)
                self._edges_by_time[time].append(module_edge)
                self._edges_by_module[source_module].append(module_edge)
                if source_module != target_module:
                    self._edges_by_module[target_module].append(module_edge)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TimeModulesNetwork(modules={len(self.modules)}, "
            f"edges={self.nb_module_edges}, "
            f"inter={self.nb_inter_module_edges}, "
            f"intra={self.nb_intra_module_edges})"
        )

    def __len__(self) -> int:
        """Number of module edges."""
        return len(self._edges)
