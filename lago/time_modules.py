"""TimeModules class for navigating temporal community detection results."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class TimeSegment:
    """A contiguous time segment.

    Attributes:
        start: Start time (inclusive).
        end: End time (inclusive).
    """

    start: int
    end: int

    @property
    def duration(self) -> int:
        """Duration of the segment."""
        return self.end - self.start + 1

    def __contains__(self, time: int) -> bool:
        """Check if a time is within this segment."""
        return self.start <= time <= self.end

    def overlaps(self, other: TimeSegment) -> bool:
        """Check if this segment overlaps with another."""
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class NodeModuleMembership:
    """A node's membership in a module with time information.

    Attributes:
        module: The module label.
        times: Set of times when the node belongs to this module.
        segments: List of contiguous time segments.
    """

    module: int
    times: frozenset[int]
    segments: tuple[TimeSegment, ...]

    @property
    def total_duration(self) -> int:
        """Total time spent in this module."""
        return len(self.times)


class TimeModules:
    """Class for navigating through nodes, times, and time modules.

    This class provides efficient lookups from multiple perspectives:
    - From a node: get all modules it belongs to and when
    - From a time: get all nodes and their module memberships
    - From a module: get all nodes and their time segments

    Example:
        ```python
        raw_modules = {
            0: {(0, 0), (0, 1), (1, 0), (1, 1)},  # Module 0: nodes 0,1 at times 0,1
            1: {(2, 0), (2, 1), (3, 0)},          # Module 1: node 2 at times 0,1; node 3 at time 0
        }
        tm = TimeModules(raw_modules)

        # Get modules of a specific node
        node_modules = tm.get_modules_of_node(0)

        # Get node-module mapping at a specific time
        membership = tm.get_nodes_modules_membership_at_time(0)
        ```
    """

    def __init__(
        self,
        raw_modules: dict[int, set[tuple[int, int]]],
    ):
        """Initialize TimeModules from raw module data.

        Args:
            raw_modules: Mapping from module label to set of (node, time) tuples.
                Example: {0: {(node1, time1), (node2, time1), ...}, 1: {...}}
        """
        self._raw_modules = raw_modules

        # Build inverted indices for efficient lookups
        self._node_to_modules: dict[int, dict[int, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )  # node -> {module -> {times}}

        self._time_to_nodes: dict[int, dict[int, int]] = defaultdict(
            dict
        )  # time -> {node -> module}

        self._module_to_nodes: dict[int, dict[int, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )  # module -> {node -> {times}}

        self._all_nodes: set[int] = set()
        self._all_times: set[int] = set()
        self._all_modules: set[int] = set()

        # Build indices
        for module_label, members in raw_modules.items():
            self._all_modules.add(module_label)
            for node, time in members:
                self._all_nodes.add(node)
                self._all_times.add(time)

                self._node_to_modules[node][module_label].add(time)
                self._time_to_nodes[time][node] = module_label
                self._module_to_nodes[module_label][node].add(time)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def nodes(self) -> frozenset[int]:
        """All nodes in the time modules."""
        return frozenset(self._all_nodes)

    @property
    def times(self) -> frozenset[int]:
        """All time points in the time modules."""
        return frozenset(self._all_times)

    @property
    def modules(self) -> frozenset[int]:
        """All module labels."""
        return frozenset(self._all_modules)

    @property
    def nb_nodes(self) -> int:
        """Number of unique nodes."""
        return len(self._all_nodes)

    @property
    def nb_times(self) -> int:
        """Number of unique time points."""
        return len(self._all_times)

    @property
    def nb_modules(self) -> int:
        """Number of modules."""
        return len(self._all_modules)

    @property
    def time_range(self) -> tuple[int, int]:
        """Min and max time values."""
        if not self._all_times:
            return (0, 0)
        return (min(self._all_times), max(self._all_times))

    # =========================================================================
    # Node-centric methods
    # =========================================================================

    def get_modules_of_node(self, node: int) -> list[NodeModuleMembership]:
        """Get all modules a node belongs to with time information.

        Args:
            node: Node identifier.

        Returns:
            List of NodeModuleMembership objects, one per module the node visits.
            Each contains the module label, times, and contiguous time segments.

        Raises:
            KeyError: If node not found in any module.
        """
        if node not in self._node_to_modules:
            raise KeyError(f"Node {node} not found in any module")

        result = []
        for module_label, times in self._node_to_modules[node].items():
            segments = self._times_to_segments(times)
            membership = NodeModuleMembership(
                module=module_label,
                times=frozenset(times),
                segments=tuple(segments),
            )
            result.append(membership)

        return result

    def get_module_of_node_at_time(self, node: int, time: int) -> int | None:
        """Get the module a node belongs to at a specific time.

        Args:
            node: Node identifier.
            time: Time point.

        Returns:
            Module label, or None if node is not active at this time.
        """
        return self._time_to_nodes.get(time, {}).get(node)

    def get_node_trajectory(self, node: int) -> dict[int, int]:
        """Get a node's module trajectory over time.

        Args:
            node: Node identifier.

        Returns:
            Dictionary mapping time -> module label for all times the node is active.
        """
        trajectory = {}
        for module_label, times in self._node_to_modules.get(node, {}).items():
            for time in times:
                trajectory[time] = module_label
        return dict(sorted(trajectory.items()))

    def get_node_switches(self, node: int) -> list[tuple[int, int, int]]:
        """Get all module switches for a node.

        Args:
            node: Node identifier.

        Returns:
            List of (time, from_module, to_module) tuples.
        """
        trajectory = self.get_node_trajectory(node)
        times = sorted(trajectory.keys())

        switches = []
        for t1, t2 in pairwise(times):
            if trajectory[t1] != trajectory[t2]:
                switches.append((t2, trajectory[t1], trajectory[t2]))

        return switches

    # =========================================================================
    # Time-centric methods
    # =========================================================================

    def get_nodes_modules_membership_at_time(self, time: int) -> dict[int, int]:
        """Get all nodes and their module membership at a specific time.

        Args:
            time: Time point.

        Returns:
            Dictionary mapping node -> module label for all nodes active at this time.
        """
        return dict(self._time_to_nodes.get(time, {}))

    def get_active_nodes_at_time(self, time: int) -> frozenset[int]:
        """Get all nodes active at a specific time.

        Args:
            time: Time point.

        Returns:
            Set of node identifiers.
        """
        return frozenset(self._time_to_nodes.get(time, {}).keys())

    def get_modules_at_time(self, time: int) -> frozenset[int]:
        """Get all modules present at a specific time.

        Args:
            time: Time point.

        Returns:
            Set of module labels.
        """
        return frozenset(self._time_to_nodes.get(time, {}).values())

    # =========================================================================
    # Module-centric methods
    # =========================================================================

    def get_time_modules_dict(
        self,
    ) -> dict[int, dict[int, list[TimeSegment]]]:
        """Get modules with nodes and their time segments.

        Returns:
            Dictionary: module_label -> {node -> [TimeSegment, ...]}.
        """
        result: dict[int, dict[int, list[TimeSegment]]] = {}

        for module_label, nodes_times in self._module_to_nodes.items():
            result[module_label] = {}
            for node, times in nodes_times.items():
                result[module_label][node] = self._times_to_segments(times)

        return result

    def get_module_nodes(self, module: int) -> frozenset[int]:
        """Get all nodes in a module.

        Args:
            module: Module label.

        Returns:
            Set of node identifiers.
        """
        return frozenset(self._module_to_nodes.get(module, {}).keys())

    def get_module_size(self, module: int) -> int:
        """Get the number of nodes in a module.

        Args:
            module: Module label.

        Returns:
            Number of unique nodes in the module.
        """
        return len(self._module_to_nodes.get(module, {}))

    def get_module_duration(self, module: int) -> int:
        """Get the total time span of a module.

        Args:
            module: Module label.

        Returns:
            Number of time points where the module has at least one node.
        """
        all_times: set[int] = set()
        for times in self._module_to_nodes.get(module, {}).values():
            all_times.update(times)
        return len(all_times)

    def get_module_members_at_time(self, module: int, time: int) -> frozenset[int]:
        """Get nodes in a module at a specific time.

        Args:
            module: Module label.
            time: Time point.

        Returns:
            Set of node identifiers in the module at this time.
        """
        nodes = set()
        for node, times in self._module_to_nodes.get(module, {}).items():
            if time in times:
                nodes.add(node)
        return frozenset(nodes)

    # =========================================================================
    # Analysis methods
    # =========================================================================

    def get_stability_score(self, node: int) -> float:
        """Calculate how stable a node's module membership is.

        A score of 1.0 means the node never switches modules.
        Lower scores indicate more switching.

        Args:
            node: Node identifier.

        Returns:
            Stability score between 0 and 1.
        """
        trajectory = self.get_node_trajectory(node)
        if len(trajectory) <= 1:
            return 1.0

        times = sorted(trajectory.keys())
        switches = sum(1 for t1, t2 in pairwise(times) if trajectory[t1] != trajectory[t2])

        return 1.0 - switches / (len(times) - 1)

    def get_module_cohesion(self, module: int) -> float:
        """Calculate how cohesive a module is over time.

        A score of 1.0 means all nodes are present at all times.

        Args:
            module: Module label.

        Returns:
            Cohesion score between 0 and 1.
        """
        nodes_times = self._module_to_nodes.get(module, {})
        if not nodes_times:
            return 0.0

        all_times: set[int] = set()
        for times in nodes_times.values():
            all_times.update(times)

        if not all_times:
            return 0.0

        # Perfect cohesion = all nodes present at all times
        max_presence = len(nodes_times) * len(all_times)
        actual_presence = sum(len(times) for times in nodes_times.values())

        return actual_presence / max_presence

    def summary(self) -> dict:
        """Get a summary of the time modules.

        Returns:
            Dictionary with summary statistics.
        """
        return {
            "nb_nodes": self.nb_nodes,
            "nb_times": self.nb_times,
            "nb_modules": self.nb_modules,
            "time_range": self.time_range,
            "module_sizes": {m: self.get_module_size(m) for m in self.modules},
            "module_durations": {m: self.get_module_duration(m) for m in self.modules},
        }

    # =========================================================================
    # Conversion methods
    # =========================================================================

    def to_communities_dict(self) -> dict[int, set[tuple[int, int]]]:
        """Convert back to raw communities format (for longitudinal_modularity).

        Returns:
            Dictionary mapping module label to set of (node, time) tuples.
        """
        return {label: set(members) for label, members in self._raw_modules.items()}

    def to_flat_labels(self) -> dict[tuple[int, int], int]:
        """Convert to flat label dictionary.

        Returns:
            Dictionary mapping (node, time) -> module label.
        """
        labels = {}
        for module_label, members in self._raw_modules.items():
            for node, time in members:
                labels[(node, time)] = module_label
        return labels

    # =========================================================================
    # Save / Load methods
    # =========================================================================

    def to_txt(self, path: str | Path) -> None:
        """Save time modules to a text file.

        Format: Each line contains "node time module" (space-separated).

        Args:
            path: Path to the output file.

        Example:
            ```python
            tm.to_txt("communities.txt")
            # File content:
            # 0 0 1
            # 0 1 1
            # 1 0 1
            # 2 0 2
            ```
        """
        path = Path(path)
        with path.open("w") as f:
            for module_label, members in self._raw_modules.items():
                for node, time in sorted(members):
                    f.write(f"{node} {time} {module_label}\n")

    @classmethod
    def from_txt(cls, path: str | Path) -> TimeModules:
        """Load time modules from a text file.

        Format: Each line contains "node time module" (space-separated).

        Args:
            path: Path to the input file.

        Returns:
            TimeModules instance.

        Example:
            ```python
            tm = TimeModules.from_txt("communities.txt")
            ```
        """
        path = Path(path)
        raw_modules: dict[int, set[tuple[int, int]]] = defaultdict(set)

        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    node, time, module = int(parts[0]), int(parts[1]), int(parts[2])
                    raw_modules[module].add((node, time))

        return cls(dict(raw_modules))

    def to_json(self, path: str | Path, indent: int | None = 2) -> None:
        """Save time modules to a JSON file.

        Args:
            path: Path to the output file.
            indent: JSON indentation level. None for compact output.

        Example:
            ```python
            tm.to_json("communities.json")
            ```
        """
        path = Path(path)
        # Convert to JSON-serializable format
        # Module labels as strings (JSON keys must be strings)
        # Members as list of [node, time] pairs
        data = {
            str(module): [[node, time] for node, time in sorted(members)]
            for module, members in self._raw_modules.items()
        }

        with path.open("w") as f:
            json.dump(data, f, indent=indent)

    @classmethod
    def from_json(cls, path: str | Path) -> TimeModules:
        """Load time modules from a JSON file.

        Args:
            path: Path to the input file.

        Returns:
            TimeModules instance.

        Example:
            ```python
            tm = TimeModules.from_json("communities.json")
            ```
        """
        path = Path(path)
        with path.open() as f:
            data = json.load(f)

        raw_modules: dict[int, set[tuple[int, int]]] = {}
        for module_str, members in data.items():
            module = int(module_str)
            raw_modules[module] = {(int(node), int(time)) for node, time in members}

        return cls(raw_modules)

    def to_csv(self, path: str | Path, sep: str = ",") -> None:
        """Save time modules to a CSV file.

        Format: node,time,module with header row.

        Args:
            path: Path to the output file.
            sep: Separator character (default: comma).

        Example:
            ```python
            tm.to_csv("communities.csv")
            ```
        """
        path = Path(path)
        with path.open("w") as f:
            f.write(f"node{sep}time{sep}module\n")
            for module_label, members in self._raw_modules.items():
                for node, time in sorted(members):
                    f.write(f"{node}{sep}{time}{sep}{module_label}\n")

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        sep: str = ",",
        has_header: bool = True,
    ) -> TimeModules:
        """Load time modules from a CSV file.

        Args:
            path: Path to the input file.
            sep: Separator character (default: comma).
            has_header: Whether file has a header row to skip.

        Returns:
            TimeModules instance.

        Example:
            ```python
            tm = TimeModules.from_csv("communities.csv")
            ```
        """
        path = Path(path)
        raw_modules: dict[int, set[tuple[int, int]]] = defaultdict(set)

        with path.open() as f:
            if has_header:
                next(f)  # Skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(sep)
                if len(parts) >= 3:
                    node, time, module = int(parts[0]), int(parts[1]), int(parts[2])
                    raw_modules[module].add((node, time))

        return cls(dict(raw_modules))

    # =========================================================================
    # Iteration
    # =========================================================================

    def __iter__(self) -> Iterator[tuple[int, set[tuple[int, int]]]]:
        """Iterate over modules."""
        yield from self._raw_modules.items()

    def __len__(self) -> int:
        """Number of modules."""
        return len(self._raw_modules)

    def __contains__(self, item: int | tuple[int, int]) -> bool:
        """Check if a module label or (node, time) tuple is in the time modules."""
        if isinstance(item, int):
            return item in self._all_modules
        if isinstance(item, tuple) and len(item) == 2:
            node, time = item
            return self._time_to_nodes.get(time, {}).get(node) is not None
        return False

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TimeModules(nb_modules={self.nb_modules}, "
            f"nb_nodes={self.nb_nodes}, nb_times={self.nb_times})"
        )

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _times_to_segments(times: set[int]) -> list[TimeSegment]:
        """Convert a set of times to contiguous segments.

        Args:
            times: Set of time points.

        Returns:
            List of TimeSegment objects representing contiguous periods.
        """
        if not times:
            return []

        sorted_times = sorted(times)
        segments = []
        start = sorted_times[0]
        end = sorted_times[0]

        for time in sorted_times[1:]:
            if time == end + 1:
                # Extend current segment
                end = time
            else:
                # Close current segment and start new one
                segments.append(TimeSegment(start=start, end=end))
                start = time
                end = time

        # Close final segment
        segments.append(TimeSegment(start=start, end=end))

        return segments
