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


class TimeModule:
    """A single temporal module/community (view into TimeModules).

    This class provides a clean API for working with a single module.
    It acts as a view into the parent TimeModules container, delegating
    data access to avoid memory duplication.

    Example:
        ```python
        tm = TimeModules(raw_modules)
        module = tm.get_module(0)

        print(module.nodes)      # Nodes in this module
        print(module.size)       # Number of nodes
        print(module.duration)   # Time span
        print(module.cohesion)   # Cohesion score
        ```

    Note:
        TimeModule objects are lightweight views. They don't copy data
        from the parent TimeModules container.
    """

    def __init__(self, label: int, parent: TimeModules) -> None:
        """Initialize a TimeModule view.

        Args:
            label: The module label.
            parent: The parent TimeModules container.

        Raises:
            KeyError: If the module label doesn't exist in parent.
        """
        if label not in parent._all_modules:
            raise KeyError(f"Module {label} not found")
        self._label = label
        self._parent = parent

    @property
    def label(self) -> int:
        """Module label/identifier."""
        return self._label

    @property
    def members(self) -> frozenset[tuple[int, int]]:
        """All (node, time) members of this module."""
        return frozenset(self._parent._raw_modules.get(self._label, set()))

    @property
    def nodes(self) -> frozenset[int]:
        """All nodes in this module."""
        return self._parent.get_module_nodes(self._label)

    @property
    def times(self) -> frozenset[int]:
        """All time points where this module has members."""
        times_set: set[int] = set()
        for times in self._parent._module_to_nodes.get(self._label, {}).values():
            times_set.update(times)
        return frozenset(times_set)

    @property
    def size(self) -> int:
        """Number of unique nodes in this module."""
        return self._parent.get_module_size(self._label)

    @property
    def duration(self) -> int:
        """Number of time points where this module has at least one node."""
        return self._parent.get_module_duration(self._label)

    @property
    def time_range(self) -> tuple[int, int]:
        """Min and max time values for this module."""
        module_times = self.times
        if not module_times:
            return (0, 0)
        return (min(module_times), max(module_times))

    @property
    def cohesion(self) -> float:
        """Cohesion score (1.0 = all nodes present at all times)."""
        return self._parent.get_module_cohesion(self._label)

    def get_nodes_at_time(self, time: int) -> frozenset[int]:
        """Get nodes in this module at a specific time.

        Args:
            time: Time point.

        Returns:
            Set of node identifiers.
        """
        return self._parent.get_module_members_at_time(self._label, time)

    def get_node_segments(self) -> dict[int, list[TimeSegment]]:
        """Get time segments for each node in this module.

        Returns:
            Dictionary mapping node -> list of TimeSegment.
        """
        result: dict[int, list[TimeSegment]] = {}
        nodes_times = self._parent._module_to_nodes.get(self._label, {})
        for node, times in nodes_times.items():
            result[node] = TimeModules._times_to_segments(times)
        return result

    def contains_node(self, node: int) -> bool:
        """Check if a node is in this module (at any time).

        Args:
            node: Node identifier.

        Returns:
            True if node is in this module.
        """
        return node in self._parent._module_to_nodes.get(self._label, {})

    def contains_at_time(self, node: int, time: int) -> bool:
        """Check if a node is in this module at a specific time.

        Args:
            node: Node identifier.
            time: Time point.

        Returns:
            True if node is in this module at this time.
        """
        return time in self._parent._module_to_nodes.get(self._label, {}).get(node, set())

    def overlaps_with(self, other: TimeModule) -> bool:
        """Check if this module shares any (node, time) with another.

        Args:
            other: Another TimeModule.

        Returns:
            True if there's any overlap.
        """
        return bool(self.members & other.members)

    def __contains__(self, item: int | tuple[int, int]) -> bool:
        """Check if a node or (node, time) is in this module."""
        if isinstance(item, int):
            return self.contains_node(item)
        if isinstance(item, tuple) and len(item) == 2:
            return self.contains_at_time(item[0], item[1])
        return False

    def __len__(self) -> int:
        """Number of (node, time) members.

        Note: This returns the total number of (node, time) pairs, not the
        number of unique nodes. Use ``.size`` for the number of unique nodes.
        """
        return len(self.members)

    def __eq__(self, other: object) -> bool:
        """Check equality based on label and parent."""
        if not isinstance(other, TimeModule):
            return NotImplemented
        return self._label == other._label and self._parent is other._parent

    def __hash__(self) -> int:
        """Hash based on label."""
        return hash(self._label)

    def __repr__(self) -> str:
        """String representation."""
        return f"TimeModule(label={self._label}, size={self.size}, duration={self.duration})"


class TimeModules:
    """Class for navigating through nodes, times, and time modules.

    This class provides efficient lookups from multiple perspectives:
    - From a node: get all modules it belongs to and when
    - From a time: get all nodes and their module memberships
    - From a module: get all nodes and their time segments

    Example:
        ```python
        # From raw modules dict
        raw_modules = {
            0: {(0, 0), (0, 1), (1, 0), (1, 1)},  # Module 0: nodes 0,1 at times 0,1
            1: {(2, 0), (2, 1), (3, 0)},          # Module 1: node 2 at times 0,1; node 3 at time 0
        }
        tm = TimeModules(raw_modules)

        # Empty declaration (then populate later)
        tm = TimeModules()

        # Direct loading from file
        tm = TimeModules(path="communities.csv")
        tm = TimeModules(path="communities.json")
        tm = TimeModules(path="communities.txt")

        # Get modules of a specific node
        node_modules = tm.get_modules_of_node(0)

        # Get node-module mapping at a specific time
        membership = tm.get_nodes_modules_membership_at_time(0)
        ```
    """

    def __init__(
        self,
        raw_modules: dict[int, set[tuple[int, int]]] | dict[int, dict] | None = None,
        *,
        path: str | Path | None = None,
    ):
        """Initialize TimeModules.

        Can be initialized in four ways:
        1. From raw module data: `TimeModules({0: {(0, 0), (1, 0)}})`
        2. From segment format: `TimeModules({0: {"nodes": {"0": [[0, 2]], "1": [[0, 0]]}}})`
        3. Empty: `TimeModules()`
        4. From file: `TimeModules(path="communities.csv")`

        Args:
            raw_modules: Module data in one of two formats:
                - Standard format: {module_label: {(node, time), ...}}
                  Example: {0: {(0, 0), (0, 1), (1, 0)}, 1: {(2, 0)}}
                - Segment format: {module_label: {"nodes": {node: [[start, end], ...]}}}
                  Example: {0: {"nodes": {"0": [[0, 2]], "1": [[0, 0]]}}}
                  Each segment [start, end] is inclusive.
                If None and no path given, creates empty TimeModules.
            path: Path to a file to load from. Supports .csv, .json, .txt formats.
                Auto-detects format from file extension. For JSON, auto-detects
                standard vs segment-based format.

        Raises:
            ValueError: If both raw_modules and path are provided, or if
                file format is not supported.
        """
        # Handle initialization from file
        if path is not None:
            if raw_modules is not None:
                raise ValueError("Cannot provide both raw_modules and path")
            raw_modules = self._load_from_file(path)

        # Default to empty dict if nothing provided
        if raw_modules is None:
            raw_modules = {}

        # Auto-detect and convert segment format to standard format
        raw_modules = self._normalize_input_format(raw_modules)

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

    @staticmethod
    def _normalize_input_format(
        data: dict[int, set[tuple[int, int]]] | dict[int, dict] | dict,
    ) -> dict[int, set[tuple[int, int]]]:
        """Convert segment format to standard format if needed.

        Args:
            data: Module data in either standard or segment format.

        Returns:
            Module data in standard format: {module_label: {(node, time), ...}}.

        Raises:
            TypeError: If the input format is not recognized.

        Supported input formats:
            1. Standard format: {module: {(node, time), ...}}
            2. Segment format with "nodes" key: {module: {"nodes": {node: [[start, end], ...]}}}
            3. Segment format without "nodes" key: {module: {node: [[start, end], ...]}}
            4. List format: {module: [[node, time], ...]}
        """
        if not data:
            return {}

        # Check first value to detect format
        first_key = next(iter(data.keys()))
        first_value = next(iter(data.values()))

        # If first value is a set, it's already standard format
        if isinstance(first_value, set):
            # Validate that it contains tuples
            if first_value:
                sample = next(iter(first_value))
                if not isinstance(sample, tuple) or len(sample) != 2:
                    raise TypeError(
                        f"Invalid standard format. Expected set of (node, time) tuples, "
                        f"but got set containing {type(sample).__name__}.\n"
                        f"Expected: {{module_label: {{(node, time), ...}}}}\n"
                        f"Example: {{0: {{(0, 0), (0, 1), (1, 0)}}}}"
                    )
            return data  # type: ignore

        # If first value is a dict, check what kind
        if isinstance(first_value, dict):
            # Check if it has the "nodes" wrapper key
            if "nodes" in first_value:
                # Segment format WITH "nodes" key
                raw_modules: dict[int, set[tuple[int, int]]] = {}
                for module_key, content in data.items():
                    try:
                        module = int(module_key)
                    except (ValueError, TypeError) as e:
                        raise TypeError(
                            f"Module key must be convertible to int, got {type(module_key).__name__}: {module_key}"
                        ) from e
                    raw_modules[module] = set()
                    nodes_data = content.get("nodes", {})
                    if not isinstance(nodes_data, dict):
                        raise TypeError(
                            f"'nodes' value must be a dict, got {type(nodes_data).__name__}.\n"
                            f"Expected: {{module: {{'nodes': {{node: [[start, end], ...]}}}}}}"
                        )
                    for node_key, segments in nodes_data.items():
                        try:
                            node = int(node_key)
                        except (ValueError, TypeError) as e:
                            raise TypeError(
                                f"Node key must be convertible to int, got {type(node_key).__name__}: {node_key}"
                            ) from e
                        if not isinstance(segments, list):
                            raise TypeError(
                                f"Segments must be a list of [start, end] pairs, got {type(segments).__name__}.\n"
                                f"Expected: [[start, end], [start, end], ...]"
                            )
                        for segment in segments:
                            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                                raise TypeError(
                                    f"Each segment must be a [start, end] pair, got {segment}.\n"
                                    f"Expected: [start_time, end_time] where both are integers"
                                )
                            start, end = int(segment[0]), int(segment[1])
                            for time in range(start, end + 1):
                                raw_modules[module].add((node, time))
                return raw_modules
            else:
                # Segment format WITHOUT "nodes" key: {module: {node: [[start, end], ...]}}
                first_node_value = next(iter(first_value.values()), None)
                if (
                    isinstance(first_node_value, list)
                    and first_node_value
                    and isinstance(first_node_value[0], (list, tuple))
                ):
                    raw_modules = {}
                    for module_key, nodes_data in data.items():
                        try:
                            module = int(module_key)
                        except (ValueError, TypeError) as e:
                            raise TypeError(
                                f"Module key must be convertible to int, got {type(module_key).__name__}: {module_key}"
                            ) from e
                        raw_modules[module] = set()
                        for node_key, segments in nodes_data.items():
                            try:
                                node = int(node_key)
                            except (ValueError, TypeError) as e:
                                raise TypeError(
                                    f"Node key must be convertible to int, got {type(node_key).__name__}: {node_key}"
                                ) from e
                            for segment in segments:
                                if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                                    raise TypeError(
                                        f"Each segment must be a [start, end] pair, got {segment}"
                                    )
                                start, end = int(segment[0]), int(segment[1])
                                for time in range(start, end + 1):
                                    raw_modules[module].add((node, time))
                    return raw_modules
                else:
                    # Unknown dict format
                    raise TypeError(
                        f"Unrecognized dict format for module {first_key}.\n"
                        f"Expected one of:\n"
                        f"  1. Standard: {{module: {{(node, time), ...}}}}\n"
                        f"  2. Segment with 'nodes': {{module: {{'nodes': {{node: [[start, end], ...]}}}}}}\n"
                        f"  3. Segment without 'nodes': {{module: {{node: [[start, end], ...]}}}}\n"
                        f"Got: {type(first_value).__name__} with first value: {first_node_value}"
                    )

        # If first value is a list, assume list format [[node, time], ...]
        if isinstance(first_value, list):
            raw_modules_list: dict[int, set[tuple[int, int]]] = {}
            for module_key, members in data.items():
                try:
                    module = int(module_key)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"Module key must be convertible to int, got {type(module_key).__name__}: {module_key}"
                    ) from e
                raw_modules_list[module] = set()
                for member in members:
                    if not isinstance(member, (list, tuple)) or len(member) < 2:
                        raise TypeError(
                            f"Each member must be a [node, time] pair, got {member}.\n"
                            f"Expected: {{module: [[node, time], [node, time], ...]}}"
                        )
                    raw_modules_list[module].add((int(member[0]), int(member[1])))
            return raw_modules_list

        # Unknown format
        raise TypeError(
            f"Unrecognized input format. Got module {first_key} with value type {type(first_value).__name__}.\n\n"
            f"Expected one of these formats:\n"
            f"  1. Standard format:\n"
            f"     {{module_label: {{(node, time), ...}}}}\n"
            f"     Example: {{0: {{(0, 0), (0, 1), (1, 0)}}}}\n\n"
            f"  2. Segment format (with 'nodes' key):\n"
            f"     {{module_label: {{'nodes': {{node: [[start, end], ...]}}}}}}\n"
            f"     Example: {{0: {{'nodes': {{'0': [[0, 5]], '1': [[2, 8]]}}}}}}\n\n"
            f"  3. Segment format (without 'nodes' key):\n"
            f"     {{module_label: {{node: [[start, end], ...}}}}\n"
            f"     Example: {{0: {{'0': [[0, 5]], '1': [[2, 8]]}}}}\n\n"
            f"  4. List format:\n"
            f"     {{module_label: [[node, time], ...]}}\n"
            f"     Example: {{0: [[0, 0], [0, 1], [1, 0]]}}"
        )

    @staticmethod
    def _load_from_file(path: str | Path) -> dict[int, set[tuple[int, int]]]:
        """Load raw modules from a file, auto-detecting format.

        Args:
            path: Path to file (.csv, .json, or .txt).

        Returns:
            Raw modules dict.

        Raises:
            ValueError: If file format is not supported.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return TimeModules._load_csv(path)
        if suffix == ".json":
            return TimeModules._load_json(path)
        if suffix == ".txt":
            return TimeModules._load_txt(path)

        raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .csv, .json, .txt")

    @staticmethod
    def _load_csv(
        path: Path,
        sep: str = ",",
        has_header: bool = True,
    ) -> dict[int, set[tuple[int, int]]]:
        """Load from CSV file."""
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

        return dict(raw_modules)

    @staticmethod
    def _load_json(path: Path) -> dict[int, set[tuple[int, int]]]:
        """Load from JSON file, auto-detecting standard vs segment format.

        Ensures all values (module labels, nodes, times) are converted to int.
        """
        with path.open() as f:
            data = json.load(f)

        raw_modules: dict[int, set[tuple[int, int]]] = {}

        for module_str, content in data.items():
            module = int(module_str)  # Ensure module label is int
            raw_modules[module] = set()

            # Detect format: segment format has {"nodes": {...}}
            if isinstance(content, dict) and "nodes" in content:
                # Segment format: {"nodes": {"0": [[0, 2], [5, 7]], "1": [[0, 0]]}}
                for node_str, segments in content["nodes"].items():
                    node = int(node_str)  # Ensure node is int
                    for segment in segments:
                        start, end = int(segment[0]), int(segment[1])  # Ensure times are int
                        for time in range(start, end + 1):
                            raw_modules[module].add((node, time))
            else:
                # Standard format: [[node, time], ...]
                for node, time in content:
                    raw_modules[module].add((int(node), int(time)))  # Ensure node/time are int

        return raw_modules

    @staticmethod
    def _load_txt(path: Path) -> dict[int, set[tuple[int, int]]]:
        """Load from TXT file (space-separated)."""
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

        return dict(raw_modules)

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

    def get_module(self, label: int) -> TimeModule:
        """Get a TimeModule view for a specific module.

        Args:
            label: Module label.

        Returns:
            TimeModule view object.

        Raises:
            KeyError: If module label doesn't exist.

        Example:
            ```python
            module = tm.get_module(0)
            print(module.nodes)
            print(module.cohesion)
            ```
        """
        return TimeModule(label=label, parent=self)

    def iter_modules(self) -> Iterator[TimeModule]:
        """Iterate over modules as TimeModule objects.

        Yields:
            TimeModule view for each module.

        Example:
            ```python
            for module in tm.iter_modules():
                print(f"Module {module.label}: {module.size} nodes")
            ```
        """
        for label in self._all_modules:
            yield TimeModule(label=label, parent=self)

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

    def to_json(
        self,
        path: str | Path,
        indent: int | None = 2,
        use_segments: bool = False,
    ) -> None:
        """Save time modules to a JSON file.

        Args:
            path: Path to the output file.
            indent: JSON indentation level. None for compact output.
            use_segments: If True, output nodes with time segments instead of
                individual time points. More compact for contiguous time ranges.

        Example:
            ```python
            # Standard format (individual time points):
            tm.to_json("communities.json")
            # Output: {"0": [[0, 0], [0, 1], [0, 2], [1, 0]]}

            # Segment format (time ranges):
            tm.to_json("communities.json", use_segments=True)
            # Output: {"0": {"nodes": {"0": [[0, 2]], "1": [[0, 0]]}}}
            # Each segment is [start, end] (inclusive)
            ```
        """
        path = Path(path)

        if use_segments:
            # Segment format: module -> {nodes: {node: [[start, end], ...]}}
            data = {}
            for module, nodes_times in self._module_to_nodes.items():
                nodes_data = {}
                for node, times in nodes_times.items():
                    segments = self._times_to_segments(times)
                    nodes_data[str(node)] = [[s.start, s.end] for s in segments]
                data[str(module)] = {"nodes": nodes_data}
        else:
            # Standard format: module -> [[node, time], ...]
            data = {
                str(module): [[node, time] for node, time in sorted(members)]
                for module, members in self._raw_modules.items()
            }

        with path.open("w") as f:
            json.dump(data, f, indent=indent)

    @classmethod
    def from_json(cls, path: str | Path) -> TimeModules:
        """Load time modules from a JSON file.

        Automatically detects format (standard or segment-based).
        Ensures all values (module labels, nodes, times) are converted to int.

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

        for module_str, content in data.items():
            module = int(module_str)  # Ensure module label is int
            raw_modules[module] = set()

            # Detect format: segment format has {"nodes": {...}}
            if isinstance(content, dict) and "nodes" in content:
                # Segment format: {"nodes": {"0": [[0, 2], [5, 7]], "1": [[0, 0]]}}
                for node_str, segments in content["nodes"].items():
                    node = int(node_str)  # Ensure node is int
                    for segment in segments:
                        start, end = int(segment[0]), int(segment[1])  # Ensure times are int
                        for time in range(start, end + 1):
                            raw_modules[module].add((node, time))
            else:
                # Standard format: [[node, time], ...]
                for node, time in content:
                    raw_modules[module].add((int(node), int(time)))  # Ensure node/time are int

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
