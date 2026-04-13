"""
Type definitions for the longitudinal plotting library.

This module defines type aliases used throughout the longitudinal plotting library
to improve code readability and type checking.
"""

from typing import Any, Optional, Union

# Basic type aliases
NodeId = int
TimePoint = int
Weight = float
ModuleLabel = Union[str, int]
"""Module label can be string or integer."""

# Backward compat alias
CommunityLabel = ModuleLabel

# Module data types
ModuleMember = tuple[NodeId, TimePoint]
"""A module member represented as (node_id, time_point)."""

# Backward compat alias
CommunityMember = ModuleMember

Modules = dict[ModuleLabel, list[ModuleMember]]
"""Dictionary mapping module labels to their members."""

# Backward compat alias
Communities = Modules

# Time segment types
TimeSegment = list[int]
"""A time segment represented as [start, end]."""

NodeTimeSegments = dict[NodeId, list[TimeSegment]]
"""Dictionary mapping node IDs to their time segments."""

ModuleNodesSegments = dict[ModuleLabel, NodeTimeSegments]
"""Dictionary mapping module labels to node time segments."""

# Backward compat alias
CommunityNodesSegments = ModuleNodesSegments

# Link/Edge types
TimeLink = tuple[NodeId, NodeId, TimePoint, Weight]
"""A time link represented as (source, target, time, weight)."""

DelayedTimeLink = tuple[NodeId, NodeId, TimePoint, TimePoint, Weight]
"""A delayed time link represented as (source, target, source_time, target_time, weight)."""

TimeLinks = list[TimeLink]
"""List of time links."""

DelayedTimeLinks = list[DelayedTimeLink]
"""List of delayed time links."""

# Mapping types
NodesMapping = dict[NodeId, int]
"""Dictionary mapping original node IDs to display indices."""

TimeNodeModuleMapping = dict[tuple[NodeId, TimePoint], ModuleLabel]
"""Dictionary mapping (node, time) tuples to module labels."""

# Backward compat alias
TimeNodeCommunityMapping = TimeNodeModuleMapping

ColorMapping = dict[ModuleLabel, str | tuple[float, float, float]]
"""Dictionary mapping module labels to colors (hex string or RGB tuple)."""

# Node state types
NodeTimeRange = dict[NodeId, TimePoint]
"""Dictionary mapping node IDs to time points (for start/end times)."""

# Focus types
NodeFocus = list[NodeId]
"""List of node IDs to focus on."""

TimeFocus = Optional[TimePoint]
"""Optional time point to focus on."""

# Geometry types
Point2D = tuple[float, float]
"""A 2D point represented as (x, y)."""

Center = Point2D
"""Alias for Point2D used as center coordinates."""

# Type for linkstream (external dependency)
LinkStream = Any
"""Type alias for the external LinkStream class."""

# Type for TimeModules (external dependency from lago)
TimeModules = Any
"""Type alias for the external TimeModules class (from lago library)."""

# Union type for modules input (dict or TimeModules)
ModulesInput = Union[Modules, Any]
"""Modules can be provided as dict or TimeModules object."""

# Backward compat alias
CommunitiesInput = ModulesInput
