"""TimeEdge: Represents a temporal edge in a link stream."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._leaf import Leaf


class TimeEdge:
    """Represents a temporal edge connecting to a target leaf.

    Attributes:
        target: The target Leaf this edge points to.
        weight: The weight of the edge.
        duration: The duration of the edge (for continuous links).
    """

    __slots__ = ("duration", "target", "weight")

    def __init__(
        self,
        target: Leaf,
        weight: float = 1.0,
        duration: int = 1,
    ) -> None:
        """Initialize a TimeEdge.

        Args:
            target: The target Leaf this edge points to.
            weight: The weight of the edge. Defaults to 1.0.
            duration: The duration of the edge. Defaults to 1.
        """
        self.target = target
        self.weight = weight
        self.duration = duration

    def __hash__(self) -> int:
        """Return hash based on target identity, weight, and duration."""
        return hash((id(self.target), self.weight, self.duration))

    def __eq__(self, other: object) -> bool:
        """Check equality based on target identity, weight, and duration."""
        if not isinstance(other, TimeEdge):
            return NotImplemented
        return (
            self.target is other.target
            and self.weight == other.weight
            and self.duration == other.duration
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TimeEdge(target={self.target}, weight={self.weight}, duration={self.duration})"
