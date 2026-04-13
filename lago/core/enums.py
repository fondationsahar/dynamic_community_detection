"""Enumerations for the lago library."""

from __future__ import annotations

from enum import Enum, auto


class LinkStreamMode(Enum):
    """Defines the type of temporal network.

    Attributes:
        INSTANTANEOUS: Discrete time, same time for both endpoints.
        CONTINUOUS: Links have duration.
        DELAYED: Different times for source and target endpoints.
    """

    INSTANTANEOUS = auto()
    CONTINUOUS = auto()
    DELAYED = auto()

    @classmethod
    def from_flags(
        cls,
        continuous: bool = False,
        delayed: bool = False,
    ) -> LinkStreamMode:
        """Create a mode from boolean flags.

        Args:
            continuous: If True, links have duration.
            delayed: If True, source and target have different timestamps.

        Returns:
            The corresponding LinkStreamMode.

        Raises:
            ValueError: If both continuous and delayed are True.
        """
        if continuous and delayed:
            raise ValueError(
                "LinkStream cannot be both continuous and delayed. Choose one mode or the other."
            )
        if continuous:
            return cls.CONTINUOUS
        if delayed:
            return cls.DELAYED
        return cls.INSTANTANEOUS


class LexType(Enum):
    """Expectation type for longitudinal modularity computation.

    Attributes:
        CM: Coexistence Modularity - uses actual co-presence times.
        JM: Joint Modularity - uses community duration.
        MM: Mean Modularity - uses geometric mean of node durations (default).
    """

    CM = auto()  # Coexistence Modularity
    JM = auto()  # Joint Modularity
    MM = auto()  # Mean Modularity

    @classmethod
    def from_string(cls, value: str) -> LexType:
        """Convert string to LexType (for backward compatibility).

        Args:
            value: String "CM", "JM", or "MM".

        Returns:
            The corresponding LexType.

        Raises:
            ValueError: If value is not a valid type.
        """
        value_upper = value.upper()
        if value_upper == "CM":
            return cls.CM
        if value_upper == "JM":
            return cls.JM
        if value_upper == "MM":
            return cls.MM
        raise ValueError(f'LexType must be "CM", "JM", or "MM", got "{value}"')
