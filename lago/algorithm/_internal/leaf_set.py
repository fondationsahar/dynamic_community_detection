"""A set difference over leaves that does not copy the larger side.

Evaluating a move asks what a module would look like without the time-nodes that
are leaving it: ``parent.leaves - M0_leaves``. Materialising that costs
O(|parent|) per evaluation, while the move itself concerns about two time-nodes
-- and parent modules grow with the network, so it is the one remaining term in
candidate evaluation that scales with module size rather than with the move. On
a stream of many nodes over few timesteps the mean parent reaches ~1000 leaves
where |M0| stays at 2, and the copying shows up as superlinear growth.

Most consumers only ever ask whether a leaf is in the difference, which needs no
copy at all. The rest -- iteration, unions -- still work, by building the set
once on first use and reusing it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ._leaf import Leaf


class LeafDifference:
    """``base - removed``, evaluated lazily.

    Construction is O(1). Membership is two set lookups. Anything that needs the
    elements themselves materialises the difference once and caches it.

    The operand sets must not be mutated while this view is alive; within a
    single candidate evaluation they are not.
    """

    __slots__ = ("_base", "_materialised", "_removed")

    def __init__(self, base: set[Leaf], removed: set[Leaf]) -> None:
        self._base = base
        self._removed = removed
        self._materialised: set[Leaf] | None = None

    def _resolve(self) -> set[Leaf]:
        """The difference as a real set, built at most once."""
        if self._materialised is None:
            self._materialised = self._base - self._removed
        return self._materialised

    def __contains__(self, item: object) -> bool:
        return item in self._base and item not in self._removed

    def __len__(self) -> int:
        if self._materialised is not None:
            return len(self._materialised)
        # O(|removed|), which is the small side
        overlap = sum(1 for leaf in self._removed if leaf in self._base)
        return len(self._base) - overlap

    def __bool__(self) -> bool:
        return len(self) > 0

    def __iter__(self) -> Iterator[Leaf]:
        return iter(self._resolve())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LeafDifference):
            other = other._resolve()
        if not isinstance(other, (set, frozenset)):
            return NotImplemented
        # Cheap reject first: the sets this is compared against are usually
        # disjoint from it, so the lengths settle it without materialising.
        if len(self) != len(other):
            return False
        return self._resolve() == other

    def __hash__(self) -> int:  # pragma: no cover - a mutable view is unhashable
        msg = "LeafDifference is a view over mutable sets and is not hashable"
        raise TypeError(msg)

    def __or__(self, other: set[Leaf]) -> set[Leaf]:
        return self._resolve() | other

    def __ror__(self, other: set[Leaf]) -> set[Leaf]:
        return other | self._resolve()

    def isdisjoint(self, other) -> bool:
        return all(leaf not in self for leaf in other)
