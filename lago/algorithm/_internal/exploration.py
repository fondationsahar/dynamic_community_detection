"""Work queue for the LAGO exploration loops.

The greedy search repeatedly takes a candidate out of a pending collection,
possibly moves it, and puts its neighbours back. Which candidate comes out first
is the "starting point" the algorithm is sensitive to: on the reference fixture,
five different orders give five different partitions spanning 0.7197 to 0.7258
L-modularity, and on a 120-node planted stream the spread is 0.6073 to 0.6247.
Trying several orders and keeping the best is exactly what ``nb_iter`` is for.

Without a generator this is a plain ``set`` consumed by ``set.pop()`` -- the
historical behaviour, kept verbatim so that default runs stay reproducible and
comparable against previously published results. With one, the same items come
out in an order that is a function of the generator's seed alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import random
    from collections.abc import Iterable, Iterator


class ExplorationQueue:
    """A set of pending candidates with a controlled removal order.

    Supports what the exploration loops need: truth-testing, ``len``, ``pop``,
    ``add``, ``update`` / ``|=``, membership and iteration.

    Args:
        items: Initial candidates.
        rng: ``None`` for the canonical order (a plain ``set.pop()``), or a
            seeded ``random.Random`` selecting a reproducible shuffled order.
            The generator is consumed, not copied, so successive queues drawn
            from the same generator explore differently while the run as a whole
            stays a function of its seed.
    """

    __slots__ = ("_order", "_pending", "_rng")

    def __init__(self, items: Iterable[Any] = (), rng: random.Random | None = None) -> None:
        self._pending: set[Any] = set(items)
        self._rng = rng
        if rng is None:
            self._order: list[Any] | None = None
        else:
            # Only a *proposed* order; _pending stays authoritative, so an item
            # removed and added back is never served twice.
            self._order = list(self._pending)
            rng.shuffle(self._order)

    def __bool__(self) -> bool:
        return bool(self._pending)

    def __len__(self) -> int:
        return len(self._pending)

    def __contains__(self, item: Any) -> bool:
        return item in self._pending

    def __iter__(self) -> Iterator[Any]:
        return iter(self._pending)

    def pop(self) -> Any:
        """Remove and return the next candidate.

        Raises:
            KeyError: If the queue is empty.
        """
        if self._order is None:
            return self._pending.pop()

        order = self._order
        pending = self._pending
        while order:
            item = order.pop()
            if item in pending:
                pending.discard(item)
                return item
        return pending.pop()

    def add(self, item: Any) -> None:
        """Add one candidate back into the pending set."""
        if item not in self._pending:
            self._pending.add(item)
            if self._order is not None:
                self._order.append(item)

    def update(self, items: Iterable[Any]) -> None:
        """Add several candidates back. Mirrors ``set |= other``."""
        if self._order is None:
            self._pending.update(items)
            return
        # Shuffle the newcomers so that a batch does not impose its own order
        fresh = [item for item in items if item not in self._pending]
        self._rng.shuffle(fresh)
        self._pending.update(fresh)
        self._order.extend(fresh)

    def __ior__(self, items: Iterable[Any]) -> ExplorationQueue:
        self.update(items)
        return self
