from ._lago_module import _LagoModule


class Leaf:
    def __init__(
        self,
        node: int,
        time: int,
    ):
        # NOTE refer to the node as str/int or specific Class ?
        self.node = node
        self.time = time

        # Hash derived from the (node, time) identity rather than from id().
        # Leaves are unique per (node, time) within a LinkStream, so this keeps
        # the default identity __eq__ consistent while making the iteration order
        # of every set of leaves a function of the data instead of of the memory
        # allocator. That is what makes lago_modules reproducible.
        self._hash = hash((node, time))

        self.left_time_active_neighbor: Leaf | None = None
        self.right_time_active_neighbor: Leaf | None = None

        # Neighbors self -> other (standard for undirected)
        self.topo_neighbors = set()
        # Neighbors other -> self
        self.topo_neighbors_from: set = set()

        self.module: _LagoModule | None = None

    def __hash__(self) -> int:
        return self._hash

    @property
    def sort_key(self) -> tuple[int, int]:
        """Stable ordering key, for call sites that need a deterministic order."""
        return (self.node, self.time)

    @property
    def neighbors(self) -> set:
        neighbors = {self.left_time_active_neighbor, self.right_time_active_neighbor}
        neighbors |= {neighb.target for neighb in self.topo_neighbors | self.topo_neighbors_from}
        neighbors.discard(None)
        return neighbors

    def __str__(self) -> str:
        return f"({self.node}, {self.time})"
