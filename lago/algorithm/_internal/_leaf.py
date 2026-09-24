from ._lago_module import _LagoModule


class Leaf:
    # Fixed layout: a stream of 10**6 time-edges holds ~10**6 of these, and a
    # __dict__ per instance is the largest single cost in that. It also makes
    # attribute reads slightly cheaper, which the hot loops do constantly.
    __slots__ = (
        "_accel_row",
        "_hash",
        "edge_duration",
        "left_time_active_neighbor",
        "module",
        "node",
        "right_time_active_neighbor",
        "time",
        "topo_neighbors",
        "topo_neighbors_from",
    )

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

        # Duration of the interactions carried by this time-node. Every edge
        # incident to a leaf has the same duration (continuous links are split on
        # one global set of instants), so it is a property of the leaf; the
        # LinkStream sets it as it adds edges. 1 for a leaf with no edge yet.
        self.edge_duration: int = 1

        # Row of this leaf in the flat topology of lago.accel, or -1 before one
        # is built. Kept on the leaf rather than in a Leaf-keyed dict because
        # __hash__ above is a Python method, so such a dict costs a Python call
        # per lookup -- once per edge, which is the whole cost being optimised.
        self._accel_row: int = -1

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
