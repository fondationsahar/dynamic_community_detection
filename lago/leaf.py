from lago.module import Module


class Leaf:
    def __init__(
        self,
        node: int,
        time: int,
    ):
        # NOTE refer to the node as str/int or specific Class ?
        self.node = node
        self.time = time

        self.left_time_active_neighbor: Leaf | None = None
        self.right_time_active_neighbor: Leaf | None = None
        self.topo_neighbors = set()

        self.module: Module | None = None

    @property
    def neighbors(self) -> set:
        neighbors = (
            set([self.left_time_active_neighbor, self.right_time_active_neighbor])
            | self.topo_neighbors
        )
        if None in neighbors:
            neighbors.remove(None)
        return neighbors

    def __str__(self) -> str:
        return f"({self.node}, {self.time})"
