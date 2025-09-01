import sys
from typing import List

from lago.leaf import Leaf

# NOTE Add a function to preprocess time scale:
# min should be 0 and min step should be 1 (use pgcd etc.)


class LinkStream:
    def __init__(
        self,
        is_stream_graph: bool = False,
    ):
        self.nodes = set[int]()
        self.degrees: dict[int, float] = {}
        self.min_time = sys.maxsize
        self.max_time = -sys.maxsize
        self.is_stream_graph = is_stream_graph

        self.leaves_dict: dict[tuple[int, int], Leaf] = {}
        self.nodes_durations: dict[int, float] = {}
        self.nb_edges: float = 0

    def add_links(self, links: List[tuple[int, ...]]):
        # NOTE times must be ints such that pgcd of all times is 1
        # Maybe add a specific step to normalize it ? With a specific option ?
        tmp_nodes_durations: dict[int, List[int]] = {}
        for source, target, time in links:
            self.nb_edges += 1
            self.min_time = min(self.min_time, time)
            self.max_time = max(self.max_time, time)
            for node in [source, target]:
                self.nodes.add(node)
                if node not in self.degrees:
                    self.degrees[node] = 0
                    tmp_nodes_durations[node] = [time, time]
                else:
                    tmp_nodes_durations[node] = [
                        min(tmp_nodes_durations[node][0], time),
                        max(tmp_nodes_durations[node][1], time),
                    ]
                self.degrees[node] += 1
                if (node, time) not in self.leaves_dict:
                    self.leaves_dict[(node, time)] = Leaf(
                        node=node,
                        time=time,
                    )

            # Increment topological neighbors
            self.leaves_dict[(source, time)].topo_neighbors.add(
                self.leaves_dict[(target, time)]
            )
            self.leaves_dict[(target, time)].topo_neighbors.add(
                self.leaves_dict[(source, time)]
            )

        if self.is_stream_graph:
            # Compute nodes durations of existence
            # NOTE This is a naive way, existences are just trimmed at left and right extremities.
            # One should allow the user to custom the phases of existence
            self.nodes_durations = {
                node: existence[1] - existence[0] + 1
                for node, existence in tmp_nodes_durations.items()
            }
        else:
            self.nodes_durations = {}

        self.network_duration = self.max_time - self.min_time + 1

        self._compute_time_neighbors()

    def _compute_time_neighbors(self) -> None:
        for node in self.nodes:
            times = sorted(
                set([time for tmp_node, time in self.leaves_dict if tmp_node == node])
            )
            for tm1, tm2 in zip(times[:-1], times[1:]):
                self.leaves_dict[
                    (node, tm1)
                ].right_time_active_neighbor = self.leaves_dict[(node, tm2)]
                self.leaves_dict[
                    (node, tm2)
                ].left_time_active_neighbor = self.leaves_dict[(node, tm1)]

    def get_time_links(self) -> set[tuple[int, int, int]]:
        # NOTE Optimize that
        time_links = set()
        for leaf in self.leaves_dict.values():
            source = leaf.node
            time = leaf.time
            for neighb in leaf.topo_neighbors:
                target = neighb.node
                fsource, ftarget = sorted([source, target])
                time_links.add((fsource, ftarget, time))
        return time_links

    def to_txt(self, path: str) -> None:
        with open(path, "w") as file:
            for triplet in self.get_time_links():
                line = " ".join(map(str, triplet))
                file.write(line + "\n")

    def read_txt(self, path: str) -> None:
        with open(path, "r") as file:
            links = [tuple(map(int, line.strip().split())) for line in file]
        self.add_links(links)

    @property
    def nb_timesteps(self) -> int:
        timesteps = set([time for _, time in self.leaves_dict.keys()])
        return len(timesteps)

    @property
    def nb_time_edges(self) -> int:
        return len(self.get_time_links())

    @property
    def nb_nodes(self) -> int:
        nodes = set([node for node, _ in self.leaves_dict.keys()])
        return len(nodes)
