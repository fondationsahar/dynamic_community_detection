import copy
import sys
from typing import List

from lago.leaf import Leaf
from lago.time_edge import TimeEdge

# NOTE Add a function to preprocess time scale:
# min should be 0 and min step should be 1 (use pgcd etc.)
# TODO Add a threshold for convergence to avoid misleading floating issues (especially in weighted networks)


class LinkStream:
    def __init__(
        self,
        continuous: bool = False,
        directed: bool = False,
        delayed: bool = False,
        partite_mapping: dict[int, int] = {},
    ):
        self.continuous: bool = continuous
        self.directed: bool = directed
        self.delayed: bool = delayed
        self.partite_mapping: dict[int, int] = partite_mapping

        # TODO Raise error if continuous and delayed

        self.nodes = set[int]()

        self.degrees: dict[int, float] = {}
        self.degrees_in: dict[int, float] = {}
        self.degrees_out: dict[int, float] = {}

        self.min_time = sys.maxsize
        self.max_time = -sys.maxsize

        self.leaves_dict: dict[tuple[int, int], Leaf] = {}
        self.nodes_durations: dict[int, float] = {}
        self.nb_edges: float = 0
        self.weight: float = 0

        self.time_instants = set[int]()

    def add_links(self, links: List[tuple[int, ...]]):
        # NOTE times must be ints such that pgcd of all times is 1
        # Maybe add a specific step to normalize it ? With a specific option ?

        # TODO Include delayed linkstreams !
        for link in links:
            if len(link) > 3:
                source, target, time, weight = link
            else:
                source, target, time = link
                weight = 1
            self.nb_edges += 1
            self.weight += weight
            self.min_time = min(self.min_time, time)
            self.max_time = max(self.max_time, time)
            for node in [source, target]:
                self.nodes.add(node)
                if (node, time) not in self.leaves_dict:
                    self.leaves_dict[(node, time)] = Leaf(
                        node=node,
                        time=time,
                    )
                if self.directed:
                    continue

                if node not in self.degrees:
                    self.degrees[node] = 0
                self.degrees[node] += weight

            if self.directed:
                if source not in self.degrees_out:
                    self.degrees_out[source] = 0
                self.degrees_out[source] += weight
                if target not in self.degrees_in:
                    self.degrees_in[target] = 0
                self.degrees_in[target] += weight

                # Increment topological neighbors
                # NOTE Same edge is declared two times
                # Or, each edge can be seen as two wires
                # TODO Clarify that
                self.leaves_dict[(source, time)].topo_neighbors.add(
                    TimeEdge(self.leaves_dict[(target, time)], weight)
                )
                self.leaves_dict[(target, time)].topo_neighbors_from.add(
                    TimeEdge(self.leaves_dict[(source, time)], weight)
                )

            else:
                # Increment topological neighbors
                # NOTE Same edge is declared two times
                # Or, each edge can be seen as two wires
                # TODO Clarify that
                self.leaves_dict[(source, time)].topo_neighbors.add(
                    TimeEdge(self.leaves_dict[(target, time)], weight)
                )
                self.leaves_dict[(target, time)].topo_neighbors.add(
                    TimeEdge(self.leaves_dict[(source, time)], weight)
                )

        self.nodes_durations = {}

        self.network_duration = self.max_time - self.min_time + 1

        self._compute_time_neighbors()

    def add_continous_links(self, links: List[tuple[int, ...]]):
        # NOTE times must be ints such that pgcd of all times is 1
        # Maybe add a specific step to normalize it ? With a specific option ?

        for link in links:
            if len(link) > 4:
                source, target, time, duration, initial_weight = link
            else:
                source, target, time, duration = link
                initial_weight = 1
            weight = (
                initial_weight * duration
            )  # Ponderation in continuous configuration
            self.nb_edges += 1  # Not relevant for continuous case, TODO arrange that
            self.weight += weight
            # NOTE A way to deal with the instantaneous / continous case is to say
            #   - instantaneous interactions are continous that last 1 (duration = 1)
            self.min_time = min(self.min_time, time)
            self.max_time = max(self.max_time, time + duration)
            self.time_instants.add(time)
            self.time_instants.add(time + duration)
            for node in [source, target]:
                self.nodes.add(node)
                if (node, time) not in self.leaves_dict:
                    self.leaves_dict[(node, time)] = Leaf(
                        node=node,
                        time=time,
                    )
                if self.directed:
                    continue

                if node not in self.degrees:
                    self.degrees[node] = 0
                self.degrees[node] += weight

            if self.directed:
                if source not in self.degrees_out:
                    self.degrees_out[source] = 0
                self.degrees_out[source] += weight
                if target not in self.degrees_in:
                    self.degrees_in[target] = 0
                self.degrees_in[target] += weight

                # Increment topological neighbors
                # NOTE Same edge is declared two times
                # Or, each edge can be seen as two wires
                # TODO Clarify that
                self.leaves_dict[(source, time)].topo_neighbors.add(
                    TimeEdge(
                        target=self.leaves_dict[(target, time)],
                        weight=initial_weight,
                        duration=duration,
                    )
                )
                self.leaves_dict[(target, time)].topo_neighbors_from.add(
                    TimeEdge(
                        target=self.leaves_dict[(source, time)],
                        weight=initial_weight,
                        duration=duration,
                    )
                )

            else:
                # Increment topological neighbors
                # NOTE Same edge is declared two times
                # Or, each edge can be seen as two wires
                # TODO Clarify that
                self.leaves_dict[(source, time)].topo_neighbors.add(
                    TimeEdge(
                        target=self.leaves_dict[(target, time)],
                        weight=initial_weight,
                        duration=duration,
                    )
                )
                self.leaves_dict[(target, time)].topo_neighbors.add(
                    TimeEdge(
                        target=self.leaves_dict[(source, time)],
                        weight=initial_weight,
                        duration=duration,
                    )
                )

        self.nodes_durations = {}

        self.network_duration = self.max_time - self.min_time + 1
        print("Splitting continuous linkstream...")
        self._split_continous_linkstream()
        print("\tOk")
        self._compute_time_neighbors()

    def _split_continous_linkstream(self) -> None:
        # Only if self.continous, raise error if not
        old_leaves_dict = copy.deepcopy(self.leaves_dict)
        self.leaves_dict: dict[tuple[int, int], Leaf] = {}

        for (node, time), leaf in old_leaves_dict.items():
            # Split all edges regarding self.time_instants
            for time_edge in leaf.topo_neighbors:
                # NOTE This may be a bottle neck. Eyes on it.
                tmp_time_instants = self.time_instants & set(
                    range(time, time + time_edge.duration + 1)
                )
                tmp_time_instants = sorted(tmp_time_instants)
                for time_start, time_end in zip(
                    tmp_time_instants[:-1], tmp_time_instants[1:]
                ):
                    duration = time_end - time_start
                    if (node, time_start) not in self.leaves_dict:
                        self.leaves_dict[(node, time_start)] = Leaf(
                            node=node,
                            time=time_start,
                        )
                    target_node = time_edge.target.node
                    if (target_node, time_start) not in self.leaves_dict:
                        self.leaves_dict[(target_node, time_start)] = Leaf(
                            node=target_node,
                            time=time_start,
                        )

                    self.leaves_dict[(node, time_start)].topo_neighbors.add(
                        TimeEdge(
                            target=self.leaves_dict[(target_node, time_start)],
                            weight=time_edge.weight,
                            duration=duration,  # stored here only for retrieval, not used in LAGO. TODO Arrange that
                        )
                    )

            for time_edge in leaf.topo_neighbors_from:
                tmp_time_instants = self.time_instants & set(
                    range(time, time + time_edge.duration + 1)
                )
                tmp_time_instants = sorted(tmp_time_instants)
                for time_start, time_end in zip(
                    tmp_time_instants[:-1], tmp_time_instants[1:]
                ):
                    duration = time_end - time_start
                    if (node, time_start) not in self.leaves_dict:
                        self.leaves_dict[(node, time_start)] = Leaf(
                            node=node,
                            time=time_start,
                        )
                    target_node = time_edge.target.node
                    if (target_node, time_start) not in self.leaves_dict:
                        self.leaves_dict[(target_node, time_start)] = Leaf(
                            node=target_node,
                            time=time_start,
                        )

                    self.leaves_dict[(node, time_start)].topo_neighbors_from.add(
                        TimeEdge(
                            target=self.leaves_dict[(target_node, time_start)],
                            weight=time_edge.weight,
                            duration=duration,  # stored here only for retrieval, not used in LAGO. TODO Arrange that
                        )
                    )

    def add_delayed_links(self, links: List[tuple[int, ...]]):
        # NOTE times must be ints such that pgcd of all times is 1
        # Maybe add a specific step to normalize it ? With a specific option ?

        # TODO Include delayed linkstreams !
        for link in links:
            if len(link) > 4:
                source, target, source_time, target_time, weight = link
            else:
                source, target, source_time, target_time = link
                weight = 1
            self.nb_edges += 1
            self.weight += weight
            for time in [source_time, target_time]:
                self.min_time = min(self.min_time, time)
                self.max_time = max(self.max_time, time)
            for node, time in [(source, source_time), (target, target_time)]:
                self.nodes.add(node)
                if (node, time) not in self.leaves_dict:
                    self.leaves_dict[(node, time)] = Leaf(
                        node=node,
                        time=time,
                    )
                if self.directed:
                    continue

                if node not in self.degrees:
                    self.degrees[node] = 0
                self.degrees[node] += weight

            if self.directed:
                if source not in self.degrees_out:
                    self.degrees_out[source] = 0
                self.degrees_out[source] += weight
                if target not in self.degrees_in:
                    self.degrees_in[target] = 0
                self.degrees_in[target] += weight

                # Increment topological neighbors
                self.leaves_dict[(source, source_time)].topo_neighbors.add(
                    TimeEdge(self.leaves_dict[(target, target_time)], weight)
                )
                self.leaves_dict[(target, target_time)].topo_neighbors_from.add(
                    TimeEdge(self.leaves_dict[(source, source_time)], weight)
                )

            else:
                # Increment topological neighbors
                self.leaves_dict[(source, source_time)].topo_neighbors.add(
                    TimeEdge(self.leaves_dict[(target, target_time)], weight)
                )
                self.leaves_dict[(target, target_time)].topo_neighbors.add(
                    TimeEdge(self.leaves_dict[(source, source_time)], weight)
                )

        self.nodes_durations = {}

        self.network_duration = self.max_time - self.min_time + 1

        # TODO Compute the continous active time nodes here

        self._compute_time_neighbors()

    def set_partite(self, partite_mapping: dict[int, int]):
        self.partite_mapping = partite_mapping

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

    def get_time_links(self) -> set[tuple[int, int, int, float]]:
        # NOTE Optimize that
        if self.continuous:
            time_links = set()
            for leaf in self.leaves_dict.values():
                source = leaf.node
                time = leaf.time
                for neighb in leaf.topo_neighbors:
                    target = neighb.target.node
                    weight = neighb.weight
                    duration = neighb.duration
                    target_time = neighb.target.time
                    time_links.add((source, target, time, duration, weight))
            return time_links

        if self.delayed:
            time_links = set()
            for leaf in self.leaves_dict.values():
                source = leaf.node
                source_time = leaf.time
                for neighb in leaf.topo_neighbors:
                    target = neighb.target.node
                    weight = neighb.weight
                    target_time = neighb.target.time
                    time_links.add((source, target, source_time, target_time, weight))
            return time_links

        time_links = set()
        for leaf in self.leaves_dict.values():
            source = leaf.node
            time = leaf.time
            for neighb in leaf.topo_neighbors:
                target = neighb.target.node
                weight = neighb.weight
                # fsource, ftarget = sorted([source, target])
                time_links.add((source, target, time, weight))
        return time_links

    def read_txt(self, path: str, columns_order=["source", "target", "time"]) -> None:
        order_mapping = {col: val for val, col in enumerate(columns_order)}
        with open(path, "r") as file:
            # must respect source target time weight (type (for k-partite networks) -> directly in nodes declarations)
            # If weight is not here, set it up to 1.
            links = []
            for rline in file:
                elements = rline.strip().split()
                weight = 1
                if "weight" in columns_order:
                    weight = elements[order_mapping["weight"]]

                if self.continuous:
                    nline = [
                        int(elements[order_mapping["source"]]),
                        int(elements[order_mapping["target"]]),
                        int(elements[order_mapping["time_start"]]),
                        int(elements[order_mapping["duration"]]),
                        float(weight),
                    ]
                else:
                    if self.delayed:
                        nline = [
                            int(elements[order_mapping["source"]]),
                            int(elements[order_mapping["target"]]),
                            int(elements[order_mapping["source_time"]]),
                            int(elements[order_mapping["target_time"]]),
                            float(weight),
                        ]
                    else:
                        nline = [
                            int(elements[order_mapping["source"]]),
                            int(elements[order_mapping["target"]]),
                            int(elements[order_mapping["time"]]),
                            float(weight),
                        ]

                links.append(nline)

        if self.continuous:
            self.add_continous_links(links)
            return
        if self.delayed:
            self.add_delayed_links(links)
            return

        self.add_links(links)

    def to_txt(self, path: str) -> None:
        with open(path, "w") as file:
            for triplet in self.get_time_links():
                line = " ".join(map(str, triplet))
                file.write(line + "\n")

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
