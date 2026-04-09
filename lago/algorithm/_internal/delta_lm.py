from collections import defaultdict
from itertools import combinations_with_replacement

import lago.core.utils as tls
from lago.core.linkstream import LinkStream

from ._leaf import Leaf


class DeltaLongitudinalModularityComputer:
    """Computer of delta longitudinal modularity
    when moving submodules for LAGO.
    """

    def __init__(
        self,
        linkstream: LinkStream,
        lex: str = "JM",
        gamma: float = 1,
        omega: float = 2,
    ):
        self.linkstream = linkstream
        self.lex = lex
        self.gamma = gamma
        self.omega = omega

    def M0_to_Mx(
        self,
        M0_leaves: set[Leaf],
        M0_time_segments: dict[int, list[list[Leaf]]],
        Mx_leaves: set[Leaf],
        partite_mapping: dict[int, int],
    ):
        """Compute Δ L-Modularity of submodule M0 joining module Mx.
            L-Modularity is made up of three terms, each computed in a dedicated function:
            - Number of edges within modules
            - Expected number of edges
            - Community Switch Counts (term for time regularisation)
            Note that it returns a non normalized Δ L-Modularity.

        Args:
            M0_leaves (set): set of M0 leaves (time nodes)
            M0_time_segments (dict): time segments of nodes leaving in module M0
            Mx_leaves (set): set of Mx leaves

        Returns:
            float: Δ L-Modularity of the movement
        """

        weight_diff = self._get_weight_diff(M0_leaves, Mx_leaves)
        if not self.linkstream.directed:
            weight_diff *= 2

        if self.lex == "MM":
            expectation_diff = self._get_expectation_mm_part(
                M0_leaves,
                Mx_leaves,
                partite_mapping,
            )
        elif self.lex == "JM":
            expectation_diff = self._get_expectation_jm_part(
                M0_leaves,
                Mx_leaves,
                partite_mapping,
            )
        else:
            raise Exception("Wrong lex value")

        csc_diff = self._get_csc_diff(M0_time_segments, Mx_leaves)

        ls = self.linkstream

        delta_lm = (
            weight_diff
            - self.gamma * expectation_diff
            + self.omega * csc_diff * ls.weight / ls.nb_edges
        )

        return delta_lm

    def _get_weight_diff(self, M0_leaves: set[Leaf], Mx_leaves: set[Leaf]):
        """Compute delta number of edges within module Mx if M0 joins Mx.

        Args:
            M0_leaves (set): M0 submodule time nodes
            Mx_leaves (set): Mx module time nodes

        Returns:
            int: delta number of edges
        """
        all_neighbs = list()
        for leaf in Mx_leaves:
            all_neighbs += list(
                [neighb for neighb in leaf.topo_neighbors | leaf.topo_neighbors_from]
            )

        # Only keep inventoried neighbors that are in M0
        # Note: cannot use a set here because we want to keep duplicated time nodes
        neighbors_weights = [
            neighb.weight * neighb.duration for neighb in all_neighbs if neighb.target in M0_leaves
        ]
        return sum(neighbors_weights)

    def _get_csc_diff(self, M0_time_segments, Mx_leaves) -> float:
        """Compute the delta CSCs (Community Switches Counts)
        if module Mx is joined by submodule M0.
        CSC for a node is the number of time it leaves and enter time module, minus 1.

        Args:
            M0_time_segments (dict): every node time segment for
                the time it belongs to module M0.
            Mx_leaves (set): Leaves (time nodes) of module Mx.

        Returns:
            float: delta CSC (Community Switch Counts)
        """

        cscs = 0
        # Iterate on node and its time segments when belonging to M0
        for _, segments in M0_time_segments.items():
            for segment in segments:
                # Get the previous active time occurence of the node
                # which is left time neighbor
                neighb_left = segment[0].left_time_active_neighbor
                # If it exists and not belong to Mx, the joined module,
                # it is counted as a leave/enter configuration for the node.
                if neighb_left and neighb_left not in Mx_leaves:
                    cscs += 1

                # Get the next active time occurence of the node
                # which is right time neighbor
                neighb_right = segment[-1].right_time_active_neighbor
                # If it exists and not belong to Mx, the joined module,
                # it is counted as a leave/enter configuration for the node.
                if neighb_right and neighb_right not in Mx_leaves:
                    cscs += 1

        # Total cscs are weighted by the time resolution parameter omega
        csc_diff = -cscs / 2

        return csc_diff

    def _get_expectation_jm_part(
        self,
        M0_leaves: set[Leaf],
        Mx_leaves: set[Leaf],
        partite_mapping: dict[int, int],
    ):
        """Compute the delta number of expected edges within module Mx
        if joined by submodule M0, regarding the Joint-Membership expectation.

        Args:
            M0_leaves (set): M0 submodule time nodes
            Mx_leaves (set): Mx module time nodes

        Returns:
            float: delta value for expected number of edges
        """

        if partite_mapping:
            return self._get_expectation_jm_kpartite_part(
                M0_leaves,
                Mx_leaves,
                partite_mapping,
            )

        duration_Cx = tls.get_module_duration(Mx_leaves)
        duration_Cx_U_C0 = tls.get_module_duration(Mx_leaves | M0_leaves)
        if self.linkstream.directed:
            degree_in_Cx = self._sum_degrees_in(Mx_leaves)
            degree_in_Cx_U_C0 = self._sum_degrees_in(Mx_leaves | M0_leaves)
            degree_out_Cx = self._sum_degrees_out(Mx_leaves)
            degree_out_Cx_U_C0 = self._sum_degrees_out(Mx_leaves | M0_leaves)
            expectation_diff = (
                degree_in_Cx_U_C0 * degree_out_Cx_U_C0 * duration_Cx_U_C0
                - degree_in_Cx * degree_out_Cx * duration_Cx
            ) / (2 * self.linkstream.weight * self.linkstream.network_duration)
            # expectation_diff = 0
            # expectation_diff = (
            #     degree_in_Cx_U_C0 * degree_out_Cx_U_C0 * duration_Cx_U_C0
            #     - degree_in_Cx * degree_out_Cx * duration_Cx
            # ) / (2 * self.linkstream.weight * self.linkstream.network_duration)
        else:
            degree_Cx = self._sum_degrees(Mx_leaves)
            degree_Cx_U_C0 = self._sum_degrees(Mx_leaves | M0_leaves)
            expectation_diff = (
                degree_Cx_U_C0**2 * duration_Cx_U_C0 - degree_Cx**2 * duration_Cx
            ) / (4 * self.linkstream.weight * self.linkstream.network_duration)

        return expectation_diff

    def _get_expectation_jm_kpartite_part(
        self,
        M0_leaves: set[Leaf],
        Mx_leaves: set[Leaf],
        partite_mapping: dict[int, int],
    ):
        """Compute the delta number of expected edges within module Mx
        if joined by submodule M0, regarding the Joint-Membership expectation.

        Args:
            M0_leaves (set): M0 submodule time nodes
            Mx_leaves (set): Mx module time nodes

        Returns:
            float: delta value for expected number of edges
        """

        nodes_in_Cx = [*{*[leaf.node for leaf in Mx_leaves]}]
        nodes_in_Cx_U_C0 = [*{*[leaf.node for leaf in Mx_leaves | M0_leaves]}]
        in_Cx = {node: node in nodes_in_Cx for node in nodes_in_Cx_U_C0}
        duration_Cx = tls.get_module_duration(Mx_leaves)
        duration_Cx_U_C0 = tls.get_module_duration(Mx_leaves | M0_leaves)

        if self.linkstream.directed:
            expectation_diff = 0
            for node1, node2 in combinations_with_replacement(nodes_in_Cx_U_C0, 2):
                if partite_mapping.get(node1, -1) == partite_mapping.get(node2, -2):
                    continue
                degree_in1 = self.linkstream.degrees_in[node1]
                degree_in2 = self.linkstream.degrees_in[node2]
                degree_out1 = self.linkstream.degrees_out[node1]
                degree_out2 = self.linkstream.degrees_out[node2]

                expectation_diff += (degree_in1 * degree_out2 + degree_in2 * degree_out1) * (
                    duration_Cx_U_C0 - in_Cx[node1] * in_Cx[node2] * duration_Cx
                )
            expectation_diff /= 2 * self.linkstream.weight * self.linkstream.network_duration

        else:
            # In k-partite networks, only interactions between different partites are expected
            expectation_diff = 0
            for node1, node2 in combinations_with_replacement(nodes_in_Cx_U_C0, 2):
                if partite_mapping.get(node1, -1) == partite_mapping.get(node2, -2):
                    continue
                degree1 = self.linkstream.degrees[node1]
                degree2 = self.linkstream.degrees[node2]

                expectation_diff += (
                    degree1
                    * degree2
                    * (duration_Cx_U_C0 - in_Cx[node1] * in_Cx[node2] * duration_Cx)
                )
            expectation_diff /= 4 * self.linkstream.weight * self.linkstream.network_duration

        return expectation_diff

    def _get_expectation_mm_part(
        self,
        M0_leaves: set[Leaf],
        Mx_leaves: set[Leaf],
        partite_mapping: dict[int, int],
    ):
        """Compute the delta number of expected edges within module Mx
        if joined by submodule M0, regarding the Mean-Membership expectation.

        Args:
            M0_leaves (set): M0 submodule time nodes
            Mx_leaves (set): Mx module time nodes

        Returns:
            float: delta value for expected number of edges
        """

        nodes_durations = {
            "raw_Cx": tls.get_nodes_durations(
                module_leaves=Mx_leaves,
            ),
            "Cx_U_C0": tls.get_nodes_durations(
                module_leaves=M0_leaves | Mx_leaves,
            ),
        }

        expectation_diff = 0

        all_nodes = {leaf.node for leaf in M0_leaves | Mx_leaves}
        for node1, node2 in combinations_with_replacement(all_nodes, 2):
            expectation_diff += self._partial_expectation_mm_diff(
                node1, node2, nodes_durations, partite_mapping
            )

        return expectation_diff

    def _partial_expectation_mm_diff(
        self,
        node1: int,
        node2: int,
        nodes_durations: dict[str, defaultdict[int, float]],
        partite_mapping: dict[int, int],
    ):
        """Compute the delta expectation number of edges between node1 and node2,
        regarding the Mean-Membership expectation.

        Args:
            node1 (int): linkstream node id
            node2 (int): linkstream node id
            nodes_durations (dict): durations of nodes existences in modules

        Returns:
            float: expected number of edges between node1 and node2.
        """

        # In k-partite networks, only interactions between different partites are expected
        if partite_mapping.get(node1, -1) == partite_mapping.get(node2, -2):
            return 0

        if self.linkstream.directed:
            degrees_part = self.linkstream.degrees_in.get(
                node1, 0
            ) * self.linkstream.degrees_out.get(node2, 0) + self.linkstream.degrees_out.get(
                node1, 0
            ) * self.linkstream.degrees_in.get(node2, 0)
            degrees_part *= 2 ** (node1 != node2)
        else:
            degrees_part = (
                2 ** (node1 != node2)
                * self.linkstream.degrees[node1]
                * self.linkstream.degrees[node2]
            )

        numerator: float = degrees_part * (
            nodes_durations["Cx_U_C0"][node1] ** 0.5 * nodes_durations["Cx_U_C0"][node2] ** 0.5
            - nodes_durations["raw_Cx"][node1] ** 0.5 * nodes_durations["raw_Cx"][node2] ** 0.5
        )

        denominator = 4 * self.linkstream.weight * self.linkstream.network_duration

        return numerator / denominator

    def _sum_degrees(self, module_leaves) -> float:
        """Compute the sum of the degrees of nodes involved in the set of time nodes.

        Args:
            module_leaves (set): module time nodes

        Returns:
            float: sum of the nodes degrees
        """
        nodes = {leaf.node for leaf in module_leaves}
        degrees = {node: self.linkstream.degrees[node] for node in nodes}
        return sum(degrees.values())

    def _sum_degrees_in(self, module_leaves) -> float:
        """Compute the sum of the in degrees of nodes involved in the set of time nodes.

        Args:
            module_leaves (set): module time nodes

        Returns:
            float: sum of the nodes degrees
        """
        nodes = {leaf.node for leaf in module_leaves}
        degrees = {node: self.linkstream.degrees_in.get(node, 0) for node in nodes}
        return sum(degrees.values())

    def _sum_degrees_out(self, module_leaves) -> float:
        """Compute the sum of the out degrees of nodes involved in the set of time nodes.

        Args:
            module_leaves (set): module time nodes

        Returns:
            float: sum of the nodes degrees
        """
        nodes = {leaf.node for leaf in module_leaves}
        degrees = {node: self.linkstream.degrees_out.get(node, 0) for node in nodes}
        return sum(degrees.values())
