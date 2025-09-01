from collections import defaultdict
from itertools import combinations_with_replacement
from typing import List

import lago.tools as tls
from lago.leaf import Leaf
from lago.linkstream import LinkStream


class DeltaLongitudinalModularityComputer:
    """Computer of delta longitudinal modularity
    when moving submodules for LAGO.
    """

    def __init__(
        self,
        linkstream: LinkStream,
        lex: str = "JM",
        omega: float = 2,
    ):
        self.linkstream = linkstream
        self.lex = lex
        self.omega = omega

    def M0_to_Mx(
        self,
        M0_leaves: set[Leaf],
        M0_time_segments: dict[int, List[List[Leaf]]],
        Mx_leaves: set[Leaf],
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
        nb_edges_diff = self._get_nb_edges_diff(M0_leaves, Mx_leaves)

        if self.lex == "MM":
            expectation_diff = self._get_expectation_mm_part(M0_leaves, Mx_leaves)
        elif self.lex == "JM":
            expectation_diff = self._get_expectation_jm_part(M0_leaves, Mx_leaves)
        else:
            raise Exception("Wrong lex value")

        csc_diff = self._get_csc_diff(M0_time_segments, Mx_leaves)

        delta_lm = nb_edges_diff - expectation_diff + csc_diff

        return delta_lm

    def _get_nb_edges_diff(self, M0_leaves: set[Leaf], Mx_leaves: set[Leaf]):
        """Compute delta number of edges within module Mx if M0 joins Mx.

        Args:
            M0_leaves (set): M0 submodule time nodes
            Mx_leaves (set): Mx module time nodes

        Returns:
            int: delta number of edges
        """
        all_neighbs = list()
        for leaf in Mx_leaves:
            all_neighbs += list(leaf.topo_neighbors)
        # Only keep inventoried neighbors that are in M0
        # NOTE cannot use a set here because we want to keep duplicated time nodes
        neighbors = [neighb for neighb in all_neighbs if neighb in M0_leaves]
        return len(neighbors)

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
        csc_diff = -self.omega / 2 * cscs

        return csc_diff

    def _get_expectation_jm_part(
        self,
        M0_leaves: set[Leaf],
        Mx_leaves: set[Leaf],
    ):
        """Compute the delta number of expected edges within module Mx
        if joined by submodule M0, regarding the Joint-Membership expectation.

        Args:
            M0_leaves (set): M0 submodule time nodes
            Mx_leaves (set): Mx module time nodes

        Returns:
            float: delta value for expected number of edges
        """
        degree_Cx = self._sum_degrees(Mx_leaves)
        duration_Cx = tls.get_module_duration(Mx_leaves)

        degree_Cx_U_C0 = self._sum_degrees(Mx_leaves | M0_leaves)
        duration_Cx_U_C0 = tls.get_module_duration(Mx_leaves | M0_leaves)

        expectation_diff = (
            degree_Cx_U_C0**2 * duration_Cx_U_C0 - degree_Cx**2 * duration_Cx
        ) / (4 * self.linkstream.nb_edges * self.linkstream.network_duration)

        return expectation_diff

    def _sum_degrees(self, module_leaves) -> float:
        """Compute the sum of the degrees of nodes involved in the set of time nodes.

        Args:
            module_leaves (set): module time nodes

        Returns:
            float: sum of the nodes degrees
        """
        nodes = set([leaf.node for leaf in module_leaves])
        degrees = {node: self.linkstream.degrees[node] for node in nodes}
        return sum(degrees.values())

    def _get_expectation_mm_part(
        self,
        M0_leaves: set[Leaf],
        Mx_leaves: set[Leaf],
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

        all_nodes = set([leaf.node for leaf in M0_leaves | Mx_leaves])
        for node1, node2 in combinations_with_replacement(all_nodes, 2):
            expectation_diff += self._partial_expectation_mm_diff(
                node1, node2, nodes_durations
            )

        return expectation_diff

    def _partial_expectation_mm_diff(
        self,
        node1: int,
        node2: int,
        nodes_durations: dict[str, defaultdict[int, float]],
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
        numerator: float = (
            2 ** (node1 != node2)
            * self.linkstream.degrees[node1]
            * self.linkstream.degrees[node2]
            * (
                nodes_durations["Cx_U_C0"][node1] ** 0.5
                * nodes_durations["Cx_U_C0"][node2] ** 0.5
                - nodes_durations["raw_Cx"][node1] ** 0.5
                * nodes_durations["raw_Cx"][node2] ** 0.5
            )
        )

        if self.linkstream.is_stream_graph:
            denominator: float = (
                4
                * self.linkstream.nb_edges
                * (
                    self.linkstream.nodes_durations[node1]
                    * self.linkstream.nodes_durations[node2]
                )
                ** 0.5
            )
        else:
            denominator = (
                4 * self.linkstream.nb_edges * self.linkstream.network_duration
            )

        return numerator / denominator
