import copy
from itertools import combinations_with_replacement

from lago.linkstream import LinkStream
from lago.tools import (
    get_module_duration,
    get_nodes_durations,
    get_nodes_times,
)


def longitudinal_modularity(
    linkstream: LinkStream,
    communities: dict,
    lex_type: str = "MM",
    omega: float = 2,
    ndigits: int = 5,
    return_time_penalty: bool = False,
):
    assert lex_type in [
        "CM",
        "JM",
        "MM",
    ], '"expectation" argument must be "CM", "JM", or "MM"'

    raw_communities = copy.deepcopy(communities)
    communities = {}
    for label, members in raw_communities.items():
        communities[label] = set()
        for leaf in members:
            if leaf not in linkstream.leaves_dict:
                # Ignore inactive time nodes
                continue
            communities[label].add(linkstream.leaves_dict[leaf])
            linkstream.leaves_dict[leaf].module = label
            # linkstream.leaves_dict[leaf].high_module = label
    del raw_communities

    ### 1 - Get nb links inside communities
    communities_nb_interactions = _get_communities_nb_interactions(linkstream)

    ### 2 - Get expectations values
    communities_expectations = {}
    if lex_type == "CM":
        communities_expectations = _get_communities_cmes(
            linkstream,
            communities,
        )
    if lex_type == "JM":
        communities_expectations = _get_communities_jmes(
            linkstream,
            communities,
        )

    if lex_type == "MM":
        communities_expectations = _get_communities_mmes(
            linkstream,
            communities,
        )

    ### 3 - Time penalty
    cscs = get_community_switch_counts(linkstream)
    time_penalty = -omega / (2 * linkstream.nb_edges) * cscs

    ### 4 - Aggregation
    lm_modularity = 0
    for community, expectation in communities_expectations.items():
        nb_links = communities_nb_interactions[community]
        lm_modularity += nb_links / (2 * linkstream.nb_edges) - expectation

    lm_modularity += time_penalty

    # NOTE Reset linkstream.leaves_dict commu tags ?
    # To avoid potential errors in next processus ?

    if return_time_penalty:
        return round(
            lm_modularity,
            ndigits=ndigits,
        ), round(
            time_penalty,
            ndigits=ndigits,
        )

    return round(lm_modularity, ndigits=ndigits)


def _get_communities_nb_interactions(
    linkstream: LinkStream,
):
    # NOTE leaves_dict must have module already initiated
    communities_nb_interactions = {}
    for tmp_leaf in linkstream.leaves_dict.values():
        community = tmp_leaf.module
        if community not in communities_nb_interactions:
            communities_nb_interactions[community] = 0
        neighbors = tmp_leaf.topo_neighbors
        for neighbor in neighbors:
            if neighbor.module != community:
                continue
            communities_nb_interactions[community] += 2 ** (neighbor == tmp_leaf)

    return communities_nb_interactions


def _get_communities_jmes(
    linkstream: LinkStream,
    communities_leaves: dict,
):
    communities_expectations = {}
    for commu, leaves in communities_leaves.items():
        expectation = 0
        community_nodes = set([leaf.node for leaf in leaves])
        community_duration = get_module_duration(leaves)
        for source, target in combinations_with_replacement(community_nodes, 2):
            # NOTE Does not make a lot of sense to apply stream graph change here
            expected_value = (
                2 ** (source != target)
                * linkstream.degrees.get(source, 0)
                * linkstream.degrees.get(target, 0)
                * (community_duration / linkstream.network_duration)
                / (2 * linkstream.nb_edges) ** 2
            )

            expectation += expected_value

        communities_expectations[commu] = expectation

    return communities_expectations


def _get_communities_mmes(
    linkstream: LinkStream,
    communities_leaves: dict,
):
    communities_expectations = {}
    for commu, leaves in communities_leaves.items():
        expectation = 0
        nodes_durations = get_nodes_durations(
            module_leaves=leaves,
        )
        community_nodes = set([leaf.node for leaf in leaves])
        for source, target in combinations_with_replacement(community_nodes, 2):
            geo_mean = (
                nodes_durations.get(source, 0) * nodes_durations.get(target, 0)
            ) ** 0.5
            if linkstream.is_stream_graph:
                expected_value = (
                    2 ** (source != target)
                    * linkstream.degrees.get(source, 0)
                    * linkstream.degrees.get(target, 0)
                    * (
                        geo_mean
                        / (
                            linkstream.nodes_durations[source]
                            * linkstream.nodes_durations[target]
                        )
                        ** 0.5
                    )
                    / (2 * linkstream.nb_edges) ** 2
                )
                # print(source, nodes_durations.get(source, 0), linkstream.nodes_durations[source])
                # print(target, nodes_durations.get(target, 0), linkstream.nodes_durations[target])
                # input()
            else:
                expected_value = (
                    2 ** (source != target)
                    * linkstream.degrees.get(source, 0)
                    * linkstream.degrees.get(target, 0)
                    * (geo_mean / linkstream.network_duration)
                    / (2 * linkstream.nb_edges) ** 2
                )

            expectation += expected_value

        communities_expectations[commu] = expectation
    return communities_expectations


def _get_communities_cmes(
    linkstream: LinkStream,
    communities_leaves: dict,
):
    communities_expectations = {}
    for commu, leaves in communities_leaves.items():
        expectation = 0
        community_nodes = sorted(set([leaf.node for leaf in leaves]))
        nodes_times = get_nodes_times(leaves)
        for ite, source in enumerate(community_nodes):
            source_times = nodes_times[source]

            for target in community_nodes[ite:]:
                target_times = nodes_times[target]
                coexistence = len(source_times & target_times)
                if not coexistence:
                    continue
                if linkstream.is_stream_graph:
                    expected_value = (
                        2 ** (source != target)
                        * linkstream.degrees.get(source, 0)
                        * linkstream.degrees.get(target, 0)
                        * (
                            coexistence
                            / (
                                linkstream.nodes_durations[source]
                                * linkstream.nodes_durations[target]
                            )
                            ** 0.5
                        )
                        / (2 * linkstream.nb_edges) ** 2
                    )
                else:
                    expected_value = (
                        2 ** (source != target)
                        * linkstream.degrees.get(source, 0)
                        * linkstream.degrees.get(target, 0)
                        * (coexistence / linkstream.network_duration)
                        / (2 * linkstream.nb_edges) ** 2
                    )
                expectation += expected_value
        communities_expectations[commu] = expectation

    return communities_expectations


def get_community_switch_counts(
    linkstream: LinkStream,
):
    cscs = 0
    for tmp_leaf in linkstream.leaves_dict.values():
        community = tmp_leaf.module
        # Left
        neigh_left = tmp_leaf.left_time_active_neighbor
        if neigh_left and neigh_left.module != community:
            cscs += 1
        # Right
        neigh_right = tmp_leaf.right_time_active_neighbor
        if neigh_right and neigh_right.module != community:
            cscs += 1
    # CSC are counted twice
    return cscs / 2
