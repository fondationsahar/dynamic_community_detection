from itertools import combinations_with_replacement

import pandas as pd


def get_longitudinal_modularity(
    time_links_df: pd.DataFrame,
    communities_df: pd.DataFrame,
    expectation: str = "mm",
    omega: float = 1,
    return_time_penalty:bool=False,
    verbose: bool = False,
):
    """_summary_

    Args:
        time_links_df (pd.DataFrame): Interactions table. Columns: source, target, time, source_commu, target_commu
        communities_df (pd.DataFrame): Nodes communities memberships. Columns: node, time, commu
        expectation (str, optional): Logitudinal expectation type. Must be "cm" for co-membership, "jm" for joint membership, or "mm" for mean membership. Defaults to "mm".
        omega (float, optional): Time penalty weigth. High values will favor continuous communities. Defaults to 1.
        verbose (bool, optional): Wether to print function status or not. Defaults to False.

    Returns:
        (float, float): Longitudinal modularity, time continuity penalty for communities
    """
    assert expectation in [
        "cm",
        "jm",
        "mm",
    ], '"expectation" argument must be "cm", "jm", or "mm"'

    nodes_set = set(time_links_df.source.tolist())
    nodes_set |= set(time_links_df.target.tolist())

    # Time steps should be normalized with 0 as first time step and min gap between two timesteps of 1.
    network_duration = time_links_df.time.max() - time_links_df.time.min() + 1

    # NOTE No weights by now
    L_dict = {edge: 0 for edge in combinations_with_replacement(nodes_set, 2)}
    for tmp_time_edge in time_links_df[["source", "target"]].values.tolist():
        L_dict[tuple(sorted(tmp_time_edge))] += 1

    degrees = {node: 0 for node in nodes_set}
    for (source, target), nb_links in L_dict.items():
        degrees[source] += nb_links
        degrees[target] += nb_links
    del L_dict

    normalisation = sum(degrees.values())

    communities_nb_links = get_communities_nb_links(
        communities_df=communities_df,
        time_links_df=time_links_df,
    )

    if verbose:
        print("Start computing communities expectations")
    if expectation == "mm":
        communities_expectations = get_communities_mmes(
            communities_df=communities_df.copy(),
            degrees=degrees.copy(),
            network_duration=network_duration,
            normalisation=normalisation,
        )

    if expectation == "cm":
        communities_expectations = get_communities_cmes(
            nodes_set=nodes_set.copy(),
            communities_df=communities_df.copy(),
            degrees=degrees.copy(),
            normalisation=normalisation,
            network_duration=network_duration,
        )

    if expectation == "jm":
        communities_expectations = get_communities_jmes(
            communities_df=communities_df.copy(),
            time_links_df=time_links_df.copy(),
            degrees=degrees.copy(),
            normalisation=normalisation,
            network_duration=network_duration,
        )

    lm_modularity = 0
    for community, expectation in communities_expectations.items():
        nb_links = communities_nb_links[community]
        lm_modularity += nb_links / normalisation - expectation

    time_penalty = get_time_penalty(
        communities_df=communities_df.copy(),
        nodes_set=nodes_set.copy(),
        normalisation=normalisation,
        omega=omega,
    )

    if verbose:
        print("time_penalty:", time_penalty)

    lm_modularity += time_penalty

    if return_time_penalty:
        return round(lm_modularity, ndigits=5), round(time_penalty, ndigits=5)

    return round(lm_modularity, ndigits=5)


def get_communities_mmes(
    communities_df,
    degrees,
    network_duration,
    normalisation,
):
    communities_set = set(communities_df.commu.tolist())

    communities_mmes = dict()

    for community in communities_set:
        tmp_commu_df = communities_df[communities_df.commu == community]
        dict_nodes_dur = dict(tmp_commu_df.groupby("node").count()["time"])
        communities_mmes[community] = dict_nodes_dur

    communities_expectations = dict()
    for community, dict_nodes_duration in communities_mmes.items():
        expectation = 0
        commu_nodes_set = set(dict_nodes_duration.keys())
        for source, target in combinations_with_replacement(commu_nodes_set, 2):
            geo_mean = (
                dict_nodes_duration.get(source, 0) * dict_nodes_duration.get(target, 0)
            ) ** 0.5
            expected_value = (
                2 ** (source != target)
                * degrees.get(source, 0)
                * degrees.get(target, 0)
                * (geo_mean / network_duration)
                / normalisation**2
            )

            expectation += expected_value

        communities_expectations[community] = expectation

    return communities_expectations


def get_communities_nb_links(
    communities_df,
    time_links_df,
):
    communities_set = set(communities_df.commu.tolist())

    communities_nb_interactions = dict()

    for community in communities_set:
        tmp_df = time_links_df[
            (time_links_df.source_commu == community)
            & (time_links_df.target_commu == community)
        ]
        tmp_df2 = pd.DataFrame(
            tmp_df.groupby(["source", "target"]).count()
        ).reset_index()
        # Count twice if not self link
        nb_links_in_commu = 2 * tmp_df2[tmp_df2.source != tmp_df2.target].time.sum()
        nb_links_in_commu += tmp_df2[tmp_df2.source == tmp_df2.target].time.sum()

        communities_nb_interactions[community] = nb_links_in_commu

    return communities_nb_interactions


def get_communities_cmes(
    nodes_set,
    communities_df,
    degrees,
    normalisation,
    network_duration,
):
    communities_set = set(communities_df.commu.tolist())

    # No need for a dict, a list works as well but is harder to debug
    communities_materials = dict()

    for community in communities_set:
        # nb coexistences
        coexistence_in_commu = dict()
        tmp_commu_df = communities_df.loc[communities_df.commu.values == community]
        nodes_list_in_commu = sorted([*{*tmp_commu_df.node.tolist()}])
        for ite, source in enumerate(nodes_list_in_commu):
            tmp_df_source = tmp_commu_df.loc[tmp_commu_df.node.values == source]
            coexistence_in_commu[(source, source)] = len(
                set(tmp_df_source.time.tolist())
            )
            for target in nodes_list_in_commu[ite + 1 :]:
                tmp_df_target = tmp_commu_df.loc[tmp_commu_df.node.values == target]
                coexistence_in_commu[(source, target)] = len(
                    set(tmp_df_source.time.unique()) & set(tmp_df_target.time.unique())
                )

        communities_materials[community] = coexistence_in_commu

    communities_expectations = dict()
    for community, coexistence_in_commu in communities_materials.items():
        expectation = 0

        for (source, target), coexistence in coexistence_in_commu.items():
            if not coexistence:
                continue

            expected_value = (
                2 ** (source != target)
                * degrees.get(source, 0)
                * degrees.get(target, 0)
                * (coexistence / network_duration)
                / normalisation**2
            )

            expectation += expected_value

        communities_expectations[community] = expectation

    return communities_expectations


def get_communities_jmes(
    communities_df,
    time_links_df,
    normalisation,
    degrees,
    network_duration,
):
    communities_set = set(communities_df.commu.tolist())

    communities_durations = dict()
    communities_nodes = dict()
    for community in communities_set:
        tmp_commu_df = communities_df[communities_df.commu == community]
        communities_nodes[community] = sorted(tmp_commu_df.node.unique())
        communities_durations[community] = len(tmp_commu_df.time.unique())

    communities_expectations = dict()
    for community, commu_duration in communities_durations.items():
        expectation = 0
        commu_nodes_set = communities_nodes[community]
        for source, target in combinations_with_replacement(commu_nodes_set, 2):
            expected_value = (
                2 ** (source != target)
                * degrees.get(source, 0)
                * degrees.get(target, 0)
                * (commu_duration / network_duration)
                / normalisation**2
            )

            expectation += expected_value

        communities_expectations[community] = expectation

    return communities_expectations


def get_time_penalty(
    nodes_set,
    communities_df,
    normalisation,
    omega,
):
    nodes_csc = dict()
    for node in nodes_set:
        tmp_commu_node = communities_df[communities_df.node == node]

        node_commus = tmp_commu_node.commu.unique()

        community_entries = []
        for community in node_commus:
            tmp_df = tmp_commu_node.loc[
                tmp_commu_node.commu.values == community
            ]
            tmp_times = sorted(tmp_df.time.tolist())
            tmp_com_ent = [tmp_times[0]]
            t2 = tmp_times[-1]
            for t1, t2 in zip(tmp_times[1:], tmp_times[:-1]):
                if t1 - t2 == 1:
                    continue
                tmp_com_ent.append(t2)
                community_entries.append(tmp_com_ent)
                tmp_com_ent = [t1]
            tmp_com_ent.append(t2)
            community_entries.append(tmp_com_ent)

        nodes_csc[node] = max(0, len(community_entries)-1)

    time_penalty = (
        omega
        * sum(
            [
                 - nb_comp_con
                for node, nb_comp_con in nodes_csc.items()
            ]
        )
        / normalisation
    )

    return time_penalty
