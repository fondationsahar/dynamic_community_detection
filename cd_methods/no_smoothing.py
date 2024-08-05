from itertools import product

import networkx as nx
import numpy as np


def get_no_smoothing_communities(
    snapshots,
    n_tries,
    merge_threshold,
):
    # NOTE the function needs python3.10
    snapshots_communities = dict()
    modularities = list()
    for snap, snapshot in enumerate(snapshots):
        modularity = 0
        communities = [{node} for node in snapshot.nodes]
        if snapshot.edges():
            for _ in range(n_tries):
                if modularity == 1:
                    break
                tmp_communities = nx.community.louvain_communities(snapshot)
                tmp_modularity = nx.community.modularity(snapshot, tmp_communities)
                if tmp_modularity > modularity:
                    communities = tmp_communities
                    modularity = tmp_modularity
        communities_dict = {
            ite: list(community) for ite, community in enumerate(communities)
        }
        snapshots_communities[snap] = communities_dict
        modularities.append(round(modularity, ndigits=5))

    mean_modularity = np.mean(modularities)

    if type(merge_threshold) is float:
        time_communities = merge_communities_over_snapshots(
            snapshots_communities, merge_threshold
        )

        return time_communities, mean_modularity

    elif type(merge_threshold) is list:
        time_communities_dict = {}
        for mthresh in merge_threshold:
            time_communities = merge_communities_over_snapshots(
                snapshots_communities.copy(), mthresh
            )
            time_communities_dict[mthresh] = time_communities
        return time_communities_dict, mean_modularity


def merge_communities_over_snapshots(snapshots_communities, threshold=0):
    # If two communities have less than threshold of jaccard coeff,
    # they are not merged

    nodes_set = set()
    for nodes in snapshots_communities[0].values():
        nodes_set |= set(nodes)

    snap_ids = list(snapshots_communities.keys())

    merge_communities = dict()

    for snap1, snap2 in zip(snap_ids[:-1], snap_ids[1:]):
        # merge_communities[(snap1, snap2)] = dict()
        # merge between two snapshots
        jaccards = np.zeros(
            (len(snapshots_communities[snap1]), len(snapshots_communities[snap2]))
        )
        for (ite1, communities1), (ite2, communities2) in product(
            snapshots_communities[snap1].items(),
            snapshots_communities[snap2].items(),
        ):
            jaccards[ite1][ite2] = jaccard(set(communities1), set(communities2))
        while sum(jaccards.flatten()):
            argmax = np.where(jaccards == max(threshold, jaccards.max()))
            if not len(argmax[0]):
                break
            idx1 = str(snap1) + "_" + str(argmax[0][0])
            idx2 = str(snap2) + "_" + str(argmax[1][0])
            merge_communities[idx1] = idx2
            jaccards[argmax[0][0], :] = 0
            jaccards[:, argmax[1][0]] = 0

    for community_from, _ in merge_communities.items():
        community_to = community_from
        while community_to in merge_communities:
            community_to = merge_communities[community_to]
        merge_communities[community_from] = community_to

    snap_nodes_community = dict()
    for snap, communities in snapshots_communities.items():
        snap_nodes_community[snap] = dict()
        for ite, commu in communities.items():
            snap_nodes_community[snap].update(
                {node: str(snap) + "_" + str(ite) for node in commu}
            )

    nodes_community_snap = {}
    all_commu = set()
    for node in nodes_set:
        snap_community = dict()
        for snap, nodes_community in snap_nodes_community.items():
            new_commu = merge_communities.get(
                nodes_community[node], nodes_community[node]
            )
            snap_community[snap] = new_commu
            all_commu.add(new_commu)
        nodes_community_snap[node] = snap_community

    # Get dict {commu : {(node, time), ...}, ...}
    commu_time_node = {commu: set() for commu in all_commu}
    for node, snap_community in nodes_community_snap.items():
        for snap, commu in snap_community.items():
            commu_time_node[commu].add((node, snap))

    list_time_commu = list(commu_time_node.values())

    return list_time_commu


def jaccard(com1: set, com2: set):
    union_size = len(com1 | com2)
    if union_size == 0:
        return 0
    return len(com1 & com2) / union_size
