import itertools
import random

import networkx as nx


def GenerateLinkStreamGivenCommunities(
    snapshots_communities,
    nodes,
    p_in,
    p_out,
):
    snapshots = list()

    for communities in snapshots_communities.values():
        nodes_communities = dict()
        for lab, commu in communities.items():
            nodes_communities.update({node: lab for node in commu})
        communities = set(nodes_communities.values())
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for edge_candidate in itertools.combinations(nodes, 2):
            commu1 = nodes_communities.get(edge_candidate[0], None)
            commu2 = nodes_communities.get(edge_candidate[1], None)
            if commu1 is not None and commu2 is not None and commu1 == commu2:
                if random.uniform(0, 1) < p_in:
                    graph.add_edge(*edge_candidate)
            else:
                if random.uniform(0, 1) < p_out:
                    graph.add_edge(*edge_candidate)

        snapshots.append(graph)

    return snapshots
