import os

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np


def get_mucha_modularity(modularities, norms, mucha_norm):
    unnormed_modularities = [mod * norm for mod, norm in zip(modularities, norms)]
    mucha_modularity = sum(unnormed_modularities) / mucha_norm

    return mucha_modularity


def get_mm_communities(
    snapshots,
    resolution_parameter=1,
    n_tries=5,
    interslice_weight=None,
    interslice_weight_ratio=0.1,
    verbose=False,
):  
    if verbose:
        print('\t\tCompute Adjacencies')
    adjacencies = {
        snap_lab: nx.adjacency_matrix(snapshot)
        for snap_lab, snapshot in enumerate(snapshots)
    }
    nb_snapshots = len(adjacencies)
    nb_nodes = adjacencies[0].shape[0]

    graphs = []

    if verbose:
        print('\t\tCompute Graphs')
    for A in adjacencies.values():
        g = ig.Graph()
        g.add_vertices(A.shape[0])
        g.vs["id"] = list(range(A.shape[0]))
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if j > i:
                    if A[i, j] > 0:
                        g.add_edge(i, j, weight=A[i, j])
        graphs.append(g)

    if not interslice_weight:
        avg_weight = np.mean([A[A > 0].mean() for A in adjacencies.values()])
        interslice_weight = avg_weight * interslice_weight_ratio

    if verbose:
        print('\t\tFormat layers')
    layers, interslice_layer, G_full = la.time_slices_to_layers(
        graphs, interslice_weight=interslice_weight
    )

    clusters_multi = []
    stab_multi_list = []
    mucha_modularity = -1

    slices_norms = [sum(val for _, val in snapshot.degree) for snapshot in snapshots]
    norm_interslice = nb_nodes * nb_snapshots * (nb_snapshots - 1) * interslice_weight
    mucha_norm = sum(slices_norms) + norm_interslice

    if verbose:
        print('\t\tStart optimization')

    if verbose:
        print('\t\t\tIte', _)
    optimiser = la.Optimiser()

    optimiser.set_rng_seed(int.from_bytes(os.urandom(3), byteorder="big"))

    if verbose:
        print('\t\t\tcompute partitions')
    partitions = [
        la.RBConfigurationVertexPartition(
            H, weights="weight", resolution_parameter=resolution_parameter,
        )
        for H in layers
    ]
    if verbose:
        print('\t\t\tcompute interslice partitions')
    interslice_partition = la.RBConfigurationVertexPartition(
        interslice_layer,
        resolution_parameter=0,
        weights="weight",
    )

    if verbose:
        print('\t\t\toptimize partitions multiplex')

    diff = optimiser.optimise_partition_multiplex(
        partitions + [interslice_partition], n_iterations=n_tries,
    )

    tmp_stab_multi_list = [
        p.modularity for p in partitions + [interslice_partition]
    ]

    if verbose:
        print('\t\t\tGet mucha modularity')
    tmp_mucha_modularity = get_mucha_modularity(
        tmp_stab_multi_list, slices_norms + [norm_interslice], mucha_norm
    )

    if verbose:
        print('\t\t\tUpdate metrics')
    if tmp_mucha_modularity > mucha_modularity or not clusters_multi:
        mucha_modularity = tmp_mucha_modularity
        stab_multi_list = tmp_stab_multi_list
        clusters_multi = partitions[0].membership

    comlabs_set = set(clusters_multi)
    time_communities = {clb: set() for clb in comlabs_set}
    for ite in range(nb_snapshots):
        comlabs = clusters_multi[ite * nb_nodes : (ite + 1) * nb_nodes]
        for node in range(nb_nodes):
            time_communities[comlabs[node]].add((node, ite))
    time_communities = list(time_communities.values())

    return time_communities, mucha_modularity
