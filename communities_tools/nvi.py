from collections import Counter
from itertools import product

import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score


def get_nvi(communities1, communities2):
    nvi = norm_var_information(
        communities1,
        communities2,
    )
    return round(nvi, ndigits=5)


def norm_var_information(
    clusters1,
    clusters2,
):
    """returns the normalized variation of information between two
    non-overlapping clustering.

    .. math::

        \hat{V}(C_1,C_2) = ({H(C1|C2)+H(C2|C1)})/{log_2 N}

    inputs can be node_to_cluster dictionaries, cluster lists of node sets
    or instances of Partition.
    """

    # num nodes
    nb_nodes = sum(len(clust) for clust in clusters1)

    nb_clus1 = len(clusters1)
    nb_clus2 = len(clusters2)

    # trivial cases
    if nb_clus1 == nb_clus2 == nb_nodes:
        return 0.0

    clusters1 = sorted(clusters1, key=lambda c: min(c))
    clusters2 = sorted(clusters2, key=lambda c: min(c))

    # other trivial case
    if (nb_clus1 == nb_clus2) and (clusters1 == clusters2):
        return 0.0

    # loop over pairs of clusters
    VI = 0.0
    for i in range(nb_clus1):
        clust1 = clusters1[i]
        ni = len(clust1)
        n_inter = 0
        for j in range(nb_clus2):
            clust2 = clusters2[j]
            nij = len(clust1.intersection(clust2))
            n_inter += nij
            if nij > 0:
                nj = len(clust2)
                VI -= nij * np.log2((nij**2) / (ni * nj))

            if n_inter >= ni:
                # we have found all the possible intersections
                break

        return VI / (nb_nodes * np.log2(nb_nodes))
