import os
import sys
from copy import deepcopy
from functools import reduce
from itertools import product

import pandas as pd

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import longitudinal_modularity_tools as lmt
from generate_synthetic_temporal_networks import GenerateLinkStreamGivenCommunities
from plotting_tools import plot_long_com

XP_TYPE = "topochrone_discontinuities"
RESULTS_PATH = "xp_time_communities_types/xp_results/"
PLOT_PATH = f"xp_time_communities_types/plot/plot_{XP_TYPE}/"

# - - - - Parameters
p_in = 0.6
p_out = 0

ls_duration = 25

nodes_scaling_factor = 3

nb_stable_nodes = (2, 1)
nb_disconnex_nodes = (4, 1, 2)

nb_nodes = nodes_scaling_factor * (sum(nb_disconnex_nodes) + sum(nb_stable_nodes))

nodes_communities = {ite: 0 for ite in range(nb_nodes)}

times_disconnex_commu = ((12, 18), (3, 10), (20, 25))

nodes = list(range(nb_nodes))

nb_communities = len(nb_stable_nodes) + len(nb_disconnex_nodes)

# - - - - Define communities
snapshots_communities = {}
for ite_snapshot in range(ls_duration):
    tmp_commu_label = 0
    tmp_communities = {}
    while tmp_commu_label < len(nb_stable_nodes):
        tmp_communities[tmp_commu_label] = set(
            range(
                nodes_scaling_factor * sum(nb_stable_nodes[:(tmp_commu_label)]),
                nodes_scaling_factor * sum(nb_stable_nodes[: (tmp_commu_label + 1)]),
            )
        )
        tmp_commu_label += 1
    ite = 0
    while tmp_commu_label < nb_communities:
        if (
            not times_disconnex_commu[ite][0]
            <= ite_snapshot
            <= times_disconnex_commu[ite][1]
        ):
            tmp_commu_label += 1
            ite += 1
            continue
        tmp_communities[tmp_commu_label] = set(
            range(
                nodes_scaling_factor
                * (sum(nb_stable_nodes) + sum(nb_disconnex_nodes[:ite])),
                nodes_scaling_factor
                * (sum(nb_stable_nodes) + sum(nb_disconnex_nodes[: ite + 1])),
            )
        )
        tmp_commu_label += 1
        ite += 1

    snapshots_communities[ite_snapshot] = tmp_communities

ground_truth_communities = list()
commulabels = set()
for snp_comm in snapshots_communities.values():
    commulabels |= set(snp_comm.keys())
for lab in commulabels:
    tmp_commu = set()
    for snap, snp_comm in snapshots_communities.items():
        tmp_snp_comm = snp_comm.get(lab)
        if not tmp_snp_comm:
            continue
        tmp_commu |= set([(node, snap) for node in tmp_snp_comm])
    ground_truth_communities.append(tmp_commu)

# - - - - Generate link stream interactions
snapshots = GenerateLinkStreamGivenCommunities(
    snapshots_communities=snapshots_communities,
    nodes=list(range(nb_nodes)),
    p_in=p_in,
    p_out=p_out,
)
temporal_network = lmt.snapshots_to_dataframe(snapshots)
### - - - - Define other community - - - - - - -

alt_snapshots_communities = deepcopy(snapshots_communities)
snapshots_communities2 = {}
for snap_ite, communities in alt_snapshots_communities.items():
    tmp_communities = {}
    tmp_commu_label = 0
    while tmp_commu_label < len(nb_stable_nodes):
        tmp_communities[tmp_commu_label] = communities[tmp_commu_label]
        tmp_commu_label += 1
    while tmp_commu_label < nb_communities:
        if not communities.get(tmp_commu_label):
            tmp_commu_label += 1
            continue
        if tmp_communities.get(len(nb_stable_nodes)):
            tmp_communities[len(nb_stable_nodes)] |= communities.get(tmp_commu_label)
        else:
            tmp_communities[len(nb_stable_nodes)] = communities.get(tmp_commu_label)
        tmp_commu_label += 1
    snapshots_communities2[snap_ite] = tmp_communities

# NOTE degueulasse, a refacto
ground_truth_communities2 = list()
commulabels = set()
for snp_comm in snapshots_communities2.values():
    commulabels |= set(snp_comm.keys())
for lab in commulabels:
    tmp_commu = set()
    for snap, snp_comm in snapshots_communities2.items():
        tmp_snp_comm = snp_comm.get(lab)
        if not tmp_snp_comm:
            continue
        tmp_commu |= set([(node, snap) for node in tmp_snp_comm])
    ground_truth_communities2.append(tmp_commu)

snapshots_communities = {}
for ite_snapshot in range(ls_duration):
    tmp_commu_label = 0
    tmp_communities = {}
    while tmp_commu_label < len(nb_stable_nodes):
        tmp_communities[tmp_commu_label] = set(
            range(
                nodes_scaling_factor * sum(nb_stable_nodes[:(tmp_commu_label)]),
                nodes_scaling_factor * sum(nb_stable_nodes[: (tmp_commu_label + 1)]),
            )
        )
        tmp_commu_label += 1
    ite = 0
    while tmp_commu_label < nb_communities:
        tmp_communities[tmp_commu_label] = set(
            range(
                nodes_scaling_factor
                * (sum(nb_stable_nodes) + sum(nb_disconnex_nodes[:ite])),
                nodes_scaling_factor
                * (sum(nb_stable_nodes) + sum(nb_disconnex_nodes[: ite + 1])),
            )
        )
        tmp_commu_label += 1
        ite += 1

    snapshots_communities[ite_snapshot] = tmp_communities

ground_truth_communities3 = list()
commulabels = set()
for snp_comm in snapshots_communities.values():
    commulabels |= set(snp_comm.keys())
for lab in commulabels:
    tmp_commu = set()
    for snap, snp_comm in snapshots_communities.items():
        tmp_snp_comm = snp_comm.get(lab)
        if not tmp_snp_comm:
            continue
        tmp_commu |= set([(node, snap) for node in tmp_snp_comm])
    ground_truth_communities3.append(tmp_commu)

# - - - - - XP - - - - -

parameters = {
    "omega": [0, 1, 2],
    "expectation": ["cm", "jm", "mm"],
}

results = []

for community_type, gtc in enumerate(
    [
        ground_truth_communities,
        ground_truth_communities2,
        ground_truth_communities3,
    ]
):
    print(community_type + 1)
    temp_net_w_commus, communities_df = lmt.format_lm_input(
        temporal_network,
        gtc,
    )

    plot_commus = plot_long_com(
        communities=gtc,
        snapshots=snapshots,
        nodes=set(range(nb_nodes)),
    )
    plot_commus.savefig(
        PLOT_PATH + f"plot_xp_{community_type+1}.png",
        bbox_inches="tight",
    )

    tmp_results = []
    for omega, expe in product(*parameters.values()):
        l_mod = lmt.get_longitudinal_modularity(
            time_links_df=temp_net_w_commus,
            communities_df=communities_df,
            expectation=expe,
            omega=omega,
        )
        tmp_results.append(
            [
                omega,
                expe,
                l_mod,
            ]
        )

    df_tmp_res = pd.DataFrame(
        tmp_results,
        columns=[
            "omega",
            "expectation",
            f"topodis_{community_type+1}",
        ],
    )
    results.append(df_tmp_res)


df_res = reduce(
    lambda left, right:  # Merge DataFrames in list
    pd.merge(
        left,
        right,
        on=[
            "omega",
            "expectation",
        ],
        how="outer",
    ),
    results,
)
df_res.to_csv(RESULTS_PATH + "xp_" + XP_TYPE + ".csv", index=False)
