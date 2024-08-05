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

XP_TYPE = "continuity"
RESULTS_PATH = "xp_time_communities_types/xp_results/"
PLOT_PATH = f"xp_time_communities_types/plot/plot_{XP_TYPE}/"

# - - - - Parameters

p_in = 3
p_out = 0

nodes_scaling_factor = 3
nb_continuous_community_nodes = (1,)

nb_discontinuous_community_nodes = (1,)

nb_nodes = nodes_scaling_factor * (
    sum(nb_continuous_community_nodes) + sum(nb_discontinuous_community_nodes)
)

ls_duration = 5

tmp_commu_unstable_label = nodes_scaling_factor * len(nb_continuous_community_nodes)

# - - - - Define communities

# Continuous community
snapshots_communities = {}
for ite_snapshot in range(ls_duration):
    tmp_commu_label = 0
    tmp_communities = {}
    while tmp_commu_label < len(nb_continuous_community_nodes):
        tmp_communities[tmp_commu_label] = set(
            range(
                nodes_scaling_factor
                * sum(nb_continuous_community_nodes[:(tmp_commu_label)]),
                nodes_scaling_factor
                * sum(nb_continuous_community_nodes[: (tmp_commu_label + 1)]),
            )
        )
        tmp_commu_label += 1
    tmp_communities[tmp_commu_unstable_label] = set(
        range(nodes_scaling_factor * sum(nb_continuous_community_nodes), nb_nodes)
    )
    snapshots_communities[ite_snapshot] = tmp_communities

snapshots_communities_list = [snapshots_communities]

# - - - - Generate link stream interactions
snapshots = GenerateLinkStreamGivenCommunities(
    snapshots_communities=snapshots_communities,
    nodes=list(range(nb_nodes)),
    p_in=p_in,
    p_out=p_out,
)
temporal_network = lmt.snapshots_to_dataframe(snapshots)


# - - - Discontinuous communities
unstability_factors = [1]

for unstab_factor in unstability_factors:
    snapshots_communities_alt = deepcopy(snapshots_communities)
    tmp_commu_unstable_label = nodes_scaling_factor * len(nb_continuous_community_nodes)
    for snapite, snapcom in snapshots_communities_alt.items():
        last_label = list(snapcom.keys())[-1]
        if snapite % unstab_factor == 0:
            tmp_commu_unstable_label += 1
        snapcom[tmp_commu_unstable_label] = deepcopy(snapcom[last_label])
        del snapcom[last_label]
    snapshots_communities_list.append(snapshots_communities_alt)

# - - - - - XP - - - - -
# Parameters to xp
parameters = {
    "omega": [0, 1, 2],
    "expectation": ["cm", "jm", "mm"],
}

results = []


for ite, snapcoms in enumerate(snapshots_communities_list):
    print(ite)

    # Prepare data for LM computing
    community_type = ite + 1
    ground_truth_communities = list()
    commulabels = set()
    for snp_comm in snapcoms.values():
        commulabels |= set(snp_comm.keys())
    for lab in commulabels:
        tmp_commu = set()
        for snap, snp_comm in snapcoms.items():
            tmp_snp_comm = snp_comm.get(lab)
            if not tmp_snp_comm:
                continue
            tmp_commu |= set([(node, snap) for node in tmp_snp_comm])
        ground_truth_communities.append(tmp_commu)
    temp_net_w_commus, communities_df = lmt.format_lm_input(
        temporal_network,
        ground_truth_communities,
    )

    plot_commus = plot_long_com(
        communities=ground_truth_communities,
        snapshots=snapshots,
        nodes=set(range(nb_nodes)),
        display_xlabel=False,
        display_xticks_labels=True,
    )
    plot_commus.savefig(
        PLOT_PATH + f"plot_xp_{community_type}.png",
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
            f"continuity_{community_type}",
        ],
    )
    results.append(df_tmp_res)


df_res = reduce(
    lambda left, right: pd.merge(
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
