import os
import sys

import networkx as nx

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from plotting_tools import plot_long_com

XP_TYPE = "win_agg_indep"
PLOT_PATH = f"xp_time_communities_types/plot/plot_{XP_TYPE}/"

nb_nodes = 5

time_edges = [
    (0, 2, 0),
    (3, 4, 0),
    (0, 1, 0),
    (1, 2, 0),
    (0, 1, 1),
    (3, 4, 1),
    (1, 3, 1),
    (1, 2, 3),
    (0, 1, 3),
    (3, 4, 3),
    (2, 3, 4),
    (0, 1, 5),
    (2, 3, 5),
    (2, 4, 6),
    (3, 4, 5),
    (2, 3, 6),
    (0, 2, 7),
    (0, 1, 7),
    (2, 3, 7),
    (3, 4, 7),
]

snapshots = [nx.Graph() for _ in range(8)]
for snapshot in snapshots:
    snapshot.add_nodes_from(list(range(nb_nodes)))

for source, target, time in time_edges:
    snapshots[time].add_edge(source, target)


snapshots_communities = {
    0: {0: {0, 1, 2}, 1: {3, 4}},
    1: {0: {0, 1, 2}, 1: {3, 4}},
    2: {0: {0, 1, 2}, 1: {3, 4}},
    3: {0: {0, 1, 2}, 1: {3, 4}},
    4: {0: {0, 1}, 1: {2, 3, 4}},
    5: {0: {0, 1}, 1: {2, 3, 4}},
    6: {0: {0, 1}, 1: {2, 3, 4}},
    7: {0: {0, 1}, 1: {2, 3, 4}},
}
# - - - Generate other communities


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


plot_commus = plot_long_com(
    communities=ground_truth_communities,
    snapshots=snapshots,
    nodes=set(range(nb_nodes)),
    width=1400,
    display_xlabel=False,
    display_ylabel=False,
    display_xticks_labels=True,
    display_yticks_labels=True,
)
plot_commus.savefig(
    PLOT_PATH + "plot_0.png",
    bbox_inches="tight",
)
