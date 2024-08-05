import os
import sys
from functools import reduce
from itertools import product
from math import gcd

import networkx as nx
import pandas as pd

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from plotting_tools import plot_long_com

file_path = "socio_pattern_highschool/High-School_data_2013.txt"

timestamps_labels = set()
nodes_labels = set()
communities_labels = set()

nodes_commus = {}

temporal_links_labels = []

with open(file_path, "r") as file:
    for line in file:
        timestamp, source, target, commu_source, commu_target = line.strip().split()
        temporal_links_labels.append([source, target, int(timestamp)])

        timestamps_labels.add(int(timestamp))
        nodes_labels.add(source)
        nodes_labels.add(target)
        communities_labels.add(commu_source)
        communities_labels.add(commu_target)

        nodes_commus[source] = commu_source
        nodes_commus[target] = commu_target


nodes_dict = {node_lab: ite for ite, node_lab in enumerate(nodes_labels)}
nodes_dict_reverse = {ite: node_lab for node_lab, ite in nodes_dict.items()}
communities_dict = {node_lab: ite for ite, node_lab in enumerate(communities_labels)}

reverse_nodes_commus = {}
for node, commu in nodes_commus.items():
    if commu not in reverse_nodes_commus:
        reverse_nodes_commus[commu] = []
    reverse_nodes_commus[commu].append(node)

temporal_links = []
for source, target, timestamp in temporal_links_labels:
    source_ = nodes_dict[source]
    target_ = nodes_dict[target]
    link = sorted([source_, target_])
    temporal_links.append(
        [
            link[0],
            link[1],
            timestamp,
        ]
    )

df_temporal_links = pd.DataFrame(
    temporal_links,
    columns=[
        "source",
        "target",
        "time",
    ],
)


def gcd_list(numbers):
    return reduce(gcd, numbers)


def pp_timesteps(timestamps: list):
    gcd = gcd_list(timestamps)
    min_date = min(timestamps)
    timesteps = {ts: int((ts - min_date) / gcd) for ts in timestamps}
    return timesteps


timesteps = pp_timesteps(sorted(list(timestamps_labels)))
df_temporal_links["snapshot"] = df_temporal_links["time"].apply(
    lambda val: timesteps[val]
)

df_grouped_interactions = (
    df_temporal_links.groupby(["source", "target", "snapshot"])["time"]
    .count()
    .reset_index()
)

snapshots_list = sorted(df_grouped_interactions.snapshot.unique().tolist())

snapshots = []
for snap in snapshots_list:
    tmp_df = df_grouped_interactions[df_grouped_interactions.snapshot == snap]
    # input(tmp_df)
    edges = tmp_df[["source", "target"]].values.tolist()
    # input(edges)
    graph = nx.Graph()
    graph.add_nodes_from(list(nodes_dict.values()))
    graph.add_edges_from(edges)
    snapshots.append(graph)

communities_time = list()
for time, node in product(timesteps.values(), nodes_labels):
    communities_time.append(
        [nodes_dict[node], time, communities_dict[nodes_commus[node]]]
    )
plot_communities_dict = {}
for node, time, commu in communities_time:
    if commu not in plot_communities_dict:
        plot_communities_dict[commu] = []
    plot_communities_dict[commu].append([node, time])
communities = list(plot_communities_dict.values())


commus_sort = ["2BIO1", "2BIO2", "2BIO3", "PSI*", "PC", "PC*", "MP", "MP*1", "MP*2"]

# Rework timesteps
timesteps_list = list(timesteps.values())
new_time_steps = {tstep: ite for ite, tstep in enumerate(timesteps_list)}
nights = []
for ts1, ts2 in zip(timesteps_list[:-1], timesteps_list[1:]):
    if ts2 - ts1 == 1:
        continue
    nights.append((new_time_steps[ts2] + new_time_steps[ts1]) / 2)


new_nodes_label = {}
tmp_nodes_sort = []
for commu in commus_sort:
    tmp_nodes_sort += reverse_nodes_commus[commu]
new_nodes_label = {nodes_dict[node]: ite for ite, node in enumerate(tmp_nodes_sort)}

new_communities = []
for commu in communities:
    tmp_commu = []
    for node, time in commu:
        tmp_commu.append((new_nodes_label[node], new_time_steps[time]))
    new_communities.append(set(tmp_commu))
del communities

title = ""
plot_commus = plot_long_com(
    communities=new_communities,
    snapshots=snapshots,
    title=title,
    width=1600,
    height=1200,
    node_alpha=0.1,
    show_edges=False,
    show_nodes=False,
    nights=nights,
)
plot_name = "classes"
plot_commus.savefig(
    f"socio_pattern_highschool/plots/plot_{plot_name}.png",
    bbox_inches="tight",
)
