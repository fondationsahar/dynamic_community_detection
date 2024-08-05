import json
import os
import sys

import networkx as nx
import pandas as pd

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from cd_methods import get_no_smoothing_communities, group_by_time

# - - - Import data - - -
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

# - - - preprocess data - - - -
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

# - - - community detection - - -

aggregation_windows = [
    "day",
    "half_day",
    "hour",
    "half_hour",
    "quarter_hour",
    "5min",
]

merge_thresholds = [
    0,
    0.2,
    0.4,
    0.6,
    0.8,
    1,
]

for agg_win in aggregation_windows:
    print("Window snapshot:", agg_win)
    grouped_timestamps = group_by_time(timestamps_labels, groupby_type=agg_win)
    timestamp_groups = {}
    for group, tss in grouped_timestamps.items():
        timestamp_groups.update({ts: group for ts in tss})

    df_temporal_links["snapshot"] = df_temporal_links["time"].apply(
        lambda val: timestamp_groups[val]
    )

    df_grouped_interactions = (
        df_temporal_links.groupby(["source", "target", "snapshot"])["time"]
        .count()
        .reset_index()
    )

    snapshots_list = sorted(df_grouped_interactions.snapshot.unique().tolist())
    snapshots_dict_reverse = {ite: snap for ite, snap in enumerate(snapshots_list)}

    snapshots = []
    for snap in snapshots_list:
        tmp_df = df_grouped_interactions[df_grouped_interactions.snapshot == snap]
        weighted_edges = tmp_df[["source", "target", "time"]].values.tolist()
        graph = nx.Graph()
        graph.add_nodes_from(list(nodes_dict.values()))
        graph.add_weighted_edges_from(weighted_edges)
        snapshots.append(graph)

    louvain_ntries = 3
    ns_communities_dict, _ = get_no_smoothing_communities(
        snapshots=snapshots,
        n_tries=louvain_ntries,
        merge_threshold=merge_thresholds,
    )

    for thresh, ns_communities in ns_communities_dict.items():
        print("\tThresh:", thresh)
        pp_ns_communities = []
        for time_commu in ns_communities:
            tmp_time_commu = []
            for node, time in time_commu:
                for timestamps in grouped_timestamps[snapshots_dict_reverse[time]]:
                    tmp_time_commu.append((nodes_dict_reverse[node], timestamps))
            pp_ns_communities.append(tmp_time_commu)

        with open(
            f"socio_pattern_highschool/commus_ns/ns_commu_groupby={agg_win}_mtresh={thresh}.json",
            "w",
        ) as file:
            json.dump(pp_ns_communities, file, indent=4)
