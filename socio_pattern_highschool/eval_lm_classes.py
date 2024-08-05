import os
import sys
from functools import reduce
from itertools import product
from math import gcd

import pandas as pd

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import longitudinal_modularity_tools as lmt

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
communities_dict = {node_lab: ite for ite, node_lab in enumerate(communities_labels)}


# - - - Preprocess Data - - -


def gcd_list(numbers):
    return reduce(gcd, numbers)


def pp_timesteps(timestamps: list):
    gcd = gcd_list(timestamps)
    min_date = min(timestamps)
    timesteps = {ts: int((ts - min_date) / gcd) for ts in timestamps}
    return timesteps


# Normalise time steps so
#   - first one is 0
#   - gap between time steps is 1
timesteps = pp_timesteps(sorted(list(timestamps_labels)))


# Build df of nodes communities membership over time
communities_time = list()
for time, node in product(timesteps.values(), nodes_labels):
    communities_time.append(
        [nodes_dict[node], time, communities_dict[nodes_commus[node]]]
    )
communities_df = pd.DataFrame(communities_time, columns=["node", "time", "commu"])


# Build dataframe of time links and the affiliated communities
temporal_links = []
for source, target, timestamp in temporal_links_labels:
    source_ = nodes_dict[source]
    target_ = nodes_dict[target]
    link = sorted([source_, target_])
    temporal_links.append(
        [
            link[0],
            link[1],
            timesteps[timestamp],
            communities_dict[nodes_commus[source]],
            communities_dict[nodes_commus[target]],
        ]
    )
df_temporal_links = pd.DataFrame(
    temporal_links,
    columns=[
        "source",
        "target",
        "time",
        "source_commu",
        "target_commu",
    ],
)


for expectation in ["cm", "jm", "mm"]:
    l_modularity = lmt.get_longitudinal_modularity(
        time_links_df=df_temporal_links,
        communities_df=communities_df,
        expectation=expectation,
        omega=2,
    )

    print(f"L-Modularity ({expectation}):", l_modularity)
