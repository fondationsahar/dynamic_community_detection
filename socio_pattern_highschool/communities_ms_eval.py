import json
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
from communities_tools import get_nvi

communities_folder_path = "socio_pattern_highschool/commus_ms/"

# To adapt
lm_expectation = "jm"
results_storage_file = (
    f"socio_pattern_highschool/mm_commu_eval_{results_storage_file}e.json"
)

# - - - - Import data - - - -

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

# - - - preprocess data - - - -


def gcd_list(numbers):
    return reduce(gcd, numbers)


def pp_timesteps(timestamps: list):
    gcd = gcd_list(timestamps)
    min_date = min(timestamps)
    timesteps = {ts: int((ts - min_date) / gcd) for ts in timestamps}
    return timesteps


timesteps = pp_timesteps(sorted(list(timestamps_labels)))

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

class_communities_dict = {}
for time, node in product(timesteps.values(), nodes_labels):
    commu = communities_dict[nodes_commus[node]]
    if commu not in class_communities_dict:
        class_communities_dict[commu] = []
    class_communities_dict[commu].append((nodes_dict[node], time))
class_communities_dict = {key: set(val) for key, val in class_communities_dict.items()}
class_communities = list(class_communities_dict.values())

# - - - compute LM for all communities

filenames = os.listdir(communities_folder_path)

results = {}

for filename in filenames:
    print("Computing", filename)

    window_aggregation = filename.split("_")[2].split("=")[-1]
    iwr = filename.split("_")[3].split("=")[-1].split(".json")[0]

    print("\tWindow agg:", window_aggregation)
    print("\tIWR:", iwr)

    with open(communities_folder_path + filename, "r", encoding="utf-8") as file:
        raw_mm_communities = json.load(file)

    print("\t\tData loaded")
    mm_communities = []
    for commu in raw_mm_communities:
        tmp_commu = []
        for node, time in commu:
            tmp_commu.append((nodes_dict[node], timesteps[time]))
        mm_communities.append(set(tmp_commu))
    del raw_mm_communities

    print("\tCompute NVI")
    mm_nvi = get_nvi(class_communities, mm_communities)

    print("\t\tNVI:", mm_nvi)

    temp_net_w_commus, communities_df = lmt.format_lm_input(
        df_temporal_links,
        mm_communities,
    )
    del mm_communities

    print("\tCompute LM")

    l_modularity, time_penalty = lmt.get_longitudinal_modularity(
        time_links_df=temp_net_w_commus,
        communities_df=communities_df,
        expectation="jm",
        omega=2,
        return_time_penalty=True,
    )
    print("\t\tLM:", l_modularity)

    results[filename] = {
        "nvi": mm_nvi,
        "lm": l_modularity,
        "time_penalty": time_penalty,
    }

    with open(
        results_storage_file,
        "w",
    ) as file:
        json.dump(results, file, indent=4)
