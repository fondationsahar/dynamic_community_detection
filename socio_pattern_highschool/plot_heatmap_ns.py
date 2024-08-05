import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

filename = "socio_pattern_highschool/ns_commu_eval_emm.json"
filename = "socio_pattern_highschool/ns_commu_eval_ejm.json"

with open(filename, "r") as f:
    data = json.load(f)

data_to_plot = {}
for filename, results in data.items():
    window_aggregation = filename.split("_")[2].split("=")[-1]
    mthresh = filename.split("_")[3].split("=")[-1].split(".json")[0]
    if window_aggregation not in data_to_plot:
        data_to_plot[window_aggregation] = {}
    data_to_plot[window_aggregation][mthresh] = results["lm"]

df_data = pd.DataFrame.from_dict(data_to_plot).sort_index(ascending=False)

df_data["1/2 day"] = df_data["halfday"]
df_data["1/2 hour"] = df_data["half"]
df_data["1/4 hour"] = df_data["quarter"]
df_data["5 min"] = df_data["5min"]
df_data = df_data[["day", "1/2 day", "hour", "1/2 hour", "1/4 hour", "5 min"]]


X_labs = list(df_data.columns)
Y_labs = list(df_data.index)

data = df_data.values.tolist()


def smooth(values, factor=0.5):
    all_values = []
    for val in values:
        for one in val:
            all_values.append(one)
    min_ = min(all_values)
    max_ = max(all_values)
    all_values_dict = {
        one: ite / len(all_values) for ite, one in enumerate(sorted(all_values))
    }

    new_values = []
    for val in values:
        tmp_new_val = []
        for one in val:
            new_one = (one - min_) / (max_ - min_) * (1 - factor) + all_values_dict[
                one
            ] * factor
            tmp_new_val.append(new_one)
        new_values.append(tmp_new_val)
    return new_values


plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    smooth(data, factor=0.75),  # data,
    annot=data,
    fmt=".4f",
    cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True),
    cbar=False,
    annot_kws={"size": 22},
)

ax.set_xlabel("Window aggregation", fontsize=26)
ax.set_ylabel("Merge threshold", fontsize=26)

ax.set_xticks(np.arange(len(X_labs)) + 0.5)
ax.set_xticklabels(
    X_labs,
    rotation=45,
    ha="right",
    fontsize=22,
)
ax.set_yticks(np.arange(len(Y_labs)) + 0.5)
ax.set_yticklabels(
    Y_labs,
    fontsize=22,
)

plt.tight_layout()
plt.savefig(
    "socio_pattern_highschool/plots/lm_mme_ns_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)
