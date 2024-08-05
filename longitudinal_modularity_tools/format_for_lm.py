import networkx as nx
import pandas as pd


def snapshots_to_dataframe(snapshots):
    dataframes = list()
    for ite, nx_graph in enumerate(snapshots):
        tmp_df = nx.to_pandas_edgelist(nx_graph)
        tmp_df["time"] = ite
        dataframes.append(tmp_df)

    temporal_network = pd.concat(dataframes)

    return temporal_network


def format_lm_input(temporal_network, time_communities):
    communities_dfs = list()
    for ite, commu in enumerate(time_communities):
        communities_dfs.append(pd.DataFrame(commu, columns=["node", "time"]))
        communities_dfs[-1]["commu"] = ite
    full_communities_df = pd.concat(communities_dfs)
    del communities_dfs

    tmp_full_communities_df = full_communities_df.rename(
        columns={"node": "source", "commu": "source_commu"}
    )

    temporal_network_w_communities = temporal_network.merge(
        tmp_full_communities_df,
        how="left",
        on=["source", "time"],
    )

    tmp_full_communities_df = full_communities_df.rename(
        columns={"node": "target", "commu": "target_commu"}
    )

    temporal_network_w_communities = temporal_network_w_communities.merge(
        tmp_full_communities_df,
        how="left",
        on=["target", "time"],
    )

    return temporal_network_w_communities, full_communities_df
