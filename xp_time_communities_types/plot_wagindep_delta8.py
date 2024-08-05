import os

import matplotlib.pyplot as plt
import networkx as nx

XP_TYPE = "win_agg_indep"
PLOT_PATH = f"xp_time_communities_types/plot/plot_{XP_TYPE}/"

os.makedirs(PLOT_PATH, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

G1 = nx.Graph()

G1.add_node(0, label="a", community=1)
G1.add_node(1, label="b", community=1)
G1.add_node(2, label="c", community=1)
G1.add_node(3, label="d", community=2)
G1.add_node(4, label="e", community=2)

G1.add_edge(0, 1, weight=3)
G1.add_edge(0, 2, weight=1)
G1.add_edge(1, 2, weight=2)
G1.add_edge(1, 3, weight=1)
G1.add_edge(3, 4, weight=3)

node_labels1 = nx.get_node_attributes(G1, "label")

edge_labels1 = nx.get_edge_attributes(G1, "weight")

edge_widths1 = [G1[u][v]["weight"] for u, v in G1.edges()]

community_colors = {1: (0.9, 0.45, 0.45), 2: (0.45, 0.9, 0.9)}
node_colors1 = [community_colors[G1.nodes[node]["community"]] for node in G1.nodes()]

pos1 = nx.spring_layout(G1)

nx.draw(
    G1,
    ax=ax1,
    pos=pos1,
    with_labels=True,
    labels=node_labels1,
    node_size=3000,
    node_color=node_colors1,
    font_size=20,
    font_weight="bold",
    width=edge_widths1,
)
nx.draw_networkx_edge_labels(
    G1, pos=pos1, edge_labels=edge_labels1, font_color="black", font_size=12, ax=ax1
)

G2 = nx.Graph()

G2.add_node(0, label="a", community=1)
G2.add_node(1, label="b", community=1)
G2.add_node(2, label="c", community=2)
G2.add_node(3, label="d", community=2)
G2.add_node(4, label="e", community=2)

G2.add_edge(0, 1, weight=2)
G2.add_edge(0, 2, weight=1)
G2.add_edge(2, 3, weight=4)
G2.add_edge(2, 4, weight=1)
G2.add_edge(3, 4, weight=2)

node_labels2 = nx.get_node_attributes(G2, "label")

edge_labels2 = nx.get_edge_attributes(G2, "weight")

edge_widths2 = [G2[u][v]["weight"] for u, v in G2.edges()]

node_colors2 = [community_colors[G2.nodes[node]["community"]] for node in G2.nodes()]

pos2 = nx.spring_layout(G2, pos=pos1, iterations=10)

nx.draw(
    G2,
    ax=ax2,
    pos=pos2,
    with_labels=True,
    labels=node_labels2,
    node_size=3000,
    node_color=node_colors2,
    font_size=20,
    font_weight="bold",
    width=edge_widths2,
)
nx.draw_networkx_edge_labels(
    G2, pos=pos2, edge_labels=edge_labels2, font_color="black", font_size=12, ax=ax2
)

for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)

ax1.text(0.5, -0.1, "1", transform=ax1.transAxes, fontsize=20, ha="center")
ax2.text(0.5, -0.1, "2", transform=ax2.transAxes, fontsize=20, ha="center")


plt.savefig(
    PLOT_PATH + "plot_2.png",
    bbox_inches="tight",
)
