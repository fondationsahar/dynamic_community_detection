import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def plot_long_com(
    communities,
    snapshots,
    nodes=set(),
    title="",
    width=800,
    height=600,
    label_font_size=18,
    display_xlabel=True,
    display_ylabel=True,
    display_xticks_labels=False,
    display_yticks_labels=False,
    monochrome=False,
    trim=False,
    edge_alpha=1,
    node_alpha=0.4,
    full_time_range=False,
    show_edges=True,
    show_nodes=True,
    nights=[],
    max_shown_communities=-1,
):
    nb_communities = len(communities)
    print("Nb communities:", nb_communities)

    if not len(nodes):
        for commu in communities:
            nodes |= set([tup[0] for tup in commu])

    timesteps = set()
    for commu in communities:
        timesteps |= set([tup[1] for tup in commu])

    if full_time_range:
        min_ts = min(timesteps)
        max_ts = max(timesteps)
        timesteps = set(range(min_ts, max_ts + 1))

    width = width / 100
    height = height / 100 * 0.75
    fig, ax = plt.subplots(1, figsize=(width, height), dpi=100)

    if trim:
        start_nodes = {node: len(timesteps) for node in nodes}
        ite_snap = 0
        while ite_snap < len(snapshots):
            snapshot = snapshots[ite_snap]
            if not snapshot:
                ite_snap += 1
                continue
            for edge in snapshot.edges:
                if start_nodes[edge[0]] > ite_snap:
                    start_nodes[edge[0]] = ite_snap
                if start_nodes[edge[1]] > ite_snap:
                    start_nodes[edge[1]] = ite_snap
            ite_snap += 1

        end_nodes = {node: 0 for node in nodes}
        ite_snap = len(snapshots) - 1
        while ite_snap > 0:
            snapshot = snapshots[ite_snap]
            if not snapshot:
                ite_snap -= 1
                continue
            for edge in snapshot.edges:
                if end_nodes[edge[0]] < ite_snap:
                    end_nodes[edge[0]] = ite_snap
                if end_nodes[edge[1]] < ite_snap:
                    end_nodes[edge[1]] = ite_snap
            ite_snap -= 1

    else:
        start_nodes = {node: min(timesteps) for node in nodes}
        end_nodes = {node: max(timesteps) for node in nodes}

    if max_shown_communities > -1:
        communities = sort_list_by_elemsize(communities, max_shown_communities)
    nb_communities = len([commu for commu in communities if len(commu) > 1])

    if monochrome:
        tmp_colors = generate_greys(nb_communities)
        colors = split_and_aggregate(tmp_colors, 3)

    else:
        colors = generate_pastel_colors(nb_communities)

    periods = []
    for ite_commu, community in enumerate(communities):
        for node, time in community:
            if time < start_nodes[node] or time > end_nodes[node]:
                continue
            if len(community) == 1:
                rect = Rectangle(
                    (time, node),
                    width=0.99,
                    height=0.99,
                    color="white",
                    edgecolor=None,
                )
            else:
                rect = Rectangle(
                    (time, node),
                    width=0.99,
                    height=0.99,
                    color=colors[ite_commu],
                    edgecolor=None,
                )

            periods.append(rect)

    pc = PatchCollection(periods, match_original=True)

    # Add collection to axes
    ax.add_collection(pc)

    # Draw nodes
    if show_nodes:
        for node in nodes:
            ax.hlines(
                y=node + 0.5,
                xmin=start_nodes[node],
                xmax=end_nodes[node] + 1,
                color="black",
                linestyle="-",
                linewidth=0.5,
                alpha=node_alpha,
            )

    # Draw edges
    if show_edges:
        for ite, snapshot in enumerate(snapshots):
            if not snapshot:
                continue
            for edge in snapshot.edges:
                center1 = (ite + 0.5, edge[0] + 0.5)
                center2 = (ite + 0.5, edge[1] + 0.5)
                arc = draw_arc(center2, center1, alpha=edge_alpha)
                ax.add_patch(arc)

    for night in nights:
        ax.vlines(
            x=night + 0.5,
            ymin=0,
            ymax=len(end_nodes),
            color="black",
            linewidth=2,
        )

    if display_xticks_labels:
        # Get the current xticks
        tmp_current_xticks = ax.get_xticks()
        current_xticks = []
        for x1, x2 in zip(tmp_current_xticks[:-1], tmp_current_xticks[1:]):
            current_xticks.append((x1 + x2) / 2)
        current_xticks.append(x2)
        # Calculate the interval
        interval = current_xticks[1] - current_xticks[0]
        # Shift xticks by half an interval
        new_xticks = current_xticks + interval
        new_xticks = list(range(int(min(new_xticks)), int(max(new_xticks)) + 1))
        new_xticks = [tck - 0.5 for tck in new_xticks]
        ax.set_xticks(new_xticks)
        new_labels = [ts + 1 for ts in timesteps]
        if len(timesteps) == 8:
            new_labels = [""] + new_labels + ["", ""]
        else:
            new_labels = new_labels + [""] * (len(new_xticks) - len(new_labels))
        ax.set_xticklabels(new_labels, fontsize=14)

    else:
        ax.set_xticklabels([])

    if display_yticks_labels:
        current_yticks = ax.get_yticks()
        interval = current_yticks[1] - current_yticks[0]
        new_yticks = current_yticks + interval / 2
        ax.set_yticks(new_yticks)
        node_labels = [""] + list(ALPHABET[: len(nodes)]) + ["", ""]
        ax.set_yticklabels(node_labels, fontsize=18)
    else:
        ax.set_yticklabels([])

    # node_labels = ["", "a", "b", "c", "d", "e", "", ""]
    # current_yticks = ax.get_yticks()
    # interval = current_yticks[1] - current_yticks[0]
    # new_yticks = current_yticks + interval / 2
    # ax.set_yticks(new_yticks)
    # ax.set_yticklabels(node_labels, fontsize=18)

    ax.margins(x=0.02)
    ax.margins(y=0.05)

    if display_xlabel:
        ax.xaxis.set_label_coords(0, 0.025)
        ax.set_xlabel("Time", fontsize=label_font_size, fontweight="bold")

    if display_ylabel:
        ax.margins(y=0.05)
        ax.yaxis.set_label_coords(0, 1)
        ax.set_ylabel("Nodes", fontsize=label_font_size, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tick_params(left=False, bottom=False)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    plt.plot()
    return fig


def generate_greys(n_colors):
    greys = []
    step = 255 // (n_colors * 2 - 1) if n_colors > 1 else 0
    for i in range(n_colors, n_colors * 2):
        # Convert the value to a hexadecimal string
        grey_value = format(i * step, "02x")
        # Format the color in hexadecimal
        grey_color = f"#{grey_value}{grey_value}{grey_value}"
        greys.append(grey_color)

    return greys


def split_and_aggregate(lst, N):
    # Split the list into N equal parts
    split_size = len(lst) // N
    split_lists = [lst[i : i + split_size] for i in range(0, len(lst), split_size)]

    # Aggregate the split lists
    aggregated_list = []
    for i in range(split_size):
        aggregated_list.extend(
            [sublist[i] for sublist in split_lists if i < len(sublist)]
        )

    return aggregated_list


def draw_arc(center1, center2, alpha=1, flatten_factor=0.7):
    radius = (
        np.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2) / 2
    )
    angle1 = np.arctan2(center2[1] - center1[1], center2[0] - center1[0])
    angle2 = angle1 + np.pi
    # Adjust angles for the arc
    angle1 = np.degrees(angle1)
    angle2 = np.degrees(angle2)

    width = 2 * radius
    height = (width / abs(center1[1] - center2[1]) ** 0.5 * 0.9) * flatten_factor

    # Draw the arc
    arc = patches.Arc(
        ((center1[0] + center2[0]) / 2, (center1[1] + center2[1]) / 2),
        height,
        width,
        angle=0,
        theta1=min(angle1, angle2),
        theta2=max(angle1, angle2),
        facecolor="black",
        edgecolor="black",
        linewidth=0.7,
        alpha=alpha,
    )

    return arc


def generate_pastel_colors(n):
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = [mcolors.hsv_to_rgb((h, 0.5, 0.9)) for h in hues]
    return colors


def sort_list_by_elemsize(lis, top_n=20):
    list_size = {ite: len(tmp_lis) for ite, tmp_lis in enumerate(lis)}
    sorted_list_size = sorted(list_size.items(), key=lambda x: x[1], reverse=True)
    outlist = []
    ite = 0
    while ite < top_n:
        outlist.append(lis[sorted_list_size[ite][0]])
        ite += 1
    return outlist
