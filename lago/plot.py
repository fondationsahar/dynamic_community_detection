import copy

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def plot_dynamic_communities(
    linkstream,
    communities,
    nodes=set(),
    title="",
    node_focus=None,
    time_focus=None,
    node_OR_time_focus=False,
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
    color_edges=False,
    show_edges=True,
    show_nodes=True,
    nights=[],
    max_shown_communities=-1,
    hide_self_communities=False,
    show_plot=False,
    return_ax=False,
    edge_flatten_factor=0.7,
    width_community_color=1,
    height_community_color=0.7,
):
    raw_communities = copy.deepcopy(communities)
    communities_monochrome = {}
    if node_focus is not None or time_focus is not None:
        communities_to_focus = get_communities_to_focus(
            communities=communities,
            node_focus=node_focus,
            time_focus=time_focus,
            node_OR_time_focus=node_OR_time_focus,
        )
        communities_monochrome = {
            label: communities[label]
            for label in communities.keys()
            if label not in communities_to_focus
        }
        communities = {label: communities[label] for label in communities_to_focus}

    nb_communities_monochrome = len(communities_monochrome)
    time_links = linkstream.get_time_links()
    nb_communities = len(communities)

    if not len(nodes):
        nodes = linkstream.nodes

    width = width / 100
    height = height / 100 * 0.75
    fig, ax = plt.subplots(1, figsize=(width, height), dpi=100)

    if trim:
        start_nodes = {}
        end_nodes = {}
        for source, target, time in time_links:
            for node in [source, target]:
                start_nodes[node] = min(
                    time, start_nodes.get(node, linkstream.network_duration)
                )
                end_nodes[node] = max(time, end_nodes.get(node, 0))

    else:
        start_nodes = {node: 0 for node in nodes}
        end_nodes = {node: linkstream.network_duration for node in nodes}

    if max_shown_communities > -1:
        communities = sort_list_by_elemsize(communities, max_shown_communities)

    if hide_self_communities:
        communities_colors_mapping_seed = [
            ite for ite, commu in communities.items() if len(commu) > 1
        ]
        communities_colors_mapping_seed_greys = [
            ite for ite, commu in communities_monochrome.items() if len(commu) > 1
        ]
    else:
        communities_colors_mapping_seed = [ite for ite, commu in communities.items()]
        communities_colors_mapping_seed_greys = [
            ite for ite, commu in communities_monochrome.items()
        ]

    nb_communities = len(communities_colors_mapping_seed)

    if monochrome:
        tmp_colors = generate_greys(nb_communities)
        colors = split_and_aggregate(tmp_colors, 3)

    else:
        greys = ["gainsboro"] * (nb_communities_monochrome + 1)
        colors = generate_pastel_colors(nb_communities)
    colors_mapping = {
        ite: color for color, ite in zip(colors, communities_colors_mapping_seed)
    }
    # print(f"{len(colors_mapping)} communities to color.")
    colors_mapping.update(
        {ite: color for color, ite in zip(greys, communities_colors_mapping_seed_greys)}
    )

    time_nodes_communities = {}
    for lab, commu in raw_communities.items():
        time_nodes_communities.update({timenode: lab for timenode in commu})
    # input(colors_mapping)

    if hide_self_communities:
        periods = []
        for ite_commu, community in raw_communities.items():
            for node, time in community:
                if time < start_nodes[node] or time > end_nodes[node]:
                    continue
                if len(community) == 1:
                    rect = Rectangle(
                        (time, node + (0.5 - height_community_color / 2)),
                        width=width_community_color,
                        height=height_community_color,
                        color="white",
                    )
                else:
                    rect = Rectangle(
                        (time, node + (0.5 - height_community_color / 2)),
                        width=width_community_color,
                        height=height_community_color,
                        color=colors_mapping.get(ite_commu, "white"),
                        edgecolor=colors_mapping.get(ite_commu, "white"),
                    )

                periods.append(rect)
    else:
        periods = []
        for ite_commu, community in raw_communities.items():
            for node, time in community:
                if time < start_nodes[node] or time > end_nodes[node]:
                    continue
                rect = Rectangle(
                    (time, node + (0.5 - height_community_color / 2)),
                    # (time, node),
                    width=width_community_color,
                    height=height_community_color,
                    facecolor=colors_mapping.get(ite_commu, "gainsboro"),
                    edgecolor=colors_mapping.get(ite_commu, "gainsboro"),
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
                linewidth=0.25,
                alpha=node_alpha,
            )

    if (node_OR_time_focus and node_focus is not None) or (time_focus is not None):
        rec = draw_rectangle(
            node=node_focus,
            time=0,
            x_decay=0,
            y_delay=0.33,
            width=linkstream.network_duration + 1,
            height=0.33,
            alpha=1,
        )
        ax.add_patch(rec)

    # Draw edges
    if show_edges:
        for source, target, time in time_links:
            center1 = (time + 0.5, source + 0.5)
            center2 = (time + 0.5, target + 0.5)

            if color_edges:
                # print(time_nodes_communities)
                source_commu = time_nodes_communities.get((source, time))
                target_commu = time_nodes_communities.get((target, time))
                if source_commu is None or target_commu is None:
                    arc = draw_arc(
                        center2,
                        center1,
                        alpha=edge_alpha,
                        flatten_factor=edge_flatten_factor,
                    )
                    ax.add_patch(arc)
                    continue
                if source_commu != target_commu:
                    arc = draw_arc(
                        center2,
                        center1,
                        alpha=edge_alpha,
                        flatten_factor=edge_flatten_factor,
                    )
                    ax.add_patch(arc)
                    continue
                color = colors_mapping.get(source_commu)
                if color is None or color == "gainsboro":
                    arc = draw_arc(
                        center2,
                        center1,
                        alpha=edge_alpha,
                        flatten_factor=edge_flatten_factor,
                    )
                    ax.add_patch(arc)
                    continue
                arc = draw_arc(
                    center2,
                    center1,
                    alpha=edge_alpha,
                    color=color,
                    linewidth=0.8,
                    flatten_factor=edge_flatten_factor,
                )
                ax.add_patch(arc)
                arc = draw_arc(
                    center2,
                    center1,
                    alpha=edge_alpha,
                    flatten_factor=edge_flatten_factor,
                )
                ax.add_patch(arc)

            else:
                arc = draw_arc(
                    center2,
                    center1,
                    alpha=edge_alpha,
                    flatten_factor=edge_flatten_factor,
                )
                ax.add_patch(arc)

    for night in nights:
        ax.vlines(
            x=night + 0.5,
            ymin=0,
            ymax=len(end_nodes),
            color="black",
            linewidth=2,
        )

    if time_focus is not None:
        if node_OR_time_focus or node_focus is None:
            rec = draw_rectangle(-0.25, time_focus, height=len(nodes) + 0.5)
            ax.add_patch(rec)
        else:
            rec = draw_rectangle(node_focus, time_focus, x_decay=0.1, width=0.8)
            ax.add_patch(rec)

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
        new_labels = [ts + 1 for ts in range(linkstream.network_duration)]
        if linkstream.network_duration == 8:
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

    if show_plot:
        plt.plot()

    if return_ax:
        return fig, ax
    return fig


def generate_greys(n_colors, alpha=1):
    alpha_value = format(int(alpha * 255), "02x")
    greys = []
    step = 255 // (n_colors * 2 - 1) if n_colors > 1 else 0
    for i in range(n_colors, n_colors * 2):
        grey_value = format(i * step, "02x")
        grey_color = f"#{grey_value}{grey_value}{grey_value}{alpha_value}"
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


def draw_arc(
    center1, center2, alpha=1, flatten_factor=0.7, color="black", linewidth=0.5
):
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
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
    )

    return arc


def draw_rectangle(
    node,
    time,
    x_decay=1 / 2,
    y_delay=0,
    width=1 / 4,
    height=1,
    alpha=0.5,
):
    rectangle = patches.Rectangle(
        (time + x_decay, node + y_delay),
        width,
        height,
        facecolor="black",
        alpha=alpha,
    )
    return rectangle


def generate_pastel_colors(n):
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = [mcolors.hsv_to_rgb((h, 0.5, 0.9)) for h in hues]
    return colors


def sort_list_by_elemsize(lis, top_n=20):
    list_size = {ite: len(tmp_lis) for ite, tmp_lis in enumerate(lis.values())}
    sorted_list_size = sorted(list_size.items(), key=lambda x: x[1], reverse=True)
    outdict = {}
    ite = 0
    while ite < top_n:
        outdict[ite] = lis[sorted_list_size[ite][0]]
        ite += 1
    # outlist = []
    # ite = 0
    # while ite < top_n:
    #     outlist.append(lis[sorted_list_size[ite][0]])
    #     ite += 1
    return outdict


def get_communities_to_focus(
    communities,
    node_focus=None,
    time_focus=None,
    node_OR_time_focus=False,
):
    labels_focus = set()
    for label, members in communities.items():
        # print(label)
        # input(members)
        for node, time in members:
            # print(node)
            # input(type(node))
            if node_focus is None:
                if time == time_focus:
                    labels_focus.add(label)
                    break
            elif time_focus is None:
                if node == node_focus:
                    labels_focus.add(label)
                    break
            else:
                if node_OR_time_focus:
                    if time == time_focus or node == node_focus:
                        labels_focus.add(label)
                        break
                else:
                    if time == time_focus and node == node_focus:
                        labels_focus.add(label)
                        break

    return labels_focus
