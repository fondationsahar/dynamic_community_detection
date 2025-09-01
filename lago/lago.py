import sys

import lago.tools as tls
from lago.lago_src.runner import lago_run
from lago.linkstream import LinkStream
from lago.module import Module


def lago_communities(
    linkstream: LinkStream,
    lex: str = "MM",
    nb_iter: int = 1,
    omega: float = 2,
    refinement: str | None = "STEM",
    fast_exploration: bool = True,
    refinement_in: bool = True,
    verbose: bool = False,
):
    """LAGO (Longitudinal Agglomerative Greedy Optimization) is a method for uncovering
        dynamic communities on link streams by optimizating L-Modularity.

    Args:
        linkstream (LinkStream): Link Stream on which find dynamic communities.
        lex (str, optional): Longitudinal Expectation. Must be either "JM" or "MM".
            "JM" is for Joint-Membership and expects dynamic communities to have a very
            consistent duration of existence, whereas "MM", which means Mean-Membership,
            allows greater freedom in the temporal evolution of communities.
            Defaults to "MM".
        nb_iter (int): Number of lago runs. Best results are returned.
        omega (float, optional): Time resolution parameter. Must be >= 0. Higher values lead
            to more smoothness in communities changes.
            Defaults to 2.
        refinement (str, optional): Whether to apply a refinement strategy or not, and which one.
            Must be None, "STEM" or "STNM". Refinement significantly improves communities quality,
            but is more time consuming. None is for no refinement strategy. "STNM" is for
            Single Time Node Movements, and "STEM" is for Single Time Edge Movements. For more details,
            see dedicated paper.
            Defaults to "STEM".
        fast_exploration (bool, optional): Whether to apply the Fast Exploration strategy or not.
            If activated, it significantly reduces the time of execution but may result in poorer
            results.
            Defaults to True.
        refinement_in (bool, optional): Whether to apply refinement strategy within the core part
            or after. Applying it within the core part implies more exploration, which may results
            in better results or more chances to get stuck in local optimum. It is also more time
            consuming.
            Defaults to True.
        verbose (bool, optional): Whether to print intermediate reports or not.
            Defaults to False.
    """

    assert lex in ["JM", "MM"], 'Wrong lex value. Must be "JM" or "MM".'
    assert omega >= 0, "Wrong omega value. Must be >= 0."
    assert refinement in [
        None,
        "STNM",
        "STEM",
    ], "Wrong refinement value. Must be None, 'STNM' or 'STEM'."

    relative_lm = -sys.maxsize

    raw_modules = set[Module]()

    for itr in range(nb_iter):
        if verbose:
            print(f"\nStart iteration {itr + 1}")

        tmp_relative_lm, raw_modules = lago_run(
            linkstream,
            lex,
            omega,
            refinement,
            fast_exploration,
            refinement_in,
            verbose,
        )
        if tmp_relative_lm < relative_lm:
            continue
        relative_lm = tmp_relative_lm

    return get_time_communities(raw_modules)


def get_time_communities(modules) -> dict[int, set[tuple[int, int]]]:
    """Returns formatted time communities (= time modules).

    Returns:
        dict: time communities. Format:
            {
                community_label_1: [[n, t], [n, t], ...],
                community_label_2: [[], ...],
            }
    """
    time_communities = {
        label: tls.get_expanded_module(module.leaves)
        for label, module in enumerate(modules)
    }
    return time_communities
