from lago.lago_src.delta_lm import (
    DeltaLongitudinalModularityComputer,
)
from lago.lago_src.rir import run_with_refinement_in_rtmm
from lago.lago_src.roor import run_with_refinement_out_of_rtmm
from lago.lago_src.stem import SingleTimeEdgeMover
from lago.lago_src.stnm import SingleTimeNodeMover
from lago.lago_src.tmm import TimeModuleMover
from lago.linkstream import LinkStream
from lago.module import Module


def lago_run(
    linkstream: LinkStream,
    lex: str,
    omega: float,
    refinement: str | None,
    fast_exploration: bool,
    refinement_in: bool,
    verbose: bool,
):
    """Apply LAGO once.
    Args:
        linkstream (LinkStream): Link Stream on which find dynamic communities.
        lex (str, optional): Longitudinal Expectation. Must be either "JM" or "MM".
            "JM" is for Joint-Membership and expects dynamic communities to have a very
            consistent duration of existence, whereas "MM", which means Mean-Membership,
            allows greater freedom in the temporal evolution of communities.
            Defaults to "JM".
        omega (float, optional): Time resolution parameter. Must be >= 0. Higher values lead
            to more smoothness in communities changes.
            Defaults to 2.
        refinement (str, optional): Whether to apply a refinement strategy or not, and which one.
            Must be None, "STEM" or "STNM". Refinement significantly improves communities quality,
            but is more time consuming. None is for no refinement strategy. "STNM" is for
            Single Time Node Movements, and "STEM" is for Single Time Edge Movements. For more details,
            see dedicated paper.
            Defaults to "STNM".
        fast_exploration (bool, optional): Whether to apply the Fast Exploration strategy or not.
            If activated, it significantly reduces the time of execution but may result in poorer
            results.
            Defaults to True.
        refinement_in (bool, optional): Whether to apply refinement strategy within the core part
            or after. Applying it within the core part implies more exploration, which may results
            in better results or more chances to get stuck in local optimum. It is also more time
            consuming.
            Defaults to False.
        verbose (bool, optional): Whether to print intermediate reports or not.
            Defaults to False.
    Returns:
        float: Δ L-Modularity between initial state and final state.
    """

    time_module_mover, refiner = _init_movers(
        linkstream,
        lex,
        omega,
        fast_exploration,
        refinement,
    )
    if refinement_in:
        return run_with_refinement_in_rtmm(
            time_module_mover,
            refiner,
            verbose,
            linkstream.nb_edges,
        )

    else:
        return run_with_refinement_out_of_rtmm(
            time_module_mover,
            refiner,
            verbose,
            linkstream.nb_edges,
        )


def _init_movers(
    linkstream: LinkStream,
    lex: str,
    omega: float,
    fast_exploration: bool,
    refinement: str | None,
):
    modules = _init_modules(linkstream)

    lm_computer = DeltaLongitudinalModularityComputer(
        linkstream,
        lex,
        omega,
    )
    time_module_mover = TimeModuleMover(
        fast_exploration,
        modules,
        lm_computer,
    )

    refiner = None

    if refinement == "STNM":
        refiner = SingleTimeNodeMover(
            fast_exploration,
            modules,
            lm_computer,
        )

    elif refinement == "STEM":
        refiner = SingleTimeEdgeMover(
            linkstream,
            fast_exploration,
            modules,
            lm_computer,
        )

    return time_module_mover, refiner


def _init_modules(linkstream: LinkStream) -> set[Module]:
    modules = set()
    iterator = linkstream.leaves_dict.values()
    for leaf in iterator:
        if leaf.module in modules:
            modules.remove(leaf.module)
        tmp_module = Module({leaf})
        leaf.module = tmp_module
        modules.add(tmp_module)

    for module in modules:
        module.compute_neighbors()

    return modules
