from lago.core.linkstream import LinkStream

from ._lago_module import _LagoModule, reset_module_counter
from .delta_lm import (
    DeltaLongitudinalModularityComputer,
)
from .rir import run_with_refinement_in_rtmm
from .roor import run_with_refinement_out_of_rtmm
from .stem import SingleTimeEdgeMover
from .stnm import SingleTimeNodeMover
from .tmm import TimeModuleMover


def lago_run(
    linkstream: LinkStream,
    lex: str,
    gamma: float,
    omega: float,
    refinement: str | None,
    fast_exploration: bool,
    refinement_in: bool,
    verbose: bool | int,
    stopping_criterion: float,
    ndigits_logs: int,
    rng=None,
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
        verbose (bool | int, optional): Verbosity level. 0=silent, 1=progress info,
            2=detailed debug, 3=infinite loop tracking.
            Defaults to False.
    Returns:
        float: Δ L-Modularity between initial state and final state.
    """

    time_module_mover, refiner = _init_movers(
        linkstream,
        lex,
        gamma,
        omega,
        fast_exploration,
        refinement,
        stopping_criterion,
        rng,
    )
    if refinement_in:
        return run_with_refinement_in_rtmm(
            time_module_mover,
            refiner,
            verbose,
            linkstream.weight,
            stopping_criterion,
            ndigits_logs,
        )

    else:
        return run_with_refinement_out_of_rtmm(
            time_module_mover,
            refiner,
            verbose,
            linkstream.weight,
            stopping_criterion,
            ndigits_logs,
        )


def _init_movers(
    linkstream: LinkStream,
    lex: str,
    gamma: float,
    omega: float,
    fast_exploration: bool,
    refinement: str | None,
    stopping_criterion: float,
    rng=None,
):
    modules = _init_modules(linkstream)

    lm_computer = DeltaLongitudinalModularityComputer(
        linkstream,
        lex,
        gamma,
        omega,
    )
    if refinement == "STNM":
        # The batch evaluation relies on `leaf.module.parent` naming the module
        # that holds the leaf. STNM breaks that from its second round on:
        # _update_modules_after_stnm rebinds its `modules` set instead of
        # mutating the one it shares with the TMM mover, so the two movers then
        # disagree about which modules are current. The per-candidate path
        # tests membership on the leaf sets themselves and does not care, so
        # STNM stays on it -- and its results stay exactly what they were.
        lm_computer.use_batch = False
    time_module_mover = TimeModuleMover(
        fast_exploration,
        modules,
        lm_computer,
        linkstream.partite_mapping,
        stopping_criterion,
        rng,
    )

    refiner = None

    if refinement == "STNM":
        refiner = SingleTimeNodeMover(
            fast_exploration,
            modules,
            lm_computer,
            linkstream.partite_mapping,
            stopping_criterion,
            rng,
        )

    elif refinement == "STEM":
        refiner = SingleTimeEdgeMover(
            linkstream,
            fast_exploration,
            modules,
            lm_computer,
            stopping_criterion,
            rng,
        )

    return time_module_mover, refiner


def _init_modules(linkstream: LinkStream) -> set[_LagoModule]:
    # Restart the module creation index so that a run's exploration order does
    # not depend on how many modules earlier runs in this process created.
    reset_module_counter()

    modules = set()
    iterator = linkstream.leaves_dict.values()
    for leaf in iterator:
        if leaf.module in modules:
            modules.remove(leaf.module)
        tmp_module = _LagoModule({leaf})
        leaf.module = tmp_module
        modules.add(tmp_module)

    for module in modules:
        module.compute_neighbors()

    return modules
