from .rtmm import rtmm_optimization
from .stem import SingleTimeEdgeMover
from .stnm import SingleTimeNodeMover
from .tmm import TimeModuleMover


def run_with_refinement_out_of_rtmm(
    time_module_mover: TimeModuleMover,
    refiner: None | SingleTimeNodeMover | SingleTimeEdgeMover,
    verbose: bool,
    nb_edges: float,  # NOTE overkill to consider this here ?
    stopping_criterion: float = 1e-6,
    ndigits_logs: int = 10,
):
    """Apply LAGO without the refinement_in. RTMM (core loop) is applied,
        then refinement is applied. Two steps are then applied
        again until no optimization can be done anymore.

    Returns:
        float: Δ L-Modularity between initial state and final state.
    """

    relative_longitudinal_modularity = 0
    improvement = True
    nb_loop = 1
    while improvement:
        tmp_delta_longitudinal_modularity = rtmm_optimization(
            time_module_mover,
            verbose,
            nb_edges,
            stopping_criterion,
            ndigits_logs,
        )
        improvement = tmp_delta_longitudinal_modularity > stopping_criterion * nb_edges
        delta_longitudinal_modularity = tmp_delta_longitudinal_modularity

        if refiner is not None:
            tmp_delta_longitudinal_modularity = refiner.run()
            improvement |= (
                tmp_delta_longitudinal_modularity > stopping_criterion * nb_edges
            )
            delta_longitudinal_modularity += tmp_delta_longitudinal_modularity

        if verbose and improvement:
            print(
                f"   ---> Optimized loop {nb_loop}  :: Δ L-Modularity = {round(delta_longitudinal_modularity / nb_edges, ndigits=ndigits_logs)}"
            )

        nb_loop += 1

        relative_longitudinal_modularity += delta_longitudinal_modularity

    return relative_longitudinal_modularity, time_module_mover.modules
