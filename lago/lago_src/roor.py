from lago.lago_src.rtmm import rtmm_optimization
from lago.lago_src.stem import SingleTimeEdgeMover
from lago.lago_src.stnm import SingleTimeNodeMover
from lago.lago_src.tmm import TimeModuleMover


def run_with_refinement_out_of_rtmm(
    time_module_mover: TimeModuleMover,
    refiner: None | SingleTimeNodeMover | SingleTimeEdgeMover,
    verbose: bool,
    nb_edges: float,  # NOTE overkill to consider this here ?
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
        )
        improvement = tmp_delta_longitudinal_modularity > 0
        delta_longitudinal_modularity = tmp_delta_longitudinal_modularity

        if refiner is not None:
            tmp_delta_longitudinal_modularity = refiner.run()
            improvement |= tmp_delta_longitudinal_modularity > 0
            delta_longitudinal_modularity += tmp_delta_longitudinal_modularity

        if verbose and improvement:
            print(
                f"   ---> Optimized loop {nb_loop}  :: Δ L-Modularity = {round(delta_longitudinal_modularity / nb_edges, ndigits=5)}"
            )

        nb_loop += 1

        relative_longitudinal_modularity += delta_longitudinal_modularity

    return relative_longitudinal_modularity, time_module_mover.modules
