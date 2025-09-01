from lago.lago_src.stem import SingleTimeEdgeMover
from lago.lago_src.stnm import SingleTimeNodeMover
from lago.lago_src.tmm import TimeModuleMover


def run_with_refinement_in_rtmm(
    time_module_mover: TimeModuleMover,
    refiner: None | SingleTimeNodeMover | SingleTimeEdgeMover,
    verbose: bool = False,
    nb_edges: float = 1,  # NOTE overkill to consider this here ?
):
    """Apply LAGO with the refinement_in option. After each
        optimized level of rtmm (core loop), refinement is applied.
        The algo keeps running until no optimization can be
        done anymore.

    Returns:
        float: Δ L-Modularity between initial state and final state.
    """
    optimize_level = 0
    relative_longitudinal_modularity = 0

    improvement = True
    while improvement:
        tmp_delta_longitudinal_modularity = time_module_mover.run()
        improvement = tmp_delta_longitudinal_modularity > 0
        delta_longitudinal_modularity = tmp_delta_longitudinal_modularity
        if verbose and improvement:
            print(
                f"\tTime Module Movements :: Δ L-Modularity = {round(tmp_delta_longitudinal_modularity / nb_edges, ndigits=5)}"
            )

        if refiner is not None:
            tmp_delta_longitudinal_modularity = refiner.run()
            improvement |= tmp_delta_longitudinal_modularity > 0
            delta_longitudinal_modularity += tmp_delta_longitudinal_modularity
            if verbose and improvement:
                print(
                    f"\tRefinement [{refiner.kind}]     :: Δ L-Modularity = {round(tmp_delta_longitudinal_modularity / nb_edges, ndigits=5)}"
                )

        if verbose and improvement:
            print(
                f"Optimized level {optimize_level}             :: Δ L-Modularity = {round(delta_longitudinal_modularity / nb_edges, ndigits=5)}"
            )

        optimize_level += 1

        relative_longitudinal_modularity += delta_longitudinal_modularity

    return relative_longitudinal_modularity, time_module_mover.modules
