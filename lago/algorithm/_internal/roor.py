from .rtmm import rtmm_optimization
from .stem import SingleTimeEdgeMover
from .stnm import SingleTimeNodeMover
from .tmm import TimeModuleMover


def run_with_refinement_out_of_rtmm(
    time_module_mover: TimeModuleMover,
    refiner: None | SingleTimeNodeMover | SingleTimeEdgeMover,
    verbose: bool | int,
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
    max_iterations_warning = 1000  # Threshold for warning about potential infinite loop

    # Track consecutive identical deltas to detect stuck loops
    previous_delta = None
    identical_delta_count = 0
    identical_delta_threshold = 5  # Warn after 5 identical deltas

    while improvement:
        # Track potential infinite loops when verbose >= 3
        if verbose >= 3:
            if nb_loop % 100 == 0:
                print(f"[LOOP TRACKING] ROOR main loop: iteration {nb_loop}")
            if nb_loop >= max_iterations_warning:
                print(
                    f"[WARNING] ROOR main loop exceeded {max_iterations_warning} iterations - possible infinite loop!"
                )

        tmp_delta_longitudinal_modularity = rtmm_optimization(
            time_module_mover,
            verbose,
            nb_edges,
            stopping_criterion,
            ndigits_logs,
        )
        improvement = tmp_delta_longitudinal_modularity > stopping_criterion
        delta_longitudinal_modularity = tmp_delta_longitudinal_modularity

        if verbose >= 3:
            threshold = stopping_criterion
            print(
                f"[LOOP TRACKING] RTMM delta: {tmp_delta_longitudinal_modularity:.10e}, threshold: {threshold:.10e}, improvement: {improvement}"
            )

        if refiner is not None:
            tmp_delta_longitudinal_modularity = refiner.run(verbose)
            refiner_improvement = tmp_delta_longitudinal_modularity > stopping_criterion
            improvement |= refiner_improvement
            delta_longitudinal_modularity += tmp_delta_longitudinal_modularity

            if verbose >= 3:
                threshold = stopping_criterion
                print(
                    f"[LOOP TRACKING] Refiner delta: {tmp_delta_longitudinal_modularity:.10e}, threshold: {threshold:.10e}, improvement: {refiner_improvement}"
                )

        # Detect repeated identical deltas (potential stuck loop)
        if verbose >= 3:
            if (
                previous_delta is not None
                and abs(delta_longitudinal_modularity - previous_delta) < 1e-15
            ):
                identical_delta_count += 1
                if identical_delta_count >= identical_delta_threshold:
                    print(
                        f"[WARNING] Same delta repeated {identical_delta_count} times: {delta_longitudinal_modularity:.10e}"
                    )
                    print(
                        "[WARNING] This may indicate the algorithm is stuck in a repetitive loop (moves being undone/redone)"
                    )
            else:
                if identical_delta_count >= identical_delta_threshold:
                    print(
                        f"[LOOP TRACKING] Delta pattern broke after {identical_delta_count} repetitions"
                    )
                identical_delta_count = 0
            previous_delta = delta_longitudinal_modularity

        if verbose and improvement:
            print(
                f"   ---> Optimized loop {nb_loop}  :: Δ L-Modularity = {round(delta_longitudinal_modularity / nb_edges, ndigits=ndigits_logs)}"
            )

        nb_loop += 1

        relative_longitudinal_modularity += delta_longitudinal_modularity

    if verbose >= 3:
        print(f"[LOOP TRACKING] ROOR main loop completed after {nb_loop - 1} iterations")
        if identical_delta_count > 0:
            print(f"[LOOP TRACKING] Final identical delta count: {identical_delta_count}")

    return relative_longitudinal_modularity, time_module_mover.modules
