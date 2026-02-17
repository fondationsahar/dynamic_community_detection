from .stem import SingleTimeEdgeMover
from .stnm import SingleTimeNodeMover
from .tmm import TimeModuleMover


def run_with_refinement_in_rtmm(
    time_module_mover: TimeModuleMover,
    refiner: None | SingleTimeNodeMover | SingleTimeEdgeMover,
    verbose: bool | int = False,
    nb_edges: float = 1,  # NOTE overkill to consider this here ?
    stopping_criterion: float = 1e-6,
    ndigits_logs: int = 10,
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
    loop_iteration = 0
    max_iterations_warning = 1000  # Threshold for warning about potential infinite loop

    # Track consecutive identical deltas to detect stuck loops
    previous_delta = None
    identical_delta_count = 0
    identical_delta_threshold = 5  # Warn after 5 identical deltas

    while improvement:
        loop_iteration += 1

        # Track potential infinite loops when verbose >= 3
        if verbose >= 3:
            print(
                f"[LOOP TRACKING] RIR starting iteration {loop_iteration}, optimize_level {optimize_level}"
            )
            if loop_iteration % 100 == 0:
                print(f"[LOOP TRACKING] RIR main loop: iteration {loop_iteration}")
            if loop_iteration >= max_iterations_warning:
                print(
                    f"[WARNING] RIR main loop exceeded {max_iterations_warning} iterations - possible infinite loop!"
                )

        if verbose >= 3:
            print("[LOOP TRACKING] RIR calling time_module_mover.run()...")

        tmp_delta_longitudinal_modularity = time_module_mover.run(verbose)

        if verbose >= 3:
            print("[LOOP TRACKING] RIR time_module_mover.run() completed")
        improvement = tmp_delta_longitudinal_modularity > stopping_criterion
        delta_longitudinal_modularity = tmp_delta_longitudinal_modularity

        if verbose >= 3:
            threshold = stopping_criterion
            print(
                f"[LOOP TRACKING] TMM delta: {tmp_delta_longitudinal_modularity:.10e}, threshold: {threshold:.10e}, improvement: {improvement}"
            )

        if verbose and improvement:
            print(
                f"\tTime Module Movements :: Δ L-Modularity = {round(tmp_delta_longitudinal_modularity / nb_edges, ndigits=ndigits_logs)}"
            )

        if refiner is not None:
            if verbose >= 3:
                print("[LOOP TRACKING] RIR calling refiner.run()...")

            tmp_delta_longitudinal_modularity = refiner.run(verbose)

            if verbose >= 3:
                print("[LOOP TRACKING] RIR refiner.run() completed")

            refiner_improvement = tmp_delta_longitudinal_modularity > stopping_criterion
            improvement |= refiner_improvement
            delta_longitudinal_modularity += tmp_delta_longitudinal_modularity

            if verbose >= 3:
                threshold = stopping_criterion
                print(
                    f"[LOOP TRACKING] Refiner delta: {tmp_delta_longitudinal_modularity:.10e}, threshold: {threshold:.10e}, improvement: {refiner_improvement}"
                )

            if verbose and improvement:
                print(
                    f"\tRefinement [{refiner.kind}]     :: Δ L-Modularity = {round(tmp_delta_longitudinal_modularity / nb_edges, ndigits=ndigits_logs)}"
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
                f"Optimized level {optimize_level}             :: Δ L-Modularity = {round(delta_longitudinal_modularity / nb_edges, ndigits=ndigits_logs)}"
            )

        optimize_level += 1

        relative_longitudinal_modularity += delta_longitudinal_modularity

    if verbose >= 3:
        print(f"[LOOP TRACKING] RIR main loop completed after {loop_iteration} iterations")
        if identical_delta_count > 0:
            print(f"[LOOP TRACKING] Final identical delta count: {identical_delta_count}")

    return relative_longitudinal_modularity, time_module_mover.modules
