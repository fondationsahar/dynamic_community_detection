from .tmm import TimeModuleMover


def rtmm_optimization(
    time_module_mover: TimeModuleMover,
    verbose: bool | int,
    nb_edges: float,
    stopping_criterion: float = 1e-6,
    ndigits_logs: int = 10,
) -> float:
    """Core vanilla loop for optimizing L-Modularity (Recursive Time Module Movements).
        For each optimization level, modules are moved to their neighbors if it
        improves L-Modularity. The process keeps runing until no move improves
        L-Modularity anymore. New modules are built by agregation, and a new
        optimization level step is applied on them.

    Returns:
        float: Δ L-Modularity between initial state and final state.
    """

    delta_longitudinal_modularity = 0
    improvement = True
    level = 0
    max_iterations_warning = 1000  # Threshold for warning about potential infinite loop

    while improvement:
        # Track potential infinite loops when verbose >= 3
        if verbose >= 3:
            if level % 100 == 0 and level > 0:
                print(f"[LOOP TRACKING] RTMM optimization loop: level {level}")
            if level >= max_iterations_warning:
                print(
                    f"[WARNING] RTMM optimization loop exceeded {max_iterations_warning} levels - possible infinite loop!"
                )

        tmp_delta_longitudinal_modularity = time_module_mover.run(verbose)
        improvement = tmp_delta_longitudinal_modularity > stopping_criterion
        delta_longitudinal_modularity += tmp_delta_longitudinal_modularity

        if verbose and improvement:
            print(
                f"\tOptimized level {level} :: Δ L-Modularity = {round(tmp_delta_longitudinal_modularity / nb_edges, ndigits=ndigits_logs)}"
            )

        level += 1

    if verbose >= 3:
        print(f"[LOOP TRACKING] RTMM optimization loop completed after {level} levels")

    return delta_longitudinal_modularity
