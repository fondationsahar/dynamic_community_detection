from lago.lago_src.tmm import TimeModuleMover


def rtmm_optimization(
    time_module_mover: TimeModuleMover,
    verbose: bool,
    nb_edges: float,
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
    while improvement:
        tmp_delta_longitudinal_modularity = time_module_mover.run()
        improvement = tmp_delta_longitudinal_modularity > 0
        delta_longitudinal_modularity += tmp_delta_longitudinal_modularity

        if verbose and improvement:
            print(
                f"\tOptimized level {level} :: Δ L-Modularity = {round(tmp_delta_longitudinal_modularity / nb_edges, ndigits=5)}"
            )

        level += 1

    return delta_longitudinal_modularity
