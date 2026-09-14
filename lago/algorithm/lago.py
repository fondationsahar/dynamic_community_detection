from __future__ import annotations

import random
import sys

import lago.core.utils as tls
from lago.algorithm._internal.runner import lago_run
from lago.core.enums import LexType
from lago.core.linkstream import LinkStream
from lago.core.time_modules import TimeModules
from lago.core.utils import log_debug, log_info


def lago_modules(
    linkstream: LinkStream,
    lex: LexType | str = LexType.MM,
    nb_iter: int = 1,
    gamma: float = 1,
    omega: float = 2,
    refinement: str | None = "STEM",
    fast_exploration: bool = True,
    refinement_in: bool = True,
    verbose: bool | int = 0,
    stopping_criterion: float = 1e-8,
    ndigits_logs: int = 8,
    seed: int | None = None,
) -> TimeModules:
    """Detect temporal modules in link streams using LAGO algorithm.

    LAGO (Longitudinal Agglomerative Greedy Optimization) detects temporal modules
    by optimizing L-Modularity.

    Args:
        linkstream: Link Stream on which to find temporal modules.
        lex: Longitudinal Expectation type. Can be LexType.JM, LexType.MM, or
            strings "JM"/"MM" for backward compatibility.
            - JM (Joint-Membership): expects modules to have a very
              consistent duration of existence.
            - MM (Mean-Membership): allows greater freedom in the temporal
              evolution of modules. Most general choice.
            Defaults to LexType.MM.
        nb_iter: Number of LAGO runs. Best result is returned. Each run explores
            the candidate moves in a different order, which is what the greedy
            optimization is sensitive to, so higher values reduce sensitivity to
            the starting point at proportionally higher cost. The orders are
            derived from `seed`, so a given (seed, nb_iter) pair is fully
            reproducible. Defaults to 1.
        gamma: Topological resolution parameter. Must be >= 0. Higher values
            produce smaller, tighter communities; gamma=1 (default) corresponds
            to standard modularity resolution. Defaults to 1.
        omega: Temporal smoothness parameter. Must be >= 0. Higher values
            penalize community membership changes over time, producing more
            stable communities. omega=0 ignores temporal continuity entirely.
            Defaults to 2.
        refinement: Refinement strategy. Must be None, "STEM" or "STNM".
            Refinement significantly improves module quality but is more
            time-consuming. None = no refinement. "STNM" = Single Time Node
            Movements. "STEM" = Single Time Edge Movements. Defaults to "STEM".
        fast_exploration: Whether to apply Fast Exploration strategy. If True,
            significantly reduces execution time but may result in poorer results.
            Defaults to True.
        refinement_in: Whether to apply refinement within the core part or after.
            Applying within implies more exploration, which may result in better
            results or more chances to get stuck in local optimum. More time-consuming.
            Defaults to True.
        verbose: Verbosity level. 0=silent, 1=progress info, 2=detailed debug,
            3=infinite loop tracking (logs iteration counts and warnings).
            Also accepts bool for backward compatibility (True=1, False=0).
            Defaults to 0.
        stopping_criterion: Convergence threshold. Defaults to 1e-8.
        ndigits_logs: Number of decimal places for logging. Defaults to 8.
        seed: Selects the order in which candidate moves are explored, which is
            the only thing that varies between runs on the same input.

            Every result is reproducible: the same arguments always give the
            same modules, in this process or any other, with or without a seed.
            What the seed chooses is *which* greedy trajectory to follow.
            Different seeds reach different local optima -- on the bundled
            120-node fixture the spread is about 3% of L-modularity -- which is
            what `nb_iter` exploits.

            None (the default) uses the canonical order. Defaults to None.

    Returns:
        TimeModules: Detected temporal modules.

    Raises:
        ValueError: If lex or refinement has invalid value.
        TypeError: If lex is not a LexType enum or valid string.

    Example:
        ```python
        from lago import LinkStream, lago_modules, LexType

        ls = LinkStream()
        ls.add_links([(0, 1, 0), (1, 2, 0), (0, 1, 1)])

        # Using LexType enum (recommended)
        tm = lago_modules(ls, lex=LexType.MM)

        # Using string (backward compatible)
        tm = lago_modules(ls, lex="MM")

        # Access results
        for module in tm.iter_modules():
            print(f"Module {module.label}: {module.nodes}")
        ```
    """
    # Validate and normalize lex
    lex_str = _validate_lex_type(lex)

    # Validate other parameters
    if gamma < 0:
        msg = "gamma must be >= 0."
        raise ValueError(msg)
    if omega < 0:
        msg = "omega must be >= 0."
        raise ValueError(msg)
    if nb_iter < 1:
        msg = "nb_iter must be >= 1."
        raise ValueError(msg)
    if refinement not in (None, "STNM", "STEM"):
        msg = "refinement must be None, 'STNM', or 'STEM'."
        raise ValueError(msg)
    if linkstream.nb_edges == 0:
        msg = "LinkStream has no edges. Add edges with add_links() before running lago_modules()."
        raise ValueError(msg)

    # Pre-process continuous linkstreams
    # Create a fresh copy to avoid modifying the user's original linkstream
    # Note: copy.deepcopy doesn't work due to circular references in Leaf objects
    if linkstream.continuous:
        # Get raw links before splitting
        raw_links = linkstream.get_time_links(include_weights=True)

        # Create new linkstream with same configuration
        linkstream_copy = LinkStream(
            continuous=True,
            directed=linkstream.directed,
            delayed=False,
            partite_mapping=linkstream.partite_mapping.copy()
            if linkstream.partite_mapping
            else None,
        )
        linkstream_copy.add_links(list(raw_links))
        linkstream_copy._split_continuous_linkstream()
        linkstream = linkstream_copy
        # Overwrite the artificial number of edges
        # resulting from segmentation phase
        linkstream.nb_edges = len(raw_links)

    # Log start info
    log_info(
        f"LAGO: {linkstream.nb_nodes} nodes, {linkstream.nb_edges} edges, "
        f"lex={lex_str}, refinement={refinement}",
        verbose,
    )

    # Run LAGO algorithm (potentially multiple iterations)
    best_modularity = -sys.maxsize
    best_segments: dict[int, dict[int, tuple[tuple[int, int], ...]]] = {}

    # Convert verbose to int for internal lago_run
    verbose_int = int(verbose) if isinstance(verbose, bool) else verbose

    for iteration in range(nb_iter):
        log_info(f"Starting iteration {iteration + 1}/{nb_iter}", verbose)

        modularity, modules = lago_run(
            linkstream,
            lex_str,
            gamma,
            omega,
            refinement,
            fast_exploration,
            refinement_in,
            verbose_int,
            stopping_criterion * linkstream.weight,
            ndigits_logs,
            _exploration_rng(seed, iteration),
        )

        # Keep best result. Recorded as segments rather than as live modules:
        # the next iteration rebuilds every module from scratch, so holding on
        # to the objects would keep a partition that no longer describes the
        # state its leaves are in.
        if modularity > best_modularity:
            best_modularity = modularity
            best_segments = {
                label: tls.get_module_segments(module.leaves)
                for label, module in enumerate(sorted(modules, key=lambda m: m.index))
            }
            log_info(
                f"Iteration {iteration + 1}/{nb_iter}: improved to {round(best_modularity, ndigits=ndigits_logs)}",
                verbose,
            )
        else:
            log_debug(f"Iteration {iteration + 1}/{nb_iter}: no improvement", verbose)

    log_info(f"Found {len(best_segments)} modules", verbose)
    return TimeModules.from_segments(best_segments)


def _exploration_rng(seed: int | None, iteration: int) -> random.Random | None:
    """Generator deciding the exploration order of one LAGO iteration.

    The greedy search is sensitive to the order in which candidates are
    considered, and that order is the only thing ``nb_iter`` can vary.

    Returns ``None`` -- meaning the canonical order -- for the very first
    iteration of an unseeded call, so that the default single run reproduces
    exactly what it always has. Every other combination gets a generator whose
    stream depends only on ``(seed, iteration)``, so restarts explore genuinely
    different orders and any given run stays reproducible.

    Args:
        seed: The caller's seed, or None.
        iteration: Zero-based iteration index within this call.

    Returns:
        A seeded ``random.Random``, or None for the canonical order.
    """
    if seed is None:
        if iteration == 0:
            return None
        return random.Random(iteration)
    # Plain arithmetic rather than hash(): reproducible across processes and
    # Python versions without depending on any hashing detail.
    return random.Random(seed * 1_000_003 + iteration)


def _validate_lex_type(lex_type: LexType | str) -> str:
    """Validate and convert lex_type to string format.

    Args:
        lex_type: LexType enum or string ('JM', 'MM').

    Returns:
        Normalized string ('JM' or 'MM').

    Raises:
        ValueError: If lex_type value is invalid.
        TypeError: If lex_type is not a LexType enum or string.
    """
    if isinstance(lex_type, LexType):
        lex_str = lex_type.name
    elif isinstance(lex_type, str):
        lex_str = lex_type.upper()
        if lex_str not in ("JM", "MM"):
            msg = f'Invalid lex_type string "{lex_type}". Must be "JM" or "MM".'
            raise ValueError(msg)
    else:
        msg = (
            f"lex_type must be a LexType enum or string ('JM', 'MM'), got {type(lex_type).__name__}"
        )
        raise TypeError(msg)

    # CM (Coexistence) is only supported in longitudinal_modularity, not in LAGO
    if lex_str == "CM":
        msg = "LexType.CM is not supported by lago_modules. Use LexType.JM or LexType.MM."
        raise ValueError(msg)

    return lex_str
