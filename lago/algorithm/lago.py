from __future__ import annotations

import multiprocessing
import os
import random
import sys
import warnings
from typing import TYPE_CHECKING

import lago.core.utils as tls
from lago.algorithm._internal.runner import lago_run
from lago.core.enums import LexType
from lago.core.linkstream import LinkStream
from lago.core.time_modules import TimeModules
from lago.core.utils import log_debug, log_info

if TYPE_CHECKING:
    from collections.abc import Iterator

# Measured on the representation (docs/FASTER_LANGUAGE_ANALYSIS.md, 1c); used only
# to warn before parallel iterations would multiply it.
_BYTES_PER_TIME_EDGE = 500

Segments = dict[int, dict[int, tuple[tuple[int, int], ...]]]


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
    n_jobs: int = 1,
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
        n_jobs: Number of processes over which to spread the `nb_iter` runs.
            1 (the default) runs them one after the other in this process,
            exactly as before. More than 1 runs them in forked copies of this
            process and returns the very partition 1 would: a fork inherits the
            link stream as it is, down to the memory layout of every set that
            steers the search, which a rebuild in a fresh process would not
            reproduce. Three things to know. **Memory**: copy-on-write does not
            hold for long -- the search touches every object -- so peak memory
            is about `n_jobs` times that of a single run, roughly 500 bytes per
            time-edge each; the value is capped to `nb_iter` and to the CPU
            count, and reduced with a warning when the estimate exceeds the
            memory available. **Logs**: worker processes are silent; this
            process logs one line per run, in run order, as results come in.
            **Platform**: forking is unavailable on Windows, and unsafe from a
            process that already runs other threads (BLAS pools, GUIs, some
            notebook kernels); without fork the runs proceed sequentially, with
            a warning. Defaults to 1.

    Returns:
        TimeModules: Detected temporal modules.

    Raises:
        ValueError: If lex, refinement, nb_iter or n_jobs has an invalid value.
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
    if n_jobs < 1:
        msg = "n_jobs must be >= 1."
        raise ValueError(msg)
    if linkstream.nb_edges == 0:
        msg = "LinkStream has no edges. Add edges with add_links() before running lago_modules()."
        raise ValueError(msg)

    linkstream = _prepare_stream(linkstream)

    # Log start info
    log_info(
        f"LAGO: {linkstream.nb_nodes} nodes, {linkstream.nb_edges} edges, "
        f"lex={lex_str}, refinement={refinement}",
        verbose,
    )

    # Convert verbose to int for internal lago_run
    verbose_int = int(verbose) if isinstance(verbose, bool) else verbose

    settings = (
        lex_str,
        gamma,
        omega,
        refinement,
        fast_exploration,
        refinement_in,
        stopping_criterion,
        ndigits_logs,
        seed,
    )

    # Run LAGO algorithm (potentially multiple iterations). Kept as segments
    # rather than as live modules: the next iteration rebuilds every module from
    # scratch, so holding on to the objects would keep a partition that no
    # longer describes the state its leaves are in.
    best_modularity = -sys.maxsize
    best_segments: Segments = {}

    def keep_best(iteration: int, modularity: float, segments: Segments) -> None:
        nonlocal best_modularity, best_segments
        if modularity > best_modularity:
            best_modularity = modularity
            best_segments = segments
            log_info(
                f"Iteration {iteration + 1}/{nb_iter}: improved to {round(best_modularity, ndigits=ndigits_logs)}",
                verbose,
            )
        else:
            log_debug(f"Iteration {iteration + 1}/{nb_iter}: no improvement", verbose)

    jobs = _effective_jobs(n_jobs, nb_iter, linkstream, verbose)
    if jobs == 1:
        for iteration in range(nb_iter):
            log_info(f"Starting iteration {iteration + 1}/{nb_iter}", verbose)
            keep_best(iteration, *_iteration(linkstream, settings, iteration, verbose_int))
    else:
        log_info(f"Running {nb_iter} iterations over {jobs} processes", verbose)
        for iteration, (modularity, segments) in enumerate(
            _parallel_iterations(linkstream, settings, nb_iter, jobs)
        ):
            keep_best(iteration, modularity, segments)

    log_info(f"Found {len(best_segments)} modules", verbose)
    return TimeModules.from_segments(best_segments)


def _prepare_stream(linkstream: LinkStream) -> LinkStream:
    """The stream LAGO runs on: a split copy for continuous streams, else itself.

    The copy is made from the raw links rather than by deep-copying: leaves refer
    to each other in cycles, and the split rebuilds them anyway.
    """
    if not linkstream.continuous:
        return linkstream
    raw_links = linkstream.get_time_links(include_weights=True)
    prepared = LinkStream(
        continuous=True,
        directed=linkstream.directed,
        delayed=False,
        partite_mapping=linkstream.partite_mapping.copy() if linkstream.partite_mapping else None,
    )
    prepared.add_links(list(raw_links))
    prepared._split_continuous_linkstream()
    # Overwrite the artificial number of edges resulting from segmentation phase
    prepared.nb_edges = len(raw_links)
    return prepared


def _iteration(
    linkstream: LinkStream, settings: tuple, iteration: int, verbose: int
) -> tuple[float, Segments]:
    """One LAGO run, returning its L-modularity and its partition as segments."""
    (
        lex,
        gamma,
        omega,
        refinement,
        fast_exploration,
        refinement_in,
        stopping_criterion,
        ndigits_logs,
        seed,
    ) = settings
    modularity, modules = lago_run(
        linkstream,
        lex,
        gamma,
        omega,
        refinement,
        fast_exploration,
        refinement_in,
        verbose,
        stopping_criterion * linkstream.weight,
        ndigits_logs,
        _exploration_rng(seed, iteration),
    )
    segments = {
        label: tls.get_module_segments(module.leaves)
        for label, module in enumerate(sorted(modules, key=lambda m: m.index))
    }
    return modularity, segments


# What a forked worker runs on. Set by the parent right before forking and read
# by the child through the inherited memory; never sent through a pipe. The
# stream must be the parent's own object: the search's exploration order follows
# the memory layout of its sets, which a rebuilt copy would not reproduce.
_fork_state: tuple[LinkStream, tuple] | None = None


def _fork_worker(iteration: int) -> tuple[float, Segments]:
    """One silent iteration on the inherited stream."""
    linkstream, settings = _fork_state
    return _iteration(linkstream, settings, iteration, 0)


def _parallel_iterations(
    linkstream: LinkStream, settings: tuple, nb_iter: int, jobs: int
) -> Iterator[tuple[float, Segments]]:
    """Run the iterations in ``jobs`` forked processes, yielding in iteration order."""
    global _fork_state
    _fork_state = (linkstream, settings)
    try:
        context = multiprocessing.get_context("fork")
        with context.Pool(jobs) as pool:
            yield from pool.imap(_fork_worker, range(nb_iter))
    finally:
        _fork_state = None


def _available_memory() -> int | None:
    """Bytes of memory available to new allocations, if it can be told."""
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().available)


def _effective_jobs(n_jobs: int, nb_iter: int, linkstream: LinkStream, verbose: bool | int) -> int:
    """How many processes to actually use: capped by runs, CPUs, memory and platform."""
    jobs = max(1, min(n_jobs, nb_iter, os.cpu_count() or 1))
    if jobs == 1:
        return 1
    if "fork" not in multiprocessing.get_all_start_methods():
        warnings.warn(
            f"n_jobs={n_jobs}: parallel iterations need the 'fork' start method, which this "
            "platform does not provide; running the iterations sequentially.",
            RuntimeWarning,
            stacklevel=3,
        )
        return 1
    per_process = linkstream.nb_edges * _BYTES_PER_TIME_EDGE
    available = _available_memory()
    if available is not None and jobs * per_process > 0.8 * available:
        affordable = max(1, int(0.8 * available // max(per_process, 1)))
        warnings.warn(
            f"n_jobs={n_jobs}: {jobs} processes would need about "
            f"{jobs * per_process / 1e9:.1f} GB ({per_process / 1e9:.2f} GB per copy of the "
            f"stream) with {available / 1e9:.1f} GB available; running {affordable}.",
            ResourceWarning,
            stacklevel=3,
        )
        jobs = affordable
    log_debug(f"n_jobs: using {jobs} process(es) for {nb_iter} iterations", verbose)
    return jobs


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
