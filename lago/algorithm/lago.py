from __future__ import annotations

import copy
import sys
from typing import TYPE_CHECKING

import lago.core.utils as tls
from lago.algorithm._internal._lago_module import _LagoModule
from lago.algorithm._internal.runner import lago_run
from lago.core.enums import LexType
from lago.core.time_modules import TimeModules

if TYPE_CHECKING:
    from lago.core.linkstream import LinkStream


def lago_modules(
    linkstream: LinkStream,
    lex: LexType | str = LexType.MM,
    nb_iter: int = 1,
    alpha: float = 1,
    omega: float = 2,
    refinement: str | None = "STEM",
    fast_exploration: bool = True,
    refinement_in: bool = True,
    verbose: bool = False,
    stopping_criterion: float = 1e-8,
    ndigits_logs: int = 8,
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
              evolution of modules.
            Defaults to LexType.MM.
        nb_iter: Number of LAGO runs. Best results are returned.
        alpha: Resolution parameter. Must be >= 0. Higher values lead to smaller
            modules topologically. Defaults to 1.
        omega: Time resolution parameter. Must be >= 0. Higher values lead to more
            smoothness in module changes. Defaults to 2.
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
        verbose: Whether to print intermediate reports. Defaults to False.
        stopping_criterion: Convergence threshold. Defaults to 1e-8.
        ndigits_logs: Number of decimal places for logging. Defaults to 8.

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
    if omega < 0:
        msg = "omega must be >= 0."
        raise ValueError(msg)
    if refinement not in (None, "STNM", "STEM"):
        msg = "refinement must be None, 'STNM', or 'STEM'."
        raise ValueError(msg)

    # Pre-process continuous linkstreams
    if linkstream.continuous:
        linkstream._split_continuous_linkstream()

    # Run LAGO algorithm (potentially multiple iterations)
    best_modularity = -sys.maxsize
    best_modules: set[_LagoModule] = set()

    for iteration in range(nb_iter):
        if verbose:
            print(f"\nStart iteration {iteration + 1}")

        modularity, modules = lago_run(
            linkstream,
            lex_str,
            alpha,
            omega,
            refinement,
            fast_exploration,
            refinement_in,
            verbose,
            stopping_criterion,
            ndigits_logs,
        )

        # Keep best result
        if modularity > best_modularity:
            best_modularity = modularity
            best_modules = copy.copy(modules)
            if verbose:
                print(
                    f"\n\t> [{iteration + 1}/{nb_iter}] Improvement: "
                    f"{round(best_modularity, ndigits=ndigits_logs)}"
                )
        elif verbose:
            print(f"\n\t> [{iteration + 1}/{nb_iter}] No improvement.")

    return _convert_to_time_modules(best_modules)


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


def _convert_to_time_modules(modules: set[_LagoModule]) -> TimeModules:
    """Convert raw _LagoModule objects to TimeModules.

    Args:
        modules: Set of _LagoModule objects from LAGO algorithm.

    Returns:
        TimeModules: A TimeModules object containing the detected modules.
    """
    raw_modules = {
        label: tls.get_expanded_module(module.leaves) for label, module in enumerate(modules)
    }
    return TimeModules(raw_modules)
