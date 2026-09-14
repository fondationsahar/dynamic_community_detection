"""Shared helpers for the LAGO benchmark / equivalence harness.

Importing this module guarantees that ``lago`` resolves to the checkout this
file lives in. That matters: an editable install of ``dcd-lago`` may point at a
different checkout, and it is picked up whenever the current working directory
does not happen to contain ``lago/``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lago  # noqa: E402

_resolved = Path(lago.__file__).resolve().parent
if _resolved != REPO_ROOT / "lago":
    msg = (
        f"benchmarks imported the wrong package: lago resolves to {_resolved}, "
        f"expected {REPO_ROOT / 'lago'}. An editable install is shadowing this checkout."
    )
    raise RuntimeError(msg)


def fingerprint_modules(time_modules) -> str:
    """Canonical, label-invariant fingerprint of a TimeModules partition.

    Two partitions have the same fingerprint iff they contain the same sets of
    ``(node, time)`` members, regardless of how the modules are labelled.
    """
    parts = sorted(tuple(sorted(members)) for members in time_modules._raw_modules.values())
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


def fingerprint_partition(modules) -> str:
    """Same, for a raw iterable of member-set iterables."""
    parts = sorted(tuple(sorted(members)) for members in modules)
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]
