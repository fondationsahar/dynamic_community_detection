"""Compile lago's core data model in place, from its own source files.

    python3.11 accel_cython/setup_core.py build_ext --inplace

Cython "pure Python mode": each module below keeps its ``.py`` as the single
source and gains a ``.pxd`` beside it declaring the C layout of its classes.
The build drops a ``.so`` next to the ``.py``; Python's import system prefers
the extension, so nothing else changes. Delete the ``.so`` (or set
``LAGO_CORE=python``) and the same ``.py`` runs as before -- that is also the
reference the compiled build is checked against.

Directives: ``annotation_typing=False`` so the ``: int`` hints in signatures
stay hints (a compiled ``int`` annotation would reject other types), and no
fast-math, for the same reason as the metric kernel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:  # pragma: no cover - build-time only
    msg = "Cython is required to build the compiled core: python -m pip install Cython"
    raise SystemExit(msg) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)

# Compiled in place. The two data-model modules have a .pxd beside them (C
# layout, C __hash__); the rest are compiled as they are, with Python object
# semantics -- what they gain is interpreter overhead: frames, loops, locals.
MODULES = [
    "lago/algorithm/_internal/_leaf.py",
    "lago/algorithm/_internal/_time_edge.py",
    "lago/algorithm/_internal/_lago_module.py",
    "lago/algorithm/_internal/delta_lm.py",
    "lago/algorithm/_internal/find_best_move.py",
    "lago/algorithm/_internal/lago_tools.py",
    "lago/algorithm/_internal/leaf_set.py",
    "lago/algorithm/_internal/stem.py",
    "lago/algorithm/_internal/tmm.py",
    "lago/core/utils.py",
    "lago/accel.py",
]


def extensions() -> list[Extension]:
    return [
        Extension(
            source[:-3].replace("/", "."),
            [source],
            extra_compile_args=["-O3"],
        )
        for source in MODULES
    ]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += ["build_ext", "--inplace"]
    setup(
        name="lago-core-compiled",
        ext_modules=cythonize(
            extensions(),
            compiler_directives={"language_level": "3", "annotation_typing": False},
            # Generated C stays out of the package tree.
            build_dir=str(REPO_ROOT / "accel_cython" / "build" / "core"),
        ),
    )
