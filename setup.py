"""Build configuration: the pure-Python package plus, where it can be built, its compiled form.

``pip install dcd-lago`` from a binary wheel needs none of this. Building from
source -- an sdist or a checkout -- compiles the modules listed below into
extension modules when a C compiler is available and falls back to pure Python
otherwise: every extension is *optional*, a failed compile is a warning, and the
``.py`` sources always ship alongside the ``.so``. Python prefers the extension
when both are present; ``LAGO_CORE=python`` forces the sources. The two paths
are verified to give identical results (``benchmarks/compare_backends.py``).

Cython is a build-time requirement only (``pyproject.toml``'s build-system),
never a runtime one. Without it, the generated C distributed in the sdist is
compiled instead.

    pip install .                          # wheel for this machine
    python setup.py build_ext --inplace    # in a checkout: a .so beside each .py
"""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import Extension, setup

HERE = Path(__file__).resolve().parent

# Compiled by Cython in pure-Python mode: the same .py files, unchanged. The
# first two have a .pxd beside them (C attributes, C __hash__); the others are
# compiled as they are, with Python object semantics. Chosen by measurement:
# docs/PERFORMANCE_ROUND2.md, step 5.
COMPILED_MODULES = [
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

# The metric's counting kernel: Cython source with no Python twin -- lago.accel
# carries the reference implementation it must match bit for bit.
KERNEL = "lago/_accel_kernel.pyx"

# annotation_typing off: the ": int" hints in signatures stay hints. No
# fast-math anywhere: the summation order is part of the result.
DIRECTIVES = {"language_level": "3", "annotation_typing": False}
COMPILE_ARGS = [] if sys.platform == "win32" else ["-O3"]


def _module_name(source: str) -> str:
    return source.rsplit(".", 1)[0].replace("/", ".")


def extensions() -> list[Extension]:
    sources = [*COMPILED_MODULES, KERNEL]
    try:
        from Cython.Build import cythonize
    except ImportError:
        cythonize = None

    if cythonize is not None:
        built = cythonize(
            [Extension(_module_name(s), [s], extra_compile_args=COMPILE_ARGS) for s in sources],
            compiler_directives=DIRECTIVES,
            quiet=True,
        )
    else:
        # No Cython: the generated C shipped in the sdist, for whatever is there.
        built = []
        for source in sources:
            c_file = source.rsplit(".", 1)[0] + ".c"
            if (HERE / c_file).exists():
                built.append(Extension(_module_name(source), [c_file], extra_compile_args=COMPILE_ARGS))

    for extension in built:
        # A compile failure must not fail the install: the .py sources are the
        # fallback and the reference.
        extension.optional = True
    return built


setup(ext_modules=extensions())
