"""Build the Cython counting kernel for ``lago.accel``.

    python -m pip install ./accel_cython

Optimisation flags are chosen for reproducibility, not for the last few
percent: ``-O3`` is safe because it preserves IEEE semantics, while anything in
the ``-ffast-math`` family would let the compiler reassociate the weight sums
and break bit-identity with the Python loops.
"""

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:  # pragma: no cover - build-time only
    msg = "Cython is required to build this backend: python -m pip install Cython"
    raise SystemExit(msg) from exc

setup(
    ext_modules=cythonize(
        [
            Extension(
                "lago_accel_cython",
                ["src/lago_accel_cython.pyx"],
                extra_compile_args=["-O3"],
            )
        ],
        compiler_directives={"language_level": "3"},
    ),
)
