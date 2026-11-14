"""What an installed build of dcd-lago contains.

Every extension in ``setup.py`` is optional, so a build that failed to compile
some of them still succeeds -- the right behaviour for a user's install, the
wrong one for a release wheel, which would ship half compiled under a platform
tag. The release workflow builds with ``LAGO_REQUIRE_COMPILED=1`` and tests each
wheel with ``LAGO_EXPECT_COMPILED=1`` (``pyproject.toml``, ``[tool.cibuildwheel]``);
under the latter, every module ``setup.py`` compiles must load from an extension.
Without it, those tests skip: a pure-Python install is a correct install.
"""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path

import pytest

from lago import accel

# setup.py's COMPILED_MODULES and KERNEL, as module names. test_matches_setup
# holds the two lists together wherever setup.py sits beside the tests.
COMPILED = (
    "lago.algorithm._internal._leaf",
    "lago.algorithm._internal._time_edge",
    "lago.algorithm._internal._lago_module",
    "lago.algorithm._internal.delta_lm",
    "lago.algorithm._internal.find_best_move",
    "lago.algorithm._internal.lago_tools",
    "lago.algorithm._internal.leaf_set",
    "lago.algorithm._internal.stem",
    "lago.algorithm._internal.tmm",
    "lago.core.utils",
    "lago.accel",
    "lago._accel_kernel",
)

SETUP = Path(__file__).resolve().parents[2] / "setup.py"
EXPECT_COMPILED = os.environ.get("LAGO_EXPECT_COMPILED") == "1"


def _setup_modules() -> set[str]:
    """COMPILED_MODULES and KERNEL from setup.py, read without running it."""
    values = {}
    for node in ast.parse(SETUP.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"COMPILED_MODULES", "KERNEL"}
        ):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    sources = [*values["COMPILED_MODULES"], values["KERNEL"]]
    return {source.rsplit(".", 1)[0].replace("/", ".") for source in sources}


@pytest.mark.skipif(not SETUP.exists(), reason="setup.py is not beside the tests")
def test_matches_setup() -> None:
    assert _setup_modules() == set(COMPILED)


@pytest.mark.skipif(not EXPECT_COMPILED, reason="LAGO_EXPECT_COMPILED=1 not set")
@pytest.mark.parametrize("name", COMPILED)
def test_module_is_compiled(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__file__.endswith((".so", ".pyd")), module.__file__


@pytest.mark.skipif(not EXPECT_COMPILED, reason="LAGO_EXPECT_COMPILED=1 not set")
def test_reported_backends_are_compiled() -> None:
    assert accel.core_name() == "compiled"
    assert accel.backend_name == "cython"
