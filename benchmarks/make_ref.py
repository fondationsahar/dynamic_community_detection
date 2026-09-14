"""Snapshot the current ``lago/`` package as an importable reference copy.

The copy's internal imports are rewritten from ``lago.`` to ``<name>.`` so both
packages can be imported in the same process, which is what lets an equivalence
check compare old and new behaviour without relying on run-to-run stability.

    python3.11 benchmarks/make_ref.py --out-dir /tmp/refs --name lago_step2
    python3.11 benchmarks/compare_ref.py --ref-dir /tmp/refs --ref-name lago_step2
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from _common import REPO_ROOT


def make_reference(out_dir: Path, name: str) -> Path:
    target = out_dir / name
    if target.exists():
        shutil.rmtree(target)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT / "lago", target, ignore=shutil.ignore_patterns("__pycache__"))

    rewritten = 0
    for path in target.rglob("*.py"):
        text = path.read_text()
        new = re.sub(r"\b(from|import)\s+lago\.", lambda m: f"{m.group(1)} {name}.", text)
        new = re.sub(r"^import lago$", f"import {name} as lago", new, flags=re.M)
        if new != text:
            path.write_text(new)
            rewritten += 1

    print(f"snapshot -> {target} ({rewritten} files rewritten)")
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="lago_ref")
    args = ap.parse_args()
    make_reference(Path(args.out_dir).resolve(), args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
