"""Bit-exactness check across ``lago.accel`` backends.

The backend is chosen once, at import, from ``LAGO_ACCEL``, so a single process
cannot hold two of them. The driver therefore runs one subprocess per backend,
each emitting every metric value it computes, and compares the streams of
values position by position. Values are compared by their ``float.hex()``, not
with a tolerance: the kernels are meant to produce the same double, and "close"
would hide exactly the reassociation bugs this is looking for.

Coverage is the point. Two families of cases:

* **Random**, from ``compare_metric.random_case`` -- mixes directed, weighted,
  k-partite and multi-community streams, which is where the argument shapes
  differ (the directed ``* 2``, self-loops, unlabelled time-nodes).
* **Named**, the generators in ``streams.py`` -- includes what random cases do
  not produce: continuous links with duration, delayed links, and streams large
  enough for the flat topology and its cache to matter.

Both families also score a LAGO partition, so the labels under test are the
ones the algorithm actually produces rather than only random ones.

    python3.11 benchmarks/compare_backends.py -n 3000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from _common import REPO_ROOT, fingerprint_modules

from lago import LexType, LinkStream, lago_modules, longitudinal_modularity

# gamma=0 and omega=0 each disable a term, so they check that the accelerated
# path is still consulted (and skipped) in the same places as the Python one.
PARAMS = ((1.0, 2.0, 5), (0.0, 2.0, 5), (0.5, 0.0, 5), (2.0, 1.0, 12), (1.0, 1.0, 10))
LEXES = (LexType.MM, LexType.JM, LexType.CM)


def _emit(values: list[str], stream, communities) -> None:
    """Append every (lex, parameter) metric value for one case."""
    for lex in LEXES:
        for gamma, omega, ndigits in PARAMS:
            result = longitudinal_modularity(
                stream, communities, lex=lex, gamma=gamma, omega=omega, ndigits=ndigits
            )
            values.append(result.value.hex())
            values.append(result.time_penalty.hex())


def run_worker(n_cases: int, seed: int) -> dict:
    """Compute every value under whichever backend this process loaded."""
    import random

    from lago import accel

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compare_metric import random_case
    from streams import GENERATORS

    values: list[str] = []
    fingerprints: list[str] = []

    rng = random.Random(seed)
    built = 0
    while built < n_cases:
        case = random_case(rng)
        if case is None:
            continue
        built += 1
        stream, communities = case
        _emit(values, stream, communities)

    for name in sorted(GENERATORS):
        stream: LinkStream = GENERATORS[name]()
        modules = lago_modules(stream, seed=7, nb_iter=2, verbose=False)
        fingerprints.append(f"{name}:{fingerprint_modules(modules)}")
        _emit(values, stream, modules)
        # A second, deliberately bad partition: one module per time-node makes
        # every adjacency a cut and every time step a switch, which is the
        # opposite extreme from a LAGO partition.
        singletons = {i: {key} for i, key in enumerate(stream.leaves_dict)}
        _emit(values, stream, singletons)

    return {
        "backend": accel.backend_name,
        "core": accel.core_name(),
        "values": values,
        "fingerprints": fingerprints,
    }


def variants() -> list[tuple[str, dict[str, str]]]:
    """The (name, environment) combinations that can be run here.

    Two independent dimensions: the metric kernel (``LAGO_ACCEL``) and the core
    data model (``LAGO_CORE``, compiled when ``setup_core.py`` has been run).
    ``python`` -- both pure -- is the reference everything else is held to.
    """
    from lago import accel

    kernels = accel.available_backends()
    cores = ["python"]
    if accel.core_name() == "compiled":
        cores.append("compiled")
    found = []
    for core in cores:
        for kernel in kernels:
            name = kernel if core == "python" else f"{kernel}+core"
            found.append((name, {"LAGO_ACCEL": kernel, "LAGO_CORE": core}))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--cases", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        json.dump(run_worker(args.cases, args.seed), sys.stdout)
        return 0

    under_test = variants()
    print(f"variants under test: {', '.join(name for name, _ in under_test)}")
    if len(under_test) == 1:
        print("  only the pure-Python variant is available; nothing to compare against.")
        print("  build one with: python -m pip install ./accel_cython")
        print("  or:             python accel_cython/setup_core.py build_ext --inplace")
        return 1

    results = {}
    for name, environment in under_test:
        env = {**os.environ, **environment, "PYTHONHASHSEED": "0"}
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", "-n", str(args.cases), "--seed", str(args.seed)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT / "benchmarks"),
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            print(f"  {name}: worker failed\n{proc.stderr}")
            return 1
        payload = json.loads(proc.stdout)
        loaded = (payload["backend"], payload["core"])
        wanted = (environment["LAGO_ACCEL"], environment["LAGO_CORE"])
        if loaded != wanted:
            print(f"  {name}: worker actually loaded kernel={loaded[0]!r} core={loaded[1]!r}")
            return 1
        results[name] = payload
        print(
            f"  {name}: {len(payload['values'])} values "
            f"in {time.perf_counter() - start:.1f}s"
        )

    baseline = "python"
    reference = results[baseline]
    failures = 0
    for backend, payload in results.items():
        if backend == baseline:
            continue
        mismatches = [
            (i, a, b)
            for i, (a, b) in enumerate(zip(reference["values"], payload["values"], strict=True))
            if a != b
        ]
        partitions = [
            (a, b)
            for a, b in zip(reference["fingerprints"], payload["fingerprints"], strict=True)
            if a != b
        ]
        total = len(reference["values"])
        print(f"\n{baseline} vs {backend}")
        print(f"  metric values:      {total - len(mismatches)}/{total} bit-identical")
        print(
            f"  lago partitions:    "
            f"{len(reference['fingerprints']) - len(partitions)}"
            f"/{len(reference['fingerprints'])} identical"
        )
        for index, first, second in mismatches[:8]:
            print(f"    [{index}] {baseline}={float.fromhex(first)!r} {backend}={float.fromhex(second)!r}")
        for first, second in partitions[:8]:
            print(f"    {first} != {second}")
        failures += len(mismatches) + len(partitions)

    print("\nOK" if not failures else f"\n{failures} DIFFERENCES")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
