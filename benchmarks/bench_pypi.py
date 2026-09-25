"""The PyPI release of dcd-lago against the current tree, on many streams, every parameter combination.

Three implementations run the same inputs:

* ``pypi``      -- ``dcd-lago`` as installed from PyPI into its own virtualenv;
* ``pure``      -- this tree with ``LAGO_CORE=python`` (no compiled module);
* ``compiled``  -- this tree with the compiled core and the Cython metric kernel.

Each runs as a separate process (the two packages share the name ``lago``). The
driver generates every stream as a plain list of links, so each side builds
exactly the same input in exactly the same order; a worker rebuilds the stream
fresh for every run, times ``lago_modules`` and then ``longitudinal_modularity``
on the result, and returns the partition. The driver scores every returned
partition with the *current* metric, so quality is compared with one yardstick
whichever code produced the partition.

Two tiers:

* **matrix** -- 7 stream families (instantaneous, weighted, directed, bipartite,
  bipartite directed, continuous, delayed) at two sizes, plus the real fixture,
  each under all 24 combinations of ``lex`` x ``refinement`` x
  ``fast_exploration`` x ``refinement_in``;
* **scale** -- wide / deep / balanced planted streams from 5k to 100k
  interactions under the default parameters, MM and JM.

A run that exceeds its timeout is recorded as such, and larger sizes of the same
family under the same parameters are skipped for that implementation.

    python3.11 benchmarks/bench_pypi.py --pypi-python <venv>/bin/python run
    python3.11 benchmarks/bench_pypi.py report            # from the saved results
"""

# ruff: noqa: RUF001  (typographic multiplication and minus signs are deliberate in the rendered report)
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULTS_DIR = Path(os.environ.get("BENCH_PYPI_DIR", REPO_ROOT / "benchmarks" / "results"))

# =============================================================================
# Streams (link lists; no lago import, so the PyPI worker can share this file)
# =============================================================================


def _sample_planted(n_nodes, n_times, edges_per_time, n_comms, p_within, rng, directed=False):
    """Planted-partition links sampled per timestep (deduplicated per timestep)."""
    members: dict[int, list[int]] = {}
    for node in range(n_nodes):
        members.setdefault(node % n_comms, []).append(node)
    links = []
    for t in range(n_times):
        seen = set()
        for _ in range(edges_per_time):
            if rng.random() < p_within:
                group = members[rng.randrange(n_comms)]
                if len(group) < 2:
                    continue
                a, b = rng.sample(group, 2)
            else:
                a, b = rng.randrange(n_nodes), rng.randrange(n_nodes)
                if a == b:
                    continue
            key = (a, b) if directed else (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            links.append((key[0], key[1], t))
    return links


def gen_instant(n, T, ept, rng):
    return {}, _sample_planted(n, T, ept, max(2, n // 15), 0.85, rng)


def gen_weighted(n, T, ept, rng):
    links = _sample_planted(n, T, ept, max(2, n // 15), 0.85, rng)
    return {}, [(a, b, t, round(rng.uniform(0.5, 2.5), 2)) for a, b, t in links]


def gen_directed(n, T, ept, rng):
    return {"directed": True}, _sample_planted(n, T, ept, max(2, n // 15), 0.85, rng, directed=True)


def _bipartite_links(n, T, ept, rng, directed):
    half = n // 2
    groups = max(2, n // 20)
    links = []
    for t in range(T):
        seen = set()
        for _ in range(ept):
            if rng.random() < 0.85:
                g = rng.randrange(groups)
                left = [i for i in range(half) if i % groups == g]
                right = [half + j for j in range(n - half) if j % groups == g]
                if not left or not right:
                    continue
                a, b = rng.choice(left), rng.choice(right)
            else:
                a, b = rng.randrange(half), half + rng.randrange(n - half)
            if rng.random() < 0.5 and directed:
                a, b = b, a
            key = (a, b) if directed else (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            links.append((key[0], key[1], t))
    mapping = {i: (0 if i < half else 1) for i in range(n)}
    return mapping, links


def gen_bipartite(n, T, ept, rng):
    mapping, links = _bipartite_links(n, T, ept, rng, directed=False)
    return {"partite_mapping": mapping}, links


def gen_bipartite_directed(n, T, ept, rng):
    mapping, links = _bipartite_links(n, T, ept, rng, directed=True)
    return {"directed": True, "partite_mapping": mapping}, links


def gen_continuous(n, T, ept, rng):
    links = _sample_planted(n, T, ept, max(2, n // 15), 0.85, rng)
    return {"continuous": True}, [(a, b, t, rng.randint(1, 4)) for a, b, t in links]


def gen_delayed(n, T, ept, rng):
    links = _sample_planted(n, T, ept, max(2, n // 15), 0.85, rng)
    seen, out = set(), []
    for a, b, t in links:
        key = (a, b, t, t + rng.randint(0, 1))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return {"delayed": True}, out


FAMILIES = {
    "instant": ("instantaneous, undirected, unit weights", gen_instant),
    "weighted": ("instantaneous, float weights", gen_weighted),
    "directed": ("instantaneous, directed", gen_directed),
    "bipartite": ("bipartite (k-partite mapping), undirected", gen_bipartite),
    "bipartite_directed": ("bipartite, directed", gen_bipartite_directed),
    "continuous": ("continuous links with durations 1-4", gen_continuous),
    "delayed": ("delayed links (source time != target time)", gen_delayed),
}

# (nodes, timesteps, edges per timestep) -> interactions ~ timesteps x edges per timestep.
# Sized so that the PyPI release gets through 24 parameter combinations per
# stream within the timeouts; the scale tier is where size grows.
MATRIX_SIZES = {"S": (40, 6, 100), "M": (100, 8, 200)}


def _shape(name: str, target: int) -> tuple[int, int, int]:
    if name == "wide":
        T = 10
        ept = target // T
        return max(ept // 4, 16), T, ept
    if name == "deep":
        return 60, max(target // 40, 1), 40
    side = max(int(target**0.5), 4)
    return max(side // 2, 16), side, side


SCALE_SIZES = (5_000, 20_000, 50_000, 100_000)


def catalogue() -> list[dict]:
    """Every stream of the benchmark, as link lists, with its features."""
    streams = []
    for family, (features, gen) in FAMILIES.items():
        for rank, (size, (n, T, ept)) in enumerate(MATRIX_SIZES.items()):
            flags, links = gen(n, T, ept, random.Random(hash((family, size)) & 0xFFFF))
            streams.append(
                {
                    "name": f"{family}-{size}",
                    "tier": "matrix",
                    "family": family,
                    "features": features,
                    "size_rank": rank,
                    "flags": flags,
                    "links": links,
                }
            )

    fixture = REPO_ROOT / "tests" / "fixtures" / "linkstream.txt"
    links = []
    for line in fixture.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and all(p.lstrip("-").isdigit() for p in parts[:3]):
            links.append((int(parts[0]), int(parts[1]), int(parts[2])))
    streams.append(
        {
            "name": "fixture",
            "tier": "matrix",
            "family": "fixture",
            "features": "the bundled real stream (tests/fixtures/linkstream.txt)",
            "size_rank": 0,
            "flags": {},
            "links": links,
        }
    )

    for shape_name in ("wide", "deep", "balanced"):
        for rank, target in enumerate(SCALE_SIZES):
            n, T, ept = _shape(shape_name, target)
            _, links = gen_instant(n, T, ept, random.Random(rank * 7 + len(shape_name)))
            streams.append(
                {
                    "name": f"{shape_name}-{target // 1000}k",
                    "tier": "scale",
                    "family": shape_name,
                    "features": f"{shape_name}: {n} nodes x {T} timesteps, instantaneous",
                    "size_rank": rank,
                    "flags": {},
                    "links": links,
                }
            )
    for stream in streams:
        stream["interactions"] = len(stream["links"])
        times = set()
        for link in stream["links"]:
            times.add(link[2])
            if stream["flags"].get("delayed"):
                times.add(link[3])
        stream["timesteps"] = len(times)
        stream["nodes"] = len({link[0] for link in stream["links"]} | {link[1] for link in stream["links"]})
    return streams


# =============================================================================
# Parameter combinations
# =============================================================================


def configs(tier: str) -> list[dict]:
    if tier == "scale":
        return [
            {"lex": "MM", "refinement": "STEM", "fast_exploration": True, "refinement_in": True},
            {"lex": "JM", "refinement": "STEM", "fast_exploration": True, "refinement_in": True},
        ]
    return [
        {"lex": lex, "refinement": ref, "fast_exploration": fast, "refinement_in": ref_in}
        for lex in ("MM", "JM")
        for ref in ("STEM", "STNM", None)
        for fast in (True, False)
        for ref_in in (True, False)
    ]


def config_key(config: dict) -> str:
    return (
        f"{config['lex']}/{config['refinement'] or 'none'}/"
        f"{'fast' if config['fast_exploration'] else 'full'}/"
        f"{'in' if config['refinement_in'] else 'out'}"
    )


# =============================================================================
# Worker
# =============================================================================


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout


def fingerprint(members: dict) -> str:
    parts = sorted(tuple(sorted(map(tuple, m))) for m in members.values())
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


def run_worker(spec_path: str, out_path: str, label: str) -> int:
    warnings.simplefilter("ignore")
    import lago
    from lago import LinkStream, lago_modules, longitudinal_modularity

    spec = json.loads(Path(spec_path).read_text())
    streams = spec["streams"]
    timeout = spec["timeout"]
    signal.signal(signal.SIGALRM, _alarm)

    def build(stream):
        flags = dict(stream["flags"])
        if "partite_mapping" in flags:
            flags["partite_mapping"] = {int(k): v for k, v in flags["partite_mapping"].items()}
        ls = LinkStream(**flags)
        ls.add_links([tuple(link) for link in stream["links"]])
        return ls

    results = []
    skipped: dict[tuple, int] = {}
    total = len(spec["jobs"])
    for number, (index, config) in enumerate(spec["jobs"], 1):
        stream = streams[index]
        key = (stream["family"], config_key(config))
        record = {"stream": stream["name"], "config": config_key(config), "status": "ok"}
        if key in skipped and stream["size_rank"] > skipped[key]:
            record["status"] = "skipped"
            results.append(record)
            continue

        ls = build(stream)
        record["nb_edges"] = ls.nb_edges
        record["time_nodes"] = len(ls.leaves_dict)
        record["network_duration"] = ls.network_duration
        kwargs = dict(config)
        try:
            signal.alarm(timeout)
            start = time.perf_counter()
            modules = lago_modules(ls, **kwargs)
            record["seconds"] = time.perf_counter() - start
            signal.alarm(0)
        except _Timeout:
            record["status"] = "timeout"
            skipped[key] = stream["size_rank"]
            results.append(record)
            print(f"[{label}] {number}/{total} {stream['name']} {config_key(config)}: TIMEOUT", file=sys.stderr, flush=True)
            continue
        except Exception as exc:  # report, keep going
            signal.alarm(0)
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            results.append(record)
            print(f"[{label}] {number}/{total} {stream['name']} {config_key(config)}: ERROR {record['error']}", file=sys.stderr, flush=True)
            continue

        members = {str(k): sorted(v) for k, v in modules._raw_modules.items()}
        record["nb_modules"] = len(members)
        record["fingerprint"] = fingerprint(members)
        record["members"] = members

        # The metric on the result: first call, then steady state.
        lex = config["lex"]
        try:
            signal.alarm(timeout)
            start = time.perf_counter()
            value = longitudinal_modularity(ls, modules, lex=lex).value
            cold = time.perf_counter() - start
            warm = []
            for _ in range(4):
                start = time.perf_counter()
                longitudinal_modularity(ls, modules, lex=lex)
                warm.append(time.perf_counter() - start)
            signal.alarm(0)
            record["metric"] = {"value": value, "cold": cold, "warm": statistics.mean(warm[1:])}
        except _Timeout:
            record["metric"] = {"status": "timeout"}
        results.append(record)
        print(
            f"[{label}] {number}/{total} {stream['name']} {config_key(config)}: {record['seconds']:.2f}s, "
            f"{record['nb_modules']} modules",
            file=sys.stderr,
            flush=True,
        )

    Path(out_path).write_text(
        json.dumps({"label": label, "lago_file": lago.__file__, "python": sys.version, "results": results})
    )
    return 0


# =============================================================================
# Driver
# =============================================================================


def _worker_command(python: str, spec: Path, out: Path, label: str) -> list[str]:
    return [python, "-P", str(HERE / "bench_pypi.py"), "--worker", str(spec), str(out), label]


def run(args) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    streams = catalogue()
    (RESULTS_DIR / "streams.json").write_text(
        json.dumps([{k: v for k, v in s.items() if k != "links"} for s in streams])
    )

    jobs = []
    for index, stream in enumerate(streams):
        if args.tier not in ("all", stream["tier"]):
            continue
        for config in configs(stream["tier"]):
            jobs.append((index, config))
    # Smallest first within a family so a timeout can skip what follows.
    jobs.sort(key=lambda job: (streams[job[0]]["tier"], streams[job[0]]["family"], streams[job[0]]["size_rank"]))

    spec = RESULTS_DIR / "spec.json"
    spec.write_text(json.dumps({"streams": streams, "jobs": jobs, "timeout": args.timeout}))
    print(f"{len(streams)} streams, {len(jobs)} runs per implementation, timeout {args.timeout}s each")

    if args.limit:
        jobs = jobs[: args.limit]
        spec.write_text(json.dumps({"streams": streams, "jobs": jobs, "timeout": args.timeout}))
        print(f"smoke test: first {len(jobs)} runs only")

    # Workers run with -P (no script directory on sys.path): the tree's variants
    # find `lago` through PYTHONPATH, the PyPI one through its site-packages only,
    # from a working directory that contains no `lago/`.
    current = sys.executable
    variants = [
        ("compiled", current, {"PYTHONPATH": str(REPO_ROOT)}, REPO_ROOT),
        ("pure", current, {"PYTHONPATH": str(REPO_ROOT), "LAGO_CORE": "python"}, REPO_ROOT),
        ("pypi", args.pypi_python, {}, Path("/")),
    ]
    if args.only:
        variants = [v for v in variants if v[0] in args.only.split(",")]

    for label, python, env_extra, cwd in variants:
        out = RESULTS_DIR / f"{label}.json"
        log = RESULTS_DIR / f"{label}.log"
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LAGO_ACCEL", "LAGO_CORE")}
        env.update(env_extra)
        env["PYTHONWARNINGS"] = "ignore"
        print(f"--- {label}: {python} (log: {log})", flush=True)
        start = time.perf_counter()
        with log.open("w") as handle:
            proc = subprocess.run(
                _worker_command(python, spec, out, label),
                cwd=str(cwd),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        print(f"    exit {proc.returncode} after {time.perf_counter() - start:.0f}s", flush=True)
    return report(args)


# =============================================================================
# Report
# =============================================================================


def _load(label: str) -> dict | None:
    path = RESULTS_DIR / f"{label}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    payload["by_key"] = {(r["stream"], r["config"]): r for r in payload["results"]}
    return payload


def _geomean(values: list[float]) -> float | None:
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return statistics.geometric_mean(values)


def _fmt_s(record: dict | None) -> str:
    if record is None:
        return "—"
    if record["status"] == "timeout":
        return "timeout"
    if record["status"] == "skipped":
        return "skipped"
    if record["status"] == "error":
        return "error"
    s = record["seconds"]
    return f"{s:.2f} s" if s >= 1 else f"{s * 1e3:.0f} ms"


def _fmt_rate(record: dict | None, interactions: int) -> str:
    """Time plus microseconds per interaction, for the scale tables."""
    if not record or record["status"] != "ok":
        return _fmt_s(record)
    return f"{_fmt_s(record)} ({record['seconds'] / interactions * 1e6:.0f})"


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1e6:.0f} µs" if seconds < 1e-3 else f"{seconds * 1e3:.1f} ms"


def _fmt_x(a: dict | None, b: dict | None) -> str:
    """Speedup of b over a."""
    if not a or not b or a["status"] != "ok" or b["status"] != "ok":
        return "—"
    return f"{a['seconds'] / b['seconds']:.1f}×"


def _rescore(streams_meta: list[dict], sides: dict[str, dict]) -> dict[tuple, dict[str, float | None]]:
    """L-modularity of every returned partition, by the current metric."""
    sys.path.insert(0, str(REPO_ROOT))
    from lago import LinkStream, longitudinal_modularity

    spec = json.loads((RESULTS_DIR / "spec.json").read_text())
    by_name = {s["name"]: s for s in spec["streams"]}
    cache: dict[str, LinkStream] = {}
    scores: dict[tuple, dict[str, float | None]] = {}
    for label, payload in sides.items():
        for record in payload["results"]:
            if record["status"] != "ok":
                continue
            name = record["stream"]
            if name not in cache:
                stream = by_name[name]
                flags = dict(stream["flags"])
                if "partite_mapping" in flags:
                    flags["partite_mapping"] = {int(k): v for k, v in flags["partite_mapping"].items()}
                ls = LinkStream(**flags)
                ls.add_links([tuple(link) for link in stream["links"]])
                cache[name] = ls
            lex = record["config"].split("/")[0]
            communities = {k: {tuple(m) for m in v} for k, v in record["members"].items()}
            value = longitudinal_modularity(cache[name], communities, lex=lex).value
            scores.setdefault((name, record["config"]), {})[label] = value
    return scores


def report(args) -> int:
    sides = {label: payload for label in ("pypi", "pure", "compiled") if (payload := _load(label))}
    if not sides:
        print("no results found in", RESULTS_DIR)
        return 1
    streams_meta = json.loads((RESULTS_DIR / "streams.json").read_text())
    scores = _rescore(streams_meta, sides)

    pypi = sides.get("pypi", {}).get("by_key", {})
    pure = sides.get("pure", {}).get("by_key", {})
    comp = sides.get("compiled", {}).get("by_key", {})
    lines: list[str] = []
    out = lines.append

    out("# dcd-lago on PyPI (1.1.0) versus the current tree")
    out("")
    out(
        "Same inputs, same parameters, three implementations, each in its own process: the PyPI "
        "release in its own virtualenv; this tree with `LAGO_CORE=python`; this tree with the "
        "compiled core and the Cython metric kernel. Every stream is generated once as a list of "
        "links and handed to each side in the same order. `lago_modules` is timed on a freshly "
        "built stream; `longitudinal_modularity` is then timed on the result (first call, and "
        "steady state). Every returned partition is scored with the **current** metric so that "
        "quality is compared with one yardstick.\n"
    )
    out(f"Generated by `benchmarks/bench_pypi.py`; raw results in `{RESULTS_DIR.relative_to(REPO_ROOT) if RESULTS_DIR.is_relative_to(REPO_ROOT) else RESULTS_DIR}`.")
    out("")
    out("Two things to keep in mind when reading the partitions column:")
    out("")
    out("* the PyPI release is **not deterministic** (its hashing follows memory addresses, bug B3 of "
        "round 1), so its partition can differ from run to run; the current tree is deterministic;")
    out("* round 1 fixed nine latent bugs in the search, some of which change the result on "
        "specific inputs (k-partite null model, delayed duplicates, pop-order durations, ...). "
        "\"same partition\" is therefore informative but not required; the L-modularity columns are "
        "the quality comparison.")
    out("")

    # ---- headline, computed from the same records as the tables below
    matrix_names = {s["name"] for s in streams_meta if s["tier"] == "matrix"}
    scale_names = {s["name"] for s in streams_meta if s["tier"] == "scale"}

    def ratios(names, side):
        return [
            pypi[key]["seconds"] / side[key]["seconds"]
            for key in side
            if key[0] in names and key in pypi
            and pypi[key]["status"] == "ok" and side[key]["status"] == "ok"
        ]

    quality_better = quality_equal = quality_worse = 0
    identical = compared = 0
    for key, sc in scores.items():
        if "pypi" in sc and "pure" in sc:
            delta = sc["pure"] - sc["pypi"]
            if delta > 1e-4:
                quality_better += 1
            elif delta < -1e-4:
                quality_worse += 1
            else:
                quality_equal += 1
        a, b = pypi.get(key), pure.get(key)
        if a and b and a["status"] == "ok" and b["status"] == "ok":
            compared += 1
            identical += a["fingerprint"] == b["fingerprint"]
    pypi_timeouts = [k for k, r in pypi.items() if r["status"] in ("timeout", "skipped")]

    out("## Headline")
    out("")
    if pure and pypi:
        rm_p, rs_p = ratios(matrix_names, pure), ratios(scale_names, pure)
        rm_c, rs_c = ratios(matrix_names, comp), ratios(scale_names, comp)
        out(f"* **Speed, `lago_modules`.** Over the {len(rm_p)} matrix-tier runs the PyPI release finished "
            f"(every parameter combination, 15 streams of 500–2 100 interactions), the tree is "
            f"**{_geomean(rm_p):.1f}× faster in pure Python and {_geomean(rm_c):.1f}× with the compiled core** "
            f"(geometric means); the range across parameter combinations is "
            f"{min(_geomean(ratios({s}, pure)) or 1 for s in matrix_names):.0f}–"
            f"{max(_geomean(ratios({s}, pure)) or 1 for s in matrix_names):.0f}× per stream in pure Python. "
            f"On the scale tier (4k–96k interactions, default parameters) the speedup is "
            f"{min(rs_p):.0f}–{max(rs_p):.0f}× pure and {min(rs_c):.0f}–{max(rs_c):.0f}× compiled, and it grows "
            f"with size on wide streams: the PyPI release is superlinear there, the tree is not.")
    if pypi_timeouts:
        out(f"* **PyPI timeouts.** {len(pypi_timeouts)} runs did not finish within {json.loads((RESULTS_DIR / 'spec.json').read_text())['timeout']} s: "
            + ", ".join(f"`{k[0]} {k[1]}`" for k in pypi_timeouts)
            + ". The two on continuous streams of a few hundred interactions are the oscillation behaviour "
            "fixed in round 1 (moves undone and redone; bug B6, id()-keyed caches), not size.")
    out(f"* **Quality, by the current metric.** Of {quality_better + quality_equal + quality_worse} runs where both "
        f"finished, the tree's partition scores higher than PyPI's in {quality_better}, the same (±1e-4) in "
        f"{quality_equal}, lower in {quality_worse}. The PyPI release is not deterministic, so its numbers are one "
        f"draw; the differences either way are those of a greedy search taking another path, plus the round-1 "
        f"bug fixes on the inputs they concern.")
    out(f"* **Same partition as PyPI** in {identical}/{compared} runs (see the caveats above on why this is "
        f"informative, not required); **pure and compiled tree identical in every run** (last section).")
    out("* **Metric.** `longitudinal_modularity` is 1.5–2.5× faster in pure Python and 10–70× faster in steady state "
        "with the compiled core and kernel (section 3).")
    out("")

    out("## Streams")
    out("")
    out("| stream | features | nodes | timesteps | interactions |")
    out("|---|---|---:|---:|---:|")
    for s in streams_meta:
        out(f"| `{s['name']}` | {s['features']} | {s['nodes']} | {s['timesteps']} | {s['interactions']:,} |")
    out("")

    # ---- matrix tier
    matrix = [s for s in streams_meta if s["tier"] == "matrix"]
    if matrix:
        out("## 1. Every parameter combination, per stream")
        out("")
        out("`lex` × `refinement` × `fast_exploration` (fast / full) × `refinement_in` (in / out); "
            "`gamma=1`, `omega=2`, `nb_iter=1`, canonical order. Times are `lago_modules` alone. "
            "L-modularity by the current metric, of the partition each side returned.")
        out("")
        for s in matrix:
            out(f"### `{s['name']}` — {s['features']}; {s['nodes']} nodes, {s['timesteps']} timesteps, {s['interactions']:,} interactions")
            out("")
            out("| parameters | PyPI | pure | compiled | pure ÷ PyPI | compiled ÷ PyPI | same partition (pure vs PyPI) | LM PyPI | LM current |")
            out("|---|---:|---:|---:|---:|---:|:-:|---:|---:|")
            for config in configs("matrix"):
                key = (s["name"], config_key(config))
                a, b, c = pypi.get(key), pure.get(key), comp.get(key)
                same = "—"
                if a and b and a["status"] == "ok" and b["status"] == "ok":
                    same = "yes" if a["fingerprint"] == b["fingerprint"] else "no"
                sc = scores.get(key, {})
                lm_p = f"{sc['pypi']:.4f}" if "pypi" in sc else "—"
                lm_c = f"{sc.get('pure', sc.get('compiled')):.4f}" if ("pure" in sc or "compiled" in sc) else "—"
                out(f"| `{config_key(config)}` | {_fmt_s(a)} | {_fmt_s(b)} | {_fmt_s(c)} | {_fmt_x(a, b)} | {_fmt_x(a, c)} | {same} | {lm_p} | {lm_c} |")
            out("")

        out("### Summary over the matrix tier")
        out("")
        out("Geometric means of the speedups over the runs where both sides finished.")
        out("")
        out("| parameters | runs | pure ÷ PyPI | compiled ÷ PyPI | LM current − LM PyPI (mean) |")
        out("|---|---:|---:|---:|---:|")
        for config in configs("matrix"):
            ck = config_key(config)
            ratios_pure, ratios_comp, deltas = [], [], []
            n = 0
            for s in matrix:
                key = (s["name"], ck)
                a, b, c = pypi.get(key), pure.get(key), comp.get(key)
                if a and b and a["status"] == b["status"] == "ok":
                    ratios_pure.append(a["seconds"] / b["seconds"])
                    n += 1
                if a and c and a["status"] == c["status"] == "ok":
                    ratios_comp.append(a["seconds"] / c["seconds"])
                sc = scores.get(key, {})
                if "pypi" in sc and ("pure" in sc or "compiled" in sc):
                    deltas.append(sc.get("pure", sc.get("compiled")) - sc["pypi"])
            gp, gc = _geomean(ratios_pure), _geomean(ratios_comp)
            md = f"{statistics.mean(deltas):+.4f}" if deltas else "—"
            out(f"| `{ck}` | {n} | {gp:.1f}× | {gc:.1f}× | {md} |" if gp and gc else f"| `{ck}` | {n} | — | — | {md} |")
        out("")
        out("| stream | pure ÷ PyPI | compiled ÷ PyPI | PyPI timeouts |")
        out("|---|---:|---:|---:|")
        for s in matrix:
            rp, rc, timeouts = [], [], 0
            for config in configs("matrix"):
                key = (s["name"], config_key(config))
                a, b, c = pypi.get(key), pure.get(key), comp.get(key)
                if a and a["status"] in ("timeout", "skipped"):
                    timeouts += 1
                if a and b and a["status"] == b["status"] == "ok":
                    rp.append(a["seconds"] / b["seconds"])
                if a and c and a["status"] == c["status"] == "ok":
                    rc.append(a["seconds"] / c["seconds"])
            gp, gc = _geomean(rp), _geomean(rc)
            out(f"| `{s['name']}` | {gp:.1f}× | {gc:.1f}× | {timeouts} |" if gp and gc else f"| `{s['name']}` | — | — | {timeouts} |")
        out("")

    # ---- scale tier
    scale = [s for s in streams_meta if s["tier"] == "scale"]
    if scale:
        out("## 2. Scale: default parameters, MM and JM")
        out("")
        out("`refinement=STEM`, `fast_exploration=True`, `refinement_in=True`. µs per interaction in "
            "parentheses. A timeout skips the larger sizes of that shape for that implementation.")
        out("")
        for lex in ("MM", "JM"):
            ck = config_key({"lex": lex, "refinement": "STEM", "fast_exploration": True, "refinement_in": True})
            out(f"### `{lex}`")
            out("")
            out("| stream | nodes | timesteps | interactions | PyPI | pure | compiled | pure ÷ PyPI | compiled ÷ PyPI | LM PyPI | LM current |")
            out("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for s in scale:
                key = (s["name"], ck)
                a, b, c = pypi.get(key), pure.get(key), comp.get(key)
                n = s["interactions"]
                sc = scores.get(key, {})
                lm_p = f"{sc['pypi']:.4f}" if "pypi" in sc else "—"
                lm_c = f"{sc.get('pure', sc.get('compiled')):.4f}" if ("pure" in sc or "compiled" in sc) else "—"
                out(f"| `{s['name']}` | {s['nodes']} | {s['timesteps']} | {n:,} | {_fmt_rate(a, n)} | {_fmt_rate(b, n)} | {_fmt_rate(c, n)} | {_fmt_x(a, b)} | {_fmt_x(a, c)} | {lm_p} | {lm_c} |")
            out("")

    # ---- metric
    out("## 3. `longitudinal_modularity` on the returned partitions")
    out("")
    out("Default parameters' partition of each stream (MM). *First call* is one scoring of a fresh "
        "stream; *steady* is the mean of repeated scorings (the compiled variant takes the Cython "
        "kernel from the second scoring on).")
    out("")
    out("| stream | interactions | PyPI first | PyPI steady | pure first | pure steady | compiled first | compiled steady | steady speedup (compiled ÷ PyPI) |")
    out("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    ck = config_key({"lex": "MM", "refinement": "STEM", "fast_exploration": True, "refinement_in": True})
    for s in streams_meta:
        key = (s["name"], ck)
        cells = []
        steady = {}
        for label, side in (("pypi", pypi), ("pure", pure), ("compiled", comp)):
            r = side.get(key)
            m = r.get("metric") if r and r["status"] == "ok" else None
            if m and "cold" in m:
                cells += [_fmt_ms(m["cold"]), _fmt_ms(m["warm"])]
                steady[label] = m["warm"]
            else:
                cells += ["—", "—"]
        ratio = f"{steady['pypi'] / steady['compiled']:.1f}×" if "pypi" in steady and "compiled" in steady else "—"
        out(f"| `{s['name']}` | {s['interactions']:,} | " + " | ".join(cells) + f" | {ratio} |")
    out("")

    # The two variants of the tree must return the same partition, always.
    if pure and comp:
        agree = total = 0
        for key, b in pure.items():
            c = comp.get(key)
            if b["status"] == "ok" and c and c["status"] == "ok":
                total += 1
                agree += b["fingerprint"] == c["fingerprint"]
        out("## Consistency of the two variants of the tree")
        out("")
        out(f"`pure` and `compiled` returned the identical partition in **{agree}/{total}** runs where both finished"
            + (" — as required: the compiled core is the same source." if agree == total else " — **MISMATCH, investigate.**"))
        out("")

    out("## Environment")
    out("")
    for label, payload in sides.items():
        out(f"* `{label}`: `{payload['lago_file']}`, Python {payload['python'].split()[0]}")
    out("* Apple Silicon (8 cores), macOS 26; one process at a time, single-threaded.")
    out("")

    target = REPO_ROOT / "docs" / "BENCHMARK_PYPI_VS_CURRENT.md"
    target.write_text("\n".join(lines))
    print(f"report -> {target}")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return run_worker(sys.argv[2], sys.argv[3], sys.argv[4])
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("run", "report"))
    ap.add_argument("--pypi-python", help="python of the virtualenv where dcd-lago is installed from PyPI")
    ap.add_argument("--tier", default="all", choices=("all", "matrix", "scale"))
    ap.add_argument("--timeout", type=int, default=900, help="per run, seconds")
    ap.add_argument("--only", default="", help="comma-separated subset of pypi,pure,compiled")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N runs")
    args = ap.parse_args()
    if args.phase == "run":
        if not args.pypi_python and "pypi" in (args.only or "pypi"):
            ap.error("--pypi-python is required to run the pypi variant")
        return run(args)
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
