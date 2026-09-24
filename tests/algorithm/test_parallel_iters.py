"""``n_jobs``: the iterations of ``nb_iter`` spread over processes.

Every iteration is a function of ``(seed, iteration)`` alone and the best is
picked the same way, so the result must be the very same partition as the
sequential run -- checked here on each stream mode. ``n_jobs=1`` is the
sequential loop itself.
"""

from __future__ import annotations

import random
import warnings

import pytest

import lago.algorithm.lago as lago_module
from lago import LinkStream, lago_modules
from lago.algorithm.lago import _effective_jobs


def _undirected():
    rng = random.Random(21)
    ls = LinkStream()
    ls.add_links(
        [
            (i, j, t)
            for t in range(5)
            for i in range(16)
            for j in range(i + 1, 16)
            if rng.random() < (0.7 if i % 2 == j % 2 else 0.06)
        ]
    )
    return ls


def _directed_weighted():
    rng = random.Random(22)
    ls = LinkStream(directed=True)
    ls.add_links(
        [
            (i, j, t, round(rng.uniform(0.5, 2.5), 2))
            for t in range(4)
            for i in range(12)
            for j in range(12)
            if i != j and rng.random() < (0.6 if i % 2 == j % 2 else 0.06)
        ]
    )
    return ls


def _continuous():
    rng = random.Random(23)
    links = []
    for _ in range(200):
        i, j = rng.randrange(12), rng.randrange(12)
        if i == j or rng.random() > (0.8 if i % 2 == j % 2 else 0.1):
            continue
        links.append((min(i, j), max(i, j), rng.randrange(0, 10), rng.randrange(1, 4)))
    ls = LinkStream(continuous=True)
    ls.add_links(links)
    return ls


def _delayed():
    rng = random.Random(24)
    ls = LinkStream(delayed=True)
    seen, links = set(), []
    for _ in range(200):
        i, j = rng.randrange(12), rng.randrange(12)
        if i == j or rng.random() > (0.8 if i % 2 == j % 2 else 0.1):
            continue
        t = rng.randrange(0, 6)
        key = (min(i, j), max(i, j), t, t + rng.randrange(0, 2))
        if key not in seen:
            seen.add(key)
            links.append((i, j, key[2], key[3]))
    ls.add_links(links)
    return ls


def _fingerprint(modules) -> tuple:
    return tuple(sorted(tuple(sorted(members)) for members in modules._raw_modules.values()))


class TestSameResultAsSequential:
    @pytest.mark.parametrize("make", [_undirected, _directed_weighted, _continuous, _delayed])
    def test_seeded(self, make) -> None:
        sequential = lago_modules(make(), nb_iter=4, seed=3)
        parallel = lago_modules(make(), nb_iter=4, seed=3, n_jobs=2)
        assert _fingerprint(parallel) == _fingerprint(sequential)

    def test_unseeded(self) -> None:
        """The canonical first iteration and the derived others, same selection."""
        sequential = lago_modules(_undirected(), nb_iter=3)
        parallel = lago_modules(_undirected(), nb_iter=3, n_jobs=3)
        assert _fingerprint(parallel) == _fingerprint(sequential)

    def test_jm_lex(self) -> None:
        sequential = lago_modules(_undirected(), lex="JM", nb_iter=3, seed=5)
        parallel = lago_modules(_undirected(), lex="JM", nb_iter=3, seed=5, n_jobs=2)
        assert _fingerprint(parallel) == _fingerprint(sequential)


class TestLogging:
    def test_parent_logs_in_iteration_order(self, capsys) -> None:
        lago_modules(_undirected(), nb_iter=4, seed=3, n_jobs=2, verbose=1)
        lines = [line for line in capsys.readouterr().out.splitlines() if "Iteration " in line]
        assert lines, "no per-iteration log line"
        indices = [int(line.split("Iteration ")[1].split("/")[0]) for line in lines]
        assert indices == sorted(indices)
        assert not any("Time Module Movements" in line for line in lines), "a worker printed"

    def test_sequential_log_is_unchanged(self, capsys) -> None:
        lago_modules(_undirected(), nb_iter=2, seed=3, verbose=1)
        out = capsys.readouterr().out
        assert "Starting iteration 1/2" in out
        assert "Running" not in out


class TestJobs:
    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            lago_modules(_undirected(), n_jobs=0)

    def test_capped_by_iterations_and_cpus(self, monkeypatch) -> None:
        monkeypatch.setattr(lago_module.os, "cpu_count", lambda: 8)
        assert _effective_jobs(16, 3, _undirected(), 0) == 3
        monkeypatch.setattr(lago_module.os, "cpu_count", lambda: 2)
        assert _effective_jobs(16, 8, _undirected(), 0) == 2

    def test_reduced_when_memory_is_short(self, monkeypatch) -> None:
        stream = _undirected()
        monkeypatch.setattr(lago_module.os, "cpu_count", lambda: 8)
        # Room for exactly two copies of the stream at the estimate.
        monkeypatch.setattr(
            lago_module,
            "_available_memory",
            lambda: int(2 * stream.nb_edges * lago_module._BYTES_PER_TIME_EDGE / 0.8),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert _effective_jobs(8, 8, stream, 0) == 2
        assert any(issubclass(w.category, ResourceWarning) for w in caught)

    def test_no_memory_information_means_no_cap(self, monkeypatch) -> None:
        monkeypatch.setattr(lago_module.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(lago_module, "_available_memory", lambda: None)
        assert _effective_jobs(4, 8, _undirected(), 0) == 4

    def test_without_fork_runs_sequentially(self, monkeypatch) -> None:
        monkeypatch.setattr(lago_module.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(lago_module.multiprocessing, "get_all_start_methods", lambda: ["spawn"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert _effective_jobs(4, 8, _undirected(), 0) == 1
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)
