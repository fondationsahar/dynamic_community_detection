"""Generators for benchmark link streams of different types.

Ten streams covering every LinkStream mode: instantaneous (plain, weighted,
directed, k-partite), continuous, delayed, and the real 2096-link fixture.
"""
from __future__ import annotations

import random

from _common import REPO_ROOT

from lago import LinkStream


def _planted(n_nodes, n_comms, n_times, p_in, p_out, rng):
    """Instantaneous planted-partition temporal network."""
    comm = {n: n % n_comms for n in range(n_nodes)}
    links = []
    for t in range(n_times):
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                p = p_in if comm[i] == comm[j] else p_out
                if rng.random() < p:
                    links.append((i, j, t))
    return links


def planted_small(seed=1):
    rng = random.Random(seed)
    ls = LinkStream()
    ls.add_links(_planted(24, 3, 6, 0.45, 0.03, rng))
    return ls


def planted_medium(seed=2):
    rng = random.Random(seed)
    ls = LinkStream()
    ls.add_links(_planted(60, 4, 12, 0.30, 0.02, rng))
    return ls


def planted_large(seed=3):
    rng = random.Random(seed)
    ls = LinkStream()
    ls.add_links(_planted(120, 5, 20, 0.20, 0.01, rng))
    return ls


def planted_xlarge(seed=4):
    rng = random.Random(seed)
    ls = LinkStream()
    ls.add_links(_planted(200, 6, 25, 0.15, 0.005, rng))
    return ls


def weighted_medium(seed=5):
    rng = random.Random(seed)
    base = _planted(50, 3, 10, 0.30, 0.02, rng)
    links = [(a, b, t, round(rng.uniform(0.2, 3.0), 3)) for (a, b, t) in base]
    ls = LinkStream()
    ls.add_links(links)
    return ls


def directed_medium(seed=6):
    rng = random.Random(seed)
    base = _planted(50, 3, 10, 0.30, 0.02, rng)
    links = []
    for a, b, t in base:
        if rng.random() < 0.5:
            links.append((a, b, t))
        else:
            links.append((b, a, t))
    ls = LinkStream(directed=True)
    ls.add_links(links)
    return ls


def bipartite_medium(seed=7):
    rng = random.Random(seed)
    left = list(range(0, 30))
    right = list(range(30, 60))
    mapping = dict.fromkeys(left, 0)
    mapping.update(dict.fromkeys(right, 1))
    links = []
    for t in range(10):
        for a in left:
            for b in right:
                p = 0.25 if (a % 3) == (b % 3) else 0.02
                if rng.random() < p:
                    links.append((a, b, t))
    ls = LinkStream(partite_mapping=mapping)
    ls.add_links(links)
    return ls


def bipartite_directed_medium(seed=7):
    """Directed AND k-partite -- the combination that exercises
    `_get_expectation_jm_kpartite_part`'s directed branch. In such a stream a
    node normally has only in-edges or only out-edges.
    """
    rng = random.Random(seed)
    left = list(range(0, 25))
    right = list(range(25, 50))
    mapping = dict.fromkeys(left, 0)
    mapping.update(dict.fromkeys(right, 1))
    links = []
    for t in range(10):
        for a in left:
            for b in right:
                p = 0.25 if (a % 3) == (b % 3) else 0.02
                if rng.random() < p:
                    links.append((a, b, t))
    ls = LinkStream(directed=True, partite_mapping=mapping)
    ls.add_links(links)
    return ls


def continuous_medium(seed=8):
    rng = random.Random(seed)
    comm = {n: n % 3 for n in range(40)}
    links = []
    for _ in range(1200):
        i = rng.randrange(40)
        j = rng.randrange(40)
        if i == j:
            continue
        p = 0.9 if comm[i] == comm[j] else 0.1
        if rng.random() > p:
            continue
        t = rng.randrange(0, 40)
        d = rng.randrange(1, 5)
        links.append((min(i, j), max(i, j), t, d))
    ls = LinkStream(continuous=True)
    ls.add_links(links)
    return ls


def delayed_medium(seed=9):
    rng = random.Random(seed)
    comm = {n: n % 3 for n in range(40)}
    seen = set()
    links = []
    for _ in range(1500):
        i = rng.randrange(40)
        j = rng.randrange(40)
        if i == j:
            continue
        p = 0.9 if comm[i] == comm[j] else 0.08
        if rng.random() > p:
            continue
        t1 = rng.randrange(0, 15)
        t2 = t1 + rng.randrange(0, 2)
        key = (min(i, j), max(i, j), t1, t2)
        if key in seen:
            continue
        seen.add(key)
        links.append((i, j, t1, t2))
    ls = LinkStream(delayed=True)
    ls.add_links(links)
    return ls


def fixture_stream():
    path = REPO_ROOT / "tests" / "fixtures" / "linkstream.txt"
    ls = LinkStream()
    ls.read_txt(str(path))
    return ls


GENERATORS = {
    "planted_small": planted_small,
    "planted_medium": planted_medium,
    "planted_large": planted_large,
    "planted_xlarge": planted_xlarge,
    "weighted_medium": weighted_medium,
    "directed_medium": directed_medium,
    "bipartite_medium": bipartite_medium,
    "bipartite_directed_medium": bipartite_directed_medium,
    "continuous_medium": continuous_medium,
    "delayed_medium": delayed_medium,
    "fixture": fixture_stream,
}
