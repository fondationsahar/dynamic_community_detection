"""The batch candidate evaluation against the per-candidate reference.

``find_best_module_for_submodule`` scores every candidate of a move in one pass
over M0 (``DeltaLongitudinalModularityComputer.evaluate_candidates``); the
original one-``M0_to_Mx``-per-candidate formulation is kept as the reference.
On every call of real runs, over every stream mode, the two must agree on the
candidate set, on each candidate's total and on the winner -- **exactly**, not
within a tolerance: the batch path performs the same additions in the same
order, so a difference of one ulp would already mean a reordering.
"""

from __future__ import annotations

import random

import pytest

import lago.algorithm._internal.find_best_move as fbm
import lago.algorithm._internal.lago_tools as lts
import lago.algorithm._internal.stem as stem
import lago.algorithm._internal.tmm as tmm
from lago import LinkStream, lago_modules
from lago.algorithm._internal.delta_lm import DeltaLongitudinalModularityComputer as Computer
from lago.algorithm._internal.leaf_set import LeafDifference
from lago.algorithm._internal.runner import _init_movers

# =============================================================================
# Streams: one of each mode, small enough for two evaluations per call
# =============================================================================


def _planted(rng, n_nodes, n_comms, n_times, p_in, p_out, *, directed=False, weighted=False):
    links = []
    for t in range(n_times):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if j <= i and not directed:
                    continue
                if i == j:
                    continue
                p = p_in if i % n_comms == j % n_comms else p_out
                if rng.random() < p:
                    links.append((i, j, t, round(rng.uniform(0.5, 2.5), 2)) if weighted else (i, j, t))
    return links


def _undirected():
    ls = LinkStream()
    ls.add_links(_planted(random.Random(1), 18, 3, 6, 0.7, 0.05))
    return ls


def _directed():
    ls = LinkStream(directed=True)
    ls.add_links(_planted(random.Random(2), 16, 2, 5, 0.6, 0.06, directed=True))
    return ls


def _weighted():
    ls = LinkStream()
    ls.add_links(_planted(random.Random(3), 16, 2, 5, 0.7, 0.05, weighted=True))
    return ls


def _bipartite():
    rng = random.Random(4)
    links = [
        (i, 10 + j, t)
        for t in range(5)
        for i in range(10)
        for j in range(8)
        if rng.random() < (0.6 if (i % 2) == (j % 2) else 0.05)
    ]
    ls = LinkStream(partite_mapping={n: (0 if n < 10 else 1) for n in range(18)})
    ls.add_links(links)
    return ls


def _continuous():
    rng = random.Random(5)
    links = []
    for _ in range(220):
        i, j = rng.randrange(14), rng.randrange(14)
        if i == j:
            continue
        if rng.random() > (0.8 if i % 2 == j % 2 else 0.1):
            continue
        links.append((min(i, j), max(i, j), rng.randrange(0, 12), rng.randrange(1, 4)))
    ls = LinkStream(continuous=True)
    ls.add_links(links)
    return ls


def _delayed():
    rng = random.Random(6)
    seen, links = set(), []
    for _ in range(260):
        i, j = rng.randrange(14), rng.randrange(14)
        if i == j or rng.random() > (0.8 if i % 2 == j % 2 else 0.1):
            continue
        t1 = rng.randrange(0, 8)
        key = (min(i, j), max(i, j), t1, t1 + rng.randrange(0, 2))
        if key not in seen:
            seen.add(key)
            links.append((i, j, key[2], key[3]))
    ls = LinkStream(delayed=True)
    ls.add_links(links)
    return ls


STREAMS = {
    "undirected": _undirected,
    "directed": _directed,
    "weighted": _weighted,
    "bipartite": _bipartite,
    "continuous": _continuous,
    "delayed": _delayed,
}


# =============================================================================
# Reference: the original candidate discovery and per-candidate deltas
# =============================================================================


def _reference(computer, submodule, partite_mapping, modules, level):
    if modules is None:
        if level == "parent":
            modules = sorted(
                {m.parent for m in submodule.neighbors if m.parent is not None},
                key=lambda m: m.index,
            )
        else:
            modules = sorted(lts.get_neighbors_modules_parents(submodule), key=lambda m: m.index)
    parent = submodule.parent
    modules = [m for m in modules if m is not parent]
    if not modules or parent is None:
        return None, {}
    M0 = submodule.leaves
    segments = lts.get_nodes_segment(module_leaves=M0)
    M1 = LeafDifference(parent.leaves, M0)
    leaving = -computer.M0_to_Mx(
        M0_leaves=M0, M0_time_segments=segments, Mx_leaves=M1,
        partite_mapping=partite_mapping, union_module=parent,
    )
    totals = {}
    for module in modules:
        disjoint = module.leaves.isdisjoint(M0)
        M2 = module.leaves if disjoint else module.leaves - M0
        if not M2 and not M1:
            continue
        totals[module] = leaving + computer.M0_to_Mx(
            M0_leaves=M0, M0_time_segments=segments, Mx_leaves=M2,
            partite_mapping=partite_mapping, mx_module=module if disjoint else None,
        )
    return leaving, totals


def _batch(computer, submodule, partite_mapping, modules, level):
    parent = submodule.parent
    if parent is None:
        return None, {}
    M0 = submodule.leaves
    segments = lts.get_nodes_segment(module_leaves=M0)
    if modules is None and level == "parent":
        # As find_best_module_for_submodule does: the neighbours recorded at
        # the start of the level, not the live adjacency.
        modules = {m.parent for m in submodule.neighbors if m.parent is not None}
    candidates, leaving, _, evaluations = computer.evaluate_candidates(
        M0, segments, parent, level == "parent", partite_mapping, modules
    )
    if not candidates:
        return None, {}
    return leaving, {module: leaving + joining for module, joining, _ in evaluations}


def _run_compared(stream, tol: float = 0.0, **lago_kwargs) -> dict:
    """Run LAGO with every find_best call also evaluated by the reference.

    ``tol`` is a relative tolerance on the deltas; 0 demands bit-identity.
    """
    original = fbm.find_best_module_for_submodule
    stats = {"calls": 0, "candidate_sets": 0, "values": 0, "winners": 0}

    def differs(a, b) -> bool:
        if a is None or b is None:
            return a is not b
        return abs(a - b) > tol * max(1.0, abs(a))

    def shadow(computer, submodule, partite_mapping={}, modules=None, verbose=0, stopping_criterion=0.0, level="parent"):  # noqa: B006 - mirrors the signature it replaces
        stats["calls"] += 1
        ref_leaving, ref_totals = _reference(computer, submodule, partite_mapping, modules, level)
        new_leaving, new_totals = _batch(computer, submodule, partite_mapping, modules, level)
        if set(ref_totals) != set(new_totals):
            stats["candidate_sets"] += 1
        elif ref_totals:
            if differs(ref_leaving, new_leaving):
                stats["values"] += 1
            stats["values"] += sum(1 for m, v in ref_totals.items() if differs(v, new_totals[m]))

        result = original(computer, submodule, partite_mapping, modules, verbose, stopping_criterion, level)
        Computer.use_batch = False
        try:
            reference = original(computer, submodule, partite_mapping, modules, verbose, stopping_criterion, level)
        finally:
            Computer.use_batch = True
        if result[0] is not reference[0] or differs(result[1], reference[1]):
            stats["winners"] += 1
        return result

    tmm.find_best_module_for_submodule = shadow
    stem.find_best_module_for_submodule = shadow
    try:
        lago_modules(stream, **lago_kwargs)
    finally:
        tmm.find_best_module_for_submodule = original
        stem.find_best_module_for_submodule = original
    return stats


# =============================================================================
# Tests
# =============================================================================


class TestBatchMatchesReference:
    @pytest.mark.parametrize("lex", ["MM", "JM"])
    @pytest.mark.parametrize("refinement", ["STEM", None])
    @pytest.mark.parametrize("name", sorted(STREAMS))
    def test_every_call_agrees(self, name: str, refinement: str | None, lex: str) -> None:
        # JM sums degrees in a canonical order where the reference sums them in
        # set order; with float weights that is a last-bit difference, held to
        # the same 1e-9 as the MM closed form. Integer degrees are exact.
        tol = 1e-9 if lex == "JM" and name == "weighted" else 0.0
        stats = _run_compared(STREAMS[name](), tol=tol, lex=lex, refinement=refinement, seed=1)
        assert stats["calls"] > 0
        assert stats["candidate_sets"] == 0, f"{stats['candidate_sets']} calls found different candidates"
        assert stats["values"] == 0, f"{stats['values']} candidate totals differ"
        assert stats["winners"] == 0, f"{stats['winners']} calls chose a different move"

    @pytest.mark.parametrize("name", ["undirected", "directed"])
    def test_canonical_order_too(self, name: str) -> None:
        """The unseeded run takes a different trajectory; check it as well."""
        stats = _run_compared(STREAMS[name](), lex="MM", refinement="STEM")
        assert stats["calls"] > 0
        assert stats["candidate_sets"] == stats["values"] == stats["winners"] == 0

    def test_gamma_zero_and_omega_zero(self) -> None:
        """The skipped terms are skipped in the same places on both paths."""
        for gamma, omega in ((0, 2), (1, 0), (0, 0)):
            stats = _run_compared(_undirected(), lex="MM", gamma=gamma, omega=omega, seed=1)
            assert stats["calls"] > 0
            assert stats["candidate_sets"] == stats["values"] == stats["winners"] == 0


class TestPathSelection:
    def test_stnm_uses_the_per_candidate_path(self) -> None:
        """STNM's shared-set handling breaks the owner invariant; see runner.py."""
        mover, _ = _init_movers(_undirected(), "MM", 1, 2, True, "STNM", 1e-8)
        assert mover.delta_lm_computer.use_batch is False

    @pytest.mark.parametrize("refinement", ["STEM", None])
    def test_others_use_the_batch_path(self, refinement) -> None:
        mover, _ = _init_movers(_undirected(), "MM", 1, 2, True, refinement, 1e-8)
        assert mover.delta_lm_computer.use_batch is True
        # ...through the class attribute, so the harnesses can switch it off.
        assert "use_batch" not in vars(mover.delta_lm_computer)


class TestMoveChanges:
    def test_returned_changes_match_recomputation(self) -> None:
        """The changes handed back describe exactly the durations after the move."""
        import lago.core.utils as tls

        original = fbm.find_best_module_for_submodule
        checked = {"n": 0}

        def shadow(computer, submodule, *args, **kwargs):
            result = original(computer, submodule, *args, **kwargs)
            best, _, changes = result
            if best is not None and changes is not None and changes.joining is not None:
                parent = submodule.parent
                for node, (before, after) in changes.joining.items():
                    assert before == best.get_durations(tls.get_nodes_durations).get(node, 0)
                    union = tls.get_nodes_durations(best.leaves | submodule.leaves)
                    assert after == union.get(node, 0)
                for node, (before, after) in changes.leaving.items():
                    assert before == parent.get_durations(tls.get_nodes_durations).get(node, 0)
                    remaining = tls.get_nodes_durations(parent.leaves - submodule.leaves)
                    assert after == remaining.get(node, 0)
                checked["n"] += 1
            return result

        tmm.find_best_module_for_submodule = shadow
        stem.find_best_module_for_submodule = shadow
        try:
            lago_modules(_continuous(), lex="MM", seed=1)
        finally:
            tmm.find_best_module_for_submodule = original
            stem.find_best_module_for_submodule = original
        assert checked["n"] > 0
