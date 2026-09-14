"""The fast MM expectation delta must equal the pair loop it replaces.

``_get_expectation_mm_part`` has an O(|M0|) path -- memoised per-module degree
aggregates plus an incremental correction -- standing in for an O(n**2)
enumeration over node pairs. The two must agree on every call, on every kind of
link stream.

This matters more than it looks. The algebra differs between undirected and
directed streams (the directed form carries a factor 2), and the k-partite mask
has to be reproduced by subtracting each partite's own aggregate. A rewrite that
gets either wrong still produces plausible numbers, and can still drive the
greedy search to the same partition on a given corpus -- so comparing partitions
does not catch it. Comparing the two implementations call by call does.

``benchmarks/compare_delta.py`` runs the same comparison over large streams;
these are the small, fast versions that run in CI.
"""

from __future__ import annotations

import pytest

from lago import LinkStream, lago_modules
from lago.algorithm._internal.delta_lm import DeltaLongitudinalModularityComputer as Computer
from lago.core.utils import get_module_duration, get_nodes_durations

TOLERANCE = 1e-9


# =============================================================================
# Independent reference
# =============================================================================


def _ordered_pair_sum(linkstream, nodes, partite_mapping, time_factor) -> float:
    """Sum a per-pair term over every ORDERED pair of nodes, skipping same-partite ones.

    Written as an explicit double loop over ordered pairs, deliberately: the
    production code enumerates the upper triangle and compensates with a
    ``2 ** (n1 != n2)`` factor, and that compensation is exactly where this code
    has gone wrong before -- once by omitting it where it was needed
    (undirected), once by adding it where it was not (directed, whose
    ``in1*out2 + in2*out1`` already covers both directions). Spelling out the
    ordered pairs removes the question.

    The normalisation follows the non-k-partite branches, which evaluate the
    same sums in closed form as ``(sum of degrees) ** 2`` undirected and
    ``(sum in)(sum out)`` directed.

    Args:
        linkstream: The temporal network.
        nodes: Nodes to sum over.
        partite_mapping: Same-partite pairs are skipped.
        time_factor: Callable ``(node1, node2) -> float``.

    Returns:
        The normalised sum.
    """
    total = 0.0
    for node1 in nodes:
        for node2 in nodes:
            if partite_mapping.get(node1, -1) == partite_mapping.get(node2, -2):
                continue
            if linkstream.directed:
                degrees_part = linkstream.degrees_in.get(node1, 0) * linkstream.degrees_out.get(
                    node2, 0
                )
            else:
                degrees_part = linkstream.degrees[node1] * linkstream.degrees[node2]
            total += degrees_part * time_factor(node1, node2)

    scale = 2 if linkstream.directed else 4
    return total / (scale * linkstream.weight * linkstream.network_duration)


def reference_expectation_mm(linkstream, M0_leaves, Mx_leaves, partite_mapping) -> float:
    """Mean-Membership expectation delta, written out independently.

    Independent of ``delta_lm`` on purpose -- it reads only the link stream and
    ``get_nodes_durations``. Comparing the production function against another of
    its own code paths would not notice a rewrite that replaces the whole
    function, which is what this file exists to police.

    The per-pair time factor is the change in the geometric mean of the two
    nodes' durations, ``sqrt(u1*u2) - sqrt(r1*r2)``, where u and r are the
    durations in ``Mx U M0`` and in ``Mx``.
    """
    union = M0_leaves | Mx_leaves
    durations_union = get_nodes_durations(union)
    durations_mx = get_nodes_durations(Mx_leaves)

    def factor(node1, node2):
        return (
            durations_union.get(node1, 0) ** 0.5 * durations_union.get(node2, 0) ** 0.5
            - durations_mx.get(node1, 0) ** 0.5 * durations_mx.get(node2, 0) ** 0.5
        )

    return _ordered_pair_sum(linkstream, {leaf.node for leaf in union}, partite_mapping, factor)


def reference_expectation_jm(linkstream, M0_leaves, Mx_leaves, partite_mapping) -> float:
    """Joint-Membership expectation delta, written out independently.

    Same ordered-pair sum as MM, but the time factor is the community's own
    duration, which does not depend on the pair: ``T_union`` minus ``T_mx`` for
    pairs already wholly inside Mx.
    """
    duration_union = get_module_duration(M0_leaves | Mx_leaves)
    duration_mx = get_module_duration(Mx_leaves)
    nodes_union = {leaf.node for leaf in M0_leaves | Mx_leaves}
    nodes_mx = {leaf.node for leaf in Mx_leaves}

    def factor(node1, node2):
        both_in_mx = node1 in nodes_mx and node2 in nodes_mx
        return duration_union - (duration_mx if both_in_mx else 0)

    return _ordered_pair_sum(linkstream, nodes_union, partite_mapping, factor)


# =============================================================================
# Streams -- one per mode, small enough to run in milliseconds
# =============================================================================


def _instantaneous() -> LinkStream:
    ls = LinkStream()
    ls.add_links(
        [(a, b, t) for t in range(4) for a, b in ((0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (2, 3))]
    )
    return ls


def _weighted() -> LinkStream:
    ls = LinkStream()
    ls.add_links(
        [
            (a, b, t, w)
            for t in range(4)
            for (a, b), w in (((0, 1), 2.5), ((1, 2), 0.3), ((0, 2), 1.7), ((3, 4), 2.2),
                              ((4, 5), 0.9), ((2, 3), 1.1))
        ]
    )
    return ls


def _directed() -> LinkStream:
    ls = LinkStream(directed=True)
    ls.add_links(
        [(a, b, t) for t in range(4) for a, b in ((0, 1), (1, 0), (1, 2), (3, 4), (4, 3), (2, 3))]
    )
    return ls


def _bipartite() -> LinkStream:
    mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    ls = LinkStream(partite_mapping=mapping)
    ls.add_links([(a, b, t) for t in range(4) for a in (0, 1, 2) for b in (3, 4, 5)])
    return ls


def _bipartite_directed() -> LinkStream:
    mapping = {0: 0, 1: 0, 2: 1, 3: 1}
    ls = LinkStream(directed=True, partite_mapping=mapping)
    ls.add_links([(a, b, t) for t in range(4) for a in (0, 1) for b in (2, 3)])
    return ls


def _continuous() -> LinkStream:
    ls = LinkStream(continuous=True)
    ls.add_links(
        [(0, 1, 0, 2), (1, 2, 1, 3), (0, 2, 2, 2), (3, 4, 0, 4), (4, 5, 2, 1), (2, 3, 3, 2)]
    )
    return ls


def _delayed() -> LinkStream:
    ls = LinkStream(delayed=True)
    ls.add_links(
        [(0, 1, 0, 1), (1, 2, 1, 2), (0, 2, 0, 2), (3, 4, 1, 3), (4, 5, 2, 3), (2, 3, 2, 4)]
    )
    return ls


STREAMS = {
    "instantaneous": _instantaneous,
    "weighted": _weighted,
    "directed": _directed,
    "bipartite": _bipartite,
    "bipartite_directed": _bipartite_directed,
    "continuous": _continuous,
    "delayed": _delayed,
}


# =============================================================================
# Helpers
# =============================================================================


def _collect_comparisons(stream: LinkStream, **lago_kwargs) -> list[tuple[float, float]]:
    """Run LAGO, returning (production, reference) for every expectation call."""
    original = Computer._get_expectation_mm_part
    pairs: list[tuple[float, float]] = []

    def shadow(self, M0_leaves, Mx_leaves, partite_mapping, mx_module=None, union_module=None):
        produced = original(self, M0_leaves, Mx_leaves, partite_mapping, mx_module, union_module)
        expected = reference_expectation_mm(
            self.linkstream, M0_leaves, Mx_leaves, partite_mapping
        )
        pairs.append((produced, expected))
        # Follow the reference value so both arms always see the same state
        return expected

    Computer._get_expectation_mm_part = shadow
    try:
        lago_modules(stream, lex="MM", **lago_kwargs)
    finally:
        Computer._get_expectation_mm_part = original
    return pairs


def _worst_relative(pairs: list[tuple[float, float]]) -> float:
    worst = 0.0
    for fast, loop in pairs:
        if fast == loop:
            continue
        worst = max(worst, abs(fast - loop) / max(abs(loop), 1e-300))
    return worst


# =============================================================================
# Tests
# =============================================================================


class TestFastPathMatchesPairLoop:
    """The O(|M0|) path must agree with the O(n**2) enumeration, everywhere."""

    @pytest.mark.parametrize("name", sorted(STREAMS))
    @pytest.mark.parametrize("refinement", ["STEM", "STNM", None])
    def test_agrees_with_reference_on_every_call(self, name: str, refinement: str | None) -> None:
        pairs = _collect_comparisons(STREAMS[name](), refinement=refinement)
        assert pairs, f"{name}/{refinement}: no expectation calls were made"
        worst = _worst_relative(pairs)
        assert worst < TOLERANCE, (
            f"{name}/{refinement}: {len(pairs)} calls, worst relative difference {worst:.3e}"
        )

    @pytest.mark.parametrize("name", sorted(STREAMS))
    def test_pair_loop_path_also_agrees(self, name: str) -> None:
        """The in-module fallback loop must match the reference too.

        The fast path is only reachable when a module hint is supplied; this
        pins the path taken when one is not.
        """
        stream = STREAMS[name]()
        original = Computer._get_expectation_mm_part
        worst = 0.0

        def shadow(self, M0, Mx, partite_mapping, mx_module=None, union_module=None):
            nonlocal worst
            saved, self._mm_closed_form = self._mm_closed_form, False
            try:
                looped = original(self, M0, Mx, partite_mapping, None, None)
            finally:
                self._mm_closed_form = saved
            expected = reference_expectation_mm(self.linkstream, M0, Mx, partite_mapping)
            if looped != expected:
                worst = max(worst, abs(looped - expected) / max(abs(expected), 1e-300))
            return expected

        Computer._get_expectation_mm_part = shadow
        try:
            lago_modules(stream, lex="MM")
        finally:
            Computer._get_expectation_mm_part = original
        assert worst < TOLERANCE, f"{name}: fallback loop differs by {worst:.3e}"

    def test_directed_is_not_off_by_a_constant_factor(self) -> None:
        """Catches a wrong factor in the directed form, which a ratio makes obvious.

        The directed aggregate is 2 * (sum in*sqrt(d))(sum out*sqrt(d)); dropping
        the 2 halves every value while leaving the search plausible.
        """
        pairs = [(f, r) for f, r in _collect_comparisons(_directed()) if r]
        assert pairs, "no non-zero directed calls"
        ratios = {round(fast / expected, 6) for fast, expected in pairs}
        assert ratios == {1.0}, f"directed fast path is off by {sorted(ratios)}"


class TestKPartiteMaskIsApplied:
    """The k-partite mask must be reproduced, not dropped.

    Dropping it is silent: values stay finite and plausible, and the search can
    still land on the same partition. Only a direct comparison exposes it.
    """

    @staticmethod
    def _stream_where_the_mask_bites(directed: bool) -> LinkStream:
        """A k-partite stream on which dropping the mask is actually detectable.

        Two properties are needed, and neither is automatic:

        * **Directed streams must have edges in both directions.** The masked
          pairs are within-partite, and their contribution is ``in_i * out_j``.
          In the usual directed k-partite network -- every node purely a source
          or purely a sink -- one of those factors is always zero, so the degrees
          already do the masking and the mask itself is unobservable.
        * **M0 must extend the module's time span**, or JM's per-pair factor
          ``T_union - T_mx`` is zero for every pair and both answers are 0.0.
        """
        mapping = {0: 0, 1: 0, 2: 1, 3: 1}
        # Directed: edges both ways across the partites, so every node has both
        # an in- and an out-degree and the masked pairs are not already zero.
        pairs = (
            ((0, 2), (2, 1), (1, 3), (3, 0))
            if directed
            else ((0, 2), (0, 3), (1, 2), (1, 3))
        )
        stream = LinkStream(directed=directed, partite_mapping=mapping)
        # times 0..2 form the module; the late link at t=9 is what M0 adds
        stream.add_links([(a, b, t) for t in range(3) for a, b in pairs] + [(0, 2, 9)])
        return stream

    @pytest.mark.parametrize("directed", [False, True], ids=["undirected", "directed"])
    @pytest.mark.parametrize("lex", ["MM", "JM"])
    def test_masked_and_unmasked_differ(self, directed: bool, lex: str) -> None:
        """Computing with and without the mask must give different answers.

        Built by hand rather than harvested from a run: whether a search happens
        to reach a call where the mask bites depends on its exploration order, so
        driving this through `lago_modules` would make the test's own premise a
        matter of luck -- it did, before the exploration order became seedable.
        """
        stream = self._stream_where_the_mask_bites(directed)
        computer = Computer(stream, lex=lex, gamma=1, omega=2)
        leaves = stream.leaves_dict

        M0 = {leaves[(0, 9)]}
        Mx = {leaf for key, leaf in leaves.items() if key[1] < 3}
        assert {stream.partite_mapping[leaf.node] for leaf in M0 | Mx} == {0, 1}

        method = getattr(
            computer,
            "_get_expectation_mm_part" if lex == "MM" else "_get_expectation_jm_part",
        )
        masked = method(M0, Mx, stream.partite_mapping)
        unmasked = method(M0, Mx, {})
        assert masked != unmasked, (
            f"directed={directed}/{lex}: the partite mask changed nothing, so a "
            "rewrite that dropped it would pass unnoticed"
        )


class TestNoOpMoveIsExactlyZero:
    """A move that changes no duration must return exactly 0.0, not float noise.

    Computing the delta as a difference of two independently accumulated sums
    leaves a residue of order the sums' own rounding -- around 1e-15 -- where the
    true value is 0. find_best_move compares deltas against a stopping criterion
    and takes a max(), so a spurious non-zero can decide a tie.
    """

    def test_empty_submodule_gives_exact_zero(self) -> None:
        from lago.algorithm._internal._lago_module import _LagoModule

        stream = _instantaneous()
        computer = Computer(stream, lex="MM", gamma=1, omega=2)
        leaves = set(stream.leaves_dict.values())
        module = _LagoModule(set(leaves))

        # Moving nothing into the module changes no duration.
        value = computer._get_expectation_mm_part(set(), module.leaves, {}, mx_module=module)
        assert value == 0.0, f"expected exactly 0.0, got {value!r}"

    @pytest.mark.parametrize("name", sorted(STREAMS))
    def test_zero_deltas_stay_exactly_zero(self, name: str) -> None:
        """Where the reference yields exactly 0, the fast path must too."""
        pairs = _collect_comparisons(STREAMS[name]())
        mismatched = [(fast, ref) for fast, ref in pairs if ref == 0.0 and fast != 0.0]
        assert not mismatched, (
            f"{name}: {len(mismatched)} calls returned float noise instead of 0.0, "
            f"e.g. {mismatched[0][0]!r}"
        )


class TestJointMembershipMatchesReference:
    """The JM expectation delta, including the k-partite branch.

    `_get_expectation_jm_kpartite_part` is a separate implementation from the
    plain branch -- it enumerates pairs where the plain one squares a sum -- so
    the two can drift apart without anything else noticing. They did: the
    k-partite branch was missing the `2 ** (n1 != n2)` factor and returned
    exactly half the plain convention.
    """

    @pytest.mark.parametrize("name", sorted(STREAMS))
    @pytest.mark.parametrize("refinement", ["STEM", None])
    def test_agrees_with_reference_on_every_call(self, name: str, refinement: str | None) -> None:
        stream = STREAMS[name]()
        original = Computer._get_expectation_jm_part
        worst = 0.0
        calls = 0

        def shadow(self, M0_leaves, Mx_leaves, partite_mapping):
            nonlocal worst, calls
            produced = original(self, M0_leaves, Mx_leaves, partite_mapping)
            expected = reference_expectation_jm(
                self.linkstream, M0_leaves, Mx_leaves, partite_mapping
            )
            calls += 1
            if produced != expected:
                worst = max(worst, abs(produced - expected) / max(abs(expected), 1e-300))
            return expected

        Computer._get_expectation_jm_part = shadow
        try:
            lago_modules(stream, lex="JM", refinement=refinement)
        finally:
            Computer._get_expectation_jm_part = original

        assert calls, f"{name}/{refinement}: no JM expectation calls were made"
        assert worst < TOLERANCE, (
            f"{name}/{refinement}: {calls} calls, worst relative difference {worst:.3e}"
        )

    @pytest.mark.parametrize("name", ["bipartite", "bipartite_directed"])
    def test_kpartite_branch_uses_the_same_convention_as_the_plain_branch(self, name: str) -> None:
        """Pins the factor the k-partite branch was missing.

        A ratio makes a constant factor obvious where an absolute comparison
        would just look like "a different number".
        """
        stream = STREAMS[name]()
        original = Computer._get_expectation_jm_part
        ratios: set[float] = set()

        def shadow(self, M0_leaves, Mx_leaves, partite_mapping):
            produced = original(self, M0_leaves, Mx_leaves, partite_mapping)
            expected = reference_expectation_jm(
                self.linkstream, M0_leaves, Mx_leaves, partite_mapping
            )
            if expected:
                ratios.add(round(produced / expected, 6))
            return expected

        Computer._get_expectation_jm_part = shadow
        try:
            lago_modules(stream, lex="JM")
        finally:
            Computer._get_expectation_jm_part = original

        assert ratios, f"{name}: no non-zero JM k-partite calls"
        assert ratios == {1.0}, f"{name}: k-partite branch is off by {sorted(ratios)}"


class TestDirectedKPartiteIsReachable:
    """A directed k-partite stream must survive JM at all.

    In a directed k-partite network a node normally has only in-edges or only
    out-edges, so reading `degrees_in[node]` rather than `degrees_in.get(node, 0)`
    raises KeyError on the very first candidate move.
    """

    @pytest.mark.parametrize("lex", ["JM", "MM"])
    def test_lago_modules_runs(self, lex: str) -> None:
        modules = lago_modules(_bipartite_directed(), lex=lex, seed=1)
        assert modules.nb_modules >= 1

    @pytest.mark.parametrize("lex", ["JM", "MM"])
    def test_nodes_with_no_in_or_out_edges(self, lex: str) -> None:
        """A pure source and a pure sink, each missing one of the two degrees."""
        stream = LinkStream(directed=True, partite_mapping={0: 0, 1: 1})
        stream.add_links([(0, 1, t) for t in range(3)])
        assert 0 not in stream.degrees_in and 1 not in stream.degrees_out
        modules = lago_modules(stream, lex=lex, seed=1)
        assert modules.nb_modules >= 1
