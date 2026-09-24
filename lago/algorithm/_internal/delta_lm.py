from collections import defaultdict
from itertools import combinations_with_replacement

import lago.core.utils as tls
from lago.core.linkstream import LinkStream

from ._leaf import Leaf
from .leaf_set import LeafDifference


class DeltaLongitudinalModularityComputer:
    """Computer of delta longitudinal modularity
    when moving submodules for LAGO.

    Two entry points compute the same numbers:

    * :meth:`M0_to_Mx` -- one candidate at a time. The original formulation and
      the reference the tests and ``benchmarks/compare_candidates.py`` hold the
      other one to.
    * :meth:`evaluate_candidates` -- every candidate of a move in one pass over
      M0. What ``find_best_module_for_submodule`` uses when :attr:`use_batch`
      is set (the default).
    """

    # Switched off by the harnesses to run the per-candidate reference instead.
    use_batch: bool = True

    def __init__(
        self,
        linkstream: LinkStream,
        lex: str = "JM",
        gamma: float = 1,
        omega: float = 2,
    ):
        self.linkstream = linkstream
        self.lex = lex
        self.gamma = gamma
        self.omega = omega
        # The MM expectation delta has an O(n) closed form (see
        # _expectation_mm_closed). It needs the k-partite mask to be a genuine
        # grouping, which it is unless some node carries one of the sentinel
        # values the mask compares against.
        self._mm_closed_form = not any(
            value in (-1, -2) for value in linkstream.partite_mapping.values()
        )
        # With unit weights every edge sum is an exact integer, so the order in
        # which a leaf's edges are visited cannot change it. That lets the batch
        # evaluation walk a leaf's edge set directly instead of allocating a
        # union per leaf per move where the union would only fix the float
        # summation order (see evaluate_candidates for where it does more).
        self._exact_weights = not linkstream.is_weighted

    def M0_to_Mx(
        self,
        M0_leaves: set[Leaf],
        M0_time_segments: dict[int, list[list[Leaf]]],
        Mx_leaves: set[Leaf],
        partite_mapping: dict[int, int],
        mx_module=None,
        union_module=None,
    ):
        """Compute Δ L-Modularity of submodule M0 joining module Mx.
            L-Modularity is made up of three terms, each computed in a dedicated function:
            - Number of edges within modules
            - Expected number of edges
            - Community Switch Counts (term for time regularisation)
            Note that it returns a non normalized Δ L-Modularity.

        Args:
            M0_leaves (set): set of M0 leaves (time nodes)
            M0_time_segments (dict): time segments of nodes leaving in module M0
            Mx_leaves (set): set of Mx leaves
            mx_module (Module, optional): the module whose leaves are exactly
                Mx_leaves, if any, so its memoised durations can be reused.
            union_module (Module, optional): the module whose leaves are exactly
                Mx_leaves | M0_leaves, if any, likewise.

        Returns:
            float: Δ L-Modularity of the movement
        """

        weight_diff = self._get_weight_diff(M0_leaves, Mx_leaves)
        if not self.linkstream.directed:
            weight_diff *= 2

        if self.gamma == 0:
            # The expectation term is multiplied by gamma below, so skip the
            # whole null model. longitudinal_modularity already does this.
            expectation_diff = 0
        elif self.lex == "MM":
            expectation_diff = self._get_expectation_mm_part(
                M0_leaves,
                Mx_leaves,
                partite_mapping,
                mx_module,
                union_module,
            )
        elif self.lex == "JM":
            expectation_diff = self._get_expectation_jm_part(
                M0_leaves,
                Mx_leaves,
                partite_mapping,
            )
        else:
            raise Exception("Wrong lex value")

        csc_diff = (
            0.0 if self.omega == 0 else self._get_csc_diff(M0_time_segments, Mx_leaves)
        )

        ls = self.linkstream

        delta_lm = (
            weight_diff
            - self.gamma * expectation_diff
            + self.omega * csc_diff * ls.weight / ls.nb_edges
        )

        return delta_lm

    # ------------------------------------------------------------------
    # Batch evaluation: every candidate of a move in one pass over M0
    #
    # The per-candidate path walks M0's edges once per candidate and tests each
    # neighbour for membership in the candidate's leaf set. But parent modules
    # partition the leaves, so a neighbour belongs to a candidate exactly when
    # that candidate *owns* it -- `leaf.module.parent` during a TMM/STNM level,
    # `leaf.module` during STEM -- and one walk over M0's edges can therefore
    # credit every candidate at once. The same goes for the time neighbours the
    # switch count looks at, and for the two time neighbours of each M0 leaf
    # that decide its duration change.
    #
    # Nothing is reordered: each candidate's weight is the same subsequence of
    # additions as before, the switch count is an integer, and the duration
    # changes are exact integers, so the results are bit-identical.
    # ------------------------------------------------------------------

    def evaluate_candidates(
        self,
        M0_leaves: set[Leaf],
        M0_time_segments: dict[int, list[list[Leaf]]],
        parent,
        via_parent: bool,
        partite_mapping: dict[int, int],
        candidates=None,
    ):
        """Δ L-modularity of taking M0 out of ``parent`` and into each candidate.

        Args:
            M0_leaves: The time-nodes being moved.
            M0_time_segments: Their per-node segments (``lago_tools.get_nodes_segment``).
            parent: The module M0 currently belongs to.
            via_parent: Whether a leaf's owning module is ``leaf.module.parent``
                (TMM / STNM levels) rather than ``leaf.module`` (STEM).
            partite_mapping: The stream's k-partite mapping.
            candidates: The candidate modules. ``None`` derives them live, the
                way STEM's ``get_neighbors_modules_parents`` does: owners reached
                from M0 by an out-edge or by a time neighbour, minus ``parent``.
                TMM/STNM levels pass the parents of the neighbours recorded when
                the level started instead (see ``find_best_module_for_submodule``).

        Returns:
            ``(candidates, leaving_delta, leaving_changes, evaluations)`` where
            ``evaluations`` is a list of ``(module, joining_delta, joining_changes)``
            and the ``*_changes`` are the per-node ``[before, after]`` durations
            of the side that changes (``None`` when not computed on this path).
        """
        linkstream = self.linkstream
        directed = linkstream.directed
        gamma = self.gamma
        omega = self.omega
        weight = linkstream.weight
        nb_edges = linkstream.nb_edges
        in_M0 = M0_leaves.__contains__
        derive = candidates is None

        # One pass over M0's edges. `weights[owner]` accumulates in exactly the
        # order _get_weight_diff would have visited the same edges. Targets in M0
        # that the parent owns are skipped: the leaving move is against
        # parent - M0.
        weights: dict = {}
        eligible: set = set()
        exact = self._exact_weights
        for leaf in M0_leaves:
            if exact and not leaf.topo_neighbors_from:
                # Integer sums: order is immaterial, skip building the union.
                # Only when there is nothing to unite with: on directed streams
                # the union also *merges* an out-edge with a reciprocal in-edge
                # of equal weight and duration, so such a pair counts once --
                # the behaviour _get_weight_diff has always had, kept as is.
                edges = leaf.topo_neighbors
            else:
                edges = leaf.topo_neighbors | leaf.topo_neighbors_from
            for edge in edges:
                target = edge.target
                owner = target.module.parent if via_parent else target.module
                if owner is parent and in_M0(target):
                    continue
                weights[owner] = weights.get(owner, 0) + edge.weight * edge.duration
            if derive and directed:
                # Candidates are reached by out-edges only (compute_neighbors and
                # get_neighbors_modules_parents both walk topo_neighbors), while
                # weights count both directions. Keep that asymmetry.
                for edge in leaf.topo_neighbors:
                    target = edge.target
                    eligible.add(target.module.parent if via_parent else target.module)
        if derive and not directed:
            eligible.update(weights)

        # Time neighbours of M0's segment endpoints: they are the time neighbours
        # of M0 that lie outside it (segments are maximal), which is both what the
        # switch count inspects and what makes a module a candidate by time.
        endpoint_total = 0
        endpoint_counts: dict = {}
        for segments in M0_time_segments.values():
            for segment in segments:
                for neighbour in (
                    segment[0].left_time_active_neighbor,
                    segment[-1].right_time_active_neighbor,
                ):
                    if neighbour is None:
                        continue
                    owner = neighbour.module.parent if via_parent else neighbour.module
                    endpoint_total += 1
                    endpoint_counts[owner] = endpoint_counts.get(owner, 0) + 1
                    if derive:
                        eligible.add(owner)

        if derive:
            eligible.discard(parent)
            candidates = eligible
        else:
            candidates = [module for module in candidates if module is not parent]

        # MM closed form: the per-candidate duration changes depend on the
        # candidate only through whether it owns each M0 leaf's two time
        # neighbours, so everything else is tabulated once.
        mm_fast = gamma != 0 and self.lex == "MM" and self._mm_closed_form
        table = None
        if mm_fast:
            rank = {leaf: index for index, leaf in enumerate(M0_leaves)}
            table = []
            for leaf in M0_leaves:
                left = leaf.left_time_active_neighbor
                right = leaf.right_time_active_neighbor
                left_owner = right_owner = None
                if left is not None:
                    left_owner = left.module.parent if via_parent else left.module
                if right is not None:
                    right_owner = right.module.parent if via_parent else right.module
                table.append(
                    (
                        leaf.node,
                        leaf.time,
                        leaf.edge_duration,
                        left,
                        left_owner,
                        rank.get(left),
                        right,
                        right_owner,
                        rank.get(right),
                    )
                )

        # JM: the same idea with the module's degree sum and time span, both
        # maintained on the module (see _JMAggregate); M0's own summary once.
        jm_fast = gamma != 0 and self.lex == "JM" and self._mm_closed_form
        m0_jm = self._m0_jm_summary(M0_leaves) if jm_fast else None

        # The view over parent - M0 is only needed by the per-candidate
        # expectation paths (k-partite sentinels); built on demand.
        M1_leaves = None
        m1_empty = len(parent.leaves) == len(M0_leaves)

        # --- M0 leaving its parent
        weight_diff = weights.get(parent, 0)
        if not directed:
            weight_diff *= 2
        leaving_changes = None
        if gamma == 0:
            expectation_diff = 0
        elif mm_fast:
            sums_union = self._module_mm_sums(parent)
            leaving_changes = self._changes_from_table(
                table, parent, parent.get_durations(tls.get_nodes_durations), removing=True
            )
            expectation_diff = self._mm_from_sums(
                sums_union, self._shift_mm_sums(sums_union, leaving_changes)
            )
        elif jm_fast:
            expectation_diff = self._jm_leaving(parent, m0_jm)
        else:
            M1_leaves = LeafDifference(parent.leaves, M0_leaves)
            if self.lex == "MM":
                expectation_diff = self._get_expectation_mm_part(
                    M0_leaves, M1_leaves, partite_mapping, None, parent
                )
            else:
                expectation_diff = self._get_expectation_jm_part(
                    M0_leaves, M1_leaves, partite_mapping
                )
        csc_diff = (
            0.0 if omega == 0 else -(endpoint_total - endpoint_counts.get(parent, 0)) / 2
        )
        leaving_delta = -(
            weight_diff - gamma * expectation_diff + omega * csc_diff * weight / nb_edges
        )

        # --- M0 joining each candidate
        evaluations = []
        for module in candidates:
            if not module.leaves.isdisjoint(M0_leaves):
                # Not a partition any more; the owner argument does not apply.
                # Never happens with the movers as written -- kept as the exact
                # original behaviour rather than an assumption.
                M2_leaves = module.leaves - M0_leaves
                if not M2_leaves and m1_empty:
                    continue
                joining_delta = self.M0_to_Mx(
                    M0_leaves, M0_time_segments, M2_leaves, partite_mapping, mx_module=None
                )
                evaluations.append((module, joining_delta, None))
                continue

            if not module.leaves and m1_empty:
                continue

            weight_diff = weights.get(module, 0)
            if not directed:
                weight_diff *= 2
            joining_changes = None
            if gamma == 0:
                expectation_diff = 0
            elif mm_fast:
                sums_mx = self._module_mm_sums(module)
                joining_changes = self._changes_from_table(
                    table, module, module.get_durations(tls.get_nodes_durations), removing=False
                )
                expectation_diff = self._mm_from_sums(
                    self._shift_mm_sums(sums_mx, joining_changes), sums_mx
                )
            elif jm_fast:
                expectation_diff = self._jm_joining(module, m0_jm)
            elif self.lex == "MM":
                expectation_diff = self._get_expectation_mm_part(
                    M0_leaves, module.leaves, partite_mapping, module, None
                )
            else:
                expectation_diff = self._get_expectation_jm_part(
                    M0_leaves, module.leaves, partite_mapping
                )
            csc_diff = (
                0.0 if omega == 0 else -(endpoint_total - endpoint_counts.get(module, 0)) / 2
            )
            joining_delta = (
                weight_diff - gamma * expectation_diff + omega * csc_diff * weight / nb_edges
            )
            evaluations.append((module, joining_delta, joining_changes))

        return candidates, leaving_delta, leaving_changes, evaluations

    @staticmethod
    def _changes_from_table(table, side, base_durations, removing: bool) -> dict:
        """Per-node ``[before, after]`` durations for one side of a move.

        The tabulated form of :meth:`_duration_changes_adding` /
        :meth:`_duration_changes_removing`: membership of a time neighbour in
        the growing (or shrinking) set is ``side`` owning it, or it being an M0
        leaf already processed. M0 leaves are processed in the table's order,
        which is M0's iteration order, so the nodes enter ``changes`` in the
        same order as before and the later float sums see the same sequence.

        Args:
            table: One row per M0 leaf, see :meth:`evaluate_candidates`.
            side: The module whose ownership means membership.
            base_durations: That module's memoised per-node durations.
            removing: Whether M0 is leaving ``side`` rather than joining it.
        """
        changes: dict[int, list[float]] = {}
        for index, (node, time, duration, left, left_owner, left_rank, right, right_owner, right_rank) in enumerate(table):
            entry = changes.get(node)
            if entry is None:
                before = base_durations.get(node, 0)
                entry = [before, before]
                changes[node] = entry

            if removing:
                # The set is side's leaves minus the M0 leaves processed so far,
                # the current one included.
                has_left = (
                    left is not None
                    and left_owner is side
                    and not (left_rank is not None and left_rank <= index)
                )
                has_right = (
                    right is not None
                    and right_owner is side
                    and not (right_rank is not None and right_rank <= index)
                )
            else:
                # The set is side's leaves plus the M0 leaves processed so far.
                has_left = left is not None and (
                    left_owner is side or (left_rank is not None and left_rank < index)
                )
                has_right = right is not None and (
                    right_owner is side or (right_rank is not None and right_rank < index)
                )

            # duration_delta_on_add, inlined.
            if has_left:
                if has_right:
                    delta = right.time - left.time - left.edge_duration
                else:
                    delta = time - left.time - left.edge_duration + duration
            elif has_right:
                delta = right.time - time
            else:
                delta = duration

            entry[1] += -delta if removing else delta
        return changes

    # ------------------------------------------------------------------
    # JM expectation: O(|M0|) path
    #
    # JM scores a leaf set by (sum of its nodes' degrees)**2 times its time
    # span -- (sum in)(sum out) when directed, minus the same product per
    # partite when k-partite. Both factors are exact integers of the set's
    # content; a module keeps them current across moves (_JMAggregate), and
    # M0 changes them only through its own nodes and its own extreme instants.
    # ------------------------------------------------------------------

    @staticmethod
    def _m0_jm_summary(M0_leaves: set[Leaf]) -> tuple:
        """Node counts, start/end instant counts and span of M0, once per move."""
        node_counts: dict[int, int] = {}
        start_counts: dict[int, int] = {}
        end_counts: dict[int, int] = {}
        first = last = None
        for leaf in M0_leaves:
            node_counts[leaf.node] = node_counts.get(leaf.node, 0) + 1
            if not leaf.topo_neighbors and not leaf.topo_neighbors_from:
                continue
            start = leaf.time
            end = start + leaf.edge_duration
            start_counts[start] = start_counts.get(start, 0) + 1
            end_counts[end] = end_counts.get(end, 0) + 1
            if first is None or start < first:
                first = start
            if last is None or end > last:
                last = end
        return node_counts, start_counts, end_counts, first, last

    def _jm_sums(self, node_counts: dict[int, int]) -> tuple[float, float, dict]:
        """Degree aggregates over a set of nodes: ``(a, b, groups)``.

        The same shape as :meth:`_mm_sums` without the duration factor: ``a``
        and ``b`` are the in and out degree sums (both the plain degree sum
        when undirected), ``groups`` the same pair per partite. Summed in
        sorted node order, so the value is a function of the node set alone.
        Integer degrees stay integers.
        """
        linkstream = self.linkstream
        partite_mapping = linkstream.partite_mapping
        groups: dict[int, list] = {}
        total_a = total_b = 0

        if linkstream.directed:
            degrees_in = linkstream.degrees_in
            degrees_out = linkstream.degrees_out
            for node in sorted(node_counts):
                value_a = degrees_in.get(node, 0)
                value_b = degrees_out.get(node, 0)
                total_a += value_a
                total_b += value_b
                partite = partite_mapping.get(node)
                if partite is not None:
                    group = groups.get(partite)
                    if group is None:
                        groups[partite] = [value_a, value_b]
                    else:
                        group[0] += value_a
                        group[1] += value_b
        else:
            degrees = linkstream.degrees
            for node in sorted(node_counts):
                value = degrees[node]
                total_a += value
                total_b += value
                partite = partite_mapping.get(node)
                if partite is not None:
                    group = groups.get(partite)
                    if group is None:
                        groups[partite] = [value, value]
                    else:
                        group[0] += value
                        group[1] += value

        return total_a, total_b, groups

    def _module_jm_sums(self, module) -> tuple:
        """Memoised :meth:`_jm_sums` of a module's nodes, with its aggregate."""
        aggregate = module.get_jm()
        sums = aggregate.sums
        if sums is None:
            sums = aggregate.sums = self._jm_sums(aggregate.node_counts)
        return aggregate, sums

    def _shift_jm_sums(self, sums: tuple, nodes: list, adding: bool) -> tuple:
        """The aggregates once ``nodes`` (sorted) enter or leave the node set."""
        total_a, total_b, groups = sums
        linkstream = self.linkstream
        partite_mapping = linkstream.partite_mapping
        directed = linkstream.directed
        degrees_in = linkstream.degrees_in
        degrees_out = linkstream.degrees_out
        degrees = linkstream.degrees
        new_groups = {key: list(value) for key, value in groups.items()} if groups else {}

        for node in nodes:
            if directed:
                value_a = degrees_in.get(node, 0)
                value_b = degrees_out.get(node, 0)
            else:
                value_a = value_b = degrees[node]
            if not adding:
                value_a = -value_a
                value_b = -value_b
            total_a += value_a
            total_b += value_b
            partite = partite_mapping.get(node)
            if partite is not None:
                group = new_groups.get(partite)
                if group is None:
                    new_groups[partite] = [value_a, value_b]
                else:
                    group[0] += value_a
                    group[1] += value_b

        return total_a, total_b, new_groups

    def _jm_from_sums(self, sums_union, span_union, sums_mx, span_mx) -> float:
        """Combine the two sides' aggregates and spans into the expectation delta.

        ``(sum in)(sum out) * span`` on each side, minus the same per partite,
        which removes the pairs inside a partite exactly as the pair loop's
        mask does (a node without a partite is never masked, itself included).
        """
        union_a, union_b, union_groups = sums_union
        mx_a, mx_b, mx_groups = sums_mx

        total = union_a * union_b * span_union - mx_a * mx_b * span_mx
        for partite, group in union_groups.items():
            other = mx_groups.get(partite)
            if other is None:
                total -= group[0] * group[1] * span_union
            else:
                total -= group[0] * group[1] * span_union - other[0] * other[1] * span_mx
        for partite, other in mx_groups.items():
            if partite not in union_groups:
                total += other[0] * other[1] * span_mx

        linkstream = self.linkstream
        denominator = (2 if linkstream.directed else 4) * linkstream.weight
        return total / (denominator * linkstream.network_duration)

    @staticmethod
    def _span_length(first, last) -> int:
        return 0 if first is None else last - first + 1

    def _jm_joining(self, module, m0_jm) -> float:
        """JM expectation delta of M0 joining ``module``, in O(|M0|)."""
        node_counts, _, _, m0_first, m0_last = m0_jm
        aggregate, sums_mx = self._module_jm_sums(module)
        added = sorted(node for node in node_counts if node not in aggregate.node_counts)
        sums_union = self._shift_jm_sums(sums_mx, added, adding=True) if added else sums_mx

        first, last = aggregate.span()
        if first is None:
            union_first, union_last = m0_first, m0_last
        elif m0_first is None:
            union_first, union_last = first, last
        else:
            union_first = first if first < m0_first else m0_first
            union_last = last if last > m0_last else m0_last

        return self._jm_from_sums(
            sums_union,
            self._span_length(union_first, union_last),
            sums_mx,
            self._span_length(first, last),
        )

    def _jm_leaving(self, parent, m0_jm) -> float:
        """JM expectation delta of M0 leaving ``parent`` (union side), in O(|M0|)."""
        node_counts, start_counts, end_counts, _, _ = m0_jm
        aggregate, sums_union = self._module_jm_sums(parent)
        parent_counts = aggregate.node_counts
        removed = sorted(node for node, count in node_counts.items() if parent_counts[node] == count)
        sums_mx = self._shift_jm_sums(sums_union, removed, adding=False) if removed else sums_union

        first, last = aggregate.span()
        mx_first, mx_last = aggregate.span_without(start_counts, end_counts)
        return self._jm_from_sums(
            sums_union,
            self._span_length(first, last),
            sums_mx,
            self._span_length(mx_first, mx_last),
        )

    def _get_weight_diff(self, M0_leaves: set[Leaf], Mx_leaves: set[Leaf]):
        """Compute delta number of edges within module Mx if M0 joins Mx.

        Walks M0, not Mx. Every edge between the two sets is recorded on both of
        its endpoints with the same weight and duration -- `_add_edge_undirected`
        and `_add_edge_directed` each store one TimeEdge per side -- so summing
        M0's edges that land in Mx covers exactly the same interactions as
        summing Mx's edges that land in M0. M0 holds one or two time-nodes where
        Mx can hold hundreds, which is the whole point.

        Args:
            M0_leaves (set): M0 submodule time nodes
            Mx_leaves (set): Mx module time nodes

        Returns:
            int: delta number of edges
        """
        total = 0
        contains = Mx_leaves.__contains__
        for leaf in M0_leaves:
            for neighb in leaf.topo_neighbors | leaf.topo_neighbors_from:
                if contains(neighb.target):
                    total += neighb.weight * neighb.duration
        return total

    def _get_csc_diff(self, M0_time_segments, Mx_leaves) -> float:
        """Compute the delta CSCs (Community Switches Counts)
        if module Mx is joined by submodule M0.
        CSC for a node is the number of time it leaves and enter time module, minus 1.

        Args:
            M0_time_segments (dict): every node time segment for
                the time it belongs to module M0.
            Mx_leaves (set): Leaves (time nodes) of module Mx.

        Returns:
            float: delta CSC (Community Switch Counts)
        """

        cscs = 0
        # Iterate on node and its time segments when belonging to M0
        for _, segments in M0_time_segments.items():
            for segment in segments:
                # Get the previous active time occurence of the node
                # which is left time neighbor
                neighb_left = segment[0].left_time_active_neighbor
                # If it exists and not belong to Mx, the joined module,
                # it is counted as a leave/enter configuration for the node.
                if neighb_left and neighb_left not in Mx_leaves:
                    cscs += 1

                # Get the next active time occurence of the node
                # which is right time neighbor
                neighb_right = segment[-1].right_time_active_neighbor
                # If it exists and not belong to Mx, the joined module,
                # it is counted as a leave/enter configuration for the node.
                if neighb_right and neighb_right not in Mx_leaves:
                    cscs += 1

        # Total cscs are weighted by the time resolution parameter omega
        csc_diff = -cscs / 2

        return csc_diff

    def _get_expectation_jm_part(
        self,
        M0_leaves: set[Leaf],
        Mx_leaves: set[Leaf],
        partite_mapping: dict[int, int],
    ):
        """Compute the delta number of expected edges within module Mx
        if joined by submodule M0, regarding the Joint-Membership expectation.

        Args:
            M0_leaves (set): M0 submodule time nodes
            Mx_leaves (set): Mx module time nodes

        Returns:
            float: delta value for expected number of edges
        """

        if partite_mapping:
            return self._get_expectation_jm_kpartite_part(
                M0_leaves,
                Mx_leaves,
                partite_mapping,
            )

        # Built once: the undirected branch needs it twice, the directed one
        # three times, and it is O(|Mx|) each time.
        union = Mx_leaves | M0_leaves

        duration_Cx = tls.get_module_duration(Mx_leaves)
        duration_Cx_U_C0 = tls.get_module_duration(union)
        if self.linkstream.directed:
            degree_in_Cx = self._sum_degrees_in(Mx_leaves)
            degree_in_Cx_U_C0 = self._sum_degrees_in(union)
            degree_out_Cx = self._sum_degrees_out(Mx_leaves)
            degree_out_Cx_U_C0 = self._sum_degrees_out(union)
            expectation_diff = (
                degree_in_Cx_U_C0 * degree_out_Cx_U_C0 * duration_Cx_U_C0
                - degree_in_Cx * degree_out_Cx * duration_Cx
            ) / (2 * self.linkstream.weight * self.linkstream.network_duration)
            # expectation_diff = 0
            # expectation_diff = (
            #     degree_in_Cx_U_C0 * degree_out_Cx_U_C0 * duration_Cx_U_C0
            #     - degree_in_Cx * degree_out_Cx * duration_Cx
            # ) / (2 * self.linkstream.weight * self.linkstream.network_duration)
        else:
            degree_Cx = self._sum_degrees(Mx_leaves)
            degree_Cx_U_C0 = self._sum_degrees(union)
            expectation_diff = (
                degree_Cx_U_C0**2 * duration_Cx_U_C0 - degree_Cx**2 * duration_Cx
            ) / (4 * self.linkstream.weight * self.linkstream.network_duration)

        return expectation_diff

    def _get_expectation_jm_kpartite_part(
        self,
        M0_leaves: set[Leaf],
        Mx_leaves: set[Leaf],
        partite_mapping: dict[int, int],
    ):
        """Compute the delta number of expected edges within module Mx
        if joined by submodule M0, regarding the Joint-Membership expectation.

        Args:
            M0_leaves (set): M0 submodule time nodes
            Mx_leaves (set): Mx module time nodes

        Returns:
            float: delta value for expected number of edges
        """

        nodes_in_Cx = {leaf.node for leaf in Mx_leaves}
        nodes_in_Cx_U_C0 = [*{*[leaf.node for leaf in Mx_leaves | M0_leaves]}]
        in_Cx = {node: node in nodes_in_Cx for node in nodes_in_Cx_U_C0}
        duration_Cx = tls.get_module_duration(Mx_leaves)
        duration_Cx_U_C0 = tls.get_module_duration(Mx_leaves | M0_leaves)

        if self.linkstream.directed:
            expectation_diff = 0
            for node1, node2 in combinations_with_replacement(nodes_in_Cx_U_C0, 2):
                if partite_mapping.get(node1, -1) == partite_mapping.get(node2, -2):
                    continue
                # .get(): in a directed network a node may have only in-edges or
                # only out-edges, which is the norm in a directed k-partite one.
                # The MM path already reads the degrees this way.
                degree_in1 = self.linkstream.degrees_in.get(node1, 0)
                degree_in2 = self.linkstream.degrees_in.get(node2, 0)
                degree_out1 = self.linkstream.degrees_out.get(node1, 0)
                degree_out2 = self.linkstream.degrees_out.get(node2, 0)

                # This enumeration is upper-triangular but the normalisation is
                # over ordered pairs, matching the plain branch's
                # (sum in)(sum out). Off the diagonal, in1*out2 + in2*out1
                # already covers both directions; on it, in1*out1 is the single
                # ordered pair (n, n) and must not be counted twice.
                if node1 == node2:
                    degrees_part = degree_in1 * degree_out1
                else:
                    degrees_part = degree_in1 * degree_out2 + degree_in2 * degree_out1

                expectation_diff += degrees_part * (
                    duration_Cx_U_C0 - in_Cx[node1] * in_Cx[node2] * duration_Cx
                )
            expectation_diff /= 2 * self.linkstream.weight * self.linkstream.network_duration

        else:
            # In k-partite networks, only interactions between different partites are expected
            expectation_diff = 0
            for node1, node2 in combinations_with_replacement(nodes_in_Cx_U_C0, 2):
                if partite_mapping.get(node1, -1) == partite_mapping.get(node2, -2):
                    continue
                degree1 = self.linkstream.degrees[node1]
                degree2 = self.linkstream.degrees[node2]

                # The 2 ** (node1 != node2) factor turns this upper-triangular
                # enumeration into the ordered-pair sum the plain branch uses,
                # (sum of degrees) ** 2.
                expectation_diff += (
                    2 ** (node1 != node2)
                    * degree1
                    * degree2
                    * (duration_Cx_U_C0 - in_Cx[node1] * in_Cx[node2] * duration_Cx)
                )
            expectation_diff /= 4 * self.linkstream.weight * self.linkstream.network_duration

        return expectation_diff

    # ------------------------------------------------------------------
    # MM expectation: O(|M0|) path
    #
    # The closed form below only needs two aggregates per side -- sum of
    # degree * sqrt(duration), overall and per partite -- not the durations
    # themselves. One side of every candidate move is a whole module, whose
    # aggregates are memoised; the other side differs from it only on the nodes
    # of M0, which holds one or two time-nodes. So a candidate costs O(|M0|)
    # rather than O(|Mx|), which is what lets this scale to large modules.
    # ------------------------------------------------------------------

    def _mm_sums(self, durations) -> tuple[float, float, dict]:
        """Aggregates of ``degree * sqrt(duration)`` over the nodes in ``durations``.

        Returns ``(a, b, groups)``. For undirected streams ``a == b``, so the
        pair sum is ``a * b == a ** 2``; for directed ones ``a`` carries the in
        degrees and ``b`` the out degrees. ``groups`` holds the same pair per
        partite, which is what the k-partite mask subtracts.

        Nodes are summed in sorted order. The durations mapping is maintained
        across moves (``_LagoModule.apply_duration_changes``), so its insertion
        order reflects the module's history; a fixed order makes these float
        sums a function of the module's content alone, identical whether the
        durations were maintained or recomputed.
        """
        linkstream = self.linkstream
        partite_mapping = linkstream.partite_mapping
        groups: dict[int, list[float]] = {}
        total_a = total_b = 0.0

        if linkstream.directed:
            degrees_in = linkstream.degrees_in
            degrees_out = linkstream.degrees_out
            for node in sorted(durations):
                duration = durations[node]
                root = duration**0.5
                value_a = degrees_in.get(node, 0) * root
                value_b = degrees_out.get(node, 0) * root
                total_a += value_a
                total_b += value_b
                partite = partite_mapping.get(node)
                if partite is not None:
                    group = groups.get(partite)
                    if group is None:
                        groups[partite] = [value_a, value_b]
                    else:
                        group[0] += value_a
                        group[1] += value_b
        else:
            degrees = linkstream.degrees
            for node in sorted(durations):
                duration = durations[node]
                value = degrees[node] * duration**0.5
                total_a += value
                total_b += value
                partite = partite_mapping.get(node)
                if partite is not None:
                    group = groups.get(partite)
                    if group is None:
                        groups[partite] = [value, value]
                    else:
                        group[0] += value
                        group[1] += value

        return total_a, total_b, groups

    @staticmethod
    def _duration_changes_adding(M0_leaves, base_leaves, base_durations) -> dict:
        """Per-node ``(before, after)`` durations for adding M0 to ``base_leaves``.

        Only M0's own nodes can change, and each time-node costs O(1), so this
        never touches the base set beyond membership tests.
        """
        added: set = set()
        in_base = base_leaves.__contains__
        in_added = added.__contains__

        def contains(leaf):
            return in_base(leaf) or in_added(leaf)

        changes: dict[int, list[float]] = {}
        for leaf in M0_leaves:
            node = leaf.node
            entry = changes.get(node)
            if entry is None:
                before = base_durations.get(node, 0)
                entry = [before, before]
                changes[node] = entry
            entry[1] += tls.duration_delta_on_add(leaf, contains)
            added.add(leaf)
        return changes

    @staticmethod
    def _duration_changes_removing(M0_leaves, base_leaves, base_durations) -> dict:
        """Per-node ``(before, after)`` durations for removing M0 from ``base_leaves``."""
        removed: set = set()
        in_base = base_leaves.__contains__
        in_removed = removed.__contains__

        def contains(leaf):
            return in_base(leaf) and not in_removed(leaf)

        changes: dict[int, list[float]] = {}
        for leaf in M0_leaves:
            node = leaf.node
            entry = changes.get(node)
            if entry is None:
                before = base_durations.get(node, 0)
                entry = [before, before]
                changes[node] = entry
            removed.add(leaf)
            entry[1] += tls.duration_delta_on_remove(leaf, contains)
        return changes

    def _module_mm_sums(self, module) -> tuple[float, float, dict]:
        """Memoised :meth:`_mm_sums` for a module's own leaves."""
        sums = module._mm_sums
        if sums is None:
            sums = self._mm_sums(module.get_durations(tls.get_nodes_durations))
            module._mm_sums = sums
        return sums

    def _shift_mm_sums(self, sums, changes) -> tuple[float, float, dict]:
        """Apply per-node duration changes to a set of aggregates.

        Args:
            sums: Aggregates of the side being shifted away from.
            changes: ``{node: (old_duration, new_duration)}`` for the nodes whose
                duration differs between the two sides -- exactly M0's nodes.

        Returns:
            The shifted aggregates. The input is not modified.
        """
        total_a, total_b, groups = sums
        linkstream = self.linkstream
        partite_mapping = linkstream.partite_mapping
        directed = linkstream.directed
        degrees_in = linkstream.degrees_in
        degrees_out = linkstream.degrees_out
        degrees = linkstream.degrees
        new_groups = {key: list(value) for key, value in groups.items()} if groups else {}

        for node, (old_duration, new_duration) in changes.items():
            root_shift = new_duration**0.5 - old_duration**0.5
            if directed:
                shift_a = degrees_in.get(node, 0) * root_shift
                shift_b = degrees_out.get(node, 0) * root_shift
            else:
                shift_a = shift_b = degrees[node] * root_shift
            total_a += shift_a
            total_b += shift_b
            partite = partite_mapping.get(node)
            if partite is not None:
                group = new_groups.get(partite)
                if group is None:
                    new_groups[partite] = [shift_a, shift_b]
                else:
                    group[0] += shift_a
                    group[1] += shift_b

        return total_a, total_b, new_groups

    def _mm_from_sums(self, sums_union, sums_mx) -> float:
        """Combine the two sides' aggregates into the expectation delta."""
        union_a, union_b, union_groups = sums_union
        mx_a, mx_b, mx_groups = sums_mx

        total = union_a * union_b - mx_a * mx_b
        for partite, group in union_groups.items():
            other = mx_groups.get(partite)
            if other is None:
                total -= group[0] * group[1]
            else:
                total -= group[0] * group[1] - other[0] * other[1]
        for partite, other in mx_groups.items():
            if partite not in union_groups:
                total += other[0] * other[1]

        # The directed diagonal term is in_i*out_i + out_i*in_i, so the whole
        # ordered sum carries a factor 2.
        if self.linkstream.directed:
            total *= 2
        return total / (4 * self.linkstream.weight * self.linkstream.network_duration)

    def _expectation_mm_closed(
        self,
        union: set[Leaf],
        durations_union,
        durations_mx,
        partite_mapping: dict[int, int],
    ) -> float:
        """O(n) form of the MM expectation delta.

        The pair loop sums, over combinations_with_replacement, a term whose
        ``2 ** (n1 != n2)`` multiplier makes the upper-triangular sum equal to
        the full ordered sum, and that factors::

            undirected  sum_{i,j} d_i d_j (sqrt(u_i u_j) - sqrt(r_i r_j))
                        = (sum_i d_i sqrt(u_i))**2 - (sum_i d_i sqrt(r_i))**2

            directed    2 * [ (sum_i in_i sqrt(u_i))(sum_i out_i sqrt(u_i))
                              - (sum_i in_i sqrt(r_i))(sum_i out_i sqrt(r_i)) ]

        The k-partite mask removes the pairs inside a partite, which is the same
        quantity computed over that partite alone.
        """
        nodes = {leaf.node for leaf in union}
        total = self._mm_group_sum(nodes, durations_union, durations_mx)

        if partite_mapping:
            groups: dict[int, list[int]] = {}
            for node in nodes:
                partite = partite_mapping.get(node)
                if partite is not None:
                    groups.setdefault(partite, []).append(node)
            for group in groups.values():
                total -= self._mm_group_sum(group, durations_union, durations_mx)

        return total / (4 * self.linkstream.weight * self.linkstream.network_duration)

    def _mm_group_sum(self, nodes, durations_union, durations_mx) -> float:
        """Full ordered-pair sum of the MM term over ``nodes``."""
        union_get = durations_union.get
        mx_get = durations_mx.get
        linkstream = self.linkstream

        if linkstream.directed:
            degrees_in = linkstream.degrees_in
            degrees_out = linkstream.degrees_out
            in_union = out_union = in_mx = out_mx = 0.0
            for node in nodes:
                root_union = union_get(node, 0) ** 0.5
                root_mx = mx_get(node, 0) ** 0.5
                degree_in = degrees_in.get(node, 0)
                degree_out = degrees_out.get(node, 0)
                in_union += degree_in * root_union
                out_union += degree_out * root_union
                in_mx += degree_in * root_mx
                out_mx += degree_out * root_mx
            # The diagonal term of the loop is in_i*out_i + out_i*in_i, so the
            # whole ordered sum carries a factor 2.
            return 2 * (in_union * out_union - in_mx * out_mx)

        degrees = linkstream.degrees
        sum_union = sum_mx = 0.0
        for node in nodes:
            degree = degrees[node]
            sum_union += degree * union_get(node, 0) ** 0.5
            sum_mx += degree * mx_get(node, 0) ** 0.5
        return sum_union * sum_union - sum_mx * sum_mx

    def _get_expectation_mm_part(
        self,
        M0_leaves: set[Leaf],
        Mx_leaves: set[Leaf],
        partite_mapping: dict[int, int],
        mx_module=None,
        union_module=None,
    ):
        """Compute the delta number of expected edges within module Mx
        if joined by submodule M0, regarding the Mean-Membership expectation.

        Args:
            M0_leaves (set): M0 submodule time nodes
            Mx_leaves (set): Mx module time nodes
            mx_module (Module, optional): module whose leaves are Mx_leaves.
            union_module (Module, optional): module whose leaves are the union.

        Returns:
            float: delta value for expected number of edges
        """

        # Fast path: one side is a whole module, whose durations and aggregates
        # are memoised, and the other differs from it only on M0's nodes. Cost
        # is O(|M0|), independent of how large the module is.
        if self._mm_closed_form:
            if mx_module is not None:
                # M0 joins Mx: the union is Mx plus M0's time-nodes.
                sums_mx = self._module_mm_sums(mx_module)
                changes = self._duration_changes_adding(
                    M0_leaves, Mx_leaves, mx_module.get_durations(tls.get_nodes_durations)
                )
                return self._mm_from_sums(self._shift_mm_sums(sums_mx, changes), sums_mx)

            if union_module is not None:
                # M0 leaves its parent: Mx is the parent minus M0's time-nodes.
                sums_union = self._module_mm_sums(union_module)
                changes = self._duration_changes_removing(
                    M0_leaves,
                    union_module.leaves,
                    union_module.get_durations(tls.get_nodes_durations),
                )
                return self._mm_from_sums(sums_union, self._shift_mm_sums(sums_union, changes))

        # One of the two duration sets is almost always a whole module's, and a
        # module only changes when a move is accepted -- hundreds of candidate
        # evaluations per accepted move -- so it is memoised on the module.
        if union_module is not None:
            union = union_module.leaves
            durations_union = union_module.get_durations(tls.get_nodes_durations)
        else:
            union = M0_leaves | Mx_leaves
            durations_union = tls.get_nodes_durations(module_leaves=union)

        if mx_module is not None:
            durations_mx = mx_module.get_durations(tls.get_nodes_durations)
        else:
            durations_mx = tls.get_nodes_durations(module_leaves=Mx_leaves)

        if self._mm_closed_form:
            return self._expectation_mm_closed(
                union, durations_union, durations_mx, partite_mapping
            )

        # A node with no time-node in M0 has exactly the same runs in Mx and in
        # Mx U M0, hence the same duration, hence a term of exactly 0.0. Only
        # pairs with at least one node from M0 can contribute, and M0 holds one
        # or two time-nodes -- so this is O(|M0| * n) instead of O(n ** 2).
        # Pairs are still visited in combinations_with_replacement order, so the
        # surviving terms are summed in the same order as before.
        m0_nodes = {leaf.node for leaf in M0_leaves}
        nodes = list({leaf.node for leaf in union})
        m0_positions = [i for i, node in enumerate(nodes) if node in m0_nodes]

        # Hoisted out of the inner loop: this runs tens of millions of times.
        union_get = durations_union.get
        mx_get = durations_mx.get
        linkstream = self.linkstream
        directed = linkstream.directed
        degrees = linkstream.degrees
        degrees_in = linkstream.degrees_in
        degrees_out = linkstream.degrees_out
        denominator = 4 * linkstream.weight * linkstream.network_duration
        partite_get = partite_mapping.get

        expectation_diff = 0
        for index, node1 in enumerate(nodes):
            if node1 in m0_nodes:
                seconds = nodes[index:]
            else:
                seconds = [nodes[j] for j in m0_positions if j >= index]

            for node2 in seconds:
                # In k-partite networks, only interactions between different
                # partites are expected
                if partite_get(node1, -1) == partite_get(node2, -2):
                    continue

                if directed:
                    degrees_part = degrees_in.get(node1, 0) * degrees_out.get(
                        node2, 0
                    ) + degrees_out.get(node1, 0) * degrees_in.get(node2, 0)
                    degrees_part *= 2 ** (node1 != node2)
                else:
                    degrees_part = 2 ** (node1 != node2) * degrees[node1] * degrees[node2]

                # .get(), not [], so a memoised durations mapping is never grown
                # by a lookup for a node that is not in it.
                numerator = degrees_part * (
                    union_get(node1, 0) ** 0.5 * union_get(node2, 0) ** 0.5
                    - mx_get(node1, 0) ** 0.5 * mx_get(node2, 0) ** 0.5
                )
                expectation_diff += numerator / denominator

        return expectation_diff

    def _partial_expectation_mm_diff(
        self,
        node1: int,
        node2: int,
        nodes_durations: dict[str, defaultdict[int, float]],
        partite_mapping: dict[int, int],
    ):
        """Compute the delta expectation number of edges between node1 and node2,
        regarding the Mean-Membership expectation.

        Args:
            node1 (int): linkstream node id
            node2 (int): linkstream node id
            nodes_durations (dict): durations of nodes existences in modules

        Returns:
            float: expected number of edges between node1 and node2.
        """

        # In k-partite networks, only interactions between different partites are expected
        if partite_mapping.get(node1, -1) == partite_mapping.get(node2, -2):
            return 0

        if self.linkstream.directed:
            degrees_part = self.linkstream.degrees_in.get(
                node1, 0
            ) * self.linkstream.degrees_out.get(node2, 0) + self.linkstream.degrees_out.get(
                node1, 0
            ) * self.linkstream.degrees_in.get(node2, 0)
            degrees_part *= 2 ** (node1 != node2)
        else:
            degrees_part = (
                2 ** (node1 != node2)
                * self.linkstream.degrees[node1]
                * self.linkstream.degrees[node2]
            )

        numerator: float = degrees_part * (
            nodes_durations["Cx_U_C0"][node1] ** 0.5 * nodes_durations["Cx_U_C0"][node2] ** 0.5
            - nodes_durations["raw_Cx"][node1] ** 0.5 * nodes_durations["raw_Cx"][node2] ** 0.5
        )

        denominator = 4 * self.linkstream.weight * self.linkstream.network_duration

        return numerator / denominator

    def _sum_degrees(self, module_leaves) -> float:
        """Compute the sum of the degrees of nodes involved in the set of time nodes.

        Args:
            module_leaves (set): module time nodes

        Returns:
            float: sum of the nodes degrees
        """
        nodes = {leaf.node for leaf in module_leaves}
        degrees = {node: self.linkstream.degrees[node] for node in nodes}
        return sum(degrees.values())

    def _sum_degrees_in(self, module_leaves) -> float:
        """Compute the sum of the in degrees of nodes involved in the set of time nodes.

        Args:
            module_leaves (set): module time nodes

        Returns:
            float: sum of the nodes degrees
        """
        nodes = {leaf.node for leaf in module_leaves}
        degrees = {node: self.linkstream.degrees_in.get(node, 0) for node in nodes}
        return sum(degrees.values())

    def _sum_degrees_out(self, module_leaves) -> float:
        """Compute the sum of the out degrees of nodes involved in the set of time nodes.

        Args:
            module_leaves (set): module time nodes

        Returns:
            float: sum of the nodes degrees
        """
        nodes = {leaf.node for leaf in module_leaves}
        degrees = {node: self.linkstream.degrees_out.get(node, 0) for node in nodes}
        return sum(degrees.values())
