"""Comprehensive tests for the longitudinal_modularity function."""

from __future__ import annotations

import numpy as np
import pytest

from lago import LexType, LinkStream, ModularityResult, lago_modules, longitudinal_modularity

# =============================================================================
# SECTION 1: Basic Functionality Tests
# =============================================================================


class TestBasicFunctionality:
    """Tests for basic longitudinal modularity functionality."""

    def test_simple_two_communities(self) -> None:
        """Test modularity with two clearly separated communities."""
        ls = LinkStream()
        ls.add_links(
            [
                # Community A (nodes 0, 1, 2)
                (0, 1, 0),
                (1, 2, 0),
                (0, 2, 0),
                # Community B (nodes 3, 4, 5)
                (3, 4, 0),
                (4, 5, 0),
                (3, 5, 0),
                # Single inter-community edge
                (2, 3, 0),
            ]
        )
        communities = {
            "A": {(0, 0), (1, 0), (2, 0)},
            "B": {(3, 0), (4, 0), (5, 0)},
        }

        result = longitudinal_modularity(ls, communities)
        assert isinstance(result, ModularityResult)
        # Good partition should have positive modularity
        assert result.value > 0

    def test_all_in_one_community(self) -> None:
        """Test modularity when all nodes in one community."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),
                (1, 2, 0),
                (0, 2, 0),
            ]
        )
        communities = {
            "all": {(0, 0), (1, 0), (2, 0)},
        }

        result = longitudinal_modularity(ls, communities)
        assert isinstance(result, ModularityResult)
        assert isinstance(result.value, float)

    def test_each_node_own_community(self) -> None:
        """Test modularity when each node is its own community."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),
                (1, 2, 0),
                (0, 2, 0),
            ]
        )
        communities = {
            "0": {(0, 0)},
            "1": {(1, 0)},
            "2": {(2, 0)},
        }

        result = longitudinal_modularity(ls, communities)
        assert isinstance(result, ModularityResult)

    def test_empty_community_ignored(self) -> None:
        """Test that empty communities are handled."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),
                (1, 2, 0),
            ]
        )
        communities = {
            "main": {(0, 0), (1, 0), (2, 0)},
            "empty": set(),
        }

        # Should not raise
        result = longitudinal_modularity(ls, communities)
        assert isinstance(result, ModularityResult)


# =============================================================================
# SECTION 2: Expectation Type Tests (CM, JM, MM)
# =============================================================================


class TestExpectationTypes:
    """Tests for different expectation types."""

    @pytest.fixture
    def simple_linkstream_and_communities(self):
        """Create a simple linkstream with communities."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),
                (0, 1, 1),
                (0, 1, 2),
                (1, 2, 0),
                (1, 2, 1),
                (2, 3, 0),
            ]
        )
        communities = {
            "A": {(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)},
            "B": {(2, 0), (3, 0), (2, 1)},
        }
        return ls, communities

    def test_mm_expectation(self, simple_linkstream_and_communities) -> None:
        """Test Mean Modularity (MM) expectation."""
        ls, communities = simple_linkstream_and_communities
        result = longitudinal_modularity(ls, communities, lex=LexType.MM)
        assert isinstance(result, ModularityResult)
        assert result.lex == LexType.MM

    def test_jm_expectation(self, simple_linkstream_and_communities) -> None:
        """Test Joint Modularity (JM) expectation."""
        ls, communities = simple_linkstream_and_communities
        result = longitudinal_modularity(ls, communities, lex=LexType.JM)
        assert isinstance(result, ModularityResult)
        assert result.lex == LexType.JM

    def test_cm_expectation(self, simple_linkstream_and_communities) -> None:
        """Test Coexistence Modularity (CM) expectation."""
        ls, communities = simple_linkstream_and_communities
        result = longitudinal_modularity(ls, communities, lex=LexType.CM)
        assert isinstance(result, ModularityResult)
        assert result.lex == LexType.CM

    def test_invalid_expectation_type_raises(self) -> None:
        """Test that invalid expectation type raises ValueError."""
        ls = LinkStream()
        ls.add_links([(0, 1, 0)])
        communities = {"A": {(0, 0), (1, 0)}}

        with pytest.raises(ValueError):
            longitudinal_modularity(ls, communities, lex="INVALID")  # type: ignore

    def test_different_expectation_types_give_different_results(
        self, simple_linkstream_and_communities
    ) -> None:
        """Test that different expectation types produce different results."""
        ls, communities = simple_linkstream_and_communities

        result_mm = longitudinal_modularity(ls, communities, lex=LexType.MM)
        result_jm = longitudinal_modularity(ls, communities, lex=LexType.JM)
        result_cm = longitudinal_modularity(ls, communities, lex=LexType.CM)

        # All should be ModularityResult
        assert all(isinstance(r, ModularityResult) for r in [result_mm, result_jm, result_cm])


# =============================================================================
# SECTION 3: Parameter Tests (gamma, omega, ndigits)
# =============================================================================


class TestParameters:
    """Tests for function parameters."""

    @pytest.fixture
    def basic_setup(self):
        """Create basic linkstream and communities."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),
                (1, 2, 0),
                (0, 2, 0),
                (3, 4, 0),
                (4, 5, 0),
                (3, 5, 0),
                (2, 3, 0),
            ]
        )
        communities = {
            "A": {(0, 0), (1, 0), (2, 0)},
            "B": {(3, 0), (4, 0), (5, 0)},
        }
        return ls, communities

    def test_gamma_affects_result(self, basic_setup) -> None:
        """Test that gamma parameter affects the result."""
        ls, communities = basic_setup

        result_gamma_1 = longitudinal_modularity(ls, communities, gamma=1.0)
        result_gamma_2 = longitudinal_modularity(ls, communities, gamma=2.0)

        # Different alpha should give different results
        assert result_gamma_1.value != result_gamma_2.value

    def test_omega_affects_time_penalty(self, basic_setup) -> None:
        """Test that omega parameter affects the time penalty."""
        ls, communities = basic_setup

        result_omega_1 = longitudinal_modularity(ls, communities, omega=1.0)
        result_omega_2 = longitudinal_modularity(ls, communities, omega=2.0)

        # Both should be ModularityResult with time_penalty attribute
        assert isinstance(result_omega_1.time_penalty, float)
        assert isinstance(result_omega_2.time_penalty, float)

    def test_ndigits_affects_precision(self, basic_setup) -> None:
        """Test that ndigits parameter affects precision."""
        ls, communities = basic_setup

        result_5 = longitudinal_modularity(ls, communities, ndigits=5)
        result_2 = longitudinal_modularity(ls, communities, ndigits=2)

        # Check that rounding works
        assert result_5.ndigits == 5
        assert result_2.ndigits == 2

    def test_modularity_result_has_time_penalty(self, basic_setup) -> None:
        """Test ModularityResult contains time_penalty."""
        ls, communities = basic_setup

        result = longitudinal_modularity(ls, communities)

        assert isinstance(result, ModularityResult)
        assert hasattr(result, "time_penalty")
        assert isinstance(result.time_penalty, float)

    def test_modularity_result_has_lex_type(self, basic_setup) -> None:
        """Test ModularityResult contains lex."""
        ls, communities = basic_setup

        result = longitudinal_modularity(ls, communities)

        assert isinstance(result, ModularityResult)
        assert result.lex == LexType.MM  # Default


# =============================================================================
# SECTION 4: Directed LinkStream Tests
# =============================================================================


class TestDirectedLinkStream:
    """Tests for directed link streams."""

    def test_directed_two_communities(self) -> None:
        """Test modularity with directed linkstream."""
        ls = LinkStream(directed=True)
        ls.add_links(
            [
                (0, 1, 0),
                (1, 0, 0),  # Bidirectional within A
                (2, 3, 0),
                (3, 2, 0),  # Bidirectional within B
                (1, 2, 0),  # Inter-community edge
            ]
        )
        communities = {
            "A": {(0, 0), (1, 0)},
            "B": {(2, 0), (3, 0)},
        }

        result = longitudinal_modularity(ls, communities)
        assert isinstance(result, ModularityResult)

    def test_directed_all_expectation_types(self) -> None:
        """Test all expectation types with directed graph."""
        ls = LinkStream(directed=True)
        ls.add_links(
            [
                (0, 1, 0),
                (1, 2, 0),
                (2, 0, 0),
            ]
        )
        communities = {"cycle": {(0, 0), (1, 0), (2, 0)}}

        for lex_type in [LexType.MM, LexType.JM, LexType.CM]:
            result = longitudinal_modularity(ls, communities, lex=lex_type)
            assert isinstance(result, ModularityResult)


# =============================================================================
# SECTION 5: Temporal Tests
# =============================================================================


class TestTemporalBehavior:
    """Tests for temporal aspects of modularity."""

    def test_community_switches_affect_time_penalty(self) -> None:
        """Test that community switches affect time penalty."""
        ls = LinkStream()
        ls.add_links(
            [
                # Node 0 and 1 together at t=0
                (0, 1, 0),
                # Node 0 with different community at t=1
                (0, 2, 1),
                # Node 1 stays
                (1, 3, 1),
            ]
        )

        # Scenario 1: Nodes stay together (no switches)
        communities_stable = {
            "main": {(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 1)},
        }

        # Scenario 2: Nodes switch communities
        communities_switch = {
            "A": {(0, 0), (1, 0)},
            "B": {(0, 1), (2, 1)},
            "C": {(1, 1), (3, 1)},
        }

        result_stable = longitudinal_modularity(ls, communities_stable)
        result_switch = longitudinal_modularity(ls, communities_switch)

        # Stable should have less penalty (time penalty is negative)
        assert result_stable.time_penalty >= result_switch.time_penalty

    def test_multiple_timesteps(self) -> None:
        """Test modularity over multiple timesteps."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),
                (0, 1, 1),
                (0, 1, 2),
                (0, 1, 3),
                (2, 3, 0),
                (2, 3, 1),
                (2, 3, 2),
                (2, 3, 3),
            ]
        )
        communities = {
            "A": {(0, t) for t in range(4)} | {(1, t) for t in range(4)},
            "B": {(2, t) for t in range(4)} | {(3, t) for t in range(4)},
        }

        result = longitudinal_modularity(ls, communities)
        assert isinstance(result, ModularityResult)
        assert result.value > 0  # Good separation


# =============================================================================
# SECTION 6: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_edge(self) -> None:
        """Test with a single edge."""
        ls = LinkStream()
        ls.add_links([(0, 1, 0)])
        communities = {"single": {(0, 0), (1, 0)}}

        result = longitudinal_modularity(ls, communities)
        assert isinstance(result, ModularityResult)

    def test_single_node_community(self) -> None:
        """Test community with single node."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),
                (1, 2, 0),
            ]
        )
        communities = {
            "pair": {(0, 0), (1, 0)},
            "single": {(2, 0)},
        }

        result = longitudinal_modularity(ls, communities)
        assert isinstance(result, ModularityResult)

    def test_weighted_edges(self) -> None:
        """Test with weighted edges."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0, 2),
                (1, 2, 0, 1),
                (2, 0, 0, 2),
            ]
        )
        communities = {"all": {(0, 0), (1, 0), (2, 0)}}

        result = longitudinal_modularity(ls, communities)
        assert isinstance(result, ModularityResult)

    def test_inactive_time_nodes_ignored(self) -> None:
        """Test that inactive time nodes are ignored."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),
                (1, 2, 0),
            ]
        )
        # Include a time node that doesn't exist in linkstream
        communities = {
            "main": {(0, 0), (1, 0), (2, 0), (99, 99)},  # (99, 99) doesn't exist
        }

        # Should not raise, should ignore inactive nodes
        result = longitudinal_modularity(ls, communities)
        assert isinstance(result, ModularityResult)


# =============================================================================
# SECTION 7: Consistency Tests
# =============================================================================


class TestConsistency:
    """Tests for result consistency."""

    def test_deterministic_results(self) -> None:
        """Test that results are deterministic."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),
                (1, 2, 0),
                (0, 2, 0),
                (3, 4, 0),
                (4, 5, 0),
                (3, 5, 0),
                (2, 3, 0),
            ]
        )
        communities = {
            "A": {(0, 0), (1, 0), (2, 0)},
            "B": {(3, 0), (4, 0), (5, 0)},
        }

        result1 = longitudinal_modularity(ls, communities)
        result2 = longitudinal_modularity(ls, communities)
        result3 = longitudinal_modularity(ls, communities)

        assert result1.value == result2.value == result3.value

    def test_multiple_calls_same_linkstream(self) -> None:
        """Test multiple calls on same linkstream don't interfere."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),
                (1, 2, 0),
            ]
        )

        communities1 = {"all": {(0, 0), (1, 0), (2, 0)}}
        communities2 = {
            "A": {(0, 0), (1, 0)},
            "B": {(2, 0)},
        }

        # Call multiple times with different communities
        result1a = longitudinal_modularity(ls, communities1)
        result2 = longitudinal_modularity(ls, communities2)
        result1b = longitudinal_modularity(ls, communities1)

        # First call with communities1 should equal third call
        assert result1a.value == result1b.value


# =============================================================================
# SECTION 8: ModularityResult Tests
# =============================================================================


class TestModularityResult:
    """Tests for the ModularityResult dataclass."""

    def test_result_attributes(self) -> None:
        """Test ModularityResult has correct attributes."""
        ls = LinkStream()
        ls.add_links([(0, 1, 0), (1, 2, 0)])
        communities = {"A": {(0, 0), (1, 0), (2, 0)}}

        result = longitudinal_modularity(ls, communities)

        assert hasattr(result, "value")
        assert hasattr(result, "time_penalty")
        assert hasattr(result, "lex")
        assert hasattr(result, "ndigits")

    def test_modularity_without_penalty(self) -> None:
        """Test modularity_without_penalty property."""
        ls = LinkStream()
        ls.add_links([(0, 1, 0), (1, 2, 0)])
        communities = {"A": {(0, 0), (1, 0), (2, 0)}}

        result = longitudinal_modularity(ls, communities)

        # modularity_without_penalty = value - time_penalty
        expected = round(result.value - result.time_penalty, result.ndigits)
        assert result.modularity_without_penalty == expected


# =============================================================================
# SECTION 9: Integration with Real Data
# =============================================================================


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_with_linkstream_from_file(self) -> None:
        """Test with linkstream loaded from file."""
        ls = LinkStream()
        ls.read_txt("tests/fixtures/linkstream.txt")

        # Create simple communities based on node IDs
        all_leaves = set(ls.leaves_dict.keys())
        half = len(all_leaves) // 2
        all_leaves_list = list(all_leaves)

        communities = {
            "first_half": set(all_leaves_list[:half]),
            "second_half": set(all_leaves_list[half:]),
        }

        result = longitudinal_modularity(ls, communities)
        assert isinstance(result, ModularityResult)


# =============================================================================
# SECTION 10: Specific Modularity Value Tests
# =============================================================================


class TestSpecificModularityValues:
    """Tests for specific expected modularity values."""

    @pytest.fixture
    def cycle_links(self):
        """Create a cycle graph links structure."""
        return [
            [0, 1, 0],
            [0, 2, 0],
            [1, 2, 0],
            [2, 3, 0],
            [3, 0, 0],
        ]

    def test_directed_two_modules_jm_and_mm(self, cycle_links) -> None:
        """Test directed graph with two modules using JM and MM expectations."""
        ls = LinkStream(directed=True)
        ls.add_links(cycle_links)

        modules = {
            0: {(0, 0), (1, 0)},
            1: {(2, 0), (3, 0)},
        }

        # Test with JM expectation
        result_jm = longitudinal_modularity(
            ls, modules, lex=LexType.JM, omega=0, gamma=1
        )
        assert result_jm.value == -0.08

        # Test with MM expectation (should give same result)
        result_mm = longitudinal_modularity(
            ls, modules, lex=LexType.MM, omega=0, gamma=1
        )
        assert result_mm.value == -0.08

    def test_directed_single_module_jm_and_mm(self, cycle_links) -> None:
        """Test directed graph with all nodes in one module using JM and MM."""
        ls = LinkStream(directed=True)
        ls.add_links(cycle_links)

        modules = {
            0: {(0, 0), (1, 0), (2, 0), (3, 0)},
        }

        # Test with JM expectation
        result_jm = longitudinal_modularity(
            ls, modules, lex=LexType.JM, omega=0, gamma=1
        )
        assert result_jm.value == 0

        # Test with MM expectation (should give same result)
        result_mm = longitudinal_modularity(
            ls, modules, lex=LexType.MM, omega=0, gamma=1
        )
        assert result_mm.value == 0

    def test_directed_singleton_modules_jm_and_mm(self, cycle_links) -> None:
        """Test directed graph with each node in separate module using JM and MM."""
        ls = LinkStream(directed=True)
        ls.add_links(cycle_links)

        modules = {
            0: {(0, 0)},
            1: {(1, 0)},
            2: {(2, 0)},
            3: {(3, 0)},
        }

        # Test with JM expectation
        result_jm = longitudinal_modularity(
            ls, modules, lex=LexType.JM, omega=0, gamma=1
        )
        assert result_jm.value == -0.24

        # Test with MM expectation (should give same result)
        result_mm = longitudinal_modularity(
            ls, modules, lex=LexType.MM, omega=0, gamma=1
        )
        assert result_mm.value == -0.24

    def test_undirected_two_modules_jm_and_mm(self, cycle_links) -> None:
        """Test undirected graph with two modules using JM and MM expectations."""
        ls = LinkStream()
        ls.add_links(cycle_links)

        modules = {
            0: {(0, 0), (1, 0)},
            1: {(2, 0), (3, 0)},
        }

        # Test with JM expectation
        result_jm = longitudinal_modularity(
            ls, modules, lex=LexType.JM, omega=0, gamma=1
        )
        assert result_jm.value == -0.1

        # Test with MM expectation (should give same result)
        result_mm = longitudinal_modularity(
            ls, modules, lex=LexType.MM, omega=0, gamma=1
        )
        assert result_mm.value == -0.1

    def test_undirected_single_module_jm_and_mm(self, cycle_links) -> None:
        """Test undirected graph with all nodes in one module using JM and MM."""
        ls = LinkStream()
        ls.add_links(cycle_links)

        modules = {
            0: {(0, 0), (1, 0), (2, 0), (3, 0)},
        }

        # Test with JM expectation
        result_jm = longitudinal_modularity(
            ls, modules, lex=LexType.JM, omega=0, gamma=1
        )
        assert result_jm.value == 0

        # Test with MM expectation (should give same result)
        result_mm = longitudinal_modularity(
            ls, modules, lex=LexType.MM, omega=0, gamma=1
        )
        assert result_mm.value == 0

    def test_undirected_singleton_modules_jm_and_mm(self, cycle_links) -> None:
        """Test undirected graph with each node in separate module using JM and MM."""
        ls = LinkStream()
        ls.add_links(cycle_links)

        modules = {
            0: {(0, 0)},
            1: {(1, 0)},
            2: {(2, 0)},
            3: {(3, 0)},
        }

        # Test with JM expectation
        result_jm = longitudinal_modularity(
            ls, modules, lex=LexType.JM, omega=0, gamma=1
        )
        assert result_jm.value == -0.26

        # Test with MM expectation (should give same result)
        result_mm = longitudinal_modularity(
            ls, modules, lex=LexType.MM, omega=0, gamma=1
        )
        assert result_mm.value == -0.26


# =============================================================================
# SECTION 11: Static vs Longitudinal Modularity Equivalence Tests
# =============================================================================


class TestStaticVsLongitudinalEquivalence:
    """Tests verifying static Q and longitudinal modularity match on single timestep."""

    @staticmethod
    def compute_directed_modularity(
        A: np.ndarray, communities: np.ndarray, k_out: np.ndarray, k_in: np.ndarray, m: int
    ) -> tuple[float, float, float]:
        """Compute directed modularity Q using the classic formula.

        Returns:
            (Q, counted_edges, expected_edges)
        """
        Q = 0.0
        counted = 0.0
        expected = 0.0
        n = len(communities)
        for i in range(n):
            for j in range(n):
                if communities[i] == communities[j]:
                    exp_ij = (k_out[i] * k_in[j]) / m if m > 0 else 0
                    counted += A[i, j]
                    expected += exp_ij
                    Q += A[i, j] - exp_ij
        Q /= m if m > 0 else 1
        return Q, counted, expected

    def test_static_vs_longitudinal_directed_three_communities(self) -> None:
        """Test that static Q equals longitudinal modularity on single timestep (directed)."""
        # Create directed graph with 12 nodes in 3 communities
        n = 12
        A = np.zeros((n, n), dtype=int)

        # Community 0: nodes 0-3 (dense intra-community edges)
        A[0, 1] = 1
        A[0, 2] = 1
        A[1, 2] = 1
        A[1, 3] = 1
        A[2, 3] = 1
        A[3, 0] = 1
        A[3, 1] = 1

        # Community 1: nodes 4-7
        A[4, 5] = 1
        A[4, 6] = 1
        A[5, 6] = 1
        A[5, 7] = 1
        A[6, 7] = 1
        A[7, 4] = 1
        A[7, 5] = 1

        # Community 2: nodes 8-11
        A[8, 9] = 1
        A[8, 10] = 1
        A[9, 10] = 1
        A[9, 11] = 1
        A[10, 11] = 1
        A[11, 8] = 1
        A[11, 9] = 1

        # Sparse inter-community edges
        A[3, 4] = 1  # 0 -> 1
        A[7, 8] = 1  # 1 -> 2
        A[11, 0] = 1  # 2 -> 0
        A[2, 5] = 1  # 0 -> 1
        A[6, 9] = 1  # 1 -> 2

        # Ground truth communities
        communities_ground_truth = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

        # Compute static modularity
        k_out = A.sum(axis=1)
        k_in = A.sum(axis=0)
        m = int(A.sum())
        Q_static, _, _ = self.compute_directed_modularity(
            A, communities_ground_truth, k_out, k_in, m
        )

        # Convert to LinkStream (single time step t=0)
        links = []
        for i in range(n):
            for j in range(n):
                if A[i, j] == 1:
                    links.append([i, j, 0])

        ls = LinkStream(directed=True)
        ls.add_links(links)

        # Convert to TimeModules format
        modules_ground_truth = {}
        for node, comm in enumerate(communities_ground_truth):
            if comm not in modules_ground_truth:
                modules_ground_truth[comm] = set()
            modules_ground_truth[comm].add((node, 0))

        # Compute longitudinal modularity with omega=0, gamma=1
        lm_result = longitudinal_modularity(
            ls, modules_ground_truth, lex=LexType.JM, omega=0, gamma=1
        )

        # They should match within 1e-5 precision
        assert abs(Q_static - lm_result.value) < 1e-5, (
            f"Static Q ({Q_static:.10f}) != Longitudinal Mod ({lm_result.value:.10f})"
        )

    def test_static_vs_longitudinal_directed_single_community(self) -> None:
        """Test static Q equals longitudinal mod for single community (directed)."""
        n = 12
        A = np.zeros((n, n), dtype=int)

        # Add edges (same as previous test)
        A[0, 1] = A[0, 2] = A[1, 2] = A[1, 3] = A[2, 3] = A[3, 0] = A[3, 1] = 1
        A[4, 5] = A[4, 6] = A[5, 6] = A[5, 7] = A[6, 7] = A[7, 4] = A[7, 5] = 1
        A[8, 9] = A[8, 10] = A[9, 10] = A[9, 11] = A[10, 11] = A[11, 8] = A[11, 9] = 1
        A[3, 4] = A[7, 8] = A[11, 0] = A[2, 5] = A[6, 9] = 1

        # Single community
        communities_single = np.zeros(n, dtype=int)

        k_out = A.sum(axis=1)
        k_in = A.sum(axis=0)
        m = int(A.sum())
        Q_static, _, _ = self.compute_directed_modularity(A, communities_single, k_out, k_in, m)

        # Convert to LinkStream
        links = [[i, j, 0] for i in range(n) for j in range(n) if A[i, j] == 1]
        ls = LinkStream(directed=True)
        ls.add_links(links)

        # Single module
        modules_single = {0: {(node, 0) for node in range(n)}}

        lm_result = longitudinal_modularity(
            ls, modules_single, lex=LexType.JM, omega=0, gamma=1
        )

        assert abs(Q_static - lm_result.value) < 1e-5

    def test_static_vs_longitudinal_directed_individual_nodes(self) -> None:
        """Test static Q equals longitudinal mod for individual nodes (directed)."""
        n = 12
        A = np.zeros((n, n), dtype=int)

        # Add edges
        A[0, 1] = A[0, 2] = A[1, 2] = A[1, 3] = A[2, 3] = A[3, 0] = A[3, 1] = 1
        A[4, 5] = A[4, 6] = A[5, 6] = A[5, 7] = A[6, 7] = A[7, 4] = A[7, 5] = 1
        A[8, 9] = A[8, 10] = A[9, 10] = A[9, 11] = A[10, 11] = A[11, 8] = A[11, 9] = 1
        A[3, 4] = A[7, 8] = A[11, 0] = A[2, 5] = A[6, 9] = 1

        # Each node in own community
        communities_individual = np.arange(n)

        k_out = A.sum(axis=1)
        k_in = A.sum(axis=0)
        m = int(A.sum())
        Q_static, _, _ = self.compute_directed_modularity(
            A, communities_individual, k_out, k_in, m
        )

        # Convert to LinkStream
        links = [[i, j, 0] for i in range(n) for j in range(n) if A[i, j] == 1]
        ls = LinkStream(directed=True)
        ls.add_links(links)

        # Individual modules
        modules_individual = {node: {(node, 0)} for node in range(n)}

        lm_result = longitudinal_modularity(
            ls, modules_individual, lex=LexType.JM, omega=0, gamma=1
        )

        assert abs(Q_static - lm_result.value) < 1e-5

    def test_static_vs_longitudinal_with_lago_modules(self) -> None:
        """Test that LAGO modules on static network match static Q computation."""
        n = 12
        A = np.zeros((n, n), dtype=int)

        # Create network with 3 communities
        A[0, 1] = A[0, 2] = A[1, 2] = A[1, 3] = A[2, 3] = A[3, 0] = A[3, 1] = 1
        A[4, 5] = A[4, 6] = A[5, 6] = A[5, 7] = A[6, 7] = A[7, 4] = A[7, 5] = 1
        A[8, 9] = A[8, 10] = A[9, 10] = A[9, 11] = A[10, 11] = A[11, 8] = A[11, 9] = 1
        A[3, 4] = A[7, 8] = A[11, 0] = A[2, 5] = A[6, 9] = 1

        # Convert to LinkStream
        links = [[i, j, 0] for i in range(n) for j in range(n) if A[i, j] == 1]
        ls = LinkStream(directed=True)
        ls.add_links(links)

        # Run LAGO to find modules
        modules_lago = lago_modules(
            ls,
            lex=LexType.JM,
            omega=0,
            gamma=1,
            refinement_in=True,
            refinement="STNM",
            nb_iter=1,
            verbose=False,
        )

        # Compute longitudinal modularity for LAGO result
        lm_lago = longitudinal_modularity(
            ls, modules_lago, lex=LexType.JM, omega=0, gamma=1
        )

        # Convert LAGO modules to static communities array
        communities_lago = np.zeros(n, dtype=int)
        for module_id, nodes_dict in modules_lago.get_time_modules_dict().items():
            for node in nodes_dict.keys():
                communities_lago[node] = module_id

        # Compute static modularity for LAGO result
        k_out = A.sum(axis=1)
        k_in = A.sum(axis=0)
        m = int(A.sum())
        Q_lago, _, _ = self.compute_directed_modularity(A, communities_lago, k_out, k_in, m)

        # Static Q should equal longitudinal modularity within 1e-5 precision
        assert abs(Q_lago - lm_lago.value) < 1e-5, (
            f"Static Q ({Q_lago:.10f}) != Longitudinal Mod ({lm_lago.value:.10f})"
        )
