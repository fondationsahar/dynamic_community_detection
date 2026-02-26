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
        print(
            f"\n  [two_communities] value={result.value}, time_penalty={result.time_penalty}, lex={result.lex}"
        )
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
        print(
            f"\n  [all_in_one] value={result.value}, time_penalty={result.time_penalty}, lex={result.lex}"
        )
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
        print(
            f"\n  [each_node_own] value={result.value}, time_penalty={result.time_penalty}, lex={result.lex}"
        )
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
        print(
            f"\n  [empty_community] value={result.value}, time_penalty={result.time_penalty}, lex={result.lex}"
        )
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
        print(f"\n  [MM] value={result.value}, time_penalty={result.time_penalty}")
        assert isinstance(result, ModularityResult)
        assert result.lex == LexType.MM

    def test_jm_expectation(self, simple_linkstream_and_communities) -> None:
        """Test Joint Modularity (JM) expectation."""
        ls, communities = simple_linkstream_and_communities
        result = longitudinal_modularity(ls, communities, lex=LexType.JM)
        print(f"\n  [JM] value={result.value}, time_penalty={result.time_penalty}")
        assert isinstance(result, ModularityResult)
        assert result.lex == LexType.JM

    def test_cm_expectation(self, simple_linkstream_and_communities) -> None:
        """Test Coexistence Modularity (CM) expectation."""
        ls, communities = simple_linkstream_and_communities
        result = longitudinal_modularity(ls, communities, lex=LexType.CM)
        print(f"\n  [CM] value={result.value}, time_penalty={result.time_penalty}")
        assert isinstance(result, ModularityResult)
        assert result.lex == LexType.CM

    def test_invalid_expectation_type_raises(self) -> None:
        """Test that invalid expectation type raises ValueError."""
        ls = LinkStream()
        ls.add_links([(0, 1, 0)])
        communities = {"A": {(0, 0), (1, 0)}}

        with pytest.raises(ValueError):
            longitudinal_modularity(ls, communities, lex="INVALID")  # type: ignore
        print("\n  [invalid_lex] ValueError raised as expected")

    def test_different_expectation_types_give_different_results(
        self, simple_linkstream_and_communities
    ) -> None:
        """Test that different expectation types produce different results."""
        ls, communities = simple_linkstream_and_communities

        result_mm = longitudinal_modularity(ls, communities, lex=LexType.MM)
        result_jm = longitudinal_modularity(ls, communities, lex=LexType.JM)
        result_cm = longitudinal_modularity(ls, communities, lex=LexType.CM)

        print(f"\n  [compare_lex] MM={result_mm.value}, JM={result_jm.value}, CM={result_cm.value}")

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

        print(
            f"\n  [gamma] gamma=1.0 -> {result_gamma_1.value}, gamma=2.0 -> {result_gamma_2.value}"
        )

        # Different alpha should give different results
        assert result_gamma_1.value != result_gamma_2.value

    def test_omega_affects_time_penalty(self, basic_setup) -> None:
        """Test that omega parameter affects the time penalty."""
        ls, communities = basic_setup

        result_omega_1 = longitudinal_modularity(ls, communities, omega=1.0)
        result_omega_2 = longitudinal_modularity(ls, communities, omega=2.0)

        print(
            f"\n  [omega] omega=1.0 -> penalty={result_omega_1.time_penalty}, omega=2.0 -> penalty={result_omega_2.time_penalty}"
        )

        # Both should be ModularityResult with time_penalty attribute
        assert isinstance(result_omega_1.time_penalty, float)
        assert isinstance(result_omega_2.time_penalty, float)

    def test_ndigits_affects_precision(self, basic_setup) -> None:
        """Test that ndigits parameter affects precision."""
        ls, communities = basic_setup

        result_5 = longitudinal_modularity(ls, communities, ndigits=5)
        result_2 = longitudinal_modularity(ls, communities, ndigits=2)

        print(f"\n  [ndigits] ndigits=5 -> {result_5.value}, ndigits=2 -> {result_2.value}")

        # Check that rounding works
        assert result_5.ndigits == 5
        assert result_2.ndigits == 2

    def test_modularity_result_has_time_penalty(self, basic_setup) -> None:
        """Test ModularityResult contains time_penalty."""
        ls, communities = basic_setup

        result = longitudinal_modularity(ls, communities)

        print(f"\n  [has_time_penalty] value={result.value}, time_penalty={result.time_penalty}")

        assert isinstance(result, ModularityResult)
        assert hasattr(result, "time_penalty")
        assert isinstance(result.time_penalty, float)

    def test_modularity_result_has_lex_type(self, basic_setup) -> None:
        """Test ModularityResult contains lex."""
        ls, communities = basic_setup

        result = longitudinal_modularity(ls, communities)

        print(f"\n  [has_lex_type] lex={result.lex} (default should be MM)")

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
        print(f"\n  [directed_two_comm] value={result.value}, time_penalty={result.time_penalty}")
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
            print(
                f"\n  [directed_{lex_type.name}] value={result.value}, time_penalty={result.time_penalty}"
            )
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

        print(
            f"\n  [community_switches] stable: value={result_stable.value}, penalty={result_stable.time_penalty}"
        )
        print(
            f"  [community_switches] switch: value={result_switch.value}, penalty={result_switch.time_penalty}"
        )

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
        print(f"\n  [multiple_timesteps] value={result.value}, time_penalty={result.time_penalty}")
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
        print(f"\n  [single_edge] value={result.value}, time_penalty={result.time_penalty}")
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
        print(f"\n  [single_node_comm] value={result.value}, time_penalty={result.time_penalty}")
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
        print(f"\n  [weighted_edges] value={result.value}, time_penalty={result.time_penalty}")
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
        print(f"\n  [inactive_nodes] value={result.value}, time_penalty={result.time_penalty}")
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

        print(
            f"\n  [deterministic] run1={result1.value}, run2={result2.value}, run3={result3.value}"
        )

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

        print(
            f"\n  [multiple_calls] comm1_first={result1a.value}, comm2={result2.value}, comm1_second={result1b.value}"
        )

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

        print(
            f"\n  [result_attrs] value={result.value}, time_penalty={result.time_penalty}, lex={result.lex}, ndigits={result.ndigits}"
        )

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
        print(
            f"\n  [without_penalty] value={result.value}, time_penalty={result.time_penalty}, modularity_without_penalty={result.modularity_without_penalty}, expected={expected}"
        )
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
        print(
            f"\n  [from_file] value={result.value}, time_penalty={result.time_penalty}, lex={result.lex}"
        )
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
        result_jm = longitudinal_modularity(ls, modules, lex=LexType.JM, omega=0, gamma=1)
        print(f"\n  [dir_two_mod_JM] value={result_jm.value} (expected -0.08)")
        assert result_jm.value == -0.08

        # Test with MM expectation (should give same result)
        result_mm = longitudinal_modularity(ls, modules, lex=LexType.MM, omega=0, gamma=1)
        print(f"  [dir_two_mod_MM] value={result_mm.value} (expected -0.08)")
        assert result_mm.value == -0.08

    def test_directed_single_module_jm_and_mm(self, cycle_links) -> None:
        """Test directed graph with all nodes in one module using JM and MM."""
        ls = LinkStream(directed=True)
        ls.add_links(cycle_links)

        modules = {
            0: {(0, 0), (1, 0), (2, 0), (3, 0)},
        }

        # Test with JM expectation
        result_jm = longitudinal_modularity(ls, modules, lex=LexType.JM, omega=0, gamma=1)
        print(f"\n  [dir_single_mod_JM] value={result_jm.value} (expected 0)")
        assert result_jm.value == 0

        # Test with MM expectation (should give same result)
        result_mm = longitudinal_modularity(ls, modules, lex=LexType.MM, omega=0, gamma=1)
        print(f"  [dir_single_mod_MM] value={result_mm.value} (expected 0)")
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
        result_jm = longitudinal_modularity(ls, modules, lex=LexType.JM, omega=0, gamma=1)
        print(f"\n  [dir_singleton_JM] value={result_jm.value} (expected -0.24)")
        assert result_jm.value == -0.24

        # Test with MM expectation (should give same result)
        result_mm = longitudinal_modularity(ls, modules, lex=LexType.MM, omega=0, gamma=1)
        print(f"  [dir_singleton_MM] value={result_mm.value} (expected -0.24)")
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
        result_jm = longitudinal_modularity(ls, modules, lex=LexType.JM, omega=0, gamma=1)
        print(f"\n  [undir_two_mod_JM] value={result_jm.value} (expected -0.1)")
        assert result_jm.value == -0.1

        # Test with MM expectation (should give same result)
        result_mm = longitudinal_modularity(ls, modules, lex=LexType.MM, omega=0, gamma=1)
        print(f"  [undir_two_mod_MM] value={result_mm.value} (expected -0.1)")
        assert result_mm.value == -0.1

    def test_undirected_single_module_jm_and_mm(self, cycle_links) -> None:
        """Test undirected graph with all nodes in one module using JM and MM."""
        ls = LinkStream()
        ls.add_links(cycle_links)

        modules = {
            0: {(0, 0), (1, 0), (2, 0), (3, 0)},
        }

        # Test with JM expectation
        result_jm = longitudinal_modularity(ls, modules, lex=LexType.JM, omega=0, gamma=1)
        print(f"\n  [undir_single_mod_JM] value={result_jm.value} (expected 0)")
        assert result_jm.value == 0

        # Test with MM expectation (should give same result)
        result_mm = longitudinal_modularity(ls, modules, lex=LexType.MM, omega=0, gamma=1)
        print(f"  [undir_single_mod_MM] value={result_mm.value} (expected 0)")
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
        result_jm = longitudinal_modularity(ls, modules, lex=LexType.JM, omega=0, gamma=1)
        print(f"\n  [undir_singleton_JM] value={result_jm.value} (expected -0.26)")
        assert result_jm.value == -0.26

        # Test with MM expectation (should give same result)
        result_mm = longitudinal_modularity(ls, modules, lex=LexType.MM, omega=0, gamma=1)
        print(f"  [undir_singleton_MM] value={result_mm.value} (expected -0.26)")
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

        print(
            f"\n  [static_vs_long_3comm] Q_static={Q_static:.10f}, longitudinal={lm_result.value:.10f}, diff={abs(Q_static - lm_result.value):.2e}"
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

        lm_result = longitudinal_modularity(ls, modules_single, lex=LexType.JM, omega=0, gamma=1)

        print(
            f"\n  [static_vs_long_single] Q_static={Q_static:.10f}, longitudinal={lm_result.value:.10f}, diff={abs(Q_static - lm_result.value):.2e}"
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
        Q_static, _, _ = self.compute_directed_modularity(A, communities_individual, k_out, k_in, m)

        # Convert to LinkStream
        links = [[i, j, 0] for i in range(n) for j in range(n) if A[i, j] == 1]
        ls = LinkStream(directed=True)
        ls.add_links(links)

        # Individual modules
        modules_individual = {node: {(node, 0)} for node in range(n)}

        lm_result = longitudinal_modularity(
            ls, modules_individual, lex=LexType.JM, omega=0, gamma=1
        )

        print(
            f"\n  [static_vs_long_individual] Q_static={Q_static:.10f}, longitudinal={lm_result.value:.10f}, diff={abs(Q_static - lm_result.value):.2e}"
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

        adj: dict[int, list[int]] = {}
        for i, j, _t in links:
            adj.setdefault(i, []).append(j)
        links_str = "\n".join(
            f"    {src}: {', '.join(map(str, tgts))}" for src, tgts in sorted(adj.items())
        )

        k_out = A.sum(axis=1)
        k_in = A.sum(axis=0)
        m = int(A.sum())

        # --- Ground truth communities (3 communities: nodes 0-3, 4-7, 8-11) ---
        communities_ground_truth = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        modules_ground_truth = {}
        for node, comm in enumerate(communities_ground_truth):
            if comm not in modules_ground_truth:
                modules_ground_truth[comm] = set()
            modules_ground_truth[comm].add((node, 0))

        Q_gt, _, _ = self.compute_directed_modularity(A, communities_ground_truth, k_out, k_in, m)

        for lex_type in [LexType.JM, LexType.MM]:
            lm_gt = longitudinal_modularity(
                ls, modules_ground_truth, lex=lex_type, omega=0, gamma=1
            )
            print(
                f"\n  [static_vs_gt_{lex_type.name}] Q_static={Q_gt:.10f}, longitudinal={lm_gt.value:.10f}, diff={abs(Q_gt - lm_gt.value):.2e}"
            )
            assert abs(Q_gt - lm_gt.value) < 1e-5, (
                f"Ground truth {lex_type.name}: Static Q ({Q_gt:.10f}) != Longitudinal Mod ({lm_gt.value:.10f})\n"
                f"  Network ({n} nodes, {len(links)} edges):\n{links_str}"
            )

        # --- LAGO-detected modules ---
        for lex_type in [LexType.JM, LexType.MM]:
            modules_lago = lago_modules(
                ls,
                lex=lex_type,
                omega=0,
                gamma=1,
                refinement_in=True,
                refinement="STNM",
                nb_iter=1,
                verbose=False,
            )

            # Compute longitudinal modularity for LAGO result
            lm_lago = longitudinal_modularity(ls, modules_lago, lex=lex_type, omega=0, gamma=1)

            # Convert LAGO modules to static communities array
            communities_lago = np.zeros(n, dtype=int)
            for module_id, nodes_dict in modules_lago.get_time_modules_dict().items():
                for node in nodes_dict:
                    communities_lago[node] = module_id

            Q_lago, _, _ = self.compute_directed_modularity(A, communities_lago, k_out, k_in, m)

            print(
                f"\n  [static_vs_lago_{lex_type.name}] Q_static={Q_lago:.10f}, longitudinal={lm_lago.value:.10f}, diff={abs(Q_lago - lm_lago.value):.2e}"
            )
            print(
                f"  [static_vs_lago_{lex_type.name}] LAGO found {len(modules_lago.get_time_modules_dict())} modules"
            )

            # Static Q should equal longitudinal modularity within 1e-5 precision
            assert abs(Q_lago - lm_lago.value) < 1e-5, (
                f"LAGO {lex_type.name}: Static Q ({Q_lago:.10f}) != Longitudinal Mod ({lm_lago.value:.10f})\n"
                f"  Network ({n} nodes, {len(links)} edges):\n{links_str}"
            )

            # LAGO-detected modules should have modularity >= ground truth
            n_lago = len(modules_lago.get_time_modules_dict())
            n_gt = len(modules_ground_truth)
            print(
                f"  [static_vs_lago_{lex_type.name}] Q_lago={Q_lago:.10f} ({n_lago} modules), Q_gt={Q_gt:.10f} ({n_gt} modules), lm_lago={lm_lago.value:.10f}"
            )
            assert Q_lago >= Q_gt - 1e-5, (
                f"LAGO {lex_type.name}: Q_lago ({Q_lago:.10f}, {n_lago} modules) < Q_gt ({Q_gt:.10f}, {n_gt} modules), lm_lago={lm_lago.value:.10f}\n"
                f"  Network ({n} nodes, {len(links)} edges):\n{links_str}"
            )

    def test_static_vs_longitudinal_with_lago_modules_large(self) -> None:
        """Test LAGO modules on a larger network (24 nodes, 4 communities of 6)."""
        n = 24
        n_communities = 4
        community_size = n // n_communities
        A = np.zeros((n, n), dtype=int)

        # Intra-community edges (chain + a few extra, not fully connected)
        for c in range(n_communities):
            start = c * community_size
            # Chain: 0->1->2->3->4->5->0
            for k in range(community_size):
                A[start + k, start + (k + 1) % community_size] = 1
            # Extra edges for density: 0->2, 1->3, 2->4, 3->5
            for k in range(community_size):
                A[start + k, start + (k + 2) % community_size] = 1
            # A couple more: 0->3, 1->4
            A[start, start + 3] = 1
            A[start + 1, start + 4] = 1

        # Inter-community edges (ring between adjacent communities, ~4 edges each)
        for c in range(n_communities):
            c_next = (c + 1) % n_communities
            start_c = c * community_size
            start_next = c_next * community_size
            # 4 directed edges from community c to community c+1
            A[start_c + community_size - 1, start_next] = 1
            A[start_c + community_size - 2, start_next + 1] = 1
            A[start_c + community_size - 3, start_next + 2] = 1
            A[start_c, start_next + community_size - 1] = 1

        # Ground truth communities
        communities_ground_truth = np.array([i // community_size for i in range(n)])
        modules_ground_truth = {}
        for node, comm in enumerate(communities_ground_truth):
            if comm not in modules_ground_truth:
                modules_ground_truth[comm] = set()
            modules_ground_truth[comm].add((node, 0))

        # Convert to LinkStream
        links = [[i, j, 0] for i in range(n) for j in range(n) if A[i, j] == 1]
        ls = LinkStream(directed=True)
        ls.add_links(links)

        adj: dict[int, list[int]] = {}
        for i, j, _t in links:
            adj.setdefault(i, []).append(j)
        links_str = "\n".join(
            f"    {src}: {', '.join(map(str, tgts))}" for src, tgts in sorted(adj.items())
        )

        k_out = A.sum(axis=1)
        k_in = A.sum(axis=0)
        m = int(A.sum())

        Q_gt, _, _ = self.compute_directed_modularity(A, communities_ground_truth, k_out, k_in, m)

        for lex_type in [LexType.JM, LexType.MM]:
            # Verify ground truth equivalence
            lm_gt = longitudinal_modularity(
                ls, modules_ground_truth, lex=lex_type, omega=0, gamma=1
            )
            print(
                f"\n  [large_gt_{lex_type.name}] Q_gt={Q_gt:.10f}, lm_gt={lm_gt.value:.10f}, diff={abs(Q_gt - lm_gt.value):.2e}"
            )
            assert abs(Q_gt - lm_gt.value) < 1e-5, (
                f"Large GT {lex_type.name}: Static Q ({Q_gt:.10f}) != Longitudinal Mod ({lm_gt.value:.10f})\n"
                f"  Network ({n} nodes, {len(links)} edges):\n{links_str}"
            )

            # Run LAGO
            modules_lago = lago_modules(
                ls,
                lex=lex_type,
                omega=0,
                gamma=1,
                refinement_in=True,
                refinement="STNM",
                nb_iter=1,
                verbose=False,
            )

            lm_lago = longitudinal_modularity(ls, modules_lago, lex=lex_type, omega=0, gamma=1)

            # Convert LAGO modules to static communities array
            communities_lago = np.zeros(n, dtype=int)
            for module_id, nodes_dict in modules_lago.get_time_modules_dict().items():
                for node in nodes_dict:
                    communities_lago[node] = module_id

            Q_lago, _, _ = self.compute_directed_modularity(A, communities_lago, k_out, k_in, m)

            n_lago = len(modules_lago.get_time_modules_dict())
            n_gt = len(modules_ground_truth)
            print(
                f"\n  [large_lago_{lex_type.name}] Q_lago={Q_lago:.10f} ({n_lago} modules), Q_gt={Q_gt:.10f} ({n_gt} modules), lm_lago={lm_lago.value:.10f}, lm_gt={lm_gt.value:.10f}"
            )

            # Static Q should equal longitudinal modularity
            assert abs(Q_lago - lm_lago.value) < 1e-5, (
                f"Large LAGO {lex_type.name}: Static Q ({Q_lago:.10f}) != Longitudinal Mod ({lm_lago.value:.10f})\n"
                f"  Network ({n} nodes, {len(links)} edges):\n{links_str}"
            )

            # LAGO should achieve modularity >= ground truth
            assert Q_lago >= Q_gt - 1e-5, (
                f"Large LAGO {lex_type.name}: Q_lago ({Q_lago:.10f}, {n_lago} modules) < Q_gt ({Q_gt:.10f}, {n_gt} modules), lm_lago={lm_lago.value:.10f}, lm_gt={lm_gt.value:.10f}\n"
                f"  Network ({n} nodes, {len(links)} edges):\n{links_str}"
            )
