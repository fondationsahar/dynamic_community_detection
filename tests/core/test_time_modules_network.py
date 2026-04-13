"""Tests for TimeModulesNetwork class."""

import warnings

import pytest

from lago.core.linkstream import LinkStream
from lago.core.time_modules import TimeModules
from lago.core.time_modules_network import ModuleTimeEdge, TimeModulesNetwork


class TestModuleTimeEdge:
    """Tests for ModuleTimeEdge dataclass."""

    def test_is_intra_module_true(self):
        """Test intra-module detection when modules match."""
        edge = ModuleTimeEdge(
            source_module=0,
            target_module=0,
            time=0,
            target_time=None,
            weight=1.0,
            duration=None,
            source_node=0,
            target_node=1,
        )
        assert edge.is_intra_module is True

    def test_is_intra_module_false(self):
        """Test intra-module detection when modules differ."""
        edge = ModuleTimeEdge(
            source_module=0,
            target_module=1,
            time=0,
            target_time=None,
            weight=1.0,
            duration=None,
            source_node=0,
            target_node=1,
        )
        assert edge.is_intra_module is False


class TestTimeModulesNetworkConstruction:
    """Tests for TimeModulesNetwork construction."""

    @pytest.fixture
    def setup_data(self):
        """Create a linkstream and time modules for testing."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),  # Edge in module 0
                (0, 2, 0),  # Edge between module 0 and 1
                (2, 3, 0),  # Edge in module 1
            ]
        )

        # Module 0: nodes 0, 1
        # Module 1: nodes 2, 3
        tm = TimeModules(
            {
                0: {(0, 0), (1, 0)},
                1: {(2, 0), (3, 0)},
            }
        )

        return ls, tm

    def test_construction(self, setup_data):
        """Test basic construction."""
        ls, tm = setup_data
        network = TimeModulesNetwork(tm, ls)

        assert network.nb_module_edges > 0
        assert network.modules == frozenset({0, 1})

    def test_properties(self, setup_data):
        """Test network properties."""
        ls, tm = setup_data
        network = TimeModulesNetwork(tm, ls)

        assert network.directed == ls.directed
        assert network.delayed == ls.delayed
        assert network.continuous == ls.continuous

    def test_include_intra_true(self, setup_data):
        """Test including intra-module edges."""
        ls, tm = setup_data
        network = TimeModulesNetwork(tm, ls, include_intra=True)

        assert network.nb_intra_module_edges > 0

    def test_include_intra_false(self, setup_data):
        """Test excluding intra-module edges."""
        ls, tm = setup_data
        network = TimeModulesNetwork(tm, ls, include_intra=False)

        assert network.nb_intra_module_edges == 0


class TestTimeModulesNetworkEdgeAccess:
    """Tests for edge access methods."""

    @pytest.fixture
    def network(self):
        """Create a network for testing."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),  # Intra module 0
                (0, 2, 0),  # Inter module 0-1
                (2, 3, 0),  # Intra module 1
                (0, 1, 1),  # Intra module 0 at time 1
            ]
        )

        tm = TimeModules(
            {
                0: {(0, 0), (1, 0), (0, 1), (1, 1)},
                1: {(2, 0), (3, 0)},
            }
        )

        return TimeModulesNetwork(tm, ls)

    def test_get_all_edges(self, network):
        """Test getting all edges."""
        edges = network.get_all_edges()
        assert len(edges) > 0
        assert all(isinstance(e, ModuleTimeEdge) for e in edges)

    def test_get_edges_at_time(self, network):
        """Test getting edges at specific time."""
        edges_t0 = network.get_edges_at_time(0)
        edges_t1 = network.get_edges_at_time(1)

        assert len(edges_t0) > 0
        assert len(edges_t1) > 0
        assert all(e.time == 0 for e in edges_t0)
        assert all(e.time == 1 for e in edges_t1)

    def test_get_edges_between(self, network):
        """Test getting edges between modules."""
        # Inter-module edges
        edges = network.get_edges_between(0, 1)
        assert len(edges) > 0
        for e in edges:
            assert {e.source_module, e.target_module} == {0, 1}

    def test_get_edges_of_module(self, network):
        """Test getting edges of a module."""
        edges = network.get_edges_of_module(0)
        assert len(edges) > 0
        for e in edges:
            assert e.source_module == 0 or e.target_module == 0

    def test_iter_edges(self, network):
        """Test edge iteration."""
        count = sum(1 for _ in network.iter_edges())
        assert count == network.nb_module_edges


class TestTimeModulesNetworkNeighbors:
    """Tests for neighbor query methods."""

    @pytest.fixture
    def network(self):
        """Create a network for testing."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),  # Intra module 0
                (0, 2, 0),  # Inter 0-1
                (1, 2, 0),  # Inter 0-1
                (2, 3, 0),  # Intra module 1
            ]
        )

        tm = TimeModules(
            {
                0: {(0, 0), (1, 0)},
                1: {(2, 0), (3, 0)},
            }
        )

        return TimeModulesNetwork(tm, ls)

    def test_get_neighbors_at_time(self, network):
        """Test getting neighbors at time."""
        neighbors = network.get_neighbors_at_time(0, 0, include_self=False)

        # Module 0 should have module 1 as neighbor
        assert 1 in neighbors
        assert 0 not in neighbors  # Self excluded

    def test_get_neighbors_at_time_with_self(self, network):
        """Test getting neighbors with self-loops."""
        neighbors = network.get_neighbors_at_time(0, 0, include_self=True)

        # Should include self-loops
        assert 1 in neighbors
        # May or may not have self depending on intra-module edges

    def test_get_neighbors_over_time(self, network):
        """Test getting neighbors across time."""
        neighbors = network.get_neighbors_over_time(0)

        assert isinstance(neighbors, dict)
        if neighbors:
            for time, time_neighbors in neighbors.items():
                assert isinstance(time_neighbors, dict)

    def test_get_aggregated_neighbors(self, network):
        """Test aggregated neighbors."""
        neighbors = network.get_aggregated_neighbors(0, include_self=False)

        assert isinstance(neighbors, dict)
        # Module 0 should have module 1 as aggregated neighbor
        assert 1 in neighbors

    def test_get_closest_neighbors_at_time(self, network):
        """Test k closest neighbors at time."""
        closest = network.get_closest_neighbors_at_time(0, 0, k=2)

        assert isinstance(closest, list)
        # Should be sorted by weight (descending)
        if len(closest) > 1:
            assert closest[0][1] >= closest[1][1]

    def test_get_closest_neighbors_overall(self, network):
        """Test k closest neighbors overall."""
        closest = network.get_closest_neighbors_overall(0, k=2)

        assert isinstance(closest, list)


class TestTimeModulesNetworkAnalysis:
    """Tests for analysis methods."""

    @pytest.fixture
    def network(self):
        """Create a network for testing."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0, 2.0),  # Intra module 0, weight 2
                (0, 2, 0, 1.0),  # Inter 0-1, weight 1
            ]
        )

        tm = TimeModules(
            {
                0: {(0, 0), (1, 0)},
                1: {(2, 0)},
            }
        )

        return TimeModulesNetwork(tm, ls)

    def test_get_inter_module_weight_at_time(self, network):
        """Test inter-module weight calculation."""
        inter = network.get_inter_module_weight_at_time(0)
        assert inter > 0

    def test_get_intra_module_weight_at_time(self, network):
        """Test intra-module weight calculation."""
        intra = network.get_intra_module_weight_at_time(0)
        assert intra > 0

    def test_get_inter_intra_over_time(self, network):
        """Test inter/intra over time."""
        result = network.get_inter_intra_over_time()

        assert isinstance(result, dict)
        for time, (inter, intra) in result.items():
            assert isinstance(inter, float)
            assert isinstance(intra, float)


class TestTimeModulesNetworkConversion:
    """Tests for conversion methods."""

    @pytest.fixture
    def network(self):
        """Create a network for testing."""
        ls = LinkStream()
        ls.add_links(
            [
                (0, 1, 0),
                (0, 2, 0),
            ]
        )

        tm = TimeModules(
            {
                0: {(0, 0), (1, 0)},
                1: {(2, 0)},
            }
        )

        return TimeModulesNetwork(tm, ls)

    def test_to_adjacency_at_time(self, network):
        """Test adjacency dict conversion."""
        adj = network.to_adjacency_at_time(0)

        assert isinstance(adj, dict)
        for module, neighbors in adj.items():
            assert isinstance(neighbors, dict)

    def test_to_edge_list(self, network):
        """Test edge list conversion."""
        edges = network.to_edge_list()

        assert isinstance(edges, list)
        for edge in edges:
            assert len(edge) == 4
            source, target, time, weight = edge
            assert isinstance(source, int)
            assert isinstance(target, int)

    def test_to_linkstream_format(self, network):
        """Test linkstream format conversion."""
        edges = network.to_linkstream_format()

        assert isinstance(edges, list)
        assert len(edges) > 0


class TestTimeModulesNetworkValidation:
    """Tests for validation and edge cases."""

    def test_incompatible_time_modules_raises(self):
        """Test that incompatible time modules raise error."""
        ls = LinkStream()
        ls.add_links([(0, 1, 0)])

        # TimeModules with member not in linkstream
        tm = TimeModules({0: {(0, 0), (1, 0), (99, 99)}})  # (99, 99) not in ls

        with pytest.raises(ValueError, match="not found in LinkStream"):
            TimeModulesNetwork(tm, ls)

    def test_partial_coverage_warns(self):
        """Test warning for partial coverage."""
        ls = LinkStream()
        ls.add_links([(0, 1, 0), (2, 3, 0)])  # 4 nodes

        # Only cover some nodes
        tm = TimeModules({0: {(0, 0), (1, 0)}})  # nodes 2, 3 uncovered

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TimeModulesNetwork(tm, ls)

            assert len(w) == 1
            assert "have no module assignment" in str(w[0].message)


class TestTimeModulesNetworkDunder:
    """Tests for dunder methods."""

    @pytest.fixture
    def network(self):
        """Create a network for testing."""
        ls = LinkStream()
        ls.add_links([(0, 1, 0), (0, 2, 0)])

        tm = TimeModules(
            {
                0: {(0, 0), (1, 0)},
                1: {(2, 0)},
            }
        )

        return TimeModulesNetwork(tm, ls)

    def test_len(self, network):
        """Test __len__."""
        assert len(network) == network.nb_module_edges

    def test_repr(self, network):
        """Test __repr__."""
        r = repr(network)
        assert "TimeModulesNetwork" in r
        assert "modules=" in r
        assert "edges=" in r
