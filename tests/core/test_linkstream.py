"""Comprehensive tests for the LinkStream class."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lago import LinkStream, LinkStreamMode

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# SECTION 1: Instantiation Tests
# =============================================================================


class TestLinkStreamInstantiation:
    """Tests for LinkStream initialization."""

    def test_default_instantiation(self) -> None:
        """Test default LinkStream creation."""
        ls = LinkStream()
        assert ls.continuous is False
        assert ls.directed is False
        assert ls.delayed is False
        assert ls.partite_mapping == {}
        assert ls.nb_edges == 0
        assert ls.weight == 0

    def test_continuous_instantiation(self) -> None:
        """Test continuous LinkStream creation."""
        ls = LinkStream(continuous=True)
        assert ls.continuous is True
        assert ls.directed is False
        assert ls.delayed is False

    def test_directed_instantiation(self) -> None:
        """Test directed LinkStream creation."""
        ls = LinkStream(directed=True)
        assert ls.continuous is False
        assert ls.directed is True
        assert ls.delayed is False

    def test_delayed_instantiation(self) -> None:
        """Test delayed LinkStream creation."""
        ls = LinkStream(delayed=True)
        assert ls.continuous is False
        assert ls.directed is False
        assert ls.delayed is True

    def test_directed_continuous_instantiation(self) -> None:
        """Test directed + continuous LinkStream creation."""
        ls = LinkStream(continuous=True, directed=True)
        assert ls.continuous is True
        assert ls.directed is True

    def test_directed_delayed_instantiation(self) -> None:
        """Test directed + delayed LinkStream creation."""
        ls = LinkStream(delayed=True, directed=True)
        assert ls.delayed is True
        assert ls.directed is True

    def test_partite_mapping(self) -> None:
        """Test partite mapping initialization."""
        mapping = {0: 0, 1: 0, 2: 1, 3: 1}
        ls = LinkStream(partite_mapping=mapping)
        assert ls.partite_mapping == mapping

    def test_set_partite_after_init(self) -> None:
        """Test setting partite mapping after initialization."""
        ls = LinkStream()
        mapping = {0: 0, 1: 1}
        ls.set_partite(mapping)
        assert ls.partite_mapping == mapping

    def test_continuous_and_delayed_raises_error(self) -> None:
        """Test that continuous + delayed raises ValueError."""
        with pytest.raises(ValueError, match="cannot be both continuous and delayed"):
            LinkStream(continuous=True, delayed=True)

    def test_mode_property_instantaneous(self) -> None:
        """Test mode property returns INSTANTANEOUS for default."""
        ls = LinkStream()
        assert ls.mode == LinkStreamMode.INSTANTANEOUS

    def test_mode_property_continuous(self) -> None:
        """Test mode property returns CONTINUOUS."""
        ls = LinkStream(continuous=True)
        assert ls.mode == LinkStreamMode.CONTINUOUS

    def test_mode_property_delayed(self) -> None:
        """Test mode property returns DELAYED."""
        ls = LinkStream(delayed=True)
        assert ls.mode == LinkStreamMode.DELAYED

    def test_unified_add_links_instantaneous(self) -> None:
        """Test unified add_links works for instantaneous mode."""
        ls = LinkStream()  # Default is instantaneous
        ls.add_links([(0, 1, 0), (1, 2, 1)])
        assert ls.nb_edges == 2
        assert ls.nb_nodes == 3

    def test_unified_add_links_continuous(self) -> None:
        """Test unified add_links works for continuous mode."""
        ls = LinkStream(continuous=True)
        ls.add_links([(0, 1, 0, 3)])  # source, target, time, duration
        assert ls.nb_edges == 1
        assert ls.weight == 3.0  # weight = duration * 1

    def test_unified_add_links_delayed(self) -> None:
        """Test unified add_links works for delayed mode."""
        ls = LinkStream(delayed=True)
        ls.add_links([(0, 1, 0, 5)])  # source, target, source_time, target_time
        assert ls.nb_edges == 1
        assert (0, 0) in ls.leaves_dict
        assert (1, 5) in ls.leaves_dict

    def test_default_columns_instantaneous(self) -> None:
        """Test default columns auto-detection for instantaneous mode."""
        ls = LinkStream()
        assert ls._get_default_columns_for_mode() == ["source", "target", "time"]

    def test_default_columns_continuous(self) -> None:
        """Test default columns auto-detection for continuous mode."""
        ls = LinkStream(continuous=True)
        assert ls._get_default_columns_for_mode() == ["source", "target", "time_start", "duration"]

    def test_default_columns_delayed(self) -> None:
        """Test default columns auto-detection for delayed mode."""
        ls = LinkStream(delayed=True)
        assert ls._get_default_columns_for_mode() == [
            "source",
            "target",
            "source_time",
            "target_time",
        ]


# =============================================================================
# SECTION 2: add_links Tests (Instantaneous/Discrete)
# =============================================================================


class TestAddLinks:
    """Tests for add_links method (instantaneous links)."""

    def test_simple_triangle(
        self,
        linkstream_factory: Callable[..., LinkStream],
        simple_triangle_links: list[tuple[int, int, int]],
    ) -> None:
        """Test adding a simple triangle."""
        ls = linkstream_factory()
        ls.add_links(simple_triangle_links)

        assert ls.nb_nodes == 3
        assert ls.nb_edges == 3
        assert ls.nodes == {0, 1, 2}
        assert ls.min_time == 0
        assert ls.max_time == 0
        assert ls.network_duration == 1

    def test_temporal_path(
        self,
        linkstream_factory: Callable[..., LinkStream],
        temporal_path_links: list[tuple[int, int, int]],
    ) -> None:
        """Test adding links across multiple timesteps."""
        ls = linkstream_factory()
        ls.add_links(temporal_path_links)

        assert ls.nb_nodes == 4
        assert ls.nb_edges == 3
        assert ls.min_time == 0
        assert ls.max_time == 2
        assert ls.network_duration == 3
        assert ls.nb_timesteps == 3

    def test_weighted_links(
        self,
        linkstream_factory: Callable[..., LinkStream],
        weighted_links: list[tuple[int, int, int, float]],
    ) -> None:
        """Test adding weighted links."""
        ls = linkstream_factory()
        ls.add_links(weighted_links)

        assert ls.nb_edges == 3
        assert ls.weight == pytest.approx(4.0)  # 2.0 + 0.5 + 1.5

    def test_empty_links(self, linkstream_factory: Callable[..., LinkStream]) -> None:
        """Test adding empty link list."""
        ls = linkstream_factory()
        ls.add_links([])

        assert ls.nb_edges == 0
        assert ls.nb_nodes == 0

    def test_single_link(
        self,
        linkstream_factory: Callable[..., LinkStream],
        single_link: list[tuple[int, int, int]],
    ) -> None:
        """Test adding a single link."""
        ls = linkstream_factory()
        ls.add_links(single_link)

        assert ls.nb_nodes == 2
        assert ls.nb_edges == 1
        assert ls.nb_timesteps == 1

    def test_duplicate_links(
        self,
        linkstream_factory: Callable[..., LinkStream],
        duplicate_links: list[tuple[int, int, int]],
    ) -> None:
        """Test adding duplicate links."""
        ls = linkstream_factory()
        ls.add_links(duplicate_links)

        # Each duplicate counts as a separate edge
        assert ls.nb_edges == 3
        assert ls.weight == 3

    def test_large_timestamps(
        self,
        linkstream_factory: Callable[..., LinkStream],
        large_timestamp_links: list[tuple[int, int, int]],
    ) -> None:
        """Test links with large timestamps."""
        ls = linkstream_factory()
        ls.add_links(large_timestamp_links)

        assert ls.min_time == 1000000
        assert ls.max_time == 2000000
        assert ls.network_duration == 1000001

    def test_degrees_undirected(
        self,
        linkstream_factory: Callable[..., LinkStream],
        simple_triangle_links: list[tuple[int, int, int]],
    ) -> None:
        """Test degree computation for undirected graph."""
        ls = linkstream_factory()
        ls.add_links(simple_triangle_links)

        # In a triangle, each node has degree 2
        assert ls.degrees[0] == 2
        assert ls.degrees[1] == 2
        assert ls.degrees[2] == 2

    def test_leaves_creation(
        self,
        linkstream_factory: Callable[..., LinkStream],
        simple_triangle_links: list[tuple[int, int, int]],
    ) -> None:
        """Test that leaves are created correctly."""
        ls = linkstream_factory()
        ls.add_links(simple_triangle_links)

        # 3 nodes at time 0
        assert len(ls.leaves_dict) == 3
        assert (0, 0) in ls.leaves_dict
        assert (1, 0) in ls.leaves_dict
        assert (2, 0) in ls.leaves_dict

    def test_topological_neighbors_undirected(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test topological neighbors for undirected graph."""
        ls = linkstream_factory()
        ls.add_links([(0, 1, 0)])

        leaf_0 = ls.leaves_dict[(0, 0)]
        leaf_1 = ls.leaves_dict[(1, 0)]

        # Both should have each other as neighbors
        neighbor_nodes_0 = {te.target.node for te in leaf_0.topo_neighbors}
        neighbor_nodes_1 = {te.target.node for te in leaf_1.topo_neighbors}

        assert 1 in neighbor_nodes_0
        assert 0 in neighbor_nodes_1


# =============================================================================
# SECTION 3: Directed Graph Tests
# =============================================================================


class TestDirectedLinkStream:
    """Tests for directed LinkStream."""

    def test_directed_triangle(
        self,
        linkstream_factory: Callable[..., LinkStream],
        directed_triangle_links: list[tuple[int, int, int]],
    ) -> None:
        """Test directed triangle graph."""
        ls = linkstream_factory(directed=True)
        ls.add_links(directed_triangle_links)

        assert ls.nb_nodes == 3
        assert ls.nb_edges == 3

    def test_degrees_in_out(
        self,
        linkstream_factory: Callable[..., LinkStream],
        directed_star_links: list[tuple[int, int, int]],
    ) -> None:
        """Test in/out degree computation for directed graph."""
        ls = linkstream_factory(directed=True)
        ls.add_links(directed_star_links)

        # Node 0 is the center, has out-degree 3, in-degree 0
        assert ls.degrees_out[0] == 3
        assert ls.degrees_in.get(0, 0) == 0

        # Nodes 1, 2, 3 have in-degree 1, out-degree 0
        for node in [1, 2, 3]:
            assert ls.degrees_in[node] == 1
            assert ls.degrees_out.get(node, 0) == 0

    def test_directed_topological_neighbors(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test topological neighbors for directed graph."""
        ls = linkstream_factory(directed=True)
        ls.add_links([(0, 1, 0)])  # 0 -> 1

        leaf_0 = ls.leaves_dict[(0, 0)]
        leaf_1 = ls.leaves_dict[(1, 0)]

        # Leaf 0 should have 1 in topo_neighbors (outgoing)
        out_neighbors_0 = {te.target.node for te in leaf_0.topo_neighbors}
        assert 1 in out_neighbors_0

        # Leaf 1 should have 0 in topo_neighbors_from (incoming)
        in_neighbors_1 = {te.target.node for te in leaf_1.topo_neighbors_from}
        assert 0 in in_neighbors_1

        # Leaf 0 should NOT have incoming from 1
        in_neighbors_0 = {te.target.node for te in leaf_0.topo_neighbors_from}
        assert 1 not in in_neighbors_0

    def test_bidirectional_edges(
        self,
        linkstream_factory: Callable[..., LinkStream],
        bidirectional_links: list[tuple[int, int, int]],
    ) -> None:
        """Test bidirectional edges in directed graph."""
        ls = linkstream_factory(directed=True)
        ls.add_links(bidirectional_links)

        assert ls.degrees_in[0] == 1
        assert ls.degrees_out[0] == 1
        assert ls.degrees_in[1] == 1
        assert ls.degrees_out[1] == 1


# =============================================================================
# SECTION 4: Continuous LinkStream Tests
# =============================================================================


class TestContinuousLinkStream:
    """Tests for continuous LinkStream (links with duration)."""

    def test_simple_continuous_links(
        self,
        linkstream_factory: Callable[..., LinkStream],
        simple_continuous_links: list[tuple[int, int, int, int]],
    ) -> None:
        """Test adding simple continuous links."""
        ls = linkstream_factory(continuous=True)
        ls.add_continous_links(simple_continuous_links)  # Note: typo in original

        assert ls.nb_nodes == 3
        assert ls.nb_edges == 2

    def test_continuous_weight_calculation(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test weight calculation for continuous links (weight * duration)."""
        ls = linkstream_factory(continuous=True)
        # Link with duration 3, weight 1 -> total weight = 3
        ls.add_continous_links([(0, 1, 0, 3)])

        assert ls.weight == 3.0

    def test_continuous_weighted_links(
        self,
        linkstream_factory: Callable[..., LinkStream],
        weighted_continuous_links: list[tuple[int, int, int, int, float]],
    ) -> None:
        """Test continuous links with explicit weights."""
        ls = linkstream_factory(continuous=True)
        ls.add_continous_links(weighted_continuous_links)

        # (0,1,0,3,2.0) -> 3 * 2.0 = 6.0
        # (1,2,1,2,0.5) -> 2 * 0.5 = 1.0
        assert ls.weight == pytest.approx(7.0)

    def test_continuous_time_instants(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test that time instants are recorded for continuous links."""
        ls = linkstream_factory(continuous=True)
        ls.add_continous_links(
            [
                (0, 1, 0, 3),  # t=0 to t=3
                (1, 2, 1, 2),  # t=1 to t=3
            ]
        )

        # Time instants should include: 0, 3, 1, 3
        assert 0 in ls.time_instants
        assert 1 in ls.time_instants
        assert 3 in ls.time_instants

    def test_continuous_min_max_time(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test min/max time calculation for continuous links."""
        ls = linkstream_factory(continuous=True)
        ls.add_continous_links(
            [
                (0, 1, 5, 10),  # t=5 to t=15
            ]
        )

        assert ls.min_time == 5
        assert ls.max_time == 15

    def test_continuous_directed(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test continuous links in directed mode."""
        ls = linkstream_factory(continuous=True, directed=True)
        ls.add_continous_links([(0, 1, 0, 3)])

        assert ls.degrees_out[0] == 3.0  # weight = duration * 1
        assert ls.degrees_in[1] == 3.0

    def test_overlapping_continuous_links(
        self,
        linkstream_factory: Callable[..., LinkStream],
        overlapping_continuous_links: list[tuple[int, int, int, int]],
    ) -> None:
        """Test overlapping continuous links."""
        ls = linkstream_factory(continuous=True)
        ls.add_continous_links(overlapping_continuous_links)

        # Should handle overlapping time intervals
        assert ls.nb_nodes == 3
        assert ls.nb_edges == 3


# =============================================================================
# SECTION 5: Delayed LinkStream Tests
# =============================================================================


class TestDelayedLinkStream:
    """Tests for delayed LinkStream (different source/target times)."""

    def test_simple_delayed_links(
        self,
        linkstream_factory: Callable[..., LinkStream],
        simple_delayed_links: list[tuple[int, int, int, int]],
    ) -> None:
        """Test adding simple delayed links."""
        ls = linkstream_factory(delayed=True)
        ls.add_delayed_links(simple_delayed_links)

        assert ls.nb_nodes == 3
        assert ls.nb_edges == 2

    def test_delayed_different_times(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test that source and target have different times in leaves."""
        ls = linkstream_factory(delayed=True)
        ls.add_delayed_links([(0, 1, 0, 5)])  # Source at t=0, target at t=5

        # Should have leaves at both times
        assert (0, 0) in ls.leaves_dict  # Source at t=0
        assert (1, 5) in ls.leaves_dict  # Target at t=5

    def test_delayed_weighted_links(
        self,
        linkstream_factory: Callable[..., LinkStream],
        weighted_delayed_links: list[tuple[int, int, int, int, float]],
    ) -> None:
        """Test delayed links with weights."""
        ls = linkstream_factory(delayed=True)
        ls.add_delayed_links(weighted_delayed_links)

        assert ls.weight == pytest.approx(4.0)  # 1.5 + 2.5

    def test_delayed_min_max_time(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test min/max time for delayed links."""
        ls = linkstream_factory(delayed=True)
        ls.add_delayed_links(
            [
                (0, 1, 2, 8),  # Source at t=2, target at t=8
            ]
        )

        assert ls.min_time == 2
        assert ls.max_time == 8

    def test_delayed_directed(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test delayed links in directed mode."""
        ls = linkstream_factory(delayed=True, directed=True)
        ls.add_delayed_links([(0, 1, 0, 5)])

        assert ls.degrees_out[0] == 1
        assert ls.degrees_in[1] == 1

    def test_delayed_topological_neighbors(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test topological neighbors for delayed links."""
        ls = linkstream_factory(delayed=True)
        ls.add_delayed_links([(0, 1, 0, 5)])

        leaf_0 = ls.leaves_dict[(0, 0)]
        leaf_1 = ls.leaves_dict[(1, 5)]

        # Check that they are connected
        neighbor_targets = {te.target for te in leaf_0.topo_neighbors}
        assert leaf_1 in neighbor_targets


# =============================================================================
# SECTION 6: Time Neighbors Tests
# =============================================================================


class TestTimeNeighbors:
    """Tests for temporal neighbor computation."""

    def test_time_neighbors_single_node_multiple_times(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test time neighbors for a node active at multiple times."""
        ls = linkstream_factory()
        ls.add_links(
            [
                (0, 1, 0),
                (0, 2, 1),
                (0, 3, 2),
            ]
        )

        # Node 0 is active at t=0, t=1, t=2
        leaf_0_t0 = ls.leaves_dict[(0, 0)]
        leaf_0_t1 = ls.leaves_dict[(0, 1)]
        leaf_0_t2 = ls.leaves_dict[(0, 2)]

        # Check forward links
        assert leaf_0_t0.right_time_active_neighbor == leaf_0_t1
        assert leaf_0_t1.right_time_active_neighbor == leaf_0_t2
        assert leaf_0_t2.right_time_active_neighbor is None

        # Check backward links
        assert leaf_0_t0.left_time_active_neighbor is None
        assert leaf_0_t1.left_time_active_neighbor == leaf_0_t0
        assert leaf_0_t2.left_time_active_neighbor == leaf_0_t1

    def test_time_neighbors_single_timestep(
        self,
        linkstream_factory: Callable[..., LinkStream],
        simple_triangle_links: list[tuple[int, int, int]],
    ) -> None:
        """Test time neighbors when all links are at same timestep."""
        ls = linkstream_factory()
        ls.add_links(simple_triangle_links)

        # All nodes only at t=0, so no time neighbors
        for node in [0, 1, 2]:
            leaf = ls.leaves_dict[(node, 0)]
            assert leaf.left_time_active_neighbor is None
            assert leaf.right_time_active_neighbor is None

    def test_time_neighbors_non_consecutive(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test time neighbors with gaps in time."""
        ls = linkstream_factory()
        ls.add_links(
            [
                (0, 1, 0),
                (0, 2, 5),  # Gap from t=0 to t=5
                (0, 3, 10),  # Gap from t=5 to t=10
            ]
        )

        leaf_0_t0 = ls.leaves_dict[(0, 0)]
        leaf_0_t5 = ls.leaves_dict[(0, 5)]
        leaf_0_t10 = ls.leaves_dict[(0, 10)]

        # Should still be linked despite gaps
        assert leaf_0_t0.right_time_active_neighbor == leaf_0_t5
        assert leaf_0_t5.right_time_active_neighbor == leaf_0_t10


# =============================================================================
# SECTION 7: File I/O Tests
# =============================================================================


class TestFileIO:
    """Tests for file reading and writing."""

    def test_read_txt_basic(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test reading basic link stream file."""
        ls = linkstream_factory()
        ls.read_txt("tests/fixtures/linkstream.txt")

        # Verify some basic properties
        assert ls.nb_edges > 0
        assert ls.nb_nodes > 0

    def test_read_txt_custom_columns(
        self,
        linkstream_factory: Callable[..., LinkStream],
        tmp_path: Path,
    ) -> None:
        """Test reading with custom column order."""
        # Create a temp file with different column order
        test_file = tmp_path / "test_links.txt"
        test_file.write_text("0 10 1\n1 20 2\n")  # time source target

        ls = linkstream_factory()
        ls.read_txt(str(test_file), columns_order=["time", "source", "target"])

        assert ls.nb_edges == 2
        assert 10 in ls.nodes
        assert 20 in ls.nodes

    def test_to_txt_basic(
        self,
        linkstream_factory: Callable[..., LinkStream],
        simple_triangle_links: list[tuple[int, int, int]],
        tmp_path: Path,
    ) -> None:
        """Test writing link stream to file."""
        ls = linkstream_factory()
        ls.add_links(simple_triangle_links)

        output_file = tmp_path / "output.txt"
        ls.to_txt(str(output_file))

        # Verify file was created and has content
        assert output_file.exists()
        content = output_file.read_text()
        # NOTE: For undirected graphs, each edge is written twice (both directions)
        # 3 edges * 2 = 6 lines
        assert len(content.strip().split("\n")) == 6

    def test_roundtrip(
        self,
        linkstream_factory: Callable[..., LinkStream],
        simple_triangle_links: list[tuple[int, int, int]],
        tmp_path: Path,
    ) -> None:
        """Test read -> write -> read roundtrip."""
        # Create and write
        ls1 = linkstream_factory()
        ls1.add_links(simple_triangle_links)
        output_file = tmp_path / "roundtrip.txt"
        ls1.to_txt(str(output_file))

        # Read back
        ls2 = linkstream_factory()
        ls2.read_txt(str(output_file), columns_order=["source", "target", "time", "weight"])

        # Compare - nb_nodes should be same
        assert ls1.nb_nodes == ls2.nb_nodes
        # NOTE: nb_edges doubles because undirected edges are written twice
        # and then read as separate edges. This is a known behavior.
        assert ls2.nb_edges == ls1.nb_edges * 2


# =============================================================================
# SECTION 8: Properties Tests
# =============================================================================


class TestProperties:
    """Tests for LinkStream properties."""

    def test_nb_nodes(
        self,
        populated_linkstream: LinkStream,
    ) -> None:
        """Test nb_nodes property."""
        assert populated_linkstream.nb_nodes == 3

    def test_nb_timesteps(
        self,
        linkstream_factory: Callable[..., LinkStream],
        temporal_path_links: list[tuple[int, int, int]],
    ) -> None:
        """Test nb_timesteps property."""
        ls = linkstream_factory()
        ls.add_links(temporal_path_links)

        assert ls.nb_timesteps == 3

    def test_nb_time_edges(
        self,
        populated_linkstream: LinkStream,
    ) -> None:
        """Test nb_time_edges property."""
        # For undirected, each edge is stored twice
        assert populated_linkstream.nb_time_edges >= 3

    def test_get_time_links_instantaneous(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test get_time_links for instantaneous links."""
        ls = linkstream_factory()
        ls.add_links([(0, 1, 5, 2.0)])

        time_links = ls.get_time_links()
        assert len(time_links) > 0

    def test_get_time_links_continuous(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test get_time_links for continuous links."""
        ls = linkstream_factory(continuous=True)
        ls.add_continous_links([(0, 1, 0, 3)])

        time_links = ls.get_time_links()
        assert len(time_links) > 0

    def test_get_time_links_delayed(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test get_time_links for delayed links."""
        ls = linkstream_factory(delayed=True)
        ls.add_delayed_links([(0, 1, 0, 5)])

        time_links = ls.get_time_links()
        assert len(time_links) > 0


# =============================================================================
# SECTION 9: Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_self_loop(
        self,
        linkstream_factory: Callable[..., LinkStream],
        self_loop_link: list[tuple[int, int, int]],
    ) -> None:
        """Test handling of self-loops."""
        ls = linkstream_factory()
        ls.add_links(self_loop_link)

        # Self-loop: node 0 connected to itself
        assert 0 in ls.nodes
        # Degree should account for self-loop
        assert ls.degrees[0] == 2  # Both endpoints contribute

    def test_zero_weight(
        self,
        linkstream_factory: Callable[..., LinkStream],
        zero_weight_links: list[tuple[int, int, int, float]],
    ) -> None:
        """Test handling of zero-weight edges."""
        ls = linkstream_factory()
        ls.add_links(zero_weight_links)

        assert ls.nb_edges == 2
        assert ls.weight == 0.0

    def test_multiple_add_links_calls(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test calling add_links multiple times."""
        ls = linkstream_factory()
        ls.add_links([(0, 1, 0)])
        ls.add_links([(1, 2, 1)])

        assert ls.nb_edges == 2
        assert ls.nb_nodes == 3

    def test_negative_timestamps(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test handling of negative timestamps (if supported)."""
        ls = linkstream_factory()
        ls.add_links(
            [
                (0, 1, -5),
                (1, 2, -3),
                (2, 3, 0),
            ]
        )

        assert ls.min_time == -5
        assert ls.max_time == 0


# =============================================================================
# SECTION 10: Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_two_communities(
        self,
        linkstream_factory: Callable[..., LinkStream],
        two_communities_links: list[tuple[int, int, int]],
    ) -> None:
        """Test network with two clear communities."""
        ls = linkstream_factory()
        ls.add_links(two_communities_links)

        assert ls.nb_nodes == 6
        assert ls.nb_edges == 7

    def test_evolving_community(
        self,
        linkstream_factory: Callable[..., LinkStream],
        evolving_community_links: list[tuple[int, int, int]],
    ) -> None:
        """Test network with evolving communities."""
        ls = linkstream_factory()
        ls.add_links(evolving_community_links)

        assert ls.nb_timesteps == 3

    def test_directed_continuous(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test directed continuous linkstream."""
        ls = linkstream_factory(continuous=True, directed=True)
        ls.add_continous_links(
            [
                (0, 1, 0, 3),
                (1, 2, 1, 2),
            ]
        )

        assert ls.continuous is True
        assert ls.directed is True
        assert ls.nb_edges == 2

    def test_directed_delayed(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test directed delayed linkstream."""
        ls = linkstream_factory(delayed=True, directed=True)
        ls.add_delayed_links(
            [
                (0, 1, 0, 5),
                (1, 0, 2, 7),  # Reverse direction
            ]
        )

        assert ls.delayed is True
        assert ls.directed is True
        assert ls.nb_edges == 2

    def test_large_network_from_file(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test loading a larger network from file."""
        ls = linkstream_factory()
        ls.read_txt("tests/fixtures/linkstream.txt")

        # The test file has many edges
        assert ls.nb_edges > 100
        assert ls.nb_nodes > 10

    def test_partite_network(
        self,
        linkstream_factory: Callable[..., LinkStream],
    ) -> None:
        """Test bipartite network."""
        # Type A: nodes 0, 1, 2
        # Type B: nodes 3, 4, 5
        mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
        ls = linkstream_factory(partite_mapping=mapping)
        ls.add_links(
            [
                (0, 3, 0),  # A-B connection
                (1, 4, 0),  # A-B connection
                (2, 5, 0),  # A-B connection
            ]
        )

        assert ls.partite_mapping == mapping
        assert ls.nb_nodes == 6
