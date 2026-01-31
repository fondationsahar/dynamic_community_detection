"""Tests for TimeModules and TimeModule classes."""

import json
import tempfile
from pathlib import Path

import pytest

from lago.core.time_modules import NodeModuleMembership, TimeModule, TimeModules, TimeSegment


class TestTimeSegment:
    """Tests for TimeSegment dataclass."""

    def test_duration(self):
        """Test segment duration calculation."""
        segment = TimeSegment(start=0, end=5)
        assert segment.duration == 6

    def test_contains(self):
        """Test __contains__ for time point membership."""
        segment = TimeSegment(start=0, end=5)
        assert 0 in segment
        assert 3 in segment
        assert 5 in segment
        assert -1 not in segment
        assert 6 not in segment

    def test_overlaps(self):
        """Test segment overlap detection."""
        s1 = TimeSegment(start=0, end=5)
        s2 = TimeSegment(start=3, end=8)
        s3 = TimeSegment(start=6, end=10)
        s4 = TimeSegment(start=0, end=0)

        assert s1.overlaps(s2)  # Partial overlap
        assert s2.overlaps(s1)  # Symmetric
        assert not s1.overlaps(s3)  # No overlap
        assert s1.overlaps(s4)  # Contained


class TestTimeModulesConstruction:
    """Tests for TimeModules construction."""

    def test_empty_construction(self):
        """Test empty TimeModules."""
        tm = TimeModules()
        assert tm.nb_modules == 0
        assert tm.nb_nodes == 0
        assert tm.nb_times == 0

    def test_from_raw_modules(self):
        """Test construction from raw modules dict."""
        raw = {
            0: {(0, 0), (0, 1), (1, 0)},
            1: {(2, 0), (2, 1)},
        }
        tm = TimeModules(raw)

        assert tm.nb_modules == 2
        assert tm.nb_nodes == 3
        assert tm.nb_times == 2
        assert 0 in tm.modules
        assert 1 in tm.modules

    def test_properties(self):
        """Test basic properties."""
        raw = {0: {(0, 0), (1, 1), (2, 2)}}
        tm = TimeModules(raw)

        assert tm.nodes == frozenset({0, 1, 2})
        assert tm.times == frozenset({0, 1, 2})
        assert tm.modules == frozenset({0})
        assert tm.time_range == (0, 2)


class TestTimeModulesNodeCentric:
    """Tests for node-centric TimeModules methods."""

    @pytest.fixture
    def tm(self):
        """Create a TimeModules for testing."""
        raw = {
            0: {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)},  # Module 0
            1: {(0, 3), (0, 4), (2, 0), (2, 1)},  # Module 1
        }
        return TimeModules(raw)

    def test_get_modules_of_node(self, tm):
        """Test getting modules for a node."""
        memberships = tm.get_modules_of_node(0)

        # Node 0 is in modules 0 and 1
        assert len(memberships) == 2
        module_labels = {m.module for m in memberships}
        assert module_labels == {0, 1}

    def test_get_modules_of_node_not_found(self, tm):
        """Test KeyError for unknown node."""
        with pytest.raises(KeyError):
            tm.get_modules_of_node(999)

    def test_get_module_of_node_at_time(self, tm):
        """Test getting module for node at specific time."""
        assert tm.get_module_of_node_at_time(0, 0) == 0
        assert tm.get_module_of_node_at_time(0, 3) == 1
        assert tm.get_module_of_node_at_time(0, 10) is None  # Not active

    def test_get_node_trajectory(self, tm):
        """Test node trajectory."""
        trajectory = tm.get_node_trajectory(0)
        expected = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
        assert trajectory == expected

    def test_get_node_switches(self, tm):
        """Test detecting node module switches."""
        switches = tm.get_node_switches(0)
        # Node 0 switches from module 0 to 1 at time 3
        assert len(switches) == 1
        assert switches[0] == (3, 0, 1)

    def test_get_stability_score(self, tm):
        """Test stability score calculation."""
        # Node 1 never switches (always in module 0)
        assert tm.get_stability_score(1) == 1.0

        # Node 0 switches once over 5 time points (4 transitions, 1 switch)
        assert tm.get_stability_score(0) == 1.0 - 1 / 4


class TestTimeModulesTimeCentric:
    """Tests for time-centric TimeModules methods."""

    @pytest.fixture
    def tm(self):
        """Create a TimeModules for testing."""
        raw = {
            0: {(0, 0), (1, 0), (2, 0)},
            1: {(3, 0), (4, 0)},
        }
        return TimeModules(raw)

    def test_get_nodes_modules_membership_at_time(self, tm):
        """Test getting node-module mapping at time."""
        mapping = tm.get_nodes_modules_membership_at_time(0)
        assert mapping == {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}

    def test_get_active_nodes_at_time(self, tm):
        """Test getting active nodes."""
        nodes = tm.get_active_nodes_at_time(0)
        assert nodes == frozenset({0, 1, 2, 3, 4})

    def test_get_modules_at_time(self, tm):
        """Test getting modules at time."""
        modules = tm.get_modules_at_time(0)
        assert modules == frozenset({0, 1})


class TestTimeModulesModuleCentric:
    """Tests for module-centric TimeModules methods."""

    @pytest.fixture
    def tm(self):
        """Create a TimeModules for testing."""
        raw = {
            0: {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)},
        }
        return TimeModules(raw)

    def test_get_module_nodes(self, tm):
        """Test getting module nodes."""
        nodes = tm.get_module_nodes(0)
        assert nodes == frozenset({0, 1})

    def test_get_module_size(self, tm):
        """Test module size."""
        assert tm.get_module_size(0) == 2

    def test_get_module_duration(self, tm):
        """Test module duration."""
        assert tm.get_module_duration(0) == 3  # Times 0, 1, 2

    def test_get_module_members_at_time(self, tm):
        """Test members at specific time."""
        assert tm.get_module_members_at_time(0, 0) == frozenset({0, 1})
        assert tm.get_module_members_at_time(0, 2) == frozenset({1})

    def test_get_module_cohesion(self, tm):
        """Test cohesion calculation."""
        # 2 nodes, 3 times = 6 possible
        # Actual: node 0 at times 0,1 (2) + node 1 at times 0,1,2 (3) = 5
        # Cohesion = 5/6
        assert tm.get_module_cohesion(0) == pytest.approx(5 / 6)

    def test_get_time_modules_dict(self, tm):
        """Test getting segment-based dict."""
        result = tm.get_time_modules_dict()
        assert 0 in result
        assert 0 in result[0]  # node 0
        assert 1 in result[0]  # node 1


class TestTimeModule:
    """Tests for TimeModule view class."""

    @pytest.fixture
    def tm(self):
        """Create a TimeModules for testing."""
        raw = {
            0: {(0, 0), (0, 1), (1, 0)},
            1: {(2, 0), (2, 1)},
        }
        return TimeModules(raw)

    def test_get_module(self, tm):
        """Test getting a TimeModule view."""
        module = tm.get_module(0)
        assert isinstance(module, TimeModule)
        assert module.label == 0

    def test_module_properties(self, tm):
        """Test TimeModule properties."""
        module = tm.get_module(0)

        assert module.nodes == frozenset({0, 1})
        assert module.times == frozenset({0, 1})
        assert module.size == 2
        assert module.duration == 2

    def test_module_members(self, tm):
        """Test members property."""
        module = tm.get_module(0)
        expected = frozenset({(0, 0), (0, 1), (1, 0)})
        assert module.members == expected

    def test_module_contains(self, tm):
        """Test __contains__ for TimeModule."""
        module = tm.get_module(0)

        # Check node
        assert 0 in module
        assert 2 not in module

        # Check (node, time)
        assert (0, 0) in module
        assert (0, 2) not in module

    def test_module_not_found(self, tm):
        """Test KeyError for unknown module."""
        with pytest.raises(KeyError):
            tm.get_module(999)

    def test_iter_modules(self, tm):
        """Test iterating over modules."""
        modules = list(tm.iter_modules())
        assert len(modules) == 2
        labels = {m.label for m in modules}
        assert labels == {0, 1}

    def test_module_overlaps(self, tm):
        """Test overlap detection between modules."""
        m0 = tm.get_module(0)
        m1 = tm.get_module(1)
        assert not m0.overlaps_with(m1)

        # Create overlapping modules
        raw = {0: {(0, 0)}, 1: {(0, 0)}}
        tm2 = TimeModules(raw)
        assert tm2.get_module(0).overlaps_with(tm2.get_module(1))


class TestTimeModulesConversion:
    """Tests for conversion methods."""

    @pytest.fixture
    def tm(self):
        """Create a TimeModules for testing."""
        raw = {
            0: {(0, 0), (0, 1), (1, 0)},
            1: {(2, 0)},
        }
        return TimeModules(raw)

    def test_to_communities_dict(self, tm):
        """Test conversion to communities dict."""
        result = tm.to_communities_dict()
        assert 0 in result
        assert 1 in result
        assert (0, 0) in result[0]

    def test_to_flat_labels(self, tm):
        """Test conversion to flat labels."""
        labels = tm.to_flat_labels()
        assert labels[(0, 0)] == 0
        assert labels[(2, 0)] == 1


class TestTimeModulesFileIO:
    """Tests for file I/O operations."""

    @pytest.fixture
    def tm(self):
        """Create a TimeModules for testing."""
        raw = {
            0: {(0, 0), (0, 1), (1, 0)},
            1: {(2, 0), (2, 1)},
        }
        return TimeModules(raw)

    def test_txt_roundtrip(self, tm):
        """Test save/load TXT roundtrip."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = Path(f.name)

        try:
            tm.to_txt(path)
            loaded = TimeModules.from_txt(path)

            assert loaded.nb_modules == tm.nb_modules
            assert loaded.nodes == tm.nodes
            assert loaded.times == tm.times
        finally:
            path.unlink()

    def test_csv_roundtrip(self, tm):
        """Test save/load CSV roundtrip."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            tm.to_csv(path)
            loaded = TimeModules.from_csv(path)

            assert loaded.nb_modules == tm.nb_modules
            assert loaded.nodes == tm.nodes
        finally:
            path.unlink()

    def test_json_standard_roundtrip(self, tm):
        """Test JSON roundtrip with standard format."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            tm.to_json(path, use_segments=False)
            loaded = TimeModules.from_json(path)

            assert loaded.nb_modules == tm.nb_modules
            assert loaded.nodes == tm.nodes
        finally:
            path.unlink()

    def test_json_segments_roundtrip(self, tm):
        """Test JSON roundtrip with segment format."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            tm.to_json(path, use_segments=True)
            loaded = TimeModules.from_json(path)

            assert loaded.nb_modules == tm.nb_modules
            assert loaded.nodes == tm.nodes
        finally:
            path.unlink()

    def test_load_from_path_constructor(self, tm):
        """Test loading from path via constructor."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            tm.to_json(path)
            loaded = TimeModules(path=path)

            assert loaded.nb_modules == tm.nb_modules
        finally:
            path.unlink()

    def test_cannot_provide_both_args(self, tm):
        """Test error when providing both raw_modules and path."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
            tm.to_json(path)

        try:
            with pytest.raises(ValueError, match="Cannot provide both"):
                TimeModules(raw_modules={0: set()}, path=path)
        finally:
            path.unlink()


class TestTimeModulesContains:
    """Tests for __contains__ magic method."""

    @pytest.fixture
    def tm(self):
        """Create a TimeModules for testing."""
        raw = {0: {(0, 0), (1, 1)}}
        return TimeModules(raw)

    def test_contains_module(self, tm):
        """Test checking if module label exists."""
        assert 0 in tm
        assert 1 not in tm

    def test_contains_node_time(self, tm):
        """Test checking if (node, time) exists."""
        assert (0, 0) in tm
        assert (0, 1) not in tm


class TestTimesToSegments:
    """Tests for _times_to_segments helper."""

    def test_empty(self):
        """Test empty input."""
        result = TimeModules._times_to_segments(set())
        assert result == []

    def test_single(self):
        """Test single time point."""
        result = TimeModules._times_to_segments({5})
        assert len(result) == 1
        assert result[0] == TimeSegment(start=5, end=5)

    def test_contiguous(self):
        """Test contiguous time points."""
        result = TimeModules._times_to_segments({0, 1, 2, 3})
        assert len(result) == 1
        assert result[0] == TimeSegment(start=0, end=3)

    def test_gaps(self):
        """Test time points with gaps."""
        result = TimeModules._times_to_segments({0, 1, 5, 6, 7, 10})
        assert len(result) == 3
        assert result[0] == TimeSegment(start=0, end=1)
        assert result[1] == TimeSegment(start=5, end=7)
        assert result[2] == TimeSegment(start=10, end=10)


class TestSummary:
    """Tests for summary method."""

    def test_summary(self):
        """Test summary output."""
        raw = {
            0: {(0, 0), (0, 1)},
            1: {(1, 0)},
        }
        tm = TimeModules(raw)
        summary = tm.summary()

        assert summary["nb_nodes"] == 2
        assert summary["nb_times"] == 2
        assert summary["nb_modules"] == 2
        assert summary["module_sizes"] == {0: 1, 1: 1}
