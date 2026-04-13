"""Tests for the data_preparation module."""

import pytest

from lago.viz.data_preparation import (
    calculate_node_time_ranges,
    create_time_node_module_mapping,
    filter_and_sort_modules,
    get_modules_to_focus,
    prepare_modules_for_display,
    sort_modules_by_size,
)


class TestPrepareModulesForDisplay:
    """Tests for prepare_modules_for_display function."""

    def test_no_focus_returns_all(self):
        """Without focus, all modules should be returned."""
        modules = {"c1": [(0, 0), (1, 0)], "c2": [(2, 1)]}
        focused, mono = prepare_modules_for_display(modules, [], None, False)
        assert focused == modules
        assert mono == {}

    def test_node_focus_filters(self):
        """Node focus should filter modules."""
        modules = {"c1": [(0, 0), (1, 0)], "c2": [(2, 1)]}
        focused, mono = prepare_modules_for_display(modules, [0], None, False)
        assert "c1" in focused
        assert "c2" in mono

    def test_time_focus_filters(self):
        """Time focus should filter modules."""
        modules = {"c1": [(0, 0), (1, 0)], "c2": [(2, 1)]}
        focused, mono = prepare_modules_for_display(modules, [], 1, False)
        assert "c2" in focused
        assert "c1" in mono


class TestCalculateNodeTimeRanges:
    """Tests for calculate_node_time_ranges function."""

    def test_no_trim(self):
        """Without trim, all nodes span full duration."""
        nodes = {0, 1, 2}
        time_links = [(0, 1, 5, 1.0), (1, 2, 10, 1.0)]
        start, end = calculate_node_time_ranges(nodes, time_links, 20, False)
        assert start == {0: 0, 1: 0, 2: 0}
        assert end == {0: 20, 1: 20, 2: 20}

    def test_with_trim(self):
        """With trim, nodes span their activity range."""
        nodes = {0, 1, 2}
        time_links = [(0, 1, 5, 1.0), (1, 2, 10, 1.0)]
        start, end = calculate_node_time_ranges(nodes, time_links, 20, True)
        assert start[0] == 5
        assert end[0] == 5
        assert start[1] == 5
        assert end[1] == 10
        assert start[2] == 10
        assert end[2] == 10


class TestFilterAndSortModules:
    """Tests for filter_and_sort_modules function."""

    def test_max_shown_limits(self):
        """max_shown_modules should limit number returned."""
        modules = {
            "c1": [(0, 0), (1, 0)],
            "c2": [(2, 1)],
            "c3": [(3, 2), (4, 2), (5, 2)],
        }
        result = filter_and_sort_modules(modules, 2, False)
        assert len(result) == 2

    def test_hide_self_modules(self):
        """Single-node modules should be hidden."""
        modules = {
            "c1": [(0, 0), (1, 0)],
            "c2": [(2, 1)],  # Single member
        }
        result = filter_and_sort_modules(modules, -1, True)
        assert "c1" in result
        assert "c2" not in result

    def test_no_filter(self):
        """Without filters, all modules returned."""
        modules = {"c1": [(0, 0)], "c2": [(1, 1)]}
        result = filter_and_sort_modules(modules, -1, False)
        assert len(result) == 2


class TestCreateTimeNodeModuleMapping:
    """Tests for create_time_node_module_mapping function."""

    def test_basic_mapping(self):
        """Should create correct mapping."""
        modules = {"c1": [(0, 0), (1, 0)], "c2": [(0, 1)]}
        mapping = create_time_node_module_mapping(modules)
        assert mapping[(0, 0)] == "c1"
        assert mapping[(1, 0)] == "c1"
        assert mapping[(0, 1)] == "c2"

    def test_empty_modules(self):
        """Empty modules should return empty mapping."""
        mapping = create_time_node_module_mapping({})
        assert mapping == {}


class TestGetModulesToFocus:
    """Tests for get_modules_to_focus function."""

    def test_node_focus_only(self):
        """Node focus should find modules containing that node."""
        modules = {"c1": [(0, 0), (1, 0)], "c2": [(2, 1)]}
        result = get_modules_to_focus(modules, [0], None, False)
        assert result == {"c1"}

    def test_time_focus_only(self):
        """Time focus should find modules at that time."""
        modules = {"c1": [(0, 0), (1, 0)], "c2": [(2, 1)]}
        result = get_modules_to_focus(modules, [], 1, False)
        assert result == {"c2"}

    def test_node_and_time_or(self):
        """OR logic should match either condition."""
        modules = {"c1": [(0, 0)], "c2": [(2, 1)]}
        result = get_modules_to_focus(modules, [0], 1, True)
        assert result == {"c1", "c2"}

    def test_node_and_time_and(self):
        """AND logic should match both conditions."""
        modules = {"c1": [(0, 0)], "c2": [(0, 1)]}
        result = get_modules_to_focus(modules, [0], 1, False)
        assert result == {"c2"}


class TestSortModulesBySize:
    """Tests for sort_modules_by_size function."""

    def test_sorts_by_size(self):
        """Should return top N largest modules."""
        modules = {
            "small": [(0, 0)],
            "medium": [(1, 0), (2, 0)],
            "large": [(3, 0), (4, 0), (5, 0)],
        }
        result = sort_modules_by_size(modules, 2)
        keys = list(result.keys())
        assert len(keys) == 2
        assert "large" in keys
        assert "medium" in keys

    def test_top_n_limits(self):
        """Should limit to top_n modules."""
        modules = {f"c{i}": [(i, 0)] for i in range(10)}
        result = sort_modules_by_size(modules, 3)
        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
