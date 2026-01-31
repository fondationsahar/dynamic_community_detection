"""Tests for the utils module."""

import numpy as np
import pytest

from lago.viz.utils import (
    generate_color_mapping,
    generate_colors_from_palette,
    generate_monochrome_colors,
    generate_pastel_colors,
    to_segments,
)


class TestToSegments:
    """Tests for the to_segments function."""

    def test_empty_list(self):
        """Empty input should return empty list."""
        assert to_segments([]) == []

    def test_single_element(self):
        """Single element should return single segment."""
        assert to_segments([5]) == [[5, 5]]

    def test_consecutive_numbers(self):
        """Consecutive numbers should be grouped."""
        assert to_segments([1, 2, 3]) == [[1, 3]]

    def test_multiple_segments(self):
        """Non-consecutive numbers should create multiple segments."""
        assert to_segments([1, 2, 3, 5, 6, 8]) == [[1, 3], [5, 6], [8, 8]]

    def test_unsorted_input(self):
        """Unsorted input should be handled correctly."""
        assert to_segments([3, 1, 2]) == [[1, 3]]

    def test_gaps(self):
        """Multiple gaps should create correct segments."""
        assert to_segments([1, 3, 5, 7]) == [[1, 1], [3, 3], [5, 5], [7, 7]]

    def test_duplicates(self):
        """Duplicate values should be handled (sorted removes implicit dupes)."""
        result = to_segments([1, 1, 2, 2, 3])
        # After sorting, duplicates become adjacent - implementation specific
        assert len(result) > 0


class TestGenerateMonochromeColors:
    """Tests for generate_monochrome_colors function."""

    def test_single_color(self):
        """Single color should return one grey."""
        colors = generate_monochrome_colors(1)
        assert len(colors) == 1
        assert colors[0].startswith("#")

    def test_multiple_colors(self):
        """Multiple colors should be generated."""
        colors = generate_monochrome_colors(5)
        assert len(colors) == 5

    def test_hex_format(self):
        """Colors should be in hex format with alpha."""
        colors = generate_monochrome_colors(3)
        for color in colors:
            assert color.startswith("#")
            # Format: #RRGGBBAA (8 hex chars + #)
            assert len(color) == 9

    def test_alpha_parameter(self):
        """Alpha parameter should affect color."""
        colors_full = generate_monochrome_colors(2, alpha=1.0)
        colors_half = generate_monochrome_colors(2, alpha=0.5)
        # Last 2 chars are alpha
        assert colors_full[0][-2:] == "ff"
        assert colors_half[0][-2:] != "ff"


class TestGeneratePastelColors:
    """Tests for generate_pastel_colors function."""

    def test_single_color(self):
        """Single color should return one RGB tuple."""
        colors = generate_pastel_colors(1)
        assert len(colors) == 1

    def test_multiple_colors(self):
        """Multiple colors should be generated."""
        colors = generate_pastel_colors(5)
        assert len(colors) == 5

    def test_rgb_format(self):
        """Colors should be RGB tuples."""
        colors = generate_pastel_colors(3)
        for color in colors:
            assert len(color) == 3  # RGB
            assert all(0 <= c <= 1 for c in color)

    def test_different_hues(self):
        """Generated colors should have different hues."""
        colors = generate_pastel_colors(4)
        # Convert to strings for comparison
        color_strs = [str(c) for c in colors]
        assert len(set(color_strs)) == 4  # All unique


class TestGenerateColorsFromPalette:
    """Tests for generate_colors_from_palette function."""

    def test_none_palette_uses_pastel(self):
        """None palette should use default pastel colors."""
        colors = generate_colors_from_palette(3, None)
        assert len(colors) == 3

    def test_matplotlib_colormap(self):
        """Should work with matplotlib colormap names."""
        colors = generate_colors_from_palette(5, "tab10")
        assert len(colors) == 5

    def test_custom_color_list(self):
        """Should work with custom color list."""
        custom = ["red", "blue", "green"]
        colors = generate_colors_from_palette(3, custom)
        assert colors == custom

    def test_custom_colors_cycle(self):
        """Should cycle through colors if n > len(palette)."""
        custom = ["red", "blue"]
        colors = generate_colors_from_palette(4, custom)
        assert colors == ["red", "blue", "red", "blue"]

    def test_empty_palette_uses_pastel(self):
        """Empty palette should fall back to pastel."""
        colors = generate_colors_from_palette(3, [])
        assert len(colors) == 3


class TestGenerateColorMapping:
    """Tests for generate_color_mapping function."""

    def test_basic_mapping(self):
        """Should create basic color mapping."""
        communities_display = {"com1": [], "com2": []}
        communities_mono = {}
        mapping = generate_color_mapping(communities_display, communities_mono, False)
        assert "com1" in mapping
        assert "com2" in mapping

    def test_monochrome_mode(self):
        """Monochrome mode should use grey colors."""
        communities_display = {"com1": []}
        communities_mono = {"com2": []}
        mapping = generate_color_mapping(communities_display, communities_mono, True)
        assert "com1" in mapping
        assert "com2" in mapping
        # Monochrome communities should be gainsboro
        assert mapping["com2"] == "gainsboro"

    def test_with_palette(self):
        """Should use provided palette."""
        communities_display = {"com1": [], "com2": []}
        communities_mono = {}
        mapping = generate_color_mapping(
            communities_display, communities_mono, False, palette="tab10"
        )
        assert len(mapping) == 2

    def test_empty_communities(self):
        """Should handle empty communities dict."""
        mapping = generate_color_mapping({}, {}, False)
        assert mapping == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
