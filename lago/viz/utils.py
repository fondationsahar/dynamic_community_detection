"""
Utility functions module for longitudinal module plotting.

This module provides utility functions for color generation, data segmentation,
and other helper operations used throughout the longitudinal module visualization pipeline.

Functions:
    - generate_monochrome_colors: Generate monochrome color palette
    - generate_pastel_colors: Generate pastel color palette
    - generate_colors_from_palette: Generate colors from custom palette or colormap
    - generate_color_mapping: Create color mapping for modules
    - to_segments: Convert time points to continuous segments for optimization
"""

from collections.abc import Sequence
import logging
from typing import Any, Union

import matplotlib
import matplotlib.colors as mcolors
import numpy as np

# Type alias for color specification
ColorType = Union[str, tuple, Any]
ColorPalette = Union[str, Sequence[ColorType]]


def generate_monochrome_colors(n_colors: int, alpha: float = 1) -> list[str]:
    """
    Generate a list of monochrome colors.

    Args:
        n_colors: Number of colors to generate
        alpha: Alpha/transparency value

    Returns:
        List of hex color strings
    """
    alpha_value = format(int(alpha * 255), "02x")
    greys = []
    step = 255 // (n_colors * 2 - 1) if n_colors > 1 else 0
    for i in range(n_colors, n_colors * 2):
        grey_value = format(i * step, "02x")
        grey_color = f"#{grey_value}{grey_value}{grey_value}{alpha_value}"
        greys.append(grey_color)

    return greys


def generate_pastel_colors(n: int) -> list:
    """
    Generate a list of pastel colors.

    Args:
        n: Number of colors to generate

    Returns:
        List of RGB color tuples
    """
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = [mcolors.hsv_to_rgb((h, 0.5, 0.9)) for h in hues]
    return colors


def generate_colors_from_palette(n: int, palette: ColorPalette | None = None) -> list[ColorType]:
    """
    Generate colors from a custom palette or matplotlib colormap.

    Args:
        n: Number of colors to generate
        palette: Color palette specification. Can be:
            - None: Uses default pastel colors
            - str: Name of a matplotlib colormap (e.g., 'viridis', 'tab10', 'Set2', 'Pastel1')
            - Sequence of colors: List/tuple of color specifications (hex strings, RGB tuples, or named colors)

    Returns:
        List of color specifications

    Example:
        >>> colors = generate_colors_from_palette(5, 'tab10')
        >>> colors = generate_colors_from_palette(5, ['red', 'blue', 'green', '#FF5733', (0.5, 0.5, 0.5)])
        >>> colors = generate_colors_from_palette(5, 'viridis')
    """
    if palette is None:
        return generate_pastel_colors(n)

    if isinstance(palette, str):
        # It's a matplotlib colormap name
        try:
            cmap = matplotlib.colormaps[palette]
            # Sample n colors from the colormap
            colors_attr = getattr(cmap, "colors", None)
            if colors_attr is not None:
                # Discrete colormap (like tab10, Set2) - cycle through exact colors
                colors = [colors_attr[i % len(colors_attr)] for i in range(n)]
            else:
                # Continuous colormap (like viridis) - sample evenly
                colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
            return colors
        except ValueError:
            # Invalid colormap name, fall back to pastel colors
            logging.getLogger(__name__).warning("Unknown colormap '%s', using default pastel colors", palette)
            return generate_pastel_colors(n)
    else:
        # It's a sequence of colors - cycle through them if needed
        palette_list = list(palette)
        if not palette_list:
            return generate_pastel_colors(n)
        colors = [palette_list[i % len(palette_list)] for i in range(n)]
        return colors


def _desaturate_color(color: ColorType, factor: float = 0.3) -> tuple:
    """
    Desaturate a color by blending it with grey.

    Args:
        color: Color to desaturate
        factor: Desaturation factor (0=no change, 1=fully grey)

    Returns:
        Desaturated color as RGB tuple
    """
    try:
        rgb = mcolors.to_rgb(color)
        grey = 0.5  # Target grey value
        return tuple(c * (1 - factor) + grey * factor for c in rgb)
    except (ValueError, TypeError):
        return (0.8, 0.8, 0.8)  # Default to light grey on error


def _lighten_color(color: ColorType, factor: float = 0.5) -> tuple:
    """
    Lighten a color by blending it with white.

    Args:
        color: Color to lighten
        factor: Lightening factor (0=no change, 1=fully white)

    Returns:
        Lightened color as RGB tuple
    """
    try:
        rgb = mcolors.to_rgb(color)
        return tuple(c + (1 - c) * factor for c in rgb)
    except (ValueError, TypeError):
        return (0.9, 0.9, 0.9)  # Default to very light grey on error


def _add_alpha_to_color(color: ColorType, alpha: float = 0.3) -> tuple:
    """
    Add alpha channel to a color.

    Args:
        color: Color to modify
        alpha: Alpha value (0=transparent, 1=opaque)

    Returns:
        Color as RGBA tuple
    """
    try:
        rgb = mcolors.to_rgb(color)
        return (*rgb, alpha)
    except (ValueError, TypeError):
        return (0.8, 0.8, 0.8, alpha)


def _normalize_color(color: ColorType) -> tuple:
    """
    Normalize a color to RGB tuple format.

    Args:
        color: Color in any format (list, tuple, hex string, named color)

    Returns:
        RGB tuple (r, g, b) with values 0-1
    """
    try:
        # Handle lists - convert to tuple first
        if isinstance(color, list):
            color = tuple(color)
        return mcolors.to_rgb(color)
    except (ValueError, TypeError):
        return (0.5, 0.5, 0.5)  # Default grey


def _colors_are_similar(color1: ColorType, color2: ColorType, threshold: float = 0.1) -> bool:
    """
    Check if two colors are similar (within a threshold).

    Args:
        color1: First color
        color2: Second color
        threshold: Maximum distance between colors to consider them similar

    Returns:
        True if colors are similar, False otherwise
    """
    try:
        rgb1 = _normalize_color(color1)
        rgb2 = _normalize_color(color2)
        # Euclidean distance in RGB space
        distance = sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5
        return distance < threshold
    except (ValueError, TypeError):
        # If we can't convert, assume they're different
        return False


def generate_color_mapping(
    focused_modules: dict,
    secondary_modules: dict,
    monochrome: bool,
    palette: ColorPalette | None = None,
    explicit_colors: dict[Any, ColorType] | None = None,
) -> dict:
    """
    Generate color mapping for modules.

    Focused modules get colors from the palette (or explicit colors if provided).
    Secondary modules get the remaining palette colors.

    When explicit_colors is provided, those colors are used for focused modules
    that have explicit assignments, and the remaining palette colors (excluding
    those similar to explicit colors) are used for other modules.

    Args:
        focused_modules: Focused modules to display with colors
        secondary_modules: Secondary modules to display with remaining colors
        monochrome: Whether to use monochrome scheme
        palette: Optional color palette specification. Can be:
            - None: Uses default colors (pastel or monochrome)
            - str: Name of a matplotlib colormap (e.g., 'viridis', 'tab10', 'Set2')
            - Sequence of colors: List of color specifications
        explicit_colors: Optional dict mapping module labels to explicit colors.
            These colors will be used directly and excluded from palette assignment.

    Returns:
        Dictionary mapping module labels to colors
    """
    explicit_colors = explicit_colors or {}

    if monochrome:
        colors = generate_monochrome_colors(len(focused_modules))
        secondary_colors = ["gainsboro"] * (len(secondary_modules) + 1)
    else:
        # Generate palette colors
        # We need enough colors for all modules that don't have explicit colors
        n_focused_no_explicit = sum(
            1 for label in focused_modules if label not in explicit_colors
        )
        n_secondary = len(secondary_modules)
        total_needed = n_focused_no_explicit + n_secondary

        # Generate more colors than needed so we can filter out similar ones
        all_colors = generate_colors_from_palette(
            max(total_needed + len(explicit_colors), 20), palette
        )

        # Filter out palette colors that are similar to explicit colors
        if explicit_colors:
            explicit_color_values = list(explicit_colors.values())
            filtered_colors = []
            for color in all_colors:
                is_similar = any(_colors_are_similar(color, ec) for ec in explicit_color_values)
                if not is_similar:
                    filtered_colors.append(color)
            all_colors = filtered_colors if filtered_colors else all_colors

        # Assign colors to focused modules
        colors = []
        focused_labels = list(focused_modules.keys())
        color_idx = 0

        for label in focused_labels:
            if label in explicit_colors:
                # Use explicit color
                colors.append(explicit_colors[label])
            else:
                # Use next palette color
                if color_idx < len(all_colors):
                    colors.append(all_colors[color_idx])
                    color_idx += 1
                else:
                    # Fallback to pastel if we run out
                    colors.append(generate_pastel_colors(1)[0])

        # Remaining colors for secondary modules
        remaining_colors = all_colors[color_idx:] if color_idx < len(all_colors) else []

        # Secondary modules get remaining palette colors
        secondary_colors = remaining_colors[: len(secondary_modules)]
        # Pad with gainsboro if not enough colors
        while len(secondary_colors) < len(secondary_modules):
            secondary_colors.append("gainsboro")

    # Create color mapping for focused modules
    color_mapping = {
        module_label: color for color, module_label in zip(colors, focused_modules.keys())
    }

    # Add secondary modules to mapping
    color_mapping.update(
        {
            module_label: color
            for color, module_label in zip(secondary_colors, secondary_modules.keys())
        }
    )

    return color_mapping


def to_segments(nums: list[int]) -> list[list[int]]:
    """
    Convert a list of numbers into continuous segments.

    This utility function identifies consecutive numbers and groups them into
    segments, which is useful for optimizing visualization of time series data.

    Args:
        nums: List of integers to segment

    Returns:
        List of [start, end] segments representing continuous ranges

    Example:
        >>> to_segments([1, 2, 3, 5, 6, 8])
        [[1, 3], [5, 6], [8, 8]]
        >>> to_segments([])
        []
        >>> to_segments([5])
        [[5, 5]]
    """
    if not nums:
        return []

    nums = sorted(nums)  # Ensure input is sorted
    segments = []

    start = prev = nums[0]

    for n in nums[1:]:
        if n == prev + 1:
            # Consecutive number, extend current segment
            prev = n
        else:
            # Gap detected, finalize current segment and start new one
            segments.append([start, prev])
            start = prev = n

    # Add the final segment
    segments.append([start, prev])
    return segments
