"""
Utility functions module for longitudinal community plotting.

This module provides utility functions for color generation, data segmentation,
and other helper operations used throughout the longitudinal community visualization pipeline.

Functions:
    - generate_monochrome_colors: Generate monochrome color palette
    - generate_pastel_colors: Generate pastel color palette
    - generate_colors_from_palette: Generate colors from custom palette or colormap
    - generate_color_mapping: Create color mapping for communities
    - to_segments: Convert time points to continuous segments for optimization
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
import matplotlib.colors as mcolors
import numpy as np

# Type alias for color specification
ColorType = Union[str, tuple, Any]
ColorPalette = Union[str, Sequence[ColorType]]


def generate_monochrome_colors(n_colors: int, alpha: float = 1) -> List[str]:
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


def generate_pastel_colors(n: int) -> List:
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


def generate_colors_from_palette(
    n: int, palette: Optional[ColorPalette] = None
) -> List[ColorType]:
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
            if colors_attr is not None and len(colors_attr) >= n:
                # Discrete colormap (like tab10, Set2) - use exact colors
                colors = [colors_attr[i % len(colors_attr)] for i in range(n)]
            else:
                # Continuous colormap (like viridis) - sample evenly
                colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
            return colors
        except ValueError:
            # Invalid colormap name, fall back to pastel colors
            print(f"Warning: Unknown colormap '{palette}', using default pastel colors")
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


def generate_color_mapping(
    communities_to_display: Dict,
    communities_monochrome: Dict,
    monochrome: bool,
    palette: Optional[ColorPalette] = None,
    unfocused_style: str = "grey",
    unfocused_alpha: float = 0.3,
) -> Dict:
    """
    Generate color mapping for communities.

    Tier 1 (focused) communities get maximally distinctive colors from the palette,
    evenly spaced across the color spectrum. Tier 2 (secondary) communities get
    the remaining colors (either de-emphasized or as-is depending on style).

    Args:
        communities_to_display: Tier 1 communities to display with full colors
        communities_monochrome: Tier 2 communities to display in de-emphasized style
        monochrome: Whether to use monochrome scheme
        palette: Optional color palette specification. Can be:
            - None: Uses default colors (pastel or monochrome)
            - str: Name of a matplotlib colormap (e.g., 'viridis', 'tab10', 'Set2')
            - Sequence of colors: List of color specifications
        unfocused_style: Style for unfocused communities. Options:
            - "grey": Display in different shades of grey
            - "transparent": Same colors but with reduced alpha
            - "desaturated": Desaturated (greyed-out) versions of colors
            - "lighter": Lighter/pastel versions of their colors
        unfocused_alpha: Intensity parameter (0.0 to 1.0)

    Returns:
        Dictionary mapping community labels to colors
    """
    if monochrome:
        colors = generate_monochrome_colors(len(communities_to_display))
        # Use light grey for monochrome communities
        unfocused_colors = ["gainsboro"] * (len(communities_monochrome) + 1)
    else:
        # Generate all colors we need for both Tier 1 and Tier 2
        total_colored = len(communities_to_display) + len(communities_monochrome)
        all_colors = generate_colors_from_palette(max(total_colored, 1), palette)

        # For Tier 1, pick maximally spaced colors from the full palette
        n_tier1 = len(communities_to_display)
        if n_tier1 > 0 and len(all_colors) > 0:
            # Calculate evenly spaced indices across the full color range
            if n_tier1 == 1:
                tier1_indices = [0]
            else:
                step = len(all_colors) / n_tier1
                tier1_indices = [int(i * step) for i in range(n_tier1)]
            colors = [all_colors[i] for i in tier1_indices]

            # Tier 2 gets the remaining colors (those not picked by Tier 1)
            tier1_indices_set = set(tier1_indices)
            remaining_colors = [
                c for i, c in enumerate(all_colors) if i not in tier1_indices_set
            ]
        else:
            colors = []
            remaining_colors = list(all_colors)

        # Generate unfocused colors based on style
        # The unfocused_alpha/intensity parameter controls de-emphasis:
        # - Lower values = closer to Tier 1 (more visible)
        # - Higher values = closer to Tier 3 (less visible)
        if unfocused_style == "grey":
            # Generate different shades of grey for Tier 2, distinct from gainsboro (Tier 3)
            # Intensity controls how dark/light the greys are (0=darker, 1=lighter/closer to gainsboro)
            n = len(communities_monochrome)
            if n > 0:
                # Generate shades from dark grey to light grey (but not as light as gainsboro)
                # gainsboro is approx (0.86, 0.86, 0.86), so we stay below that
                base_grey = (
                    0.4 + 0.3 * unfocused_alpha
                )  # Range: 0.4 (dark) to 0.7 (medium grey)
                step = 0.15 / max(n - 1, 1)  # Small variation between shades
                unfocused_colors = [
                    (base_grey + i * step, base_grey + i * step, base_grey + i * step)
                    for i in range(n)
                ]
            else:
                unfocused_colors = []
        elif unfocused_style == "transparent":
            # Use remaining colors (those not picked by Tier 1) and add alpha
            # intensity directly controls opacity (0=invisible, 1=fully opaque)
            base_unfocused = remaining_colors[: len(communities_monochrome)]
            unfocused_colors = [
                _add_alpha_to_color(c, unfocused_alpha) for c in base_unfocused
            ]
        elif unfocused_style == "desaturated":
            # Use remaining colors (those not picked by Tier 1) and desaturate
            # intensity controls desaturation (0=no change, 1=fully grey)
            base_unfocused = remaining_colors[: len(communities_monochrome)]
            unfocused_colors = [
                _desaturate_color(c, unfocused_alpha) for c in base_unfocused
            ]
        elif unfocused_style == "lighter":
            # Use remaining colors (those not picked by Tier 1) and lighten
            # intensity controls lightening (0=no change, 1=fully white)
            base_unfocused = remaining_colors[: len(communities_monochrome)]
            unfocused_colors = [
                _lighten_color(c, unfocused_alpha) for c in base_unfocused
            ]
        else:
            unfocused_colors = ["gainsboro"] * (len(communities_monochrome) + 1)

    # Create color mapping for display communities
    color_mapping = {
        community_label: color
        for color, community_label in zip(colors, communities_to_display.keys())
    }

    # Add unfocused communities to mapping
    color_mapping.update(
        {
            community_label: color
            for color, community_label in zip(
                unfocused_colors, communities_monochrome.keys()
            )
        }
    )

    return color_mapping


def to_segments(nums: List[int]) -> List[List[int]]:
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
