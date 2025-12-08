"""
Color Palette Utilities for Genomic Data Visualization
========================================================

This module provides functions for creating and managing color palettes
specifically designed for genomic data visualization, including:

- Chromosome color palettes with high visual distinction
- Cell type color schemes based on developmental lineages
- Timepoint color gradients for temporal data
- Utility functions for color manipulation and contrast detection

The palettes are optimized for creating clear, publication-quality figures
for multi-omic single-cell genomics data.

Example Usage
-------------
>>> # Create a distinctive chromosome palette
>>> chrom_colors = create_distinctive_chromosome_palette(n_chromosomes=25)
>>>
>>> # Create a custom high-contrast chromosome palette
>>> custom_colors = create_custom_chromosome_palette()
>>>
>>> # Generate a circular palette visualization
>>> fig = create_circular_chromosome_palette(
...     chromosome_colors=chrom_colors,
...     save_path="chromosome_palette.png"
... )
>>>
>>> # Create a timepoint color palette
>>> timepoints = ['0somites', '5somites', '10somites', '15somites']
>>> tp_palette = make_timepoint_palette(timepoints, cmap_name='viridis')
>>>
>>> # Create cell type color palette
>>> celltypes = ['neural', 'neural_crest', 'PSM', 'endoderm']
>>> ct_palette = _create_color_palette(celltypes, metadata_type='celltype')
>>>
>>> # Order cell types by lineage
>>> ordered_celltypes = create_celltype_order_from_lineages()

Notes
-----
- Color palettes are designed to be perceptually distinct
- Supports both matplotlib color specifications and hex color codes
- Includes utilities for dark/light color detection for optimal text contrast
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import re


# Default lineage mapping for zebrafish developmental cell types
DEFAULT_LINEAGE_MAPPING = {
    "CNS": [
        "neural",
        "neural_optic",
        "neural_posterior",
        "neural_telencephalon",
        "neurons",
        "hindbrain",
        "midbrain_hindbrain_boundary",
        "optic_cup",
        "spinal_cord",
        "differentiating_neurons",
        "floor_plate",
        "neural_floor_plate",
        "enteric_neurons",
    ],

    "Neural Crest": [
        "neural_crest"
    ],

    "Paraxial Mesoderm": [
        "somites",
        "fast_muscle",
        "muscle",
        "PSM",
        "NMPs",
        "tail_bud",
        "notochord",
    ],

    "Lateral Mesoderm": [
        "lateral_plate_mesoderm",
        "heart_myocardium",
        "hematopoietic_vasculature",
        "pharyngeal_arches",
        "pronephros",
        "hemangioblasts",
        "hatching_gland",
    ],

    "Endoderm": [
        "endoderm",
        "endocrine_pancreas",
    ],

    "Epiderm": [
        "epidermis"
    ],

    "Germline": [
        "primordial_germ_cells"
    ],
}


def create_distinctive_chromosome_palette(n_chromosomes=25):
    """
    Create a highly distinctive color palette for chromosome visualization.

    Combines multiple qualitative color palettes (tab10, Set1, Set2) to generate
    maximally distinct colors for visualizing multiple chromosomes in genomic plots.

    Parameters
    ----------
    n_chromosomes : int, default=25
        Number of chromosomes to generate colors for. Typically 25 for zebrafish
        (24 autosomes + 1 sex chromosome).

    Returns
    -------
    list of tuples
        List of RGBA color tuples with length n_chromosomes. Each tuple has
        format (R, G, B, A) with values in [0, 1].

    Examples
    --------
    >>> # Create palette for 25 chromosomes
    >>> colors = create_distinctive_chromosome_palette(n_chromosomes=25)
    >>> len(colors)
    25
    >>> colors[0]  # First chromosome color (RGBA tuple)
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0)

    >>> # Use with scanpy for UMAP visualization
    >>> import scanpy as sc
    >>> chrom_colors = create_distinctive_chromosome_palette()
    >>> sc.pl.umap(adata, color="chromosome", palette=chrom_colors)

    Notes
    -----
    - Uses perceptually distinct qualitative colormaps from matplotlib
    - Color order: tab10 (10) -> Set1 (9) -> Set2 (6) -> tab20c (as needed)
    - Colors are optimized for visual distinction, not biological meaning
    - For custom color assignments, consider using create_custom_chromosome_palette()

    See Also
    --------
    create_custom_chromosome_palette : Predefined high-contrast chromosome palette
    create_circular_chromosome_palette : Visualize chromosome color palette
    """

    # Combine multiple qualitative palettes for maximum distinction
    colors = []

    # Start with tab10 (10 distinct colors)
    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    colors.extend(tab10_colors)

    # Add Set1 colors (9 distinct colors, avoid overlap)
    set1_colors = plt.cm.Set1(np.linspace(0, 1, 9))
    colors.extend(set1_colors)

    # Add some distinct colors from Set2 and Set3
    set2_colors = plt.cm.Set2(np.linspace(0, 1, 6))
    colors.extend(set2_colors[:6])  # Take first 6 to reach 25 total

    # Ensure we have exactly n_chromosomes colors
    if len(colors) > n_chromosomes:
        colors = colors[:n_chromosomes]
    elif len(colors) < n_chromosomes:
        # Add more colors if needed
        additional_colors = plt.cm.tab20c(np.linspace(0, 1, n_chromosomes - len(colors)))
        colors.extend(additional_colors)

    return colors[:n_chromosomes]


def create_custom_chromosome_palette():
    """
    Create a custom high-contrast color palette for 25 chromosomes.

    Returns a predefined set of 25 carefully selected hex colors that provide
    high visual contrast and distinction. Colors are organized in sets of
    primary, light, and dark variants to maximize perceptual separation.

    Returns
    -------
    list of str
        List of 25 hex color codes (e.g., '#1f77b4').

    Examples
    --------
    >>> # Get custom chromosome palette
    >>> colors = create_custom_chromosome_palette()
    >>> len(colors)
    25
    >>> colors[0]
    '#1f77b4'

    >>> # Use with scanpy
    >>> import scanpy as sc
    >>> chrom_colors = create_custom_chromosome_palette()
    >>> sc.pl.umap(adata, color="chromosome", palette=chrom_colors)

    >>> # Convert to matplotlib colors for plotting
    >>> import matplotlib.colors as mcolors
    >>> rgb_colors = [mcolors.to_rgb(c) for c in colors]

    Notes
    -----
    - Contains 25 predefined hex colors organized as:
        * Primary colors (10): blue, orange, green, red, purple, brown, pink, gray, olive, cyan
        * Light variants (10): corresponding light versions of primary colors
        * Dark variants (5): corresponding dark versions of some primary colors
    - Fixed color scheme ensures reproducibility across analyses
    - Alternative to create_distinctive_chromosome_palette() for consistent styling

    See Also
    --------
    create_distinctive_chromosome_palette : Generate palette from colormaps
    create_circular_chromosome_palette : Visualize the palette
    """

    # Define 25 highly distinctive colors
    custom_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#aec7e8',  # light blue
        '#ffbb78',  # light orange
        '#98df8a',  # light green
        '#ff9896',  # light red
        '#c5b0d5',  # light purple
        '#c49c94',  # light brown
        '#f7b6d3',  # light pink
        '#c7c7c7',  # light gray
        '#dbdb8d',  # light olive
        '#9edae5',  # light cyan
        '#393b79',  # dark blue
        '#637939',  # dark green
        '#8c6d31',  # dark orange
        '#843c39',  # dark red
        '#7b4173'   # dark purple
    ]

    return custom_colors


def create_circular_chromosome_palette(chromosome_colors, save_path=None,
                                     figsize=(10, 6), circles_per_row=5):
    """
    Create a circular visualization of chromosome color palette.

    Generates a figure displaying chromosome colors as circles arranged in a grid,
    with chromosome numbers labeled. Useful for creating color legends and
    verifying visual distinction between chromosomes.

    Parameters
    ----------
    chromosome_colors : list
        List of colors (hex codes, RGB tuples, or matplotlib color names).
    save_path : str or None, default=None
        Path to save the figure. If None, figure is displayed but not saved.
    figsize : tuple of float, default=(10, 6)
        Figure size in inches as (width, height).
    circles_per_row : int, default=5
        Number of circles to display per row in the grid layout.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Examples
    --------
    >>> # Create and visualize chromosome palette
    >>> colors = create_distinctive_chromosome_palette(n_chromosomes=25)
    >>> fig = create_circular_chromosome_palette(
    ...     chromosome_colors=colors,
    ...     save_path="figures/chromosome_palette.png"
    ... )

    >>> # Custom layout with more circles per row
    >>> fig = create_circular_chromosome_palette(
    ...     chromosome_colors=colors,
    ...     figsize=(15, 4),
    ...     circles_per_row=8
    ... )

    >>> # Use custom palette
    >>> custom_colors = create_custom_chromosome_palette()
    >>> fig = create_circular_chromosome_palette(custom_colors)

    Notes
    -----
    - Each circle is labeled with chromosome number (Chr 1, Chr 2, ...)
    - Text color is automatically adjusted (white/black) based on background darkness
    - Includes reference text showing expected uniform distribution (0.040 for 25 chromosomes)
    - Figure has gray border and white background for clean presentation
    - Uses _is_dark_color() internally to determine optimal text contrast

    See Also
    --------
    _is_dark_color : Determine if color is dark for text contrast
    create_distinctive_chromosome_palette : Generate distinctive colors
    create_custom_chromosome_palette : Get predefined color palette
    """

    n_chromosomes = len(chromosome_colors)
    n_rows = int(np.ceil(n_chromosomes / circles_per_row))

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate circle positions
    circle_radius = 0.3
    spacing_x = 2.0
    spacing_y = 1.5

    positions = []
    labels = []

    for i, color in enumerate(chromosome_colors):
        row = i // circles_per_row
        col = i % circles_per_row

        # Center the circles in each row
        x_offset = (circles_per_row - 1) * spacing_x / 2
        x = col * spacing_x - x_offset
        y = -row * spacing_y

        # Create circle
        circle = patches.Circle((x, y), circle_radius,
                              facecolor=color,
                              edgecolor='black',
                              linewidth=1.5)
        ax.add_patch(circle)

        # Add chromosome label
        ax.text(x, y, f'Chr\n{i+1}',
               ha='center', va='center',
               fontsize=9, fontweight='bold',
               color='white' if _is_dark_color(color) else 'black')

        positions.append((x, y))
        labels.append(f'Chr {i+1}')

    # Add expected uniform reference
    ref_y = -n_rows * spacing_y - 0.5
    ax.text(0, ref_y, 'Expected uniform (0.040)',
           ha='center', va='center',
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                    edgecolor='red', linestyle='--', linewidth=2))

    # Set axis limits and properties
    max_x = max([pos[0] for pos in positions]) + circle_radius + 0.5
    min_x = min([pos[0] for pos in positions]) - circle_radius - 0.5
    max_y = max([pos[1] for pos in positions]) + circle_radius + 0.5
    min_y = min([pos[1] for pos in positions]) - circle_radius - 1.0

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add title
    ax.set_title('Chromosome Color Palette',
                fontsize=16, fontweight='bold', pad=20)

    # Add border
    border = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                             linewidth=2, edgecolor='gray', facecolor='none',
                             linestyle='-', alpha=0.5)
    ax.add_patch(border)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Circular palette saved to: {save_path}")

    plt.show()
    return fig


def _is_dark_color(color):
    """
    Determine if a color is dark (for optimal text contrast).

    Calculates the perceptual luminance of a color using the standard formula
    for converting RGB to grayscale. Returns True if the color is dark enough
    that white text should be used, False if black text is more appropriate.

    Parameters
    ----------
    color : str, tuple, or array-like
        Color specification in any format accepted by matplotlib.colors.to_rgb()
        (hex code, RGB tuple, color name, etc.).

    Returns
    -------
    bool
        True if color is dark (luminance < 0.5), False otherwise.

    Examples
    --------
    >>> # Check if colors are dark
    >>> _is_dark_color('#1f77b4')  # Blue
    True
    >>> _is_dark_color('#ffff00')  # Yellow
    False
    >>> _is_dark_color((0.1, 0.1, 0.1))  # Dark gray RGB
    True

    >>> # Use for text color selection
    >>> bg_color = '#2ca02c'  # Green
    >>> text_color = 'white' if _is_dark_color(bg_color) else 'black'
    >>> print(text_color)
    'white'

    Notes
    -----
    - Luminance formula: 0.299*R + 0.587*G + 0.114*B
    - Weights reflect human perception sensitivity (most sensitive to green)
    - Threshold of 0.5 provides good contrast in most cases
    - Used internally by create_circular_chromosome_palette()

    See Also
    --------
    create_circular_chromosome_palette : Uses this for text color selection
    """
    rgb = mcolors.to_rgb(color)
    # Calculate luminance using standard formula
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return luminance < 0.5


def make_timepoint_palette(timepoints, cmap_name='viridis'):
    """
    Build a timepoint-to-color mapping using a matplotlib colormap.

    Creates a dictionary mapping developmental timepoints to colors sampled
    from a sequential colormap. Useful for visualizing temporal progression
    in developmental biology datasets.

    Parameters
    ----------
    timepoints : list or array-like
        List of timepoint identifiers (e.g., ['0somites', '5somites', '10somites']).
        Order determines color assignment along the colormap gradient.
    cmap_name : str, default='viridis'
        Name of matplotlib colormap to use. Common choices:
        - 'viridis': perceptually uniform, purple to yellow
        - 'plasma': perceptually uniform, purple to pink/yellow
        - 'inferno': perceptually uniform, black to yellow
        - 'coolwarm': diverging blue to red
        - 'RdYlBu': diverging red to blue

    Returns
    -------
    dict
        Dictionary mapping timepoint -> RGBA color tuple.
        Colors are sampled evenly across the colormap range.

    Examples
    --------
    >>> # Create timepoint palette
    >>> timepoints = ['0somites', '5somites', '10somites', '15somites']
    >>> palette = make_timepoint_palette(timepoints, cmap_name='viridis')
    >>> palette['5somites']
    (0.282623, 0.140926, 0.457517, 1.0)

    >>> # Use with scanpy
    >>> import scanpy as sc
    >>> tp_colors = make_timepoint_palette(adata.obs['timepoint'].unique())
    >>> sc.pl.umap(adata, color='timepoint', palette=tp_colors)

    >>> # Try different colormaps
    >>> plasma_pal = make_timepoint_palette(timepoints, cmap_name='plasma')
    >>> coolwarm_pal = make_timepoint_palette(timepoints, cmap_name='coolwarm')

    Notes
    -----
    - Colors are evenly spaced across the full colormap range [0, 1]
    - For single timepoint, returns color at middle of colormap (0.5)
    - Sequential colormaps (viridis, plasma) work best for ordered temporal data
    - Diverging colormaps (coolwarm, RdYlBu) can highlight early vs late timepoints

    See Also
    --------
    _create_color_palette : General palette creation for different metadata types
    """
    cmap = plt.get_cmap(cmap_name)
    n = max(1, len(timepoints))
    colors = [cmap(i/(n-1) if n > 1 else 0.5) for i in range(n)]
    return {tp: col for tp, col in zip(timepoints, colors)}


def _create_color_palette(categories, metadata_type):
    """
    Create color palette for different types of categorical metadata.

    Generates biologically-informed color schemes for cell types based on
    developmental lineages, or generic palettes for other metadata types.

    Parameters
    ----------
    categories : list or array-like
        List of category names to assign colors to.
    metadata_type : str
        Type of metadata. Supported values:
        - 'celltype': Uses biologically-informed lineage-based colors
        - 'timepoint': Uses sequential viridis colormap
        - Other: Uses qualitative Set3 colormap

    Returns
    -------
    dict
        Dictionary mapping category name -> color.
        For celltype: predefined hex colors or matplotlib colors
        For timepoint: RGBA tuples from viridis
        For other: RGBA tuples from Set3

    Examples
    --------
    >>> # Cell type palette with biological colors
    >>> celltypes = ['neural', 'PSM', 'endoderm', 'epidermis']
    >>> ct_palette = _create_color_palette(celltypes, 'celltype')
    >>> ct_palette['neural']
    '#1f77b4'

    >>> # Timepoint palette
    >>> timepoints = ['0somites', '5somites', '10somites']
    >>> tp_palette = _create_color_palette(timepoints, 'timepoint')

    >>> # Generic palette for other metadata
    >>> stages = ['early', 'mid', 'late']
    >>> stage_palette = _create_color_palette(stages, 'stage')

    Notes
    -----
    Cell type color scheme (lineage-based):
    - CNS/Neural: blues (#1f77b4, #aec7e8, #4682b4, etc.)
    - Neural Crest: purples (#9467bd, #c5b0d5)
    - Early Mesoderm: dark greens (#2ca02c, #98df8a)
    - Axial Mesoderm: brown (#8c564b)
    - Paraxial Mesoderm: greens
    - Lateral Plate Mesoderm: reds (#d62728, #ff7f0e)
    - Endoderm: yellows (#bcbd22, #dbdb8d)
    - Ectoderm: grays (#7f7f7f)
    - Germline: pink (#e377c2)

    Unknown cell types are assigned colors using hash-based Set3 sampling.

    See Also
    --------
    make_timepoint_palette : Specialized function for timepoint palettes
    create_celltype_order_from_lineages : Get lineage-based cell type ordering
    """

    if metadata_type == 'celltype':
        # Create biologically-informed color palette
        color_scheme = {
            # CNS/Neural - blues
            'neural': '#1f77b4', 'neural_optic': '#aec7e8', 'neural_posterior': '#4682b4',
            'neural_telencephalon': '#6495ed', 'neurons': '#0000cd', 'differentiating_neurons': '#4169e1',
            'hindbrain': '#1e90ff', 'midbrain_hindbrain_boundary': '#87ceeb', 'spinal_cord': '#00bfff',
            'optic_cup': '#87cefa', 'floor_plate': '#b0e0e6', 'neural_floor_plate': '#add8e6',

            # Neural Crest - purples
            'neural_crest': '#9467bd', 'enteric_neurons': '#c5b0d5',

            # Early Mesoderm - dark greens
            'NMPs': '#2ca02c', 'tail_bud': '#98df8a',

            # Axial Mesoderm - brown
            'notochord': '#8c564b',

            # Paraxial Mesoderm - greens
            'PSM': '#2ca02c', 'somites': '#98df8a', 'fast_muscle': '#c5b0d5', 'muscle': '#bcbd22',

            # Lateral Plate Mesoderm - reds
            'lateral_plate_mesoderm': '#d62728', 'heart_myocardium': '#ff7f0e',
            'hematopoietic_vasculature': '#ff9896', 'hemangioblasts': '#ffbb78',

            # Other Mesoderm - oranges
            'pharyngeal_arches': '#ff7f0e', 'pronephros': '#ffbb78', 'hatching_gland': '#ffd700',

            # Endoderm - yellows
            'endoderm': '#bcbd22', 'endocrine_pancreas': '#dbdb8d',

            # Ectoderm - grays
            'epidermis': '#7f7f7f',

            # Germline - pink
            'primordial_germ_cells': '#e377c2'
        }

        # Use predefined colors if available, otherwise generate
        palette = {}
        for cat in categories:
            if cat in color_scheme:
                palette[cat] = color_scheme[cat]
            else:
                # Generate color for unknown categories
                palette[cat] = plt.cm.Set3(hash(cat) % 12 / 12)

        return palette

    elif metadata_type == 'timepoint':
        # Use sequential palette for timepoints (temporal progression)
        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
        return {cat: colors[i] for i, cat in enumerate(categories)}
    else:
        # Default qualitative palette for other metadata types
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        return {cat: colors[i] for i, cat in enumerate(categories)}


def create_celltype_order_from_lineages(lineage_mapping=None):
    """
    Create ordered cell type list based on developmental lineage groupings.

    Orders cell types following the developmental trajectory from neural
    to mesodermal to endodermal to ectodermal lineages. Useful for
    consistent ordering in heatmaps, bar plots, and other visualizations.

    Parameters
    ----------
    lineage_mapping : dict or None, default=None
        Dictionary mapping lineage names to lists of cell types.
        If None, uses DEFAULT_LINEAGE_MAPPING for zebrafish cell types.
        Expected format:
        {
            'CNS': ['neural', 'neurons', ...],
            'Neural Crest': ['neural_crest'],
            'Paraxial Mesoderm': ['somites', 'PSM', ...],
            ...
        }

    Returns
    -------
    list
        Ordered list of cell types grouped by lineages following
        developmental progression:
        CNS -> Neural Crest -> Paraxial Mesoderm -> Lateral Mesoderm ->
        Endoderm -> Epiderm -> Germline

    Examples
    --------
    >>> # Use default zebrafish lineage mapping
    >>> ordered_celltypes = create_celltype_order_from_lineages()
    >>> ordered_celltypes[:5]
    ['neural', 'neural_optic', 'neural_posterior', 'neural_telencephalon', 'neurons']

    >>> # Use custom lineage mapping
    >>> custom_lineages = {
    ...     'Neural': ['neuron_A', 'neuron_B'],
    ...     'Muscle': ['muscle_fast', 'muscle_slow']
    ... }
    >>> ordered = create_celltype_order_from_lineages(custom_lineages)

    >>> # Use for plotting
    >>> import scanpy as sc
    >>> celltype_order = create_celltype_order_from_lineages()
    >>> sc.pl.matrixplot(adata, var_names=genes, groupby='celltype',
    ...                  categories_order=celltype_order)

    Notes
    -----
    Default lineage order (zebrafish):
    1. CNS (Central Nervous System)
    2. Neural Crest
    3. Paraxial Mesoderm
    4. Lateral Mesoderm
    5. Endoderm
    6. Epiderm
    7. Germline

    The default mapping includes common zebrafish developmental cell types
    organized by their embryonic origins and differentiation trajectories.

    See Also
    --------
    _create_color_palette : Creates colors organized by same lineages
    DEFAULT_LINEAGE_MAPPING : Default lineage-to-celltype mapping
    """

    if lineage_mapping is None:
        lineage_mapping = DEFAULT_LINEAGE_MAPPING

    # Define lineage order (CNS → Neural Crest → Mesoderm → Endoderm → Epiderm → Germline)
    lineage_order = ["CNS", "Neural Crest", "Paraxial Mesoderm", "Lateral Mesoderm",
                     "Endoderm", "Epiderm", "Germline"]

    ordered_celltypes = []

    for lineage in lineage_order:
        if lineage in lineage_mapping:
            ordered_celltypes.extend(lineage_mapping[lineage])

    return ordered_celltypes
