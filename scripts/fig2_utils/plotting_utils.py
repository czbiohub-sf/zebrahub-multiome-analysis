"""
Plotting utilities for Figure 2 metacell analysis.

This module provides consistent plotting styles and visualization functions
for RNA-ATAC correlation analysis via metacells.

Dependencies:
    - matplotlib: For plotting configuration
    - seaborn: For aesthetics
"""

import matplotlib.pyplot as plt


def set_plotting_style() -> None:
    """
    Set publication-quality matplotlib plotting style.

    Configures matplotlib with:
        - Seaborn 'paper' style for clean aesthetics
        - Arial font family for consistency
        - Appropriate font sizes for labels, titles, legends
        - LaTeX math text configuration
        - SVG font embedding for editability

    Returns:
        None (modifies matplotlib rcParams globally)

    Example:
        >>> set_plotting_style()
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> plt.savefig("figure.pdf")

    Notes:
        - Call this function once at the beginning of notebooks
        - Settings persist for the entire Python session
        - Designed for publication-quality figures
        - SVG fonttype='none' ensures editable text in vector formats
    """
    plt.style.use('seaborn-paper')
    plt.rc('axes', labelsize=12)
    plt.rc('axes', titlesize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=10)
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('xtick.major', pad=2)
    plt.rc('ytick.major', pad=2)
    plt.rc('mathtext', fontset='stixsans', sf='sansserif')
    plt.rc('figure', figsize=[10, 9])
    plt.rc('svg', fonttype='none')

    # Override to ensure Arial is used
    plt.rc('font', family='Arial')
