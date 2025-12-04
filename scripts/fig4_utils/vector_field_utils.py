"""
Vector Field Visualization Utilities

Functions for visualizing cell fate transitions as vector fields on UMAPs,
including metacell-level aggregation and perturbation score overlays.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def average_2D_trans_vecs_metacells(adata, metacell_col="SEACell",
                                    basis='umap_aligned', key_added='WT'):
    """
    Compute average 2D transition vectors at the metacell level.

    Averages cell-level 2D transition vectors and UMAP positions within each metacell
    to produce metacell-level vector fields for visualization.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with UMAP embedding and transition vectors
    metacell_col : str, default "SEACell"
        Key in adata.obs for metacell assignments
    basis : str, default 'umap_aligned'
        Name of the 2D embedding in adata.obsm (without 'X_' prefix)
    key_added : str, default 'WT'
        Prefix for the transition vector key in adata.obsm
        (e.g., 'WT' for 'WT_umap_aligned')

    Returns
    -------
    X_metacell : np.ndarray, shape (n_metacells, 2)
        Average UMAP positions for each metacell
    V_metacell : np.ndarray, shape (n_metacells, 2)
        Average transition vectors for each metacell

    Examples
    --------
    >>> # Average WT transition vectors at metacell level
    >>> X_meta, V_meta = average_2D_trans_vecs_metacells(
    ...     oracle.adata,
    ...     metacell_col="SEACell",
    ...     basis='umap_aligned',
    ...     key_added='WT'
    ... )
    >>> # Then visualize with plot_metacell_transitions()
    """
    X_umap = adata.obsm[f'X_{basis}']
    V_cell = adata.obsm[f"{key_added}_{basis}"]

    # Convert metacell column to categorical if it's not already
    if not pd.api.types.is_categorical_dtype(adata.obs[metacell_col]):
        metacells = pd.Categorical(adata.obs[metacell_col])
    else:
        metacells = adata.obs[metacell_col]

    # number of metacells
    n_metacells = len(metacells.categories)

    # X_metacell is the average UMAP position of the metacells
    # V_metacell is the average transition vector of the metacells
    X_metacell = np.zeros((n_metacells, 2))
    V_metacell = np.zeros((n_metacells, 2))

    for i, category in enumerate(metacells.categories):
        mask = metacells == category
        X_metacell[i] = X_umap[mask].mean(axis=0)
        V_metacell[i] = V_cell[mask].mean(axis=0)

    return X_metacell, V_metacell


def plot_metacell_transitions(adata, X_metacell, V_metacell, data_id,
                              figpath=None,
                              metacell_col="SEACell",
                              annotation_class="manual_annotation",
                              basis='umap_aligned', genotype="WT",
                              cell_type_color_dict=None,
                              cell_size=10, SEACell_size=20,
                              scale=1, arrow_scale=15, arrow_width=0.002, dpi=120):
    """
    Plot metacell transitions as vector field overlay on UMAP.

    Creates a publication-quality figure showing:
    1. Single cells colored by cell type
    2. Metacells (larger points) colored by most prevalent cell type
    3. Arrows showing transition directions

    Parameters
    ----------
    adata : AnnData
        Annotated data object with UMAP embedding
    X_metacell : np.ndarray, shape (n_metacells, 2)
        Average UMAP positions for metacells (from average_2D_trans_vecs_metacells)
    V_metacell : np.ndarray, shape (n_metacells, 2)
        Average transition vectors for metacells
    data_id : str
        Dataset identifier for file naming
    figpath : str, optional
        Path to save figure (PNG and PDF). If None, figure is not saved.
    metacell_col : str, default "SEACell"
        Key in adata.obs for metacell assignments
    annotation_class : str, default "manual_annotation"
        Key in adata.obs for cell type annotations
    basis : str, default 'umap_aligned'
        Name of the 2D embedding
    genotype : str, default "WT"
        Genotype label for figure title and file naming
    cell_type_color_dict : dict, optional
        Cell type to color mapping. If None, uses default seaborn palette.
    cell_size : int, default 10
        Point size for single cells
    SEACell_size : int, default 20
        Point size for metacells
    arrow_scale : float, default 15
        Arrow scaling factor (larger = shorter arrows)
    arrow_width : float, default 0.002
        Arrow line width
    dpi : int, default 120
        Figure DPI

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object

    Examples
    --------
    >>> X_meta, V_meta = average_2D_trans_vecs_metacells(adata, key_added='WT')
    >>> fig = plot_metacell_transitions(
    ...     adata, X_meta, V_meta,
    ...     data_id="TDR126",
    ...     figpath="/path/to/figures/",
    ...     genotype="WT"
    ... )
    """
    # Default color dict if not provided
    if cell_type_color_dict is None:
        cell_type_color_dict = {
            'NMPs': '#8dd3c7',
            'PSM': '#008080',
            'fast_muscle': '#df4b9b',
            'neural_posterior': '#393b7f',
            'somites': '#1b9e77',
            'spinal_cord': '#d95f02',
            'tail_bud': '#7570b3'
        }

    # create a figure object (matplotlib)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

    # Prepare data for plotting
    umap_coords = pd.DataFrame(adata.obsm[f'X_{basis}'], columns=[0, 1], index=adata.obs_names)
    umap_data = umap_coords.join(adata.obs[[metacell_col, annotation_class]])
    umap_data = umap_data.rename(columns={annotation_class: 'celltype'})

    # Plot single cells
    sns.scatterplot(
        x=0, y=1, hue='celltype', data=umap_data, s=cell_size,
        palette=cell_type_color_dict, legend=None, ax=ax, alpha=0.7
    )

    # Calculate most prevalent cell type for each metacell
    most_prevalent = adata.obs.groupby(metacell_col)[annotation_class].agg(lambda x: x.value_counts().idxmax())

    # Prepare metacell data
    mcs = umap_data.groupby(metacell_col).mean().reset_index()
    mcs['celltype'] = most_prevalent.values

    # Plot metacells
    sns.scatterplot(
        x=0, y=1, s=SEACell_size, hue='celltype', data=mcs,
        palette=cell_type_color_dict, edgecolor='black', linewidth=1.25,
        legend=None, ax=ax
    )

    # Plot transition vectors
    Q = ax.quiver(X_metacell[:, 0], X_metacell[:, 1], V_metacell[:, 0], V_metacell[:, 1],
               angles='xy', scale_units='xy', scale=1/arrow_scale, width=arrow_width,
                color='black', alpha=0.8)

    # Customize the plot
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    plt.title(f'Metacell Transitions on {basis.upper()}')
    plt.tight_layout()
    plt.grid(False)

    if figpath:
        plt.savefig(figpath + f"umap_{genotype}_metacell_aggre_trans_probs_{data_id}.png")
        plt.savefig(figpath + f"umap_{genotype}_metacell_aggre_trans_probs_{data_id}.pdf")

    return fig


def plot_metacell_cosine_sims(adata, X_metacell, cosine_sim_df, gene="",
                              vmin=0, vmax=1,
                              figpath=None,
                              metacell_col="SEACell",
                              annotation_class="manual_annotation",
                              basis='umap_aligned',
                              cmap="viridis",
                              cell_type_color_dict=None,
                              cell_size=10, SEACell_size=20, dpi=120):
    """
    Plot metacell-level perturbation scores (cosine similarities) on UMAP.

    Creates a heatmap-style visualization showing how much each metacell's
    transition probabilities are perturbed by a gene knockout.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with UMAP embedding
    X_metacell : np.ndarray, shape (n_metacells, 2)
        Average UMAP positions for metacells
    cosine_sim_df : pd.DataFrame
        Cosine similarity scores per metacell (from compute_row_cosine_similarities)
    gene : str, optional
        Gene name for title/legend
    vmin : float, default 0
        Minimum value for colormap
    vmax : float, default 1
        Maximum value for colormap
    figpath : str, optional
        Path to save figure. If None, figure is not saved.
    metacell_col : str, default "SEACell"
        Key in adata.obs for metacell assignments
    annotation_class : str, default "manual_annotation"
        Key in adata.obs for cell type annotations
    basis : str, default 'umap_aligned'
        Name of the 2D embedding
    cmap : str, default "viridis"
        Matplotlib colormap name
    cell_type_color_dict : dict, optional
        Cell type to color mapping
    cell_size : int, default 10
        Point size for single cells
    SEACell_size : int, default 20
        Point size for metacells
    dpi : int, default 120
        Figure DPI

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object

    Examples
    --------
    >>> # Compute metacell-level cosine similarities for a gene
    >>> trans_wt = compute_metacell_transitions(adata, "T_fwd_WT")
    >>> trans_ko = compute_metacell_transitions(adata, "T_fwd_tbx16_KO")
    >>> cosine_sims = compute_row_cosine_similarities(trans_wt, trans_ko)
    >>> # Plot
    >>> fig = plot_metacell_cosine_sims(
    ...     adata, X_metacell, cosine_sims,
    ...     gene="tbx16",
    ...     vmin=0.5, vmax=1.0
    ... )
    """
    # Default color dict if not provided
    if cell_type_color_dict is None:
        cell_type_color_dict = {
            'NMPs': '#8dd3c7',
            'PSM': '#008080',
            'fast_muscle': '#df4b9b',
            'neural_posterior': '#393b7f',
            'somites': '#1b9e77',
            'spinal_cord': '#d95f02',
            'tail_bud': '#7570b3'
        }

    # create a figure object (matplotlib)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

    # Prepare data for plotting
    umap_coords = pd.DataFrame(adata.obsm[f'X_{basis}'], columns=[0, 1], index=adata.obs_names)
    umap_data = umap_coords.join(adata.obs[[metacell_col, annotation_class]])
    umap_data = umap_data.rename(columns={annotation_class: 'celltype'})

    # Plot single cells
    sns.scatterplot(
        x=0, y=1, hue='celltype', data=umap_data, s=cell_size,
        palette=cell_type_color_dict, legend=None, ax=ax, alpha=0.7
    )

    # Calculate most prevalent cell type for each metacell
    most_prevalent = adata.obs.groupby(metacell_col)[annotation_class].agg(lambda x: x.value_counts().idxmax())

    # Prepare metacell data
    mcs = umap_data.groupby(metacell_col).mean().reset_index()
    mcs['celltype'] = most_prevalent.values
    mcs.set_index("SEACell", inplace=True)

    mcs_merged = pd.concat([mcs, cosine_sim_df], axis=1)

    # Plot metacells with cosine similarity as color
    # Use the first column of cosine_sim_df if it exists, otherwise use gene name
    color_col = cosine_sim_df.columns[0] if len(cosine_sim_df.columns) > 0 else gene
    scatter = sns.scatterplot(
        x=0, y=1, s=20, hue=color_col, data=mcs_merged, edgecolor='black', linewidth=1.25,
        legend=None, palette=cmap, vmin=vmin, vmax=vmax)

    # Set the colormap and the color limits
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    scatter.collections[0].set_cmap(cmap)
    scatter.collections[0].set_norm(norm)
    plt.colorbar(scatter.collections[0])

    # Customize the plot
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    plt.title(f'Metacell Transitions on {basis.upper()}')
    plt.tight_layout()
    plt.grid(False)

    return fig
