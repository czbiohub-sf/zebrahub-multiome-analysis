"""
Knockout Analysis Functions

Functions for analyzing in-silico knockout experiments by computing
transition probabilities at celltype and metacell levels.
"""

import numpy as np
import pandas as pd


def compute_celltype_transitions(adata, trans_key="T_fwd_WT", celltype_key="manual_annotation"):
    """
    Compute cell type-to-cell type transition probabilities.

    Aggregates cell-cell transition probabilities to celltype-celltype level
    by averaging transitions between cells of different types.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with cell-cell transition matrix in .obsp
    trans_key : str, default "T_fwd_WT"
        Key in adata.obsp for cell-cell transition probability matrix
    celltype_key : str, default "manual_annotation"
        Key in adata.obs for cell type annotations

    Returns
    -------
    pd.DataFrame
        Cell type-to-cell type transition probability matrix,
        with rows normalized to sum to 1

    Examples
    --------
    >>> celltype_trans = compute_celltype_transitions(
    ...     oracle.adata,
    ...     trans_key="T_fwd_WT",
    ...     celltype_key="manual_annotation"
    ... )
    >>> celltype_trans.loc["NMPs", "PSM"]  # Transition prob from NMPs to PSM
    """
    # Get the cell-cell transition matrix
    T_cell = adata.obsp[trans_key]

    # Get celltype labels
    celltypes = adata.obs[celltype_key]

    # Get unique celltypes
    unique_celltypes = celltypes.cat.categories

    # Initialize the celltype transition matrix
    n_celltypes = len(unique_celltypes)
    T_celltype = np.zeros((n_celltypes, n_celltypes))

    # Create a mapping of celltype to cell indices
    celltype_to_indices = {ct: np.where(celltypes == ct)[0] for ct in unique_celltypes}

    # Compute celltype transitions
    for i, source_type in enumerate(unique_celltypes):
        source_indices = celltype_to_indices[source_type]
        for j, target_type in enumerate(unique_celltypes):
            target_indices = celltype_to_indices[target_type]

            # Extract the submatrix of transitions from source to target celltype
            submatrix = T_cell[source_indices][:, target_indices]

            # Sum all transitions and normalize by the number of source cells
            T_celltype[i, j] = submatrix.sum() / len(source_indices)

    # Create a DataFrame for easier interpretation
    T_celltype_df = pd.DataFrame(T_celltype, index=unique_celltypes, columns=unique_celltypes)

    # Normalize rows to sum to 1
    T_celltype_df = T_celltype_df.div(T_celltype_df.sum(axis=1), axis=0)

    return T_celltype_df


def compute_metacell_transitions(adata, trans_key="T_fwd_WT", metacell_key="SEACell"):
    """
    Compute metacell-to-metacell transition probabilities.

    Aggregates cell-cell transition probabilities to metacell-metacell level
    by averaging transitions between cells belonging to different metacells.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with cell-cell transition matrix in .obsp
    trans_key : str, default "T_fwd_WT"
        Key in adata.obsp for cell-cell transition probability matrix
    metacell_key : str, default "SEACell"
        Key in adata.obs for metacell assignments

    Returns
    -------
    pd.DataFrame
        Metacell-to-metacell transition probability matrix,
        with rows normalized to sum to 1

    Examples
    --------
    >>> metacell_trans = compute_metacell_transitions(
    ...     oracle.adata,
    ...     trans_key="T_fwd_meox1_KO",
    ...     metacell_key="SEACell"
    ... )
    >>> # Compare WT vs KO metacell transitions
    >>> trans_wt = compute_metacell_transitions(adata, "T_fwd_WT")
    >>> trans_ko = compute_metacell_transitions(adata, "T_fwd_tbx16_KO")
    """
    # Get the cell-cell transition matrix
    T_cell = adata.obsp[trans_key]

    # Get metacell labels
    metacells = adata.obs[metacell_key]

    # Get unique metacells
    unique_metacells = metacells.unique()

    # Initialize the metacell transition matrix
    n_metacells = len(unique_metacells)
    T_metacell = np.zeros((n_metacells, n_metacells))

    # Create a mapping of metacell to cell indices
    metacell_to_indices = {mc: np.where(metacells == mc)[0] for mc in unique_metacells}

    # Compute metacell transitions
    for i, source_metacell in enumerate(unique_metacells):
        source_indices = metacell_to_indices[source_metacell]
        for j, target_metacell in enumerate(unique_metacells):
            target_indices = metacell_to_indices[target_metacell]

            # Extract the submatrix of transitions from source to target metacell
            submatrix = T_cell[source_indices][:, target_indices]

            # Sum all transitions and normalize by the number of source cells
            T_metacell[i, j] = submatrix.sum() / len(source_indices)

    # Create a DataFrame for easier interpretation
    T_metacell_df = pd.DataFrame(T_metacell, index=unique_metacells, columns=unique_metacells)

    # Normalize rows to sum to 1
    T_metacell_df = T_metacell_df.div(T_metacell_df.sum(axis=1), axis=0)

    return T_metacell_df


def get_top_genes_for_celltype(df, celltype, n=10):
    """
    Get the top n genes with the lowest cosine similarity scores for a given celltype.

    Useful for identifying genes whose knockout most perturbs a specific cell type's
    differentiation trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        Cosine similarity DataFrame with celltypes as rows and genes as columns
    celltype : str
        The celltype to analyze
    n : int, default 10
        Number of top genes to return

    Returns
    -------
    pd.Series
        Top n genes with their cosine similarity scores,
        sorted from lowest (most perturbed) to highest

    Examples
    --------
    >>> # Get genes that most perturb fast_muscle differentiation
    >>> top_genes = get_top_genes_for_celltype(
    ...     cosine_sim_df,
    ...     celltype="fast_muscle",
    ...     n=10
    ... )
    >>> print(top_genes)
    tbx16    0.42
    meox1    0.51
    ...

    Notes
    -----
    Lower cosine similarity indicates greater perturbation.
    A score of 1.0 means no change from WT.
    """
    # Sort the row for the given celltype
    sorted_genes = df.loc[celltype].sort_values()

    # Return the top n genes
    return sorted_genes.head(n)
