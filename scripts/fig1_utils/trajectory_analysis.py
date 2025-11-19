"""
Developmental trajectory analysis for multimodal single-cell data.

This module evaluates how well different modalities (RNA, ATAC, WNN) preserve
developmental trajectories and bifurcation events during zebrafish embryogenesis.

Key features:
- Pseudotime ordering evaluation against expected developmental sequences
- Robust diffusion map computation with error handling
- Bifurcated trajectory conservation metrics (mesodermal vs neural branches)
- Cross-modality trajectory comparison and visualization

The module is specifically designed for analyzing neuromesodermal progenitor (NMP)
bifurcation into mesodermal and neural lineages, but can be adapted for other
developmental bifurcations.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from typing import Dict, List, Optional, Tuple
import warnings


def compute_trajectory_ordering_score(
    pseudotime: np.ndarray,
    cell_types: np.ndarray,
    expected_order: List[str]
) -> float:
    """
    Evaluate how well pseudotime preserves expected developmental ordering.

    Computes the Pearson correlation between observed pseudotime ordering
    and the expected developmental sequence of cell types.

    Args:
        pseudotime: Pseudotime values for cells
        cell_types: Cell type annotations for each cell
        expected_order: List of cell types in expected developmental order
                       (e.g., ['NMPs', 'PSM', 'somites', 'fast_muscle'])

    Returns:
        Ordering score between 0 and 1, where:
        - 1.0 = perfect preservation of expected ordering
        - 0.5 = random ordering
        - 0.0 = completely reversed ordering

    Example:
        >>> pseudotime = adata.obs['dpt_pseudotime'].values
        >>> cell_types = adata.obs['celltype'].values
        >>> expected = ['progenitor', 'intermediate', 'mature']
        >>> score = compute_trajectory_ordering_score(pseudotime, cell_types, expected)
        >>> print(f"Trajectory ordering score: {score:.3f}")

    Notes:
        - Only considers cell types present in both pseudotime and expected_order
        - Uses median pseudotime for each cell type
        - Robust to uneven cell type distributions
    """
    # For each cell type, compute median pseudotime
    type_medians = {}
    for cell_type in expected_order:
        type_mask = cell_types == cell_type
        if type_mask.sum() > 0:
            type_medians[cell_type] = np.median(pseudotime[type_mask])

    if len(type_medians) < 2:
        return 0.0

    # Get median values in expected order
    median_values = [
        type_medians.get(ct, np.inf)
        for ct in expected_order
        if ct in type_medians
    ]

    if len(median_values) < 2:
        return 0.0

    # Compute Pearson correlation with expected ordering
    expected_ranks = list(range(len(median_values)))

    # Get actual ranks based on sorted median values
    sorted_indices = np.argsort(median_values)
    rank_correlation = np.corrcoef(expected_ranks, sorted_indices)[0, 1]

    # Convert correlation to 0-1 scale
    ordering_score = (rank_correlation + 1) / 2 if not np.isnan(rank_correlation) else 0.0

    return ordering_score


def robust_diffmap_and_dpt(
    adata_mod,
    n_dcs: int = 10,
    max_cells: int = 15000,
    root_cell_type: str = 'NMPs'
) -> bool:
    """
    Compute diffusion map and diffusion pseudotime (DPT) with robust error handling.

    Performs several preprocessing steps to ensure numerical stability:
    - Subsampling large datasets
    - Removing disconnected cells
    - Symmetrizing and normalizing connectivity matrix
    - Adding diagonal regularization

    Args:
        adata_mod: AnnData object with 'connectivities' in obsp and
                  'annotation_ML_coarse' in obs
        n_dcs: Number of diffusion components to compute
        max_cells: Maximum cells to use (subsamples if exceeded)
        root_cell_type: Cell type to use as root for DPT
                       (must exist in adata_mod.obs['annotation_ML_coarse'])

    Returns:
        True if successful, False if computation failed

    Side Effects:
        Modifies adata_mod in place:
        - Adds 'X_diffmap' to obsm
        - Adds 'dpt_pseudotime' to obs
        - Sets 'iroot' in uns

    Example:
        >>> # Prepare AnnData with connectivity matrix
        >>> adata_traj = adata[trajectory_mask].copy()
        >>> adata_traj.obsp['connectivities'] = wnn_graph
        >>>
        >>> # Compute diffusion map and DPT
        >>> success = robust_diffmap_and_dpt(adata_traj, n_dcs=10)
        >>> if success:
        ...     pseudotime = adata_traj.obs['dpt_pseudotime']

    Notes:
        - Requires precomputed connectivity matrix in adata_mod.obsp['connectivities']
        - Uses root_cell_type cells as starting point for DPT
        - May subsample cells for numerical stability
    """
    # Subsample if too many cells
    if adata_mod.n_obs > max_cells:
        print(f"Subsampling from {adata_mod.n_obs} to {max_cells} cells for stability")
        sc.pp.subsample(adata_mod, n_obs=max_cells)

    # Clean up connectivity matrix
    connectivities = adata_mod.obsp['connectivities'].copy()

    # Remove cells with no connections (isolated nodes)
    node_degrees = np.array(connectivities.sum(axis=1)).flatten()
    connected_mask = node_degrees > 0

    if connected_mask.sum() < adata_mod.n_obs * 0.8:
        print(f"Warning: {(~connected_mask).sum()} cells have no connections, removing them")
        adata_mod = adata_mod[connected_mask].copy()
        connectivities = adata_mod.obsp['connectivities']

    # Ensure matrix is symmetric and normalized
    connectivities = (connectivities + connectivities.T) / 2

    # Add small diagonal regularization to avoid numerical issues
    connectivities.setdiag(1e-6)

    # Normalize rows to sum to 1
    row_sums = np.array(connectivities.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalizer = sparse.diags(1.0 / row_sums)
    connectivities = normalizer @ connectivities

    adata_mod.obsp['connectivities'] = connectivities
    adata_mod.obsp['distances'] = connectivities.copy()  # Use same for distances

    try:
        # Try with fewer components first for stability
        reduced_comps = min(n_dcs, 10)
        sc.tl.diffmap(adata_mod, n_comps=reduced_comps)

        # Set root and compute DPT
        root_mask = adata_mod.obs['annotation_ML_coarse'] == root_cell_type
        if root_mask.sum() > 0:
            root_indices = np.where(root_mask)[0]
            if len(root_indices) > 0:
                # Use first root cell for simplicity
                adata_mod.uns['iroot'] = root_indices[0]
                sc.tl.dpt(adata_mod, n_dcs=reduced_comps)
                return True

        return False

    except Exception as e:
        print(f"Diffusion map failed even with regularization: {str(e)}")
        return False


def compute_simple_bifurcation_score(
    adata_mod,
    mesodermal_trajectory: List[str],
    neural_trajectory: List[str],
    root_cluster: str
) -> float:
    """
    Compute simplified bifurcation quality score using pseudotime separation.

    Evaluates how well terminal cell types from the two branches are separated
    in pseudotime space.

    Args:
        adata_mod: AnnData object with 'dpt_pseudotime' in obs
        mesodermal_trajectory: List of cell types in mesodermal branch
        neural_trajectory: List of cell types in neural branch
        root_cluster: Root cell type (not used in current implementation)

    Returns:
        Bifurcation score between 0 and 1, where higher values indicate
        better separation between the two branches

    Example:
        >>> meso = ['NMPs', 'PSM', 'somites', 'fast_muscle']
        >>> neural = ['NMPs', 'spinal_cord', 'neural_posterior']
        >>> score = compute_simple_bifurcation_score(adata, meso, neural, 'NMPs')
        >>> print(f"Bifurcation quality: {score:.3f}")

    Notes:
        - Uses terminal cell types from each branch
        - Computes mean pseudotime for terminals
        - Separation score uses sigmoid normalization
    """
    try:
        pseudotime = adata_mod.obs['dpt_pseudotime'].values
        cell_types = adata_mod.obs['annotation_ML_coarse'].values

        # Get terminal cell types from each branch
        # (typically the last 1-2 cell types in the trajectory)
        meso_terminals = ['fast_muscle', 'somites']
        neural_terminals = ['neural_posterior', 'spinal_cord']

        # Collect pseudotime values for mesodermal terminals
        meso_pseudotimes = []
        for ct in meso_terminals:
            mask = cell_types == ct
            if mask.sum() > 0:
                pt_values = pseudotime[mask]
                valid_pt = pt_values[~np.isnan(pt_values)]
                if len(valid_pt) > 0:
                    meso_pseudotimes.extend(valid_pt)

        # Collect pseudotime values for neural terminals
        neural_pseudotimes = []
        for ct in neural_terminals:
            mask = cell_types == ct
            if mask.sum() > 0:
                pt_values = pseudotime[mask]
                valid_pt = pt_values[~np.isnan(pt_values)]
                if len(valid_pt) > 0:
                    neural_pseudotimes.extend(valid_pt)

        if len(meso_pseudotimes) > 0 and len(neural_pseudotimes) > 0:
            meso_mean = np.mean(meso_pseudotimes)
            neural_mean = np.mean(neural_pseudotimes)

            # Simple separation score based on pseudotime difference
            separation = abs(meso_mean - neural_mean)

            # Normalize to 0-1 using sigmoid
            bifurcation_score = 1 / (1 + np.exp(-separation * 2))
            return bifurcation_score

        return 0.0

    except Exception:
        return 0.0


def compute_bifurcated_trajectory_conservation_robust(
    adata: sc.AnnData,
    neighbors_keys: Dict[str, str],
    root_cluster: str = 'NMPs',
    mesodermal_trajectory: List[str] = None,
    neural_trajectory: List[str] = None,
    n_dcs: int = 10,
    max_cells: int = 15000
) -> Dict[str, Dict[str, float]]:
    """
    Robust computation of bifurcated trajectory conservation scores.

    Evaluates how well each modality (RNA, ATAC, WNN) preserves the bifurcation
    from neuromesodermal progenitors (NMPs) into mesodermal and neural lineages.

    Args:
        adata: AnnData object with cell type annotations in obs['annotation_ML_coarse']
        neighbors_keys: Dict mapping modality names to connectivity keys in obsp
                       (e.g., {'RNA': 'RNA_connectivities', 'ATAC': 'ATAC_connectivities'})
        root_cluster: Root cell type for DPT (default: 'NMPs')
        mesodermal_trajectory: Cell types in mesodermal branch. If None, uses default:
                              ['NMPs', 'tail_bud', 'PSM', 'somites', 'fast_muscle']
        neural_trajectory: Cell types in neural branch. If None, uses default:
                          ['NMPs', 'spinal_cord', 'neural_posterior']
        n_dcs: Number of diffusion components
        max_cells: Maximum cells per analysis

    Returns:
        Dictionary mapping modality names to result dictionaries with keys:
        - 'mesodermal_conservation': Score for mesodermal branch (0-1)
        - 'neural_conservation': Score for neural branch (0-1)
        - 'bifurcation_quality': Separation between branches (0-1)
        - 'overall_trajectory_score': Mean of the three scores

    Example:
        >>> neighbors_keys = {
        ...     'RNA': 'RNA_connectivities',
        ...     'ATAC': 'ATAC_connectivities',
        ...     'WNN': 'connectivities_wnn'
        ... }
        >>> results = compute_bifurcated_trajectory_conservation_robust(
        ...     adata, neighbors_keys, max_cells=10000
        ... )
        >>> for modality, scores in results.items():
        ...     print(f"{modality}: {scores['overall_trajectory_score']:.3f}")

    Notes:
        - Requires connectivity matrices precomputed in adata.obsp
        - Subsets to trajectory-relevant cell types before analysis
        - Robust to missing connectivity matrices or failed computations
    """
    # Set default trajectories if not provided
    if mesodermal_trajectory is None:
        mesodermal_trajectory = ['NMPs', 'tail_bud', 'PSM', 'somites', 'fast_muscle']
    if neural_trajectory is None:
        neural_trajectory = ['NMPs', 'spinal_cord', 'neural_posterior']

    # Combine all trajectory cell types
    all_trajectory_types = list(set(mesodermal_trajectory + neural_trajectory))
    trajectory_mask = adata.obs['annotation_ML_coarse'].isin(all_trajectory_types)
    adata_traj = adata[trajectory_mask].copy()

    print(f"Analyzing bifurcated trajectories with {adata_traj.n_obs} cells "
          f"across {len(all_trajectory_types)} cell types")
    print(f"Mesodermal branch: {' -> '.join(mesodermal_trajectory)}")
    print(f"Neural branch: {' -> '.join(neural_trajectory)}")

    trajectory_results = {}

    for modality_name, neighbors_key in neighbors_keys.items():
        print(f"\n=== Analyzing {modality_name} modality using {neighbors_key} ===")

        try:
            # Check if the connectivity matrix exists
            if neighbors_key not in adata_traj.obsp.keys():
                print(f"Warning: {neighbors_key} not found in adata.obsp")
                continue

            # Prepare data for this modality
            adata_mod = adata_traj.copy()

            # Set up connectivity matrix
            adata_mod.obsp['connectivities'] = adata_traj.obsp[neighbors_key].copy()

            # Set up neighbors metadata (required by scanpy)
            n_neighbors = min(30, int(adata_mod.obsp['connectivities'].getnnz() / adata_mod.n_obs))

            adata_mod.uns['neighbors'] = {
                'connectivities_key': 'connectivities',
                'distances_key': 'connectivities',  # Use same matrix
                'params': {
                    'n_neighbors': n_neighbors,
                    'method': 'precomputed',
                    'metric': 'precomputed'
                }
            }

            # Robust diffusion map computation
            diffmap_success = robust_diffmap_and_dpt(adata_mod, n_dcs, max_cells, root_cluster)

            if not diffmap_success:
                print(f"Failed to compute diffusion map for {modality_name}")
                trajectory_results[modality_name] = {
                    'mesodermal_conservation': 0.0,
                    'neural_conservation': 0.0,
                    'bifurcation_quality': 0.0,
                    'overall_trajectory_score': 0.0
                }
                continue

            # Get pseudotime values
            if 'dpt_pseudotime' not in adata_mod.obs.columns:
                print(f"No pseudotime computed for {modality_name}")
                continue

            pseudotime = adata_mod.obs['dpt_pseudotime'].values
            cell_types = adata_mod.obs['annotation_ML_coarse'].values

            # Check for valid pseudotime
            if np.isnan(pseudotime).all():
                print(f"All pseudotime values are NaN for {modality_name}")
                continue

            # Analyze each trajectory branch
            branch_results = {}

            # 1. Mesodermal trajectory analysis
            meso_cells = np.isin(cell_types, mesodermal_trajectory)
            if meso_cells.sum() > 5:
                meso_pseudotime = pseudotime[meso_cells]
                meso_celltypes = cell_types[meso_cells]

                # Filter out NaN values
                valid_mask = ~np.isnan(meso_pseudotime)
                if valid_mask.sum() > 2:
                    meso_score = compute_trajectory_ordering_score(
                        meso_pseudotime[valid_mask],
                        meso_celltypes[valid_mask],
                        mesodermal_trajectory
                    )
                    branch_results['mesodermal_conservation'] = meso_score
                else:
                    branch_results['mesodermal_conservation'] = 0.0
            else:
                branch_results['mesodermal_conservation'] = 0.0

            # 2. Neural trajectory analysis
            neural_cells = np.isin(cell_types, neural_trajectory)
            if neural_cells.sum() > 5:
                neural_pseudotime = pseudotime[neural_cells]
                neural_celltypes = cell_types[neural_cells]

                # Filter out NaN values
                valid_mask = ~np.isnan(neural_pseudotime)
                if valid_mask.sum() > 2:
                    neural_score = compute_trajectory_ordering_score(
                        neural_pseudotime[valid_mask],
                        neural_celltypes[valid_mask],
                        neural_trajectory
                    )
                    branch_results['neural_conservation'] = neural_score
                else:
                    branch_results['neural_conservation'] = 0.0
            else:
                branch_results['neural_conservation'] = 0.0

            # 3. Bifurcation quality
            bifurcation_score = compute_simple_bifurcation_score(
                adata_mod, mesodermal_trajectory, neural_trajectory, root_cluster
            )
            branch_results['bifurcation_quality'] = bifurcation_score

            # 4. Overall trajectory score
            branch_results['overall_trajectory_score'] = np.mean([
                branch_results['mesodermal_conservation'],
                branch_results['neural_conservation'],
                branch_results['bifurcation_quality']
            ])

            trajectory_results[modality_name] = branch_results

            print(f"{modality_name} results:")
            print(f"  Mesodermal conservation: {branch_results['mesodermal_conservation']:.3f}")
            print(f"  Neural conservation: {branch_results['neural_conservation']:.3f}")
            print(f"  Bifurcation quality: {branch_results['bifurcation_quality']:.3f}")
            print(f"  Overall score: {branch_results['overall_trajectory_score']:.3f}")

        except Exception as e:
            print(f"Error analyzing {modality_name}: {str(e)}")
            trajectory_results[modality_name] = {
                'mesodermal_conservation': 0.0,
                'neural_conservation': 0.0,
                'bifurcation_quality': 0.0,
                'overall_trajectory_score': 0.0
            }

    return trajectory_results


def run_robust_bifurcated_trajectory_analysis(
    adata: sc.AnnData,
    neighbors_keys: Optional[Dict[str, str]] = None,
    max_cells: int = 15000
) -> Dict[str, Dict[str, float]]:
    """
    High-level wrapper for complete bifurcated trajectory analysis.

    Runs comprehensive trajectory analysis across all modalities with default
    NMP bifurcation settings and prints formatted results.

    Args:
        adata: AnnData object with connectivity matrices and cell type annotations
        neighbors_keys: Dict mapping modality names to connectivity keys.
                       If None, uses defaults for RNA, ATAC, and WNN.
        max_cells: Maximum cells per analysis for numerical stability

    Returns:
        Dictionary mapping modality names to result dictionaries

    Example:
        >>> # Run complete analysis with defaults
        >>> results = run_robust_bifurcated_trajectory_analysis(adata)
        >>>
        >>> # Run with custom connectivity keys
        >>> custom_keys = {
        ...     'RNA': 'RNA_connectivities',
        ...     'ATAC_v2': 'ATAC_connectivities_alternative'
        ... }
        >>> results = run_robust_bifurcated_trajectory_analysis(
        ...     adata, neighbors_keys=custom_keys, max_cells=10000
        ... )

    Notes:
        - Prints formatted summary table of results
        - Uses default NMP bifurcation trajectories
        - Robust to errors in individual modality analyses
    """
    if neighbors_keys is None:
        neighbors_keys = {
            'RNA': 'RNA_connectivities',
            'ATAC': 'ATAC_connectivities',
            'WNN': 'connectivities_wnn'
        }

    print("=== Robust Bifurcated NMP Trajectory Analysis ===")
    print(f"Using max {max_cells} cells per analysis for numerical stability")

    # Define trajectories
    mesodermal_trajectory = ['NMPs', 'tail_bud', 'PSM', 'somites', 'fast_muscle']
    neural_trajectory = ['NMPs', 'spinal_cord', 'neural_posterior']

    # Compute trajectory conservation scores
    trajectory_results = compute_bifurcated_trajectory_conservation_robust(
        adata=adata,
        neighbors_keys=neighbors_keys,
        mesodermal_trajectory=mesodermal_trajectory,
        neural_trajectory=neural_trajectory,
        max_cells=max_cells
    )

    # Print formatted summary
    print(f"\n=== Final Results Summary ===")
    print(f"{'Modality':<8} {'Mesodermal':<12} {'Neural':<10} {'Bifurcation':<12} {'Overall':<8}")
    print("-" * 60)

    for modality, results in trajectory_results.items():
        print(f"{modality:<8} {results['mesodermal_conservation']:<12.3f} "
              f"{results['neural_conservation']:<10.3f} "
              f"{results['bifurcation_quality']:<12.3f} "
              f"{results['overall_trajectory_score']:<8.3f}")

    return trajectory_results


def plot_trajectory_conservation_results(
    trajectory_results: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Visualize trajectory conservation results across modalities.

    Creates a two-panel figure:
    - Left: Bar plot comparing mesodermal vs neural conservation
    - Right: Heatmap representation of conservation scores

    Args:
        trajectory_results: Output from compute_bifurcated_trajectory_conservation_robust()
        figsize: Figure size as (width, height) tuple

    Returns:
        matplotlib Figure object

    Example:
        >>> results = run_robust_bifurcated_trajectory_analysis(adata)
        >>> fig = plot_trajectory_conservation_results(results)
        >>> plt.savefig('trajectory_conservation.pdf')
        >>> plt.show()

    Notes:
        - Values are displayed with 3 decimal places
        - Uses red-yellow-green colormap for heatmap
        - Scores range from 0 to 1
    """
    # Extract data
    modalities = list(trajectory_results.keys())
    meso_scores = [trajectory_results[mod]['mesodermal_conservation'] for mod in modalities]
    neural_scores = [trajectory_results[mod]['neural_conservation'] for mod in modalities]

    # Create subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar plot comparison
    x = np.arange(len(modalities))
    width = 0.35

    bars1 = ax1.bar(x - width/2, meso_scores, width, label='Mesodermal',
                    alpha=0.8, color='#d62728')
    bars2 = ax1.bar(x + width/2, neural_scores, width, label='Neural',
                    alpha=0.8, color='#2ca02c')

    ax1.set_xlabel('Modality')
    ax1.set_ylabel('Trajectory Conservation Score')
    ax1.set_title('Developmental Trajectory Preservation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(modalities)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Heatmap representation
    data_matrix = np.array([meso_scores, neural_scores])
    im = ax2.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax2.set_xticks(range(len(modalities)))
    ax2.set_xticklabels(modalities)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Mesodermal', 'Neural'])
    ax2.set_title('Conservation Score Heatmap')

    # Add text annotations to heatmap
    for i in range(2):
        for j in range(len(modalities)):
            text = ax2.text(j, i, f'{data_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Conservation Score', rotation=270, labelpad=15)

    plt.tight_layout()
    return fig
