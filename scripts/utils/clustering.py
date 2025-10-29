# A module for clustering analysis (leiden, etc.)

# import libraries
import scanpy as sc
import numpy as np

# function to group singletons in seurat style
def group_singletons_seurat_style(
    adata,
    leiden_key: str = "leiden",
    adjacency_key: str = "connectivities",
    merged_key_suffix: str = "merged",
    group_singletons: bool = True,
    random_seed: int = 1
):
    """
    Replicates Seurat's 'GroupSingletons' post-processing step.
    - Finds clusters of size 1 (singletons) in adata.obs[leiden_key].
    - For each singleton, measures average connectivity to each other cluster
      by summing the adjacency submatrix SNN[i_cell, j_cells] and dividing
      by (# i_cells * # j_cells).
    - Reassigns the singleton cell to whichever cluster has the highest connectivity.
    - If there's a tie, picks randomly (set by random_seed).
    - Writes the merged labels to adata.obs[f"{leiden_key}_{merged_key_suffix}"].
    - If group_singletons=False, singletons remain in a “singleton” label.

    Parameters
    ----------
    adata : AnnData
        Your annotated data matrix.
    leiden_key : str
        Column in adata.obs where the initial Leiden (or other) clustering is stored.
    adjacency_key : str
        Key in adata.obsp containing an NxN adjacency matrix (e.g. "connectivities").
        Must be the same dimension as number of cells.
    merged_key_suffix : str
        Suffix to append when creating the merged labels column. The merged labels go
        in adata.obs[f"{leiden_key}_{merged_key_suffix}"].
    group_singletons : bool
        If True, merge singletons. If False, label them all "singleton".
    random_seed : int
        RNG seed for tie-breaking among equally connected clusters.

    Returns
    -------
    None
        (Modifies adata.obs in place, adding a column with merged labels.)
    """
    # Copy cluster labels
    old_labels = adata.obs[leiden_key].astype(str).values  # ensure string
    unique_labels, counts = np.unique(old_labels, return_counts=True)

    # Identify the singleton clusters (size=1)
    singleton_labels = unique_labels[counts == 1]

    # If not grouping them, just mark them as "singleton" and return
    if not group_singletons:
        new_labels = old_labels.copy()
        for s in singleton_labels:
            new_labels[new_labels == s] = "singleton"
        adata.obs[f"{leiden_key}_{merged_key_suffix}"] = new_labels
        adata.obs[f"{leiden_key}_{merged_key_suffix}"] = adata.obs[
            f"{leiden_key}_{merged_key_suffix}"
        ].astype("category")
        return

    # Otherwise, proceed to merge each singleton
    adjacency = adata.obsp[adjacency_key]
    new_labels = old_labels.copy()
    cluster_names = [cl for cl in unique_labels if cl not in singleton_labels]

    rng = np.random.default_rng(seed=random_seed)  # for tie-breaking

    for s_label in singleton_labels:
        i_cells = np.where(new_labels == s_label)[0]
        if len(i_cells) == 0:
            # Possibly already reassigned if something changed mid-loop
            continue

        # Seurat only has 1 cell for a singleton cluster, but let's be robust:
        # We'll compute the average connectivity for all i_cells anyway.
        # Usually i_cells will be length 1.
        sub_row_count = len(i_cells)

        best_cluster = None
        best_conn = -1  # track maximum average connectivity

        for j_label in cluster_names:
            j_cells = np.where(new_labels == j_label)[0]
            if len(j_cells) == 0:
                continue
            # Extract adjacency submatrix
            # shape is (len(i_cells), len(j_cells))
            sub_snn = adjacency[i_cells[:, None], j_cells]
            avg_conn = sub_snn.sum() / (sub_snn.shape[0] * sub_snn.shape[1])
            if np.isclose(avg_conn, best_conn):
                # tie => randomly pick
                if rng.integers(2) == 0:
                    best_cluster = j_label
                    best_conn = avg_conn
            elif avg_conn > best_conn:
                best_cluster = j_label
                best_conn = avg_conn

        if best_cluster is None:
            # If the singleton has zero connectivity to everything, you could:
            # (A) leave it in its own cluster, or
            # (B) label it "disconnected_singleton"
            # We'll leave it as is for now:
            continue

        # Reassign all i_cells to the chosen cluster
        new_labels[i_cells] = best_cluster

    # Store merged labels in adata.obs
    adata.obs[f"{leiden_key}_{merged_key_suffix}"] = new_labels
    # Remove any unused categories
    adata.obs[f"{leiden_key}_{merged_key_suffix}"] = adata.obs[
        f"{leiden_key}_{merged_key_suffix}"
    ].astype("category")
    adata.obs[f"{leiden_key}_{merged_key_suffix}"].cat.remove_unused_categories()
    
    return adata


# function to group small clusters based on size threshold
def merge_small_clusters(
    adata,
    leiden_key: str = "leiden",
    adjacency_key: str = "connectivities",
    merged_key_suffix: str = "merged",
    threshold: int = 100,
    merge_small_clusters: bool = True,
    random_seed: int = 1
):
    """
    Merges small clusters (below threshold size) with their most connected neighbors.
    - Finds clusters smaller than the specified threshold in adata.obs[leiden_key].
    - For each small cluster, measures average connectivity to each other cluster
      by summing the adjacency submatrix SNN[i_cells, j_cells] and dividing
      by (# i_cells * # j_cells).
    - Reassigns cells from small clusters to whichever cluster has the highest connectivity.
    - If there's a tie, picks randomly (set by random_seed).
    - Writes the merged labels to adata.obs[f"{leiden_key}_{merged_key_suffix}"].
    - If merge_small_clusters=False, small clusters remain labeled as "small_cluster".

    Parameters
    ----------
    adata : AnnData
        Your annotated data matrix.
    leiden_key : str
        Column in adata.obs where the initial Leiden (or other) clustering is stored.
    adjacency_key : str
        Key in adata.obsp containing an NxN adjacency matrix (e.g. "connectivities").
        Must be the same dimension as number of cells.
    merged_key_suffix : str
        Suffix to append when creating the merged labels column. The merged labels go
        in adata.obs[f"{leiden_key}_{merged_key_suffix}"].
    threshold : int
        Size threshold below which clusters are considered "small" and eligible for merging.
    merge_small_clusters : bool
        If True, merge small clusters. If False, label them all "small_cluster".
    random_seed : int
        RNG seed for tie-breaking among equally connected clusters.

    Returns
    -------
    adata : AnnData
        Returns the modified adata object with merged labels.
    """
    # Copy cluster labels
    old_labels = adata.obs[leiden_key].astype(str).values  # ensure string
    unique_labels, counts = np.unique(old_labels, return_counts=True)

    # Identify the small clusters (size < threshold)
    small_labels = unique_labels[counts < threshold]
    
    print(f"Found {len(small_labels)} clusters smaller than threshold {threshold}")
    for i, label in enumerate(small_labels):
        print(f"  Cluster {label}: {counts[unique_labels == label][0]} cells")

    # If not merging them, just mark them as "small_cluster" and return
    if not merge_small_clusters:
        new_labels = old_labels.copy()
        for s in small_labels:
            new_labels[new_labels == s] = "small_cluster"
        adata.obs[f"{leiden_key}_{merged_key_suffix}"] = new_labels
        adata.obs[f"{leiden_key}_{merged_key_suffix}"] = adata.obs[
            f"{leiden_key}_{merged_key_suffix}"
        ].astype("category")
        return adata

    # Otherwise, proceed to merge each small cluster
    adjacency = adata.obsp[adjacency_key]
    new_labels = old_labels.copy()
    
    # Get clusters that are large enough to be merge targets
    large_cluster_names = [cl for cl in unique_labels if cl not in small_labels]
    
    rng = np.random.default_rng(seed=random_seed)  # for tie-breaking

    # Sort small clusters by size (smallest first) to handle them systematically
    small_cluster_sizes = [(label, counts[unique_labels == label][0]) for label in small_labels]
    small_cluster_sizes.sort(key=lambda x: x[1])
    
    for s_label, cluster_size in small_cluster_sizes:
        i_cells = np.where(new_labels == s_label)[0]
        if len(i_cells) == 0:
            # Possibly already reassigned if something changed mid-loop
            continue

        print(f"Processing small cluster {s_label} with {len(i_cells)} cells...")

        best_cluster = None
        best_conn = -1  # track maximum average connectivity
        connectivity_scores = []

        # Check connectivity to all potential target clusters
        potential_targets = [cl for cl in large_cluster_names if np.sum(new_labels == cl) > 0]
        
        for j_label in potential_targets:
            j_cells = np.where(new_labels == j_label)[0]
            if len(j_cells) == 0:
                continue
                
            # Extract adjacency submatrix
            # shape is (len(i_cells), len(j_cells))
            sub_snn = adjacency[i_cells[:, None], j_cells]
            avg_conn = sub_snn.sum() / (sub_snn.shape[0] * sub_snn.shape[1])
            connectivity_scores.append((j_label, avg_conn))
            
            if np.isclose(avg_conn, best_conn, rtol=1e-10):
                # tie => randomly pick
                if rng.integers(2) == 0:
                    best_cluster = j_label
                    best_conn = avg_conn
            elif avg_conn > best_conn:
                best_cluster = j_label
                best_conn = avg_conn

        if best_cluster is None:
            # If the small cluster has zero connectivity to everything
            print(f"  Warning: Cluster {s_label} has no connectivity to other clusters, keeping as is")
            continue

        print(f"  Merging cluster {s_label} -> {best_cluster} (connectivity: {best_conn:.6f})")
        
        # Show top 3 connectivity scores for transparency
        connectivity_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top connectivity scores: {connectivity_scores[:3]}")

        # Reassign all i_cells to the chosen cluster
        new_labels[i_cells] = best_cluster
        
        # Update the list of large clusters since the target cluster grew
        # (This doesn't affect the current loop since we're using the original large_cluster_names)

    # Store merged labels in adata.obs
    adata.obs[f"{leiden_key}_{merged_key_suffix}"] = new_labels
    # Remove any unused categories
    adata.obs[f"{leiden_key}_{merged_key_suffix}"] = adata.obs[
        f"{leiden_key}_{merged_key_suffix}"
    ].astype("category")
    adata.obs[f"{leiden_key}_{merged_key_suffix}"].cat.remove_unused_categories()
    
    # Print summary
    new_unique_labels, new_counts = np.unique(new_labels, return_counts=True)
    print(f"\nMerging complete!")
    print(f"Original clusters: {len(unique_labels)}")
    print(f"Final clusters: {len(new_unique_labels)}")
    print(f"Clusters merged: {len(unique_labels) - len(new_unique_labels)}")
    
    return adata