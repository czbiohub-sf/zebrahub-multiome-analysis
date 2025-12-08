"""
Data loading and processing functions for peak cluster analysis.

This module handles loading data from CSV files and preparing it for LLM analysis.
"""

from typing import Dict, List

import pandas as pd
from arbol import asection
from numpy import log10

from scripts.litemind_peak_analysis import config


def convert_clusters_genes_to_lists(df: pd.DataFrame, method: str = 'nonzero') -> Dict:
    """
    Convert cluster-gene DataFrame to dictionary of gene lists.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with clusters as rows and genes as columns,
        with numeric values (e.g., 0/1 or expression values)
    method : str
        Method to use: 'nonzero' keeps genes with non-zero values

    Returns
    -------
    Dict
        Dictionary mapping cluster IDs to lists of gene names
    """
    cluster_genes_dict = {}

    for cluster_id in df.index:
        if method == 'nonzero':
            # Get genes with non-zero values for this cluster
            genes = df.loc[cluster_id][df.loc[cluster_id] != 0].index.tolist()
        else:
            # Default: all genes
            genes = df.columns.tolist()

        cluster_genes_dict[cluster_id] = genes

    return cluster_genes_dict


def load_coarse_cluster_data():
    """
    Load all data required for coarse cluster analysis.

    Returns
    -------
    tuple
        (df_peak_stats, df_num_cells, df_clusters_groups, cluster_genes_dict,
         df_peak_details_overlap, df_peak_details_corr, df_peak_details_anticorr,
         df_clusters_motifs, df_motif_info)
    """
    with asection("=== Loading Coarse Clusters Data ==="):
        # (0) peak statistics per cluster
        df_peak_stats_coarse = pd.read_csv(
            config.DATA_DIR / "coarse_cluster_statistics.csv",
            index_col=0
        )

        # (1) number of cells per pseudobulk group
        df_num_cells = pd.read_csv(
            config.DATA_DIR / "num_cells_per_pseudobulk_group.csv",
            index_col=0
        )

        # (2) peak clusters-by-pseudobulk groups
        df_clusters_groups_coarse = pd.read_csv(
            config.DATA_DIR / "leiden_by_pseudobulk.csv",
            index_col=0
        )

        # (3) peak clusters-by-associated genes
        df_clusters_genes_coarse = pd.read_csv(
            config.DATA_DIR / "leiden_by_assoc_genes.csv",
            index_col=0
        )

        # (4) peak details:
        df_peak_details_overlap_coarse, df_peak_details_corr_coarse, df_peak_details_anticorr_coarse = coarse_peaks_details(
            str(config.DATA_DIR / "peaks_assoc_genes_filtered.csv")
        )

        # (5) peak clusters-by-TF motifs
        df_clusters_motifs_coarse = pd.read_csv(
            config.DATA_DIR / "leiden_by_motifs_maelstrom.csv",
            index_col=0
        )

        # (6) motifs-by-factors
        df_motif_info_coarse = pd.read_csv(
            config.DATA_DIR / "info_cisBP_v2_danio_rerio_motif_factors_consensus.csv",
            index_col=0
        )

        # parse the genes for each cluster (to reduce the size of the input table tokens)
        cluster_genes_dict_coarse = convert_clusters_genes_to_lists(
            df_clusters_genes_coarse,
            method='nonzero'
        )

        return (df_peak_stats_coarse,
                df_num_cells,
                df_clusters_groups_coarse,
                cluster_genes_dict_coarse,
                df_peak_details_overlap_coarse,
                df_peak_details_corr_coarse,
                df_peak_details_anticorr_coarse,
                df_clusters_motifs_coarse,
                df_motif_info_coarse)


def load_fine_cluster_data():
    """
    Load all data required for fine cluster analysis.

    Returns
    -------
    tuple
        (df_peak_stats, df_num_cells, df_clusters_groups, cluster_genes_dict,
         df_peak_details_overlap, df_peak_details_corr, df_peak_details_anticorr,
         df_clusters_motifs, df_motif_info)
    """
    with asection("=== Loading Fine Clusters Data ==="):
        # (0) peak statistics per cluster
        df_peak_stats_fine = pd.read_csv(
            config.DATA_DIR / "fine_cluster_statistics.csv",
            index_col=0
        )

        # (1) number of cells per pseudobulk group
        df_num_cells = pd.read_csv(
            config.DATA_DIR / "num_cells_per_pseudobulk_group.csv",
            index_col=0
        )

        # (2) peak clusters-by-pseudobulk groups
        df_clusters_groups_fine = pd.read_csv(
            config.DATA_DIR / "leiden_fine_by_pseudobulk.csv",
            index_col=0
        )

        # (3) peak clusters-by-associated genes
        df_clusters_genes_fine = pd.read_csv(
            config.DATA_DIR / "leiden_fine_by_assoc_genes.csv",
            index_col=0
        )

        # (4) peak details:
        df_peak_details_overlap_fine, df_peak_details_corr_fine, df_peak_details_anticorr_fine = fine_peaks_details(
            str(config.DATA_DIR / "peaks_assoc_genes_filtered.csv")
        )

        # (5) peak clusters-by-TF motifs
        df_clusters_motifs_fine = pd.read_csv(
            config.DATA_DIR / "leiden_fine_by_motifs_maelstrom.csv",
            index_col=0
        )

        # (6) motifs-by-factors
        df_motif_info_fine = pd.read_csv(
            config.DATA_DIR / "info_cisBP_v2_danio_rerio_motif_factors_fine_clusters_consensus.csv",
            index_col=0
        )

        # parse the genes for each cluster (to reduce the size of the input table tokens)
        cluster_genes_dict_fine = convert_clusters_genes_to_lists(
            df_clusters_genes_fine,
            method='nonzero'
        )

        return (df_peak_stats_fine,
                df_num_cells,
                df_clusters_groups_fine,
                cluster_genes_dict_fine,
                df_peak_details_overlap_fine,
                df_peak_details_corr_fine,
                df_peak_details_anticorr_fine,
                df_clusters_motifs_fine,
                df_motif_info_fine)


def process_cluster_data(cluster_id,
                         df_peak_stats: pd.DataFrame,
                         df_clusters_groups: pd.DataFrame,
                         df_num_cells: pd.DataFrame,
                         cluster_genes_dict: Dict[int, List[str]],
                         df_peak_details_overlap,
                         df_peak_details_corr,
                         df_peak_details_anticorr,
                         df_clusters_motifs: pd.DataFrame,
                         df_motif_info: pd.DataFrame) -> tuple:
    """
    Process data for a single cluster and prepare it for LLM analysis.

    Parameters
    ----------
    cluster_id : int or str
        ID of the cluster to process
    df_peak_stats : pd.DataFrame
        DataFrame with peak statistics per cluster
    df_clusters_groups : pd.DataFrame
        DataFrame with cluster expression across groups
    df_num_cells : pd.DataFrame
        DataFrame with number of cells per pseudobulk group
    cluster_genes_dict : Dict[int, List[str]]
        Dictionary mapping cluster IDs to gene lists
    df_peak_details_overlap : pd.DataFrame
        Peak details for overlapping genes
    df_peak_details_corr : pd.DataFrame
        Peak details for correlated genes
    df_peak_details_anticorr : pd.DataFrame
        Peak details for anti-correlated genes
    df_clusters_motifs : pd.DataFrame
        DataFrame with cluster motif enrichment
    df_motif_info : pd.DataFrame
        DataFrame with motif information

    Returns
    -------
    tuple
        (peak_stats, groups_data, genes_text, overlap_details, corr_details,
         anticorr_details, motifs_data)

    Raises
    ------
    KeyError
        If cluster_id is not found in required DataFrames
    """
    # Check if cluster_id exists in all required DataFrames
    missing_from = []
    if cluster_id not in df_peak_stats.index:
        missing_from.append("df_peak_stats")
    if cluster_id not in df_clusters_groups.index:
        missing_from.append("df_clusters_groups")
    if cluster_id not in cluster_genes_dict:
        missing_from.append("cluster_genes_dict")
    if cluster_id not in df_clusters_motifs.index:
        missing_from.append("df_clusters_motifs")

    if missing_from:
        raise KeyError(f"Cluster ID '{cluster_id}' not found in: {', '.join(missing_from)}")

    # Subset the dataframes for the current cluster
    df_peak_stats_cluster = df_peak_stats.loc[cluster_id].to_frame()
    df_clusters_groups_cluster = df_clusters_groups.loc[cluster_id].to_frame()
    df_clusters_genes_cluster = cluster_genes_dict[cluster_id]
    df_clusters_motifs_cluster = df_clusters_motifs.loc[cluster_id].to_frame()

    # merge df_num_cells_cluster into df_clusters_groups_cluster:
    df_clusters_groups_cluster = df_clusters_groups_cluster.join(
        df_num_cells,  # the "right" table – the one whose extra columns you want
        how='inner',  # keep every row of df1; fill NaNs where df2 has no match
    )

    # Rename the first column to 'CIS-BP_motif_id' for merging:
    df_clusters_motifs_cluster.index.name = "CisBP_motif_id"
    df_motif_info.index.name = "CisBP_motif_id"

    # Rename the first column to 'peak_id' for consistency:
    df_clusters_motifs_cluster = df_clusters_motifs_cluster.rename(
        columns={df_clusters_motifs_cluster.columns[0]: "z-score"})

    df_clusters_motifs_cluster_merged = (
        df_clusters_motifs_cluster  # the "left" table – the one you want to keep intact
        .join(
            df_motif_info,  # the "right" table – the one whose extra columns you want
            how='inner',  # keep every row of df1; fill NaNs where df2 has no match
        )
    )

    # Sort the tables:
    df_clusters_groups_cluster = df_clusters_groups_cluster.sort_values(
        by=df_clusters_groups_cluster.columns[0],
        ascending=False
    )
    df_clusters_motifs_cluster_merged = df_clusters_motifs_cluster_merged.sort_values(
        by=df_clusters_motifs_cluster_merged.columns[0],
        ascending=False
    )

    # Sort the list of genes:
    df_clusters_genes_cluster = sorted(df_clusters_genes_cluster)

    # Convert df_clusters_genes_cluster to a string
    genes_text = ', '.join(df_clusters_genes_cluster)

    # Select rows for cluster:
    df_peak_details_overlap_cluster = (
        df_peak_details_overlap.loc[cluster_id]
        if cluster_id in df_peak_details_overlap.index
        else pd.DataFrame()
    )
    df_peak_details_corr_cluster = (
        df_peak_details_corr.loc[cluster_id]
        if cluster_id in df_peak_details_corr.index
        else pd.DataFrame()
    )
    df_peak_details_anticorr_cluster = (
        df_peak_details_anticorr.loc[cluster_id]
        if cluster_id in df_peak_details_anticorr.index
        else pd.DataFrame()
    )

    # Keep only the top 128 rows:
    df_peak_details_overlap_cluster = df_peak_details_overlap_cluster.head(128)
    df_peak_details_corr_cluster = df_peak_details_corr_cluster.head(128)
    df_peak_details_anticorr_cluster = df_peak_details_anticorr_cluster.head(128)

    # Remove the index from the tables:
    df_peak_details_overlap_cluster.reset_index(drop=True, inplace=True)
    df_peak_details_corr_cluster.reset_index(drop=True, inplace=True)
    df_peak_details_anticorr_cluster.reset_index(drop=True, inplace=True)

    return (df_peak_stats_cluster,
            df_clusters_groups_cluster,
            genes_text,
            df_peak_details_overlap_cluster,
            df_peak_details_corr_cluster,
            df_peak_details_anticorr_cluster,
            df_clusters_motifs_cluster_merged)


def coarse_peaks_details(csv_path: str):
    """
    Summarize peaks-gene associations for coarse clusters.

    Parameters
    ----------
    csv_path : str
        Path to `peaks_assoc_genes_filtered.csv`.

    Returns
    -------
    tuple
        (overlap_df, corr_df, anticorr_df)

        overlap_df : pd.DataFrame
            coarse_cluster_id | gene | overlap_peak_count
        corr_df : pd.DataFrame
            coarse_cluster_id | gene | correlation_coefficient | log10_p-value
            (ρ > 0)
        anticorr_df : pd.DataFrame
            coarse_cluster_id | gene | correlation_coefficient | log10_p-value
            (ρ < 0)
    """
    # Load data
    df = pd.read_csv(csv_path)

    # 1. Overlap table (coarse clusters)
    overlap_df = (
        df.loc[df["association_type"] == "overlap",
        ["leiden_coarse", "associated_gene"]]
        .groupby(["leiden_coarse", "associated_gene"], as_index=False)
        .size()  # each row = one peak
        .rename(columns={
            "leiden_coarse": "coarse_cluster_id",
            "associated_gene": "gene",
            "size": "overlap_peak_count"})
        .sort_values("overlap_peak_count", ascending=False,
                     ignore_index=True)
    )

    # 2 & 3. Linked-gene tables (coarse clusters)
    linked = df[df["association_type"] == "linked"].copy()
    linked["abs_corr"] = linked["correlation_coefficient"].abs()

    sort_order = ["p_value", "abs_corr"]  # ascending p-value, then ↓|ρ|
    ascending = [True, False]

    base_cols = ["leiden_coarse", "associated_gene",
                 "correlation_coefficient", "abs_corr", "p_value"]

    corr_df = (
        linked[linked["correlation_coefficient"] > 0][base_cols]
        .rename(columns={
            "leiden_coarse": "coarse_cluster_id",
            "associated_gene": "gene"})
        .sort_values(sort_order, ascending=ascending,
                     ignore_index=True)
    )

    anticorr_df = (
        linked[linked["correlation_coefficient"] < 0][base_cols]
        .rename(columns={
            "leiden_coarse": "coarse_cluster_id",
            "associated_gene": "gene"})
        .sort_values(sort_order, ascending=ascending,
                     ignore_index=True)
    )

    # replace the p-value column with log_10 of the p-value instead, rename column accordingly:
    corr_df["log10_p-value"] = log10(corr_df["p_value"].clip(lower=1e-300)).astype(int)
    anticorr_df["log10_p-value"] = log10(anticorr_df["p_value"].clip(lower=1e-300)).astype(int)

    # Remove the p_value and abs_corr columns
    corr_df = corr_df.drop(columns=["p_value", "abs_corr"])
    anticorr_df = anticorr_df.drop(columns=["p_value", "abs_corr"])

    # Set the index to the coarse cluster ID
    overlap_df = overlap_df.set_index("coarse_cluster_id")
    corr_df = corr_df.set_index("coarse_cluster_id")
    anticorr_df = anticorr_df.set_index("coarse_cluster_id")

    return overlap_df, corr_df, anticorr_df


def fine_peaks_details(csv_path: str):
    """
    Summarize peaks-gene associations for fine clusters.

    Parameters
    ----------
    csv_path : str
        Path to `peaks_assoc_genes_filtered.csv`.

    Returns
    -------
    tuple
        (overlap_df, corr_df, anticorr_df)

        overlap_df : pd.DataFrame
            fine_cluster_id | gene | overlap_peak_count
        corr_df : pd.DataFrame
            fine_cluster_id | gene | correlation_coefficient | log10_p-value
            (ρ > 0)
        anticorr_df : pd.DataFrame
            fine_cluster_id | gene | correlation_coefficient | log10_p-value
            (ρ < 0)
    """
    # Load the CSV
    df = pd.read_csv(csv_path)

    # 1. Overlap table (fine clusters)
    overlap_fine_df = (
        df.loc[df["association_type"] == "overlap",
        ["leiden_fine", "associated_gene"]]
        .groupby(["leiden_fine", "associated_gene"], as_index=False)
        .size()  # rows = individual peaks
        .rename(columns={
            "leiden_fine": "fine_cluster_id",
            "associated_gene": "gene",
            "size": "overlap_peak_count"})
        .sort_values("overlap_peak_count", ascending=False,
                     ignore_index=True)
    )

    # 2 & 3. Linked-gene tables (fine clusters)
    linked = df[df["association_type"] == "linked"].copy()
    linked["abs_corr"] = linked["correlation_coefficient"].abs()

    # Rank first by lower p-value (more significant),
    # then by larger |ρ| (stronger effect size).
    sort_cols = ["p_value", "abs_corr"]
    ascending = [True, False]

    base_cols = ["leiden_fine", "associated_gene",
                 "correlation_coefficient", "abs_corr", "p_value"]

    corr_df = (
        linked[linked["correlation_coefficient"] > 0][base_cols]
        .rename(columns={
            "leiden_fine": "fine_cluster_id",
            "associated_gene": "gene"})
        .sort_values(sort_cols, ascending=ascending,
                     ignore_index=True)
    )

    anticorr_df = (
        linked[linked["correlation_coefficient"] < 0][base_cols]
        .rename(columns={
            "leiden_fine": "fine_cluster_id",
            "associated_gene": "gene"})
        .sort_values(sort_cols, ascending=ascending,
                     ignore_index=True)
    )

    # replace the p-value column with log_10 of the p-value instead, rename column accordingly:
    corr_df["log10_p-value"] = log10(corr_df["p_value"].clip(lower=1e-300)).astype(int)
    anticorr_df["log10_p-value"] = log10(anticorr_df["p_value"].clip(lower=1e-300)).astype(int)

    # Remove the p_value and abs_corr columns
    corr_df = corr_df.drop(columns=["p_value", "abs_corr"])
    anticorr_df = anticorr_df.drop(columns=["p_value", "abs_corr"])

    # Set the index to the fine cluster ID
    overlap_fine_df = overlap_fine_df.set_index("fine_cluster_id")
    corr_df = corr_df.set_index("fine_cluster_id")
    anticorr_df = anticorr_df.set_index("fine_cluster_id")

    return overlap_fine_df, corr_df, anticorr_df
