# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: celloracle_env
#     language: python
#     name: celloracle_env
# ---

# %% [markdown]
# ## Use GimmeMotifs to find over-represented motifs from scATAC-seq datasets
#
# - We will use the leiden clusters from the peak UMAP to find the de novo/known motifs that are over-represented in each peak cluster.
# - We will take the one vs rest approach - where we subset the peaks from each cluster, and peaks from the rest, then compute the over-represented motifs from each peak cluster. (the rest will be used as the background here.)
#
# - last updated: 2/12/2025
#

# %%
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the custom module
import os
import sys
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/gimmemotifs")

from atac_seq_motif_analysis import ATACSeqMotifAnalysis

# %%
# figure parameter setting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')
# Set default DPI for saved figures
mpl.rcParams['savefig.dpi'] = 600

# Plotting style function (run this before plotting the final figure)
def set_plotting_style():
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
    plt.rc('figure', figsize=[10,9])
    plt.rc('svg', fonttype='none')


# %%
import logging
# Suppress INFO-level logs for the entire logger
logging.getLogger().setLevel(logging.WARNING)

# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peak_psuedobulk_dynamics/"
os.makedirs(figpath, exist_ok=True)
sc.settings.figdir = figpath

# %%
# Initialize the analysis class ("atac_analysis")
genomes_dir = "/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/fasta/"  # Replace with your genome data directory
ref_genome = "" # "danRer11" for zebrafish, GRCz11

atac_analysis = ATACSeqMotifAnalysis(ref_genome, genomes_dir)

# %%
# import the peak data
adata_peaks = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_celltype_timepoint_pseudobulked_hvp_50k_EDA2.h5ad")
adata_peaks

# %%
adata_peaks.obs_names

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## split the peaks into one vs rest (for each cluster)

# %%
# take each peak cluster (leiden)
cluster_col = "leiden_0.7"
# compute the unique clusters
unique_clusters = adata_peaks.obs[cluster_col].unique()

# define the directory for the output fasta files
output_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/one_vs_rest/"
os.makedirs(output_dir, exist_ok=True)

# for each cluster, split the peaks in/out of the cluster
for cluster_label in unique_clusters:
    print(f"\n--- Processing cluster: {cluster_label} ---")

    # Boolean masks for cluster vs rest
    cluster_mask = (adata_peaks.obs[cluster_col] == cluster_label)
    rest_mask = ~cluster_mask

    # Get the peak strings
    cluster_peaks = adata_peaks.obs_names[cluster_mask]
    rest_peaks = adata_peaks.obs_names[rest_mask]

    print(f"  Number of peaks in cluster {cluster_label}: {len(cluster_peaks)}")
    print(f"  Number of peaks in rest: {len(rest_peaks)}")

    # ----------------------------------------------------------------------
    # (A) Convert cluster_peaks to FASTA
    # ----------------------------------------------------------------------
    # 1. Convert from list of "chr-start-end" to DataFrame
    cluster_peaks_df = atac_analysis.list_peakstr_to_df(cluster_peaks)

    # 2. Validate peaks (remove those not in genome or <5 bp, etc.)
    valid_cluster_df = atac_analysis.check_peak_format(cluster_peaks_df)

    # 3. Convert to FASTA
    valid_cluster_str = (
        valid_cluster_df["chr"]
        + "-"
        + valid_cluster_df["start"].astype(str)
        + "-"
        + valid_cluster_df["end"].astype(str)
    )
    cluster_fasta_obj = atac_analysis.peak_to_fasta(valid_cluster_str)

    # 4. Remove zero-length sequences
    cluster_fasta_obj = atac_analysis.remove_zero_seq(cluster_fasta_obj)

    # 5. Save to file
    cluster_fasta_path = os.path.join(output_dir, f"cluster_{cluster_label}.fasta")
    with open(cluster_fasta_path, "w") as f_out:
        for name, seq in zip(cluster_fasta_obj.ids, cluster_fasta_obj.seqs):
            f_out.write(f">{name}\n{seq}\n")

    print(f"  Saved cluster FASTA: {cluster_fasta_path}")

    # ----------------------------------------------------------------------
    # (B) Convert rest_peaks to FASTA
    # ----------------------------------------------------------------------
    rest_peaks_df = atac_analysis.list_peakstr_to_df(rest_peaks)
    valid_rest_df = atac_analysis.check_peak_format(rest_peaks_df)
    valid_rest_str = (
        valid_rest_df["chr"]
        + "-"
        + valid_rest_df["start"].astype(str)
        + "-"
        + valid_rest_df["end"].astype(str)
    )
    rest_fasta_obj = atac_analysis.peak_to_fasta(valid_rest_str)
    rest_fasta_obj = atac_analysis.remove_zero_seq(rest_fasta_obj)

    rest_fasta_path = os.path.join(output_dir, f"cluster_{cluster_label}_rest.fasta")
    with open(rest_fasta_path, "w") as f_out:
        for name, seq in zip(rest_fasta_obj.ids, rest_fasta_obj.seqs):
            f_out.write(f">{name}\n{seq}\n")

    print(f"  Saved rest FASTA: {rest_fasta_path}")

print("\nAll cluster FASTA files have been created.\n")

# %% [markdown]
# ### NOTE that the leiden clusters 1 and 6 are likely one cluster (early optic_cup)
#
# - we will merge this cluster into one, as we will sub-cluster this later.

# %%
# merge the cluster "1" and "6" into "1" - as they are early optic cup
adata_peaks.obs['leiden'] = adata_peaks.obs['leiden_0.7'].copy()
adata_peaks.obs['leiden'] = adata_peaks.obs['leiden'].replace({'1': '1', '6': '1'})

# %%
# Re-compute the cluster 1 vs rest
# Boolean masks for cluster vs rest
cluster_col = "leiden"
cluster_mask = (adata_peaks.obs[cluster_col] == "1")
rest_mask = ~cluster_mask

# Get the peak strings
cluster_peaks = adata_peaks.obs_names[cluster_mask]
rest_peaks = adata_peaks.obs_names[rest_mask]

print(f"  Number of peaks in cluster {cluster_label}: {len(cluster_peaks)}")
print(f"  Number of peaks in rest: {len(rest_peaks)}")

# ----------------------------------------------------------------------
# (A) Convert cluster_peaks to FASTA
# ----------------------------------------------------------------------
# 1. Convert from list of "chr-start-end" to DataFrame
cluster_peaks_df = atac_analysis.list_peakstr_to_df(cluster_peaks)

# 2. Validate peaks (remove those not in genome or <5 bp, etc.)
valid_cluster_df = atac_analysis.check_peak_format(cluster_peaks_df)

# 3. Convert to FASTA
valid_cluster_str = (
    valid_cluster_df["chr"]
    + "-"
    + valid_cluster_df["start"].astype(str)
    + "-"
    + valid_cluster_df["end"].astype(str)
)
cluster_fasta_obj = atac_analysis.peak_to_fasta(valid_cluster_str)

# 4. Remove zero-length sequences
cluster_fasta_obj = atac_analysis.remove_zero_seq(cluster_fasta_obj)

# 5. Save to file
cluster_fasta_path = os.path.join(output_dir, f"cluster_merged_{cluster_label}.fasta")
with open(cluster_fasta_path, "w") as f_out:
    for name, seq in zip(cluster_fasta_obj.ids, cluster_fasta_obj.seqs):
        f_out.write(f">{name}\n{seq}\n")

print(f"  Saved cluster FASTA: {cluster_fasta_path}")

# ----------------------------------------------------------------------
# (B) Convert rest_peaks to FASTA
# ----------------------------------------------------------------------
rest_peaks_df = atac_analysis.list_peakstr_to_df(rest_peaks)
valid_rest_df = atac_analysis.check_peak_format(rest_peaks_df)
valid_rest_str = (
    valid_rest_df["chr"]
    + "-"
    + valid_rest_df["start"].astype(str)
    + "-"
    + valid_rest_df["end"].astype(str)
)
rest_fasta_obj = atac_analysis.peak_to_fasta(valid_rest_str)
rest_fasta_obj = atac_analysis.remove_zero_seq(rest_fasta_obj)

rest_fasta_path = os.path.join(output_dir, f"cluster_merged_{cluster_label}_rest.fasta")
with open(rest_fasta_path, "w") as f_out:
    for name, seq in zip(rest_fasta_obj.ids, rest_fasta_obj.seqs):
        f_out.write(f">{name}\n{seq}\n")

print(f"  Saved rest FASTA: {rest_fasta_path}")


# %% [markdown]
# ## export the peaks with "cluster" for gimme maelstrom

# %%
def export_peaks_for_gimmemotifs(adata_peaks):
    # Create a DataFrame with properly formatted peak locations
    export_df = pd.DataFrame({
        'loc': 'chr' + adata_peaks.obs_names.str.replace('-', ':', 1).str.replace('-', '-', 1),
        'cluster': adata_peaks.obs['leiden']
    })
    # Optional: Remove any potential duplicates
    export_df = export_df.drop_duplicates()
    
    # Export to a tab-separated file
    output_file = 'peaks_50k_hvps_leiden.txt'
    export_df.to_csv(output_file, sep='\t', index=False)
    
    print(f"Exported {len(export_df)} peaks to {output_file}")
    print("Clusters present:", export_df['cluster'].unique())
    
    return export_df

# Run the export
peaks_export = export_peaks_for_gimmemotifs(adata_peaks)

# %%
peaks_export.head()

# %%
# import pyfaidx

# # Load the genome
# genome = pyfaidx.Fasta("/hpc/mydata/yang-joon.kim/genomes/danRer11/danRer11.fa")

# # Print chromosome names
# print(list(genome.keys()))

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## [differential motif detection] Running gimme maelstrom for differential motifs between peak groups/clusters
#
# - reference: https://gimmemotifs.readthedocs.io/en/master/tutorials.html#find-differential-motifs
#
#

# %%
# %matplotlib inline

# %%
# check out the result of the maelstrom
from gimmemotifs.maelstrom import MaelstromResult

mr = MaelstromResult("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_diff_motifs_leiden/maelstrom_peaks_50k_hvps_leiden_out/")
mr.plot_heatmap(threshold=2)

# %%
# check out the result of the maelstrom
from gimmemotifs.maelstrom import MaelstromResult

mr = MaelstromResult("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_diff_motifs_leiden/maelstrom_peaks_50k_hvps_leiden_out/")
# mr.plot_heatmap(threshold=2)
# plt.savefig("maelstrom.blood.1k.out/heatmap.png", dpi=300)

# %%
mr.scores

# %%
mr.result

# %%
# save the cluster-by-motifs dataframe
mr.result.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peak_clusts_by_motifs_maelstrom.csv")


# %%

# %%
# define a function to extract the direct/indirect factors for each motif, and create a dataframe of motifs
def extract_motif_factors(mr_motifs):
    """
    Extracts direct and indirect factors from mr.motifs and converts them into a DataFrame.

    Parameters:
    -----------
    mr_motifs : dict
        Dictionary of motifs (mr.motifs) where each motif contains a `.factors` dictionary.

    Returns:
    --------
    pd.DataFrame
        DataFrame with motif names as the index and two columns: "direct" and "indirect" factors.
    """
    motif_data = []

    for motif_name, motif_obj in mr_motifs.items():
        # Extract factors (handle cases where direct/indirect keys are missing)
        direct_factors = motif_obj.factors.get('direct', [])
        indirect_factors = motif_obj.factors.get('indirect\nor predicted', [])

        # Convert lists to comma-separated strings
        direct_str = ", ".join(direct_factors) if direct_factors else "None"
        indirect_str = ", ".join(indirect_factors) if indirect_factors else "None"

        # Append to list
        motif_data.append([motif_name, direct_str, indirect_str])

    # Convert to DataFrame
    df_motif_factors = pd.DataFrame(motif_data, columns=["motif", "direct", "indirect"])
    df_motif_factors.set_index("motif", inplace=True)

    return df_motif_factors

# Example usage
df_motifs = extract_motif_factors(mr.motifs)

# check the df
df_motifs.head()

# Save to CSV
df_motifs.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/info_GM_5.0_motif_factors.csv")

# # Display first few rows
# print(df_motifs.head())


# %%

# %%

# %%
adata_peaks.obs_names

# %%
mr.plot_heatmap(threshold=3)

# %%
mr.plot_heatmap(threshold=4)
plt.savefig(figpath + "maelstrom_peak_leiden_clusts_thresh_4.png")
plt.savefig(figpath + "maelstrom_peak_leiden_clusts_thresh_4.pdf")
plt.show()

# %%
mr.plot_heatmap(threshold=3.5)
plt.savefig(figpath + "maelstrom_peak_leiden_clusts_thresh_3.5.png")
plt.savefig(figpath + "maelstrom_peak_leiden_clusts_thresh_3.5.pdf")
plt.show()

# %%
sc.pl.umap(adata_peaks, color="leiden", legend_loc="on data", save="_peaks_leiden_merged.png")

# %%
sc.pl.umap(adata_peaks, color="leiden", legend_loc="on data")

# %%
adata_peaks[adata_peaks.obs["leiden"]=="9"].obs["celltype"].value_counts()

# %%
mr.result.head()

# %%
motif_obj = mr.motifs["GM.5.0.Forkhead.0031"]
print(motif_obj.id)
print(motif_obj.factors)   # e.g., ["NR1H3", "NR1H2", "RXR", ...] if known

# %%
# 1) Filter based on threshold
threshold = 3.5 # for example
df = mr.result.copy()

# 2) Keep only those rows where at least one absolute value >= threshold
mask = df.abs().ge(threshold).any(axis=1)
df_filt = df[mask]
# Example: remove "z-score " prefix
df_filt.columns = [col.replace("z-score ", "") for col in df_filt.columns]

# 3) RENAME ROWS to factor names (similar to 'plot_heatmap(name=True)')
# ---------------------------------------------------------
df_filt_named = df_filt.copy()
factors = []
for motif_id in df_filt_named.index:
    # The same logic that mr.plot_heatmap() uses:
    # mr.motifs[motif_id].format_factors(...)
    motif_obj = mr.motifs.get(motif_id, None)
    if motif_obj is not None:
        # Example parameters (tweak as needed)
        factor_str = motif_obj.format_factors(
            max_length=5,        # how many factors to show
            html=False,
            include_indirect=True,
            extra_str=",..",
        )
    else:
        factor_str = motif_id  # fallback if motif is missing
    factors.append(factor_str)

# Put these factors in a new column, then set as index
df_filt_named["factors"] = factors
df_filt_named = df_filt_named.set_index("factors")


# Optional: check how many motifs remain
print(f"Number of motifs passing threshold: {df_filt.shape[0]}")


# 4) CLUSTER + PLOT using seaborn.clustermap
# ---------------------------------------------------------
g = sns.clustermap(
    df_filt_named,
    metric="euclidean",   # or "correlation"
    method="ward",        # or "average", "complete", etc.
    cmap="RdBu_r",        # diverging colormap for +/- z-scores
    center=0,             # center colormap on zero
    linewidths=0.5,
    figsize=(8, 6),      # adjust as you like
    xticklabels=True,
    yticklabels=True
)

# Rotate x-axis tick labels if desired
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
# # 4) Use clustermap to cluster both rows and columns
# g = sns.clustermap(
#     df_filt,
#     metric="euclidean",    # or "correlation", "cosine", etc.
#     method="ward",         # or "average", "complete", etc.
#     cmap="RdBu_r",         # typical diverging colormap
#     center=0,              # center on zero for z-scores
#     linewidths=0.5,        # helps separate cells visually
#     figsize=(8,6)        # adjust figure size as needed
# )
plt.savefig(figpath + "motifs_heatmap_thresh_3.5_clustered.png")
plt.savefig(figpath + "motifs_heatmap_thresh_3.5_clustered.pdf")
plt.show()

# %%
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

# 1) FILTER based on a threshold in mr.result
threshold = 3.5
df = mr.result.copy()  # the aggregated motif activity DataFrame

# 2) Keep only rows where at least one absolute value >= threshold
mask = df.abs().ge(threshold).any(axis=1)
df_filt = df[mask]

# Remove "z-score " prefix from columns (optional, if your columns have that prefix)
df_filt.columns = [col.replace("z-score ", "") for col in df_filt.columns]

print(f"Number of motifs passing threshold: {df_filt.shape[0]}")

# 2) Convert motif IDs to factor names (as rows)
df_filt_named = df_filt.copy()
factors = []
for motif_id in df_filt_named.index:
    motif_obj = mr.motifs.get(motif_id, None)
    if motif_obj is not None:
        # Adjust parameters as you like
        factor_str = motif_obj.format_factors(
            max_length=5,
            html=False,
            include_indirect=True,
            extra_str=",..",
        )
    else:
        factor_str = motif_id  # fallback if no motif object
    factors.append(factor_str)

df_filt_named["factors"] = factors
df_filt_named = df_filt_named.set_index("factors")

# 3) Build a custom row linkage based on motif family
# Gather the original motif IDs (before we replaced them with factor names)
# So we can parse the family from the motif ID string.
motif_ids = list(df_filt.index)

# Extract family name from motif ID
def get_family(motif_id):
    # Typically: "GM.5.0.Nuclear_receptor.0001" -> parts[3] = "Nuclear_receptor"
    parts = motif_id.split(".")
    return parts[3] if len(parts) > 3 else "Unknown"

families = [get_family(m) for m in motif_ids]
N = len(motif_ids)

# Build NxN distance matrix: 0 if same family, 1 otherwise
dist_fam = np.zeros((N, N), dtype=float)
for i in range(N):
    for j in range(N):
        dist_fam[i, j] = 0.0 if families[i] == families[j] else 1.0

# Convert to condensed form and build a linkage for the rows
dist_condensed = squareform(dist_fam)
Z_rows = linkage(dist_condensed, method="complete")  # or "average", "single", "ward", etc.

# Now we do a clustermap, specifying the row linkage for family-based clustering.
g = sns.clustermap(
    df_filt_named,
    row_linkage=Z_rows,   # force row clustering by family
    col_cluster=True,     # cluster columns by data
    metric="euclidean",   # used for columns only
    method="ward",        # used for columns only
    cmap="RdBu_r",
    center=0,
    linewidths=0.5,
    figsize=(8, 6),
    xticklabels=True,
    yticklabels=True
)

# Rotate x-axis labels if desired
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)

plt.show()

# %%
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import linkage
# import matplotlib.patches as mpatches

# ---------------------------------------------------------
# 1) FILTER motifs based on threshold in mr.result
# ---------------------------------------------------------
threshold = 3.5
df = mr.result.copy()

mask = df.abs().ge(threshold).any(axis=1)
df_filt = df[mask]
df_filt.columns = [col.replace("z-score ", "") for col in df_filt.columns]
print(f"Number of motifs passing threshold: {df_filt.shape[0]}")

# ---------------------------------------------------------
# 2) Cluster rows by family
# ---------------------------------------------------------
def get_family(motif_id):
    parts = motif_id.split(".")
    return parts[3] if len(parts) > 3 else "Unknown"

motif_ids = df_filt.index.tolist()
families = [get_family(m) for m in motif_ids]
N = len(motif_ids)

# 0/1 distance: 0 if same family, else 1
dist_fam = np.zeros((N, N), dtype=float)
for i in range(N):
    for j in range(N):
        dist_fam[i, j] = 0.0 if families[i] == families[j] else 1.0

dist_condensed = squareform(dist_fam)
Z_rows = linkage(dist_condensed, method="complete")  # or "average", etc.

# ---------------------------------------------------------
# 3) Create a DataFrame with simpler row labels (no "Direct/Indirect" text)
# ---------------------------------------------------------
df_filt_named = df_filt.copy()
row_labels = []
row_is_direct = []  # True if there's at least one direct factor

for motif_id in df_filt_named.index:
    motif_obj = mr.motifs.get(motif_id, None)
    direct_list = []
    indirect_list = []

    if motif_obj is not None and isinstance(motif_obj.factors, dict):
        direct_list = motif_obj.factors.get("direct", [])
        # Some GimmeMotifs versions store indirect as "indirect or predicted" or "predicted"
        indirect_list = motif_obj.factors.get("indirect or predicted", [])
        if not indirect_list:
            indirect_list = motif_obj.factors.get("predicted", [])

    # Build a single label from all factors (direct + indirect)
    all_factors = direct_list + indirect_list
    if all_factors:
        label_str = ",".join(all_factors)
    else:
        # fallback to motif ID if we have no factors
        label_str = motif_id

    row_labels.append(label_str)
    row_is_direct.append(len(direct_list) > 0)  # if any direct factor

df_filt_named["row_label"] = row_labels
df_filt_named = df_filt_named.set_index("row_label")

# ---------------------------------------------------------
# 4) Build row color strips (A) for family, (B) for directness
# ---------------------------------------------------------
unique_fams = sorted(set(families))
palette_fams = sns.color_palette("hls", len(unique_fams))
family2color = dict(zip(unique_fams, palette_fams))

direct_colors = {"direct": "black", "indirect": "grey"}

row_colors_family = []
row_colors_direct = []

for motif_id in motif_ids:
    fam = get_family(motif_id)
    fam_color = family2color.get(fam, (0.5, 0.5, 0.5))
    row_colors_family.append(fam_color)

    motif_obj = mr.motifs.get(motif_id)
    if motif_obj is not None and isinstance(motif_obj.factors, dict):
        is_direct = len(motif_obj.factors.get("direct", [])) > 0
    else:
        is_direct = False

    color = direct_colors["direct"] if is_direct else direct_colors["indirect"]
    row_colors_direct.append(color)

row_colors = [row_colors_family, row_colors_direct]

# ---------------------------------------------------------
# 5) Plot with clustermap
# ---------------------------------------------------------
g = sns.clustermap(
    df_filt_named,
    row_linkage=Z_rows,      # rows by family
    col_cluster=True,        # columns by data
    metric="euclidean",      # for columns
    method="ward",           # for columns
    cmap="RdBu_r",
    center=0,
    linewidths=0.5,
    figsize=(8, 6),
    row_colors=row_colors,   # color strips on left
    xticklabels=True,
    yticklabels=True
)

# Rotate x-axis labels
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)

# ---------------------------------------------------------
# 6) Color the entire row label black if direct, grey if purely indirect
# ---------------------------------------------------------
row_order = g.dendrogram_row.reordered_ind
yticklabels = g.ax_heatmap.get_yticklabels()

for label, idx in zip(yticklabels, row_order):
    label_color = "black" if row_is_direct[idx] else "grey"
    label.set_color(label_color)

# ---------------------------------------------------------
# 7) Move the legends to the right side
# ---------------------------------------------------------
# Build legend patches for families
family_patches = [
    mpatches.Patch(color=color, label=fam)
    for fam, color in family2color.items()
]

# Build legend patches for directness
direct_patches = [
    mpatches.Patch(color=col, label=status)
    for status, col in direct_colors.items()
]

# Place them on the right side. 
fam_legend = g.ax_heatmap.legend(
    handles=family_patches,
    title="Family",
    bbox_to_anchor=(1.3, 1.0),  # shift further right
    loc="upper left"
)
g.ax_heatmap.add_artist(fam_legend)

# g.ax_heatmap.legend(
#     handles=direct_patches,
#     title="Directness",
#     bbox_to_anchor=(1.3, 0.7),
#     loc="upper left"
# )
plt.savefig(figpath + "motifs_heatmap_thresh_3.5_annotated_families.png")
plt.savefig(figpath + "motifs_heatmap_thresh_3.5_annotated_families.pdf")
plt.show()


# %%
mr.motifs.keys()

# %%
motif_obj = mr.motifs["GM.5.0.Ets.0033"]
print(motif_obj.id)
print(motif_obj.factors)   # e.g., ["NR1H3", "NR1H2", "RXR", ...] if known

# %% [markdown]
# ### heatmap (tranposed - leiden clusters-by-motifs)

# %%
# 1) Filter based on threshold
threshold = 3.5 # for example
df = mr.result.copy()

# 2) Keep only those rows where at least one absolute value >= threshold
mask = df.abs().ge(threshold).any(axis=1)
df_filt = df[mask]
# Example: remove "z-score " prefix
df_filt.columns = [col.replace("z-score ", "") for col in df_filt.columns]

# 3) RENAME Columns to factor names (similar to 'plot_heatmap(name=True)')
# ---------------------------------------------------------
df_filt_named = df_filt.copy()
factors = []
for motif_id in df_filt_named.index:
    # The same logic that mr.plot_heatmap() uses:
    # mr.motifs[motif_id].format_factors(...)
    motif_obj = mr.motifs.get(motif_id, None)
    if motif_obj is not None:
        # Example parameters (tweak as needed)
        factor_str = motif_obj.format_factors(
            max_length=3,        # how many factors to show
            html=False,
            include_indirect=True,
            # extra_str=",..",
        )
    else:
        factor_str = motif_id  # fallback if motif is missing
    factors.append(factor_str)

# Put these factors in a new column, then set as index
df_filt_named["factors"] = factors
df_filt_named = df_filt_named.set_index("factors")


# Optional: check how many motifs remain
print(f"Number of motifs passing threshold: {df_filt.shape[0]}")

# transpose the matrix
df_filt_trans = df_filt_named.transpose()


# 4) CLUSTER + PLOT using seaborn.clustermap
# ---------------------------------------------------------
g = sns.clustermap(
    df_filt_trans,
    metric="euclidean",   # or "correlation"
    method="ward",        # or "average", "complete", etc.
    cmap="RdBu_r",        # diverging colormap for +/- z-scores
    center=0,             # center colormap on zero
    linewidths=0.5,
    figsize=(6, 8),      # adjust as you like
    xticklabels=True,
    yticklabels=True
)

# Rotate x-axis tick labels if desired
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
# # 4) Use clustermap to cluster both rows and columns
# g = sns.clustermap(
#     df_filt,
#     metric="euclidean",    # or "correlation", "cosine", etc.
#     method="ward",         # or "average", "complete", etc.
#     cmap="RdBu_r",         # typical diverging colormap
#     center=0,              # center on zero for z-scores
#     linewidths=0.5,        # helps separate cells visually
#     figsize=(8,6)        # adjust figure size as needed
# )
plt.savefig(figpath + "motifs_heatmap_thresh_3.5_clustered_transposed.png")
plt.savefig(figpath + "motifs_heatmap_thresh_3.5_clustered_transposed.pdf")
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Filter based on threshold
threshold = 3.5  # example threshold
df = mr.result.copy()

# 2) Keep only those rows where at least one absolute value >= threshold
mask = df.abs().ge(threshold).any(axis=1)
df_filt = df[mask]

# Example: remove "z-score " prefix from column names
df_filt.columns = [col.replace("z-score ", "") for col in df_filt.columns]

# 3) Store TF factor names separately while keeping motif IDs as index
factors = []
for motif_id in df_filt.index:
    motif_obj = mr.motifs.get(motif_id, None)
    if motif_obj is not None:
        factor_str = motif_obj.format_factors(
            max_length=3,  # how many factors to show
            html=False,
            include_indirect=True,
        )
    else:
        factor_str = "Unknown"  # fallback if motif is missing
    factors.append(factor_str)

# Store factor names in a separate column (not as index)
df_filt["TF_factors"] = factors  # This ensures we do not replace motif IDs

# Transpose the matrix for clustering
df_filt_trans = df_filt.drop(columns=["TF_factors"]).transpose()

# 4) CLUSTER + PLOT using seaborn.clustermap
g = sns.clustermap(
    df_filt_trans,
    metric="euclidean",
    method="ward",
    cmap="RdBu_r",
    center=0,
    linewidths=0.5,
    figsize=(6, 8),
    xticklabels=True,
    yticklabels=True  # Keep y-ticks as motif IDs
)

# Rotate x-axis tick labels if desired
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)

# Save the figure
plt.savefig(figpath + "motifs_heatmap_thresh_3.5_clustered_transposed_motifnames.png")
plt.savefig(figpath + "motifs_heatmap_thresh_3.5_clustered_transposed_motifnames.pdf")
plt.show()


# %% [markdown]
# ## motif enrichment scores projected onto peak UMAP

# %%
mr_scores = mr.scores
mr_scores.head()

# %%
# Convert index from 'chr1:start-end' to '1-start-end'
mr_scores.index = mr_scores.index.str.replace(r'chr(\d+):', r'\1-', regex=True)

# map the columns to the 
for col in mr_scores.columns:
    adata_peaks.obs[col] = adata_peaks.obs_names.map(mr_scores[col])

# %%
mr_scores.columns

# %%
mr.plot_heatmap(threshold=3.5, name=False)
plt.show()

# %%
# Define threshold
threshold = 3.5

# Filter rows where max or min value exceeds the threshold
filtered_tfs = mr.result[(mr.result.max(axis=1) > threshold) | (mr.result.min(axis=1) < -threshold)].index.tolist()

# Print the list of TFs
print(filtered_tfs)


# %%
sc.pl.umap(adata_peaks, color=["GM.5.0.Ets.0033","GM.5.0.Ets.0009",
                               "GM.5.0.Homeodomain.0053","GM.5.0.T-box.0014",
                               "GM.5.0.Homeodomain.0062", "GM.5.0.Unknown.0094",
                               "GM.5.0.C2H2_ZF.0071","GM.5.0.C2H2_ZF.0177", "GM.5.0.p53.0011",
                               "GM.5.0.Nuclear_receptor.0008", "GM.5.0.bHLH.0043"], 
           ncols=3, cmap="plasma", vmin=1, vmax=3.5)

# %%
# sc.pl.umap(adata_peaks, color=["GM.5.0.Ets.0033","GM.5.0.Ets.0009",
#                                "GM.5.0.Homeodomain.0053","GM.5.0.T-box.0014",
#                                "GM.5.0.Homeodomain.0062", "GM.5.0.Unknown.0094",
#                                "GM.5.0.C2H2_ZF.0071","GM.5.0.C2H2_ZF.0177", "GM.5.0.p53.0011",
#                                "GM.5.0.Nuclear_receptor.0008", "GM.5.0.bHLH.0043"], 
#            ncols=3, cmap="RdBu_r", vmin=-3, vmax=3)

# 
sc.pl.umap(adata_peaks, color=filtered_tfs, \
           ncols=3, cmap="plasma", vmin=1, vmax=3)

# %%
motif_name = "GM.5.0.Ets.0033"
motif_name = "GM.5.0.C2H2_ZF.0177"
adata_peaks.obs[motif_name].sort_values(ascending=False)

# %%
list_motifs = ["GM.5.0.C2H2_ZF.0177", "GM.5.0.Unknown.0055", "GM.5.0.Unknown.0094", "GM.5.0.Unknown.0094",# from the cluster 1
"GM.5.0.Homeodomain.0135","GM.5.0.C2H2_ZF.0071", # from the cluster 21
"GM.5.0.Nuclear_receptor.0008", # clusters 3 and 17
"GM.5.0.Ets.0033", # cluster 9 (hematopoesis)
"GM.5.0.Grainyhead.0004", # cluster 7
"GM.5.0.Mixed.0049",
"GM.5.0.bHLH.0043",
"GM.5.0.T-box.0014"]

sc.pl.umap(adata_peaks, color=list_motifs, ncols=3, cmap="plasma", vmin=1, vmax=3, save="_peaks_motifs_maelstrom_plasma.png")


# %%
list_motifs = ["GM.5.0.C2H2_ZF.0071","GM.5.0.Ets.0033","GM.5.0.bHLH.0028"]

sc.pl.umap(adata_peaks, color=list_motifs, ncols=3, cmap="magma", vmin=1, vmax=3)
sc.pl.umap(adata_peaks, color=list_motifs, ncols=3, cmap="plasma", vmin=1, vmax=3)
# sc.pl.umap(adata_peaks, color=list_motifs, ncols=3, cmap="YlOrBr", vmin=1, vmax=3)
sc.pl.umap(adata_peaks, color=list_motifs, ncols=3, cmap="cividis", vmin=1, vmax=3)
sc.pl.umap(adata_peaks, color=list_motifs, ncols=3, cmap="viridis", vmin=1, vmax=3)
# sc.pl.umap(adata_peaks, color=list_motifs, ncols=3, cmap="rocket", vmin=1, vmax=3)
# sc.pl.umap(adata_peaks, color=list_motifs, ncols=3, cmap="BuPu", vmin=1, vmax=3)


# %%
sc.pl.umap(adata_peaks, color=list_motifs, ncols=3, cmap="viridis_r", vmin=1, vmax=3)

# %%
sc.pl.umap(adata_peaks, color=list_motifs, ncols=3, cmap="plasma", vmin=1, vmax=3, save="_peaks_example_motifs_maelstrom_plasma.png")
sc.pl.umap(adata_peaks, color=list_motifs, ncols=3, cmap="cividis", vmin=1, vmax=3, save="_peaks_example_motifs_maelstrom_cividis.png")
sc.pl.umap(adata_peaks, color=list_motifs, ncols=3, cmap="viridis", vmin=1, vmax=3, save="_peaks_example_motifs_maelstrom_viridis.png")

# %%
list_motifs = ["GM.5.0.C2H2_ZF.0071","GM.5.0.Ets.0033","GM.5.0.bHLH.0028"]

# %%
sc.pl.umap(adata_peaks, color=list_motifs[0], cmap="plasma", 
           vmin=1, vmax=3, save="_peaks_maelstrom_C2H2_ZF_0071_plasma.png")

# %%
sc.pl.umap(adata_peaks, color=list_motifs[1], cmap="plasma", 
           vmin=1, vmax=3, save="_peaks_maelstrom_Ets_0033_plasma.png")

# %%
sc.pl.umap(adata_peaks, color=list_motifs[2], cmap="plasma", 
           vmin=1, vmax=3, save="_peaks_maelstrom_bHLH_0028_plasma.png")

# %%
sc.pl.umap(adata_peaks, color=list_motifs, ncols=3, cmap="plasma", vmin=1, vmax=3, save="_peaks_example_motifs_maelstrom_plasma.pdf")

# %%
# Create a figure and an axes for the colorbar
fig, ax = plt.subplots(figsize=(1, 6))  # Adjust size as needed

# Define the colormap and normalization
cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=1, vmax=3)

# Create a ScalarMappable (required for colorbar)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy array since we don't have actual data

# Add the colorbar to the axes
cbar = plt.colorbar(sm, cax=ax, orientation='vertical')

# Customize the colorbar (optional)
cbar.set_label('Colorbar Label')
cbar.set_ticks([1, 2, 3])
cbar.set_ticklabels(['1', '2', '3'])

# Display the plot
plt.savefig("colormap.pdf")
plt.show()

# %% [markdown]
# ## create an adata object for peaks-by-motifs (differential)
# - We will create an adata object using the peaks-by-motifs count matrix - this will keep the same metadata for the peaks (obs)
#
#
#

# %%
mr_scores.values

# %%
import scipy.sparse as sp

# extract the count matrix and convert it to a sparse matrix
sparse_matrix = sp.csr_matrix(mr_scores.values)

# create an adata object
peaks_by_motifs = sc.AnnData(X = sparse_matrix)
peaks_by_motifs.obs_names = mr_scores.index
peaks_by_motifs.var_names = mr_scores.columns

# %%
peaks_by_motifs

# %%
# # copy over the obs fields
fields_to_copy = ['chrom', 'peak_type', 'timepoint', 'timepoint_contrast', 
                 'celltype', 'celltype_contrast', 'associated_gene', 
                 'association_type', 'n_overlapping_genes', 'celltype_timepoint', 
                 'leiden_0.7', 'leiden_0.5', 'leiden_0.3', 'leiden_1', 
                 'optic_cup_subclust', 'linked_gene', 'link_score', 'link_zscore', 
                 'link_pvalue', 'is_linked', 'hemato_manual', 'leiden']

# # copy over the metadata
peaks_by_motifs.obs[fields_to_copy] = adata_peaks.obs[fields_to_copy].loc[peaks_by_motifs.obs_names]

# %%
# # copy over the PCA/UMAP coordinates
peaks_by_motifs.obsm["X_pca_pseudobulk"] = adata_peaks.obsm["X_pca"]
peaks_by_motifs.obsm["X_umap_pseudobulk"] = adata_peaks.obsm["X_umap"]

# %%
sc.tl.pca(peaks_by_motifs)
sc.pl.pca_variance_ratio(peaks_by_motifs)

# %%
sc.pl.pca(peaks_by_motifs, components=['1,2'], color=['leiden',"celltype","timepoint"])

# %%
sc.pp.neighbors(peaks_by_motifs, n_neighbors=20, n_pcs=10, metric="cosine") 
sc.tl.umap(peaks_by_motifs, min_dist=0.05)
sc.pl.umap(peaks_by_motifs, color=["leiden"])

# %%
sc.pl.umap(peaks_by_motifs, color="GM.5.0.Ets.0033", cmap="plasma", vmin=1, vmax=3)

# %%
sc.tl.leiden(peaks_by_motifs, key_added="leiden_motifs", resolution=0.7)
sc.pl.umap(peaks_by_motifs, color="leiden_motifs")

# %%
peaks_by_motifs[peaks_by_motifs.obs["leiden_motifs"]=="6"].obs["celltype"].value_counts()

# %%
sc.tl.rank_genes_groups(peaks_by_motifs, groupby="leiden_motifs")

# %%
sc.pl.rank_genes_groups_dotplot(peaks_by_motifs, n_genes=2)

# %%
# save the peaks-by-motifs adata object
peaks_by_motifs.write_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/peaks_by_motifs_50k_hvps_GM5Motifs.h5ad")

# %% [markdown]
# ## plot the UMAP with aggreagated/ranked scores (per cluster)

# %%
# Step 1: Create a mapping from cluster ID to its row in mr.result
cluster_to_scores = {}

# Iterate through columns in mr.result (each representing a leiden cluster)
for col in mr.result.columns:
    if col.startswith("z-score"):
        # Extract cluster ID
        cluster_id = col.split(" ")[1]
        # Store this column as the scores for this cluster
        cluster_to_scores[cluster_id] = mr.result[col]

# Step 2: Add all motif scores directly to peaks_by_motifs.obs
for motif in mr.result.index:
    motif_name = motif.replace('.', '_')  # Clean up motif name for column naming
    
    # For each peak, find its leiden cluster and assign the corresponding motif score
    peaks_by_motifs.obs[f'motif_{motif_name}'] = peaks_by_motifs.obs['leiden'].apply(
        lambda x: cluster_to_scores.get(str(x), {}).get(motif, 0)
    )
    # ensure that the scores are "float64", not "category"
    peaks_by_motifs.obs[f'motif_{motif_name}'] = peaks_by_motifs.obs[f'motif_{motif_name}'].astype("float64")

# Now you can visualize specific motifs
motif_cols = [f'motif_{motif.replace(".", "_")}' for motif in mr.result.index[:5]]  # First 5 motifs
sc.pl.umap(peaks_by_motifs, color=motif_cols, cmap='bwr')

# %%
peaks_by_motifs

# %%
# plot the UMAP and save the plot for TF motif
motif = "motif_GM_5_0_Ets_0033"
extracted_motif = motif.split("motif_GM_5_0_")[1]
sc.pl.embedding(peaks_by_motifs, basis="X_umap_pseudobulk", color=["motif_GM_5_0_Ets_0033"], 
                cmap="RdBu_r", vmin=-4, vmax=4, save=f"_peaks_leiden_{extracted_motif}")

# %%
sc.pl.embedding(peaks_by_motifs, basis="X_umap", color=["motif_GM_5_0_Ets_0033"], 
                cmap="RdBu_r", vmin=-4, vmax=4)

# %%
peaks_by_motifs.obs["motif_GM_5_0_C2H2_ZF_0071"] = peaks_by_motifs.obs["motif_GM_5_0_C2H2_ZF_0071"].astype('float64')

# %%
# plot the UMAP and save the plot for TF motif
motif = "motif_GM_5_0_C2H2_ZF_0071"
extracted_motif = motif.split("motif_GM_5_0_")[1]
sc.pl.embedding(peaks_by_motifs, basis="X_umap_pseudobulk", color=motif, 
                cmap="RdBu_r", vmin=-4, vmax=4, save=f"_peaks_leiden_{extracted_motif}")

# %%
# plot the UMAP and save the plot for TF motif
motif = "motif_GM_5_0_bHLH_0028"
extracted_motif = motif.split("motif_GM_5_0_")[1]
sc.pl.embedding(peaks_by_motifs, basis="X_umap_pseudobulk", color=[motif], 
                cmap="RdBu_r", vmin=-4, vmax=4, save=f"_peaks_leiden_{extracted_motif}")

# %%
# plot the UMAP and save the plot for TF motif
motif = "motif_GM_5_0_Mixed_0049"
extracted_motif = motif.split("motif_GM_5_0_")[1]
sc.pl.embedding(peaks_by_motifs, basis="X_umap_pseudobulk", color=motif, 
                cmap="RdBu_r", vmin=-4, vmax=4, save=f"_peaks_leiden_{extracted_motif}")

# %%
# plot the UMAP and save the plot for TF motif
motif = "motif_GM_5_0_Grainyhead_0004"
extracted_motif = motif.split("motif_GM_5_0_")[1]
sc.pl.embedding(peaks_by_motifs, basis="X_umap_pseudobulk", color=motif, 
                cmap="RdBu_r", vmin=-4, vmax=4, save=f"_peaks_leiden_{extracted_motif}")

# %%
motif = "motif_GM_5_0_Grainyhead_0004"
sc.pl.pca(peaks_by_motifs, components=['1,2'], color=[motif], 
          cmap="RdBu_r", vmin=-4, vmax=4)

# %%
for motif in peaks_by_motifs.obs.columns[peaks_by_motifs.obs.columns.str.startswith("motif_GM_5_0")]:
    extracted_motif = motif.split("motif_GM_5_0_")[1]
    sc.pl.embedding(peaks_by_motifs, basis="X_umap_pseudobulk", color=motif, 
                    cmap="RdBu_r", vmin=-4, vmax=4, save=f"_peaks_leiden_{extracted_motif}.png", show=False)

# %%

# %%
mr.counts

# %%
peaks_by_motifs.obs.head()

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## generate a UMAP of cluster-by-motifs (from maelstrom scores)

# %%
mr.result

# %%

# %%

# %% [markdown]
# ## GimmeMotifs to generate the sequence logo plots

# %%
from gimmemotifs.motif import Motif, read_motifs

# Load motifs from the default database
motif_file = "/hpc/mydata/yang-joon.kim/genomes/danRer11/gimme.vertebrate.v5.0.pfm"  # Use the correct database file
motifs = read_motifs(motif_file)

# %%
# import logomaker

# Get the specific motif (e.g., "GM.5.0.Ets.0033")
motif_name = "GM.5.0.Ets.0033"
selected_motif = next(m for m in motifs if m.id == motif_name)

# Extract the Position Frequency Matrix (PFM) and Position Weight Matrix (PWM)
pfm = selected_motif.to_pfm()  # Transpose to make it compatible with logomaker
pwm = selected_motif.to_ppm()

# consensus sequence
print(selected_motif.to_consensus())

# %%
# Use either PWM or PFM (they are already probability matrices)
matrix = selected_motif.pwm  # Or use selected_motif.pfm if preferred

# Convert matrix to a Pandas DataFrame with correct column labels
df = pd.DataFrame(matrix, columns=["A", "C", "G", "T"])

# Plot the sequence logo
plt.figure(figsize=(10, 3))
logomaker.Logo(df)

# Add title
plt.title(f"Sequence Logo for {selected_motif.id}")

# Show the plot
plt.grid(False)
# plt.savefig(figpath + "GM.5.0.Ets.0033.seq.logo.png")
# plt.savefig(figpath + "GM.5.0.Ets.0033.seq.logo.pdf")
plt.show()


# %%
# Compute information content (IC) from PWM
def compute_information_content(pwm):
    background = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform background probabilities
    ic_matrix = pwm * np.log2(pwm / background)  # Shannon information content
    ic_matrix = np.nan_to_num(ic_matrix)  # Replace NaNs with 0
    return ic_matrix

# Use PWM and normalize to information content
pwm = selected_motif.pwm  # Or use selected_motif.pfm if necessary
ic_matrix = compute_information_content(pwm)

# Convert to Pandas DataFrame for Logomaker
df = pd.DataFrame(ic_matrix, columns=["A", "C", "G", "T"])

# Plot with Logomaker
plt.figure(figsize=(12, 4))  # Increase figure size
logomaker.Logo(df)

plt.title(f"Sequence Logo for {selected_motif.id}")
plt.ylabel("bits")
plt.xlabel("bases")
plt.grid(False)
plt.savefig(figpath + "GM.5.0.Ets.0033.seq.logo.png")
plt.savefig(figpath + "GM.5.0.Ets.0033.seq.logo.pdf")
plt.show()

# %% [markdown]
# ## Use a module to generate the seq.logo plots systematically

# %%
# Add the path to your module
sys.path.append('data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/gimmemotifs')
import module_logo_plots
import importlib
importlib.reload(module_logo_plots)
from module_logo_plots import LogoPlot

# %%
# List of motifs you want to plot
# motif_list = ["GM.5.0.Ets.0033", "GM.5.0.Ets.0034", "GM.5.0.Ets.0035"]  # Replace with your motif IDs
motif_list = filtered_tfs

# Create a directory for output
output_dir = figpath + "logos/"

# Generate plots for each motif
for motif_name in motif_list:
    fig = logo_plotter.generate_logo_plot(
        motif_name=motif_name,
        output_dir=output_dir,
        figsize=(12, 4)
    )
    plt.show()
    plt.close()  # Close the figure to free memory

# Optional: Print consensus sequences
for motif_name in motif_list:
    consensus = logo_plotter.get_consensus_sequence(motif_name)
    print(f"Consensus sequence for {motif_name}: {consensus}")

# %%
# # import logomaker

# # Get the specific motif (e.g., "GM.5.0.Ets.0033")
# motif_list = ["GM.5.0.Ets.0033", "GM.5.0.Paired.box.0010", "GM.5.0.Forkhead.0043"]

# # Create a directory for output
# output_dir = figpath + "logos/"

# # Initialize the LogoPlot class
# logo_plotter = LogoPlot(motif_file)

# # Generate plots for each motif
# for motif_name in motif_list:
#     fig = logo_plotter.generate_logo_plot(
#         motif_name=motif_name,
#         output_dir=output_dir,
#         figsize=(12, 4)
#     )
#     plt.show()
#     plt.close()  # Close the figure to free memory

# # Optional: Print consensus sequences
# for motif_name in motif_list:
#     consensus = logo_plotter.get_consensus_sequence(motif_name)
#     print(f"Consensus sequence for {motif_name}: {consensus}")

# %%
import logomaker

# %%
# Use either PWM or PFM (they are already probability matrices)
motif_list = ["GM.5.0.Ets.0033", "GM.5.0.Forkhead.0043", "GM.5.0.Paired_box.0010"]
motif_name = motif_list[2]
selected_motif = next(m for m in motifs if m.id == motif_name)
matrix = selected_motif.pwm  # Or use selected_motif.pfm if preferred

# Convert matrix to a Pandas DataFrame with correct column labels
df = pd.DataFrame(matrix, columns=["A", "C", "G", "T"])

# Plot the sequence logo
plt.figure(figsize=(10, 3))
logomaker.Logo(df)

# Add title
plt.title(f"Sequence Logo for {selected_motif.id}")

# Show the plot
plt.grid(False)
# plt.savefig(figpath + "GM.5.0.Ets.0033.seq.logo.png")
# plt.savefig(figpath + "GM.5.0.Ets.0033.seq.logo.pdf")
plt.show()

# %%
import numpy as np


# %%
# Compute information content (IC) from PWM
def compute_information_content(pwm):
    background = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform background probabilities
    ic_matrix = pwm * np.log2(pwm / background)  # Shannon information content
    ic_matrix = np.nan_to_num(ic_matrix)  # Replace NaNs with 0
    return ic_matrix

# Use PWM and normalize to information content
pwm = selected_motif.pwm  # Or use selected_motif.pfm if necessary
ic_matrix = compute_information_content(pwm)

# Convert to Pandas DataFrame for Logomaker
df = pd.DataFrame(ic_matrix, columns=["A", "C", "G", "T"])

# Plot with Logomaker
plt.figure(figsize=(12, 4))  # Increase figure size
logomaker.Logo(df)

plt.title(f"Sequence Logo for {selected_motif.id}")
plt.ylabel("bits")
plt.xlabel("bases")
plt.grid(False)
plt.savefig(figpath + "GM.5.0.Paired_box.0010.seq.logo.png")
plt.savefig(figpath + "GM.5.0.Paired_box.0010.seq.logo.pdf")
plt.show()

# %%
len(motifs)

# %%
[m.id for m in motifs if m.id.startswith("GM.5.0.Forkhead.0043")]

# %%
motif_name = "GM.5.0.Forkhead.0043"
selected_motif = next(m for m in motifs if m.id == motif_name)
selected_motif


# %%
# Compute information content (IC) from PWM
def compute_information_content(pwm):
    background = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform background probabilities
    ic_matrix = pwm * np.log2(pwm / background)  # Shannon information content
    ic_matrix = np.nan_to_num(ic_matrix)  # Replace NaNs with 0
    return ic_matrix

# Use PWM and normalize to information content
pwm = selected_motif.pwm  # Or use selected_motif.pfm if necessary
ic_matrix = compute_information_content(pwm)

# Convert to Pandas DataFrame for Logomaker
df = pd.DataFrame(ic_matrix, columns=["A", "C", "G", "T"])

# Plot with Logomaker
plt.figure(figsize=(12, 4))  # Increase figure size
logomaker.Logo(df)

plt.title(f"Sequence Logo for {selected_motif.id}")
plt.ylabel("bits")
plt.xlabel("bases")
plt.grid(False)
plt.savefig(figpath + "GM.5.0.Forkhead.0043.seq.logo.png")
plt.savefig(figpath + "GM.5.0.Forkhead.0043.seq.logo.pdf")
plt.show()

# %%
# Use either PWM or PFM (they are already probability matrices)
motif_name = "GM.5.0.C2H2_ZF.0032"
selected_motif = next(m for m in motifs if m.id == motif_name)
matrix = selected_motif.pwm  # Or use selected_motif.pfm if preferred

# Use PWM and normalize to information content
pwm = selected_motif.pwm  # Or use selected_motif.pfm if necessary
ic_matrix = compute_information_content(pwm)

# Convert to Pandas DataFrame for Logomaker
df = pd.DataFrame(ic_matrix, columns=["A", "C", "G", "T"])

# Plot with Logomaker
plt.figure(figsize=(12, 4))  # Increase figure size
logomaker.Logo(df)

plt.title(f"Sequence Logo for {selected_motif.id}")
plt.ylabel("bits")
plt.xlabel("bases")
plt.grid(False)
plt.savefig(figpath + "GM.5.0.C2H2.ZF.0032.seq.logo.png")
plt.savefig(figpath + "GM.5.0.C2H2.ZF.0032.seq.logo.pdf")
plt.show()

# %%
motif_name = "GM.5.0.Paired_box.0010"
selected_motif = next(m for m in motifs if m.id == motif_name)
selected_motif.factor_info

# %% [markdown]
# # GimmeMotifs maelstrom for sub-clusters
#
# - (Background) The leiden clusters 0, 1, and (8,11) were used for sub-clustering analysis
# - We will use the adata_sub (subsetted adata objects) to create additional labels in the .obs field
# - Re-export these peaks with new labels, and re-run gimme maelstrom to re-compute the motif activity scores
#
#

# %%
adata_peaks.obs["leiden"].unique()

# %%
# 1) Import the adata_sub objects
adata_sub = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/peak_cluster_tailbud_late.h5ad")
adata_sub2 = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/peak_cluster_optic_cup_early.h5ad")
adata_sub3 = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/peak_cluster_mesoderm.h5ad")

# 2) Define Leiden clusters associated with each subset
dict_adata_sub = {
    "0": adata_sub,      # adata_sub belongs to Leiden cluster "0"
    "1": adata_sub2,     # adata_sub2 belongs to Leiden cluster "1"
    ("8", "11"): adata_sub3  # adata_sub3 belongs to Leiden clusters "8" and "11"
}

# 3) Convert the "leiden" column in adata_peaks to string to allow new values
adata_peaks.obs["leiden_subclust_append"] = adata_peaks.obs["leiden"].astype(str)

# 4) Iterate through each subset and update leiden labels
for parent_leiden, adata_sub_obj in dict_adata_sub.items():
    
    # Ensure parent_leiden is iterable (handles single and multiple Leiden clusters)
    leiden_parents = (parent_leiden,) if isinstance(parent_leiden, str) else parent_leiden  
    
    # Extract sub-cluster labels as strings
    sub_labels = adata_sub_obj.obs["leiden"].astype(str)

    for leiden_parent in leiden_parents:
        # Generate new leiden labels combining parent cluster with sub-clusters
        new_labels = leiden_parent + "-" + sub_labels.values  # Ensure proper concatenation

        # Assign new values to the correct subset of adata_peaks
        adata_peaks.obs.loc[adata_sub_obj.obs.index, "leiden_subclust_append"] = new_labels

# 5) Convert back to categorical
adata_peaks.obs["leiden_subclust_append"] = adata_peaks.obs["leiden_subclust_append"].astype("category")

# Check final Leiden category values
print(adata_peaks.obs["leiden_subclust_append"].value_counts())


# %%
adata_peaks.obs["leiden_subclust_append"][adata_peaks.obs["leiden_subclust_append"].str.startswith("0-")].unique()

# %%
sc.pl.umap(adata_peaks, color="leiden_subclust_append")


# %%
# export the peaks for gimme maelstrom
def export_peaks_for_gimmemotifs(adata_peaks, cluster="leiden", out_dir=None):
    """
    Export peaks in a format suitable for GimmeMotifs.

    Parameters:
    -----------
    adata_peaks : AnnData
        AnnData object containing peak information.
    cluster : str, default="leiden"
        Column in `adata_peaks.obs` specifying cluster labels.
    out_dir : str or None, default=None
        Directory where the output file will be saved. If None, the file will not be saved.

    Returns:
    --------
    export_df : pd.DataFrame
        DataFrame containing the formatted pjjeak locations and cluster labels.
    """
    # Format peak locations (assuming 'chr' prefix is needed)
    export_df = pd.DataFrame({
        'loc': 'chr' + adata_peaks.obs_names.str.replace('-', ':', 1).str.replace('-', '-', 1),
        'cluster': adata_peaks.obs[cluster]  # Use the specified cluster column
    })

    # Remove potential duplicates
    export_df = export_df.drop_duplicates()

    # Save file only if out_dir is specified
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        output_file = os.path.join(out_dir, f'peaks_{len(export_df)}_{cluster}.txt')
        export_df.to_csv(output_file, sep='\t', index=False)
        print(f"Exported {len(export_df)} peaks to {output_file}")
        print("Clusters present:", export_df['cluster'].unique())
    
    return export_df


# Run the export
out_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_diff_motifs_leiden_subclust/"
peaks_export = export_peaks_for_gimmemotifs(adata_peaks, 
                                            cluster="leiden_subclust_append", 
                                            out_dir=out_dir)

# %% [markdown]
# ## Looking at the maelstrom results from the sub-clusters
#
# - sub-clustering done for the three leiden clusters in the main figure
#

# %%
# check out the result of the maelstrom
from gimmemotifs.maelstrom import MaelstromResult

mr = MaelstromResult("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/maelstrom_diff_motifs_leiden_subclust/maelstrom_peaks_50k_hvps_leiden_subclust_out_v3/")
mr.plot_heatmap(threshold=3.5)
# plt.savefig("maelstrom.blood.1k.out/heatmap.png", dpi=300)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Filter based on threshold
threshold = 3.5  # example threshold
df = mr.result.copy()

# 2) Keep only those rows where at least one absolute value >= threshold
mask = df.abs().ge(threshold).any(axis=1)
df_filt = df[mask]

# Example: remove "z-score " prefix from column names
df_filt.columns = [col.replace("z-score ", "") for col in df_filt.columns]

# 3) Store TF factor names separately while keeping motif IDs as index
factors = []
for motif_id in df_filt.index:
    motif_obj = mr.motifs.get(motif_id, None)
    if motif_obj is not None:
        factor_str = motif_obj.format_factors(
            max_length=2,  # how many factors to show
            html=False,
            include_indirect=True,
        )
    else:
        factor_str = "Unknown"  # fallback if motif is missing
    factors.append(factor_str)

# Store factor names in a separate column (not as index)
df_filt["TF_factors"] = factors  # This ensures we do not replace motif IDs

# Transpose the matrix for clustering
df_filt_trans = df_filt.drop(columns=["TF_factors"]).transpose()

# 4) CLUSTER + PLOT using seaborn.clustermap
g = sns.clustermap(
    df_filt_trans,
    metric="euclidean",
    method="ward",
    cmap="RdBu_r",
    center=0,
    linewidths=0.5,
    figsize=(6, 8),
    xticklabels=True,
    yticklabels=True  # Keep y-ticks as motif IDs
)

# Rotate x-axis tick labels if desired
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)

# Save the figure
plt.savefig(figpath + "motifs_heatmap_subclust_thresh_3.5_clustered_transposed_motifnames.png")
plt.savefig(figpath + "motifs_heatmap_subclust_thresh_3.5_clustered_transposed_motifnames.pdf")
plt.show()


# %%
# # copy the peaks-by-motifs adata object
adata_peaks_subclust = adata_peaks.copy()

# %%
# Step 1: Create a mapping from cluster ID to its row in mr.result
cluster_to_scores = {}

# Iterate through columns in mr.result (each representing a leiden cluster)
for col in mr.result.columns:
    if col.startswith("z-score"):
        # Extract cluster ID
        cluster_id = col.split(" ")[1]
        # Store this column as the scores for this cluster
        cluster_to_scores[cluster_id] = mr.result[col]

# Step 2: Add all motif scores directly to peaks_by_motifs.obs
for motif in mr.result.index:
    motif_name = motif.replace('.', '_')  # Clean up motif name for column naming
    
    # For each peak, find its leiden cluster and assign the corresponding motif score
    adata_peaks_subclust.obs[f'motif_{motif_name}'] = adata_peaks_subclust.obs['leiden_subclust_append'].apply(
        lambda x: cluster_to_scores.get(str(x), {}).get(motif, 0)
    )
    # ensure that the scores are "float64", not "category"
    adata_peaks_subclust.obs[f'motif_{motif_name}'] = adata_peaks_subclust.obs[f'motif_{motif_name}'].astype("float64")

# Now you can visualize specific motifs
motif_cols = [f'motif_{motif.replace(".", "_")}' for motif in mr.result.index[:5]]  # First 5 motifs
# sc.pl.umap(adata_peaks_subclust, color=motif_cols, cmap='bwr')

# %%
# plot the UMAP and save the plot for TF motif
motif = "motif_GM_5_0_Paired_box_0002"
extracted_motif = motif.split("motif_GM_5_0_")[1]
sc.pl.embedding(adata_peaks_subclust, basis="X_umap", color=motif, 
                cmap="RdBu_r", vmin=-4, vmax=4, save=f"_peaks_leiden_{extracted_motif}.png")

# %%
# plot the UMAP and save the plot for TF motif
motif = "motif_GM_5_0_GCM_0004"
extracted_motif = motif.split("motif_GM_5_0_")[1]
sc.pl.embedding(adata_peaks_subclust, basis="X_umap", color=motif, 
                cmap="RdBu_r", vmin=-4, vmax=4, save=f"_peaks_leiden_{extracted_motif}.png")

# %%
# plot the UMAP and save the plot for TF motif
motif = "motif_GM_5_0_Paired_box_0010"
extracted_motif = motif.split("motif_GM_5_0_")[1]
sc.pl.embedding(adata_peaks_subclust, basis="X_umap", color=motif, 
                cmap="RdBu_r", vmin=-4, vmax=4, save=f"_peaks_leiden_{extracted_motif}.png")

# %%
# plot the UMAP and save the plot for TF motif
motif = "motif_GM_5_0_Homeodomain_0195"
extracted_motif = motif.split("motif_GM_5_0_")[1]
sc.pl.embedding(adata_peaks_subclust, basis="X_umap", color=motif, 
                cmap="RdBu_r", vmin=-4, vmax=4, save=f"_peaks_leiden_{extracted_motif}.png")

# %%

# %%
for motif in adata_peaks_subclust.obs.columns[adata_peaks_subclust.obs.columns.str.startswith("motif_GM_5_0")]:
    extracted_motif = motif.split("motif_GM_5_0_")[1]
    sc.pl.embedding(adata_peaks_subclust, basis="X_umap", color=motif, 
                    cmap="RdBu_r", vmin=-4, vmax=4, save=f"_peaks_subclust_leiden_{extracted_motif}.png", show=False)

# %%

# %%

# %%

# %%

# %% [markdown]
# ## constructing mini GRNs for some peak clusters 
#
# - For peak clusters with enriched with specific TF motifs, we'll check out the downstream genes
# - hematopoesis, etc.

# %%
adata_peaks[adata_peaks.obs["leiden"]=="9"].obs["linked_gene"].unique()

# %%
adata_peaks.obs["GM.5.0.Ets.0033"].sort_values(ascending=False)[0:20].index

# %%
adata_peaks[adata_peaks.obs_names.isin(adata_peaks.obs["GM.5.0.Ets.0033"].sort_values(ascending=False)[0:30].index)].obs["associated_gene"]

# %%
adata_peaks[adata_peaks.obs_names.isin(adata_peaks.obs["GM.5.0.Ets.0033"].sort_values(ascending=False)[0:30].index)].obs["associated_gene"]

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Run GimmeMotifs for one vs rest (test) - this part was moved to CLI (GimmeMotifs, instead of python API)

# %%
from gimmemotifs.denovo import gimme_motifs
from gimmemotifs.config import MotifConfig

import celloracle as co
from celloracle import motif_analysis as ma

# %%
# import the util module
sys.path.append("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/scripts/gimmemotifs")
from atac_seq_motif_analysis import ATACSeqMotifAnalysis

# %%
config = MotifConfig()
config.set_motif_dir("/hpc/mydata/yang-joon.kim/genomes/danRer11/")
config.set_bg_dir("/hpc/mydata/yang-joon.kim/genomes/danRer11/")
config.write(open(config.user_config, "w"))

print(f"Motif database directory: {config.get_motif_dir()}")
print(f"Background directory: {config.get_bg_dir()}")

# %%
# # load the peaks from a cluster and the rest of the clusters
# cluster_id = "0"
# input_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/one_vs_rest/"
# output_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/motifs_result_one_vs_all/"

# # Ensure output directory exists
# os.makedirs(output_dir, exist_ok=True)

# # Paths to cluster & background FASTA files
# cluster_fasta = os.path.join(input_dir, f"cluster_{cluster_id}.fasta")
# background_fasta = os.path.join(input_dir, f"cluster_{cluster_id}_rest.fasta")

# # Set genome (make sure it's installed with genomepy)
# genome = "danRer11"
# # Output directory
# output_prefix = os.path.join(output_dir, f"cluster_{cluster_id}_motifs")

# # Define parameters
# params = {
#     "genome": genome,
#     # "background": background_fasta,  # Custom background
#     "tools": "Homer,BioProspector,MEME",  # Specify motif tools if needed
#     # "size": "given",  # Default region size
# }


# # Run GimmeMotifs directly in Python
# print("\n Running GimmeMotifs in Python (Motif Discovery)...")

# try:
#     motifs = gimme_motifs(cluster_fasta, output_prefix, params=params)
#     print(f"\n Motif Discovery completed for Cluster {cluster_id}!\n")
# except Exception as e:
#     print(f"\n Error running GimmeMotifs De Novo: {e}")


# %%

# %%

# %%
os.environ["PATH"] += ":/home/user/.conda/envs/celloracle_env/bin"  # Update path if necessary

# %%
output_dir

# %%
# Parameters
params = {
    "genome": "danRer11",  # Specify the genome name
    "tools": "Homer,BioProspector,MEME",
    "size": 200,
}

# Run GimmeMotifs
motifs = gimme_motifs(inputfile=peaks_fasta, 
                      outdir=output_dir, params=params,
                      cluster=True, create_report=True)
print("GimmeMotifs completed successfully.")

# output_prefix = os.path.join(output_dir, f"cluster_{cluster_id}_motifs")

# gimme_command = [
#     "gimme", "motifs", cluster_fasta, output_prefix,
#     "-g", genome,          # Specify genome (e.g., danRer11)
#     "--denovo",            # Run de novo motif discovery
#     "--background", background_fasta  # Specify background sequences
# ]

# print("\nRunning GimmeMotifs command:\n" + " ".join(gimme_command))

# # Run command and capture output
# try:
#     subprocess.run(gimme_command, check=True)
#     print(f"\n✅ GimmeMotifs completed for Cluster {cluster_id}!\n")
# except subprocess.CalledProcessError as e:
#     print(f"\n❌ Error running GimmeMotifs: {e}")

# # -------------------------------------------------------------------
# # 4️⃣ Print Output Files
# # -------------------------------------------------------------------
# print(f"\n📂 Results saved in: {output_prefix}")
# print(f"🔍 Check {output_prefix}/gimme.denovo.html for motif results visualization.")

# %%

# %%

# %%

# %%

# %%
