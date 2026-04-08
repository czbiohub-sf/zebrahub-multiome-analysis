# %% Script 09e: RNA-ATAC concordance — Tier 1 validation
#
# For the top 50 V3-specific peaks per celltype (filtered to those with
# linked/nearest gene annotations), compare:
#   x-axis: ATAC V3 celltype-level z-score (peak specificity)
#   y-axis: RNA expression z-score for the same gene in the same celltype
#
# RNA z-score: leave-one-out z-score across celltypes (same formula as ATAC V3),
# computed from the pseudobulk RNA HVG matrix (5077 genes × 32 celltypes × 6 tp).
#
# Outputs: figures/peak_parts_list/V3/rna_atac_concordance/
#   per_celltype_scatter.{pdf,png}   — grid of per-celltype scatter plots
#   combined_scatter.{pdf,png}       — all celltypes overlaid, color-coded
#   concordance_table.csv            — full peak-gene pair table with both z-scores
#
# Env: /home/yang-joon.kim/.conda/envs/gReLu

import os, re, time
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication-quality defaults
matplotlib.rcParams['pdf.fonttype'] = 42   # editable text in Illustrator
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['savefig.dpi']  = 300
import matplotlib.patches as mpatches
from scipy import stats


# ── Publication figure settings (exact pattern from 01_EDA_annotate_peak_umap.py) ──
import matplotlib as _mpl
_mpl.rcParams.update(_mpl.rcParamsDefault)   # 1. reset all rcParams to defaults
_mpl.rcParams['font.family'] = 'Arial'      # 2. explicit Arial font
_mpl.rcParams["pdf.fonttype"] = 42          # 3. editable text in Illustrator
_mpl.rcParams["ps.fonttype"]  = 42
import seaborn as _sns
_sns.set(style="whitegrid", context="paper") # 4. seaborn (after fonttype)
_mpl.rcParams["savefig.dpi"]  = 300         # 5. DPI re-set after sns.set()
# ────────────────────────────────────────────────────────────────────────────────

print("=== Script 09e: RNA-ATAC Concordance ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUT_DIR = f"{REPO}/figures/peak_parts_list/V3/rna_atac_concordance"
V3_DIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
os.makedirs(OUT_DIR, exist_ok=True)

RNA_Z_MAT   = f"{V3_DIR}/rna_specificity_matrix_celltype_level.h5ad"
V3_PEAKS    = f"{V3_DIR}/V3_celltype_level_top_peaks.csv"
V3_ZMAT     = f"{V3_DIR}/V3_specificity_matrix_celltype_level.h5ad"

# %% Color palette
import sys as _sys
_sys.path.insert(0, f"{REPO}/scripts/utils")
from module_dict_colors import cell_type_color_dict

# %% Load ATAC V3 z-score matrix
print("\nLoading ATAC V3 z-score matrix ...", flush=True)
z_adata = ad.read_h5ad(V3_ZMAT)
Z_ct    = np.array(z_adata.X)          # (640830, ~31)
ct_names = list(z_adata.var_names)

# %% Load top peaks table
print("Loading top peaks table ...", flush=True)
top_peaks = pd.read_csv(V3_PEAKS, index_col=0)
print(f"  {len(top_peaks)} peaks across {top_peaks['celltype'].nunique()} celltypes")

# Gene assignment: linked_gene > nearest_gene
top_peaks["gene"] = top_peaks["linked_gene"].astype(str)
mask_no_linked = top_peaks["gene"].isin(["nan", "", "None"])
top_peaks.loc[mask_no_linked, "gene"] = \
    top_peaks.loc[mask_no_linked, "nearest_gene"].astype(str)
top_peaks["has_gene"] = ~top_peaks["gene"].isin(["nan", "", "None"])

annotated = top_peaks[top_peaks["has_gene"]].copy()
print(f"  Peaks with gene annotation: {len(annotated)} / {len(top_peaks)}")

# %% Load precomputed RNA celltype-level z-score matrix
# Built by 09e_rna_pseudobulk.py using raw counts + same median-scale + log1p
# pipeline as the ATAC pseudobulk (peaks_pb_preprocessing.py).
print("\nLoading RNA celltype-level z-score matrix ...", flush=True)
t0 = time.time()
rna_z_adata = ad.read_h5ad(RNA_Z_MAT)
print(f"  RNA z-score shape: {rna_z_adata.shape}  ({time.time()-t0:.1f}s)")
# obs = genes, var = celltypes

Z_rna = np.array(rna_z_adata.X)        # (n_genes, n_celltypes)
gene_names   = list(rna_z_adata.obs_names)
rna_celltypes = list(rna_z_adata.var_names)

gene_to_row   = {g: i for i, g in enumerate(gene_names)}
rna_ct_to_col = {ct: i for i, ct in enumerate(rna_celltypes)}

print(f"  Genes: {len(gene_names)}  |  Celltypes: {len(rna_celltypes)}")
print(f"  RNA z-score range: [{Z_rna.min():.1f}, {Z_rna.max():.1f}]")
print(f"  Celltypes: {rna_celltypes[:5]} ...")

# %% Build concordance table
print("\nBuilding concordance table ...", flush=True)
rows = []
for _, peak_row in annotated.iterrows():
    ct  = peak_row["celltype"]
    gene = peak_row["gene"]
    atac_z = peak_row["V3_zscore"]

    # Get RNA z-score for this gene in this celltype
    if gene not in gene_to_row:
        continue
    if ct not in rna_ct_to_col:
        continue

    rna_z = Z_rna[gene_to_row[gene], rna_ct_to_col[ct]]
    peak_id = peak_row["peak_id"] if "peak_id" in peak_row else peak_row.name

    rows.append({
        "celltype":  ct,
        "peak_id":   peak_id,
        "gene":      gene,
        "atac_z":    atac_z,
        "rna_z":     rna_z,
        "chrom":     str(peak_row.get("chrom", "")),
        "start":     int(peak_row.get("start", 0)),
        "end":       int(peak_row.get("end", 0)),
        "peak_type": str(peak_row.get("peak_type", "")),
    })

concordance_df = pd.DataFrame(rows)
concordance_df.to_csv(f"{OUT_DIR}/concordance_table.csv", index=False)
print(f"  {len(concordance_df)} peak-gene pairs with both ATAC and RNA z-scores")
print(concordance_df.groupby("celltype").size().rename("n_pairs"))

# %% Per-celltype scatter plots (grid)
print("\nGenerating per-celltype scatter plots ...", flush=True)

focal_cts = sorted(concordance_df["celltype"].unique())
n_cts = len(focal_cts)
ncols = 4
nrows = int(np.ceil(n_cts / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
axes = axes.flatten()

for ax_i, ct in enumerate(focal_cts):
    ax = axes[ax_i]
    ct_df = concordance_df[concordance_df["celltype"] == ct]
    color = cell_type_color_dict.get(ct, "#888888")

    ax.scatter(ct_df["atac_z"], ct_df["rna_z"],
               s=60, color=color, edgecolors="black",
               linewidths=0.5, alpha=0.85, zorder=3)

    # Regression line
    if len(ct_df) >= 3:
        r, p = stats.pearsonr(ct_df["atac_z"], ct_df["rna_z"])
        x_line = np.linspace(ct_df["atac_z"].min(), ct_df["atac_z"].max(), 100)
        slope, intercept, *_ = stats.linregress(ct_df["atac_z"], ct_df["rna_z"])
        ax.plot(x_line, slope * x_line + intercept,
                color="gray", lw=1.2, ls="--", zorder=2)
        stats_str = f"r = {r:.2f}  p = {p:.2e}\nn = {len(ct_df)}"
    else:
        stats_str = f"n = {len(ct_df)}"

    # Label top genes (highest ATAC z-score)
    top_label = ct_df.nlargest(5, "atac_z")
    for _, row in top_label.iterrows():
        ax.annotate(row["gene"],
                    xy=(row["atac_z"], row["rna_z"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="gray", alpha=0.75))

    ax.axhline(0, color="#aaaaaa", lw=0.8, ls=":")
    ax.axvline(0, color="#aaaaaa", lw=0.8, ls=":")
    ax.set_xlabel("ATAC V3 z-score", fontsize=9)
    ax.set_ylabel("RNA expression z-score", fontsize=9)
    ax.set_title(f"{ct.replace('_', ' ')}\n{stats_str}", fontsize=10)
    ax.grid(True, alpha=0.2)

# Hide unused axes
for j in range(n_cts, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("RNA–ATAC concordance: top-50 V3-specific peaks per celltype\n"
             "(x = ATAC specificity z-score, y = RNA expression z-score, same celltype)",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/per_celltype_scatter.pdf", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/per_celltype_scatter.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: per_celltype_scatter")

# %% Combined scatter (all celltypes, color-coded)
print("Generating combined scatter ...", flush=True)

fig, ax = plt.subplots(figsize=(8, 7))

legend_handles = []
for ct in focal_cts:
    ct_df = concordance_df[concordance_df["celltype"] == ct]
    color = cell_type_color_dict.get(ct, "#888888")
    ax.scatter(ct_df["atac_z"], ct_df["rna_z"],
               s=55, color=color, edgecolors="black", linewidths=0.4,
               alpha=0.82, zorder=3, label=ct.replace("_", " "))
    legend_handles.append(
        mpatches.Patch(facecolor=color, edgecolor="black",
                       label=ct.replace("_", " ")))

# Overall regression
r_all, p_all = stats.pearsonr(concordance_df["atac_z"], concordance_df["rna_z"])
slope_all, intercept_all, *_ = stats.linregress(
    concordance_df["atac_z"], concordance_df["rna_z"])
x_line = np.linspace(concordance_df["atac_z"].min(),
                     concordance_df["atac_z"].max(), 200)
ax.plot(x_line, slope_all * x_line + intercept_all,
        color="#333333", lw=1.5, ls="--", zorder=4,
        label=f"Overall fit (r={r_all:.2f}, p={p_all:.1e})")

ax.axhline(0, color="#aaaaaa", lw=0.8, ls=":")
ax.axvline(0, color="#aaaaaa", lw=0.8, ls=":")
ax.set_xlabel("ATAC V3 z-score (peak specificity)", fontsize=11)
ax.set_ylabel("RNA expression z-score (gene specificity)", fontsize=11)
ax.set_title(f"RNA–ATAC concordance across all celltypes\n"
             f"Overall: r = {r_all:.2f},  p = {p_all:.1e},  "
             f"n = {len(concordance_df)} peak-gene pairs",
             fontsize=11)
ax.legend(handles=legend_handles, fontsize=7.5, ncol=2,
          loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/combined_scatter.pdf")
fig.savefig(f"{OUT_DIR}/combined_scatter.png", dpi=300)
plt.close(fig)
print("  Saved: combined_scatter")

# %% Summary statistics
print("\n" + "=" * 70)
print("CONCORDANCE SUMMARY")
print("=" * 70)
print(f"{'Celltype':<25} {'n':>4}  {'Pearson r':>9}  {'p-value':>10}")
print("-" * 70)
for ct in focal_cts:
    ct_df = concordance_df[concordance_df["celltype"] == ct]
    if len(ct_df) >= 3:
        r, p = stats.pearsonr(ct_df["atac_z"], ct_df["rna_z"])
        print(f"{ct:<25} {len(ct_df):>4}  {r:>9.3f}  {p:>10.2e}")
    else:
        print(f"{ct:<25} {len(ct_df):>4}  {'n/a':>9}  {'n/a':>10}")

print(f"\nOverall (all celltypes): r = {r_all:.3f},  p = {p_all:.2e},  "
      f"n = {len(concordance_df)}")
print(f"\nNote: RNA coverage = {len(concordance_df)} / {len(annotated)} annotated peaks "
      f"({100*len(concordance_df)/len(annotated):.0f}%). "
      f"Missing genes are not expressed in the RNA pseudobulk (not in top-32K detected genes).")
print(f"\nDone. Figures: {OUT_DIR}/")
print(f"End: {time.strftime('%c')}")
