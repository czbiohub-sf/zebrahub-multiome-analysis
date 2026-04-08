# %% [markdown]
# # Step 03: Visualize Parts List
#
# Publication-quality figures demonstrating the parts list concept:
#
#   Fig 1: Distribution of max specificity z-scores across all 640K peaks
#   Fig 2: Condition coverage tile (32 celltypes × 6 timepoints, color = n_specific_peaks)
#   Fig 3: UMAP with top query peaks highlighted (one per example query)
#   Fig 4: Accessibility heatmap for top-5 peaks (raw log_norm, query condition highlighted)
#   Fig 5: Specificity z-score heatmap for top-5 peaks
#   Fig 6: Enriched motifs for top-100 query peaks (via leiden_coarse → maelstrom z-scores)
#
# Inputs:
#   outputs/specificity_matrix.h5ad
#   outputs/specificity_summary.csv
#   peaks_by_ct_tp_master_anno.h5ad      (for raw log_norm values)
#   leiden_by_motifs_maelstrom.csv       (36 × 115 motif z-scores)
#   info_cisBP_v2_danio_rerio_motif_factors.csv  (motif → TF name)

# %% Imports
import os, re, gc
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

print(f"anndata {ad.__version__}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs"
FIG_DIR = f"{REPO}/figures/peak_parts_list"
OBJ_DIR = f"{BASE}/data/annotated_data/objects_v2"
os.makedirs(FIG_DIR, exist_ok=True)

SPEC_H5AD    = f"{OUTDIR}/specificity_matrix.h5ad"
MASTER_H5AD  = f"{OBJ_DIR}/peaks_by_ct_tp_master_anno.h5ad"
MOTIF_CSV    = f"{OBJ_DIR}/leiden_by_motifs_maelstrom.csv"
MOTIF_INFO   = f"{OBJ_DIR}/info_cisBP_v2_danio_rerio_motif_factors.csv"

# Style
sns.set(style="whitegrid", context="paper")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42

EXAMPLE_QUERIES = [
    ("heart_myocardium",   "20somites"),
    ("neural",             "10somites"),
    ("neural_crest",       "15somites"),
    ("somites",            "5somites"),
    ("endoderm",           "20somites"),
]

# %% Load specificity matrix
import time
print("Loading specificity matrix ...", flush=True)
t0 = time.time()
Z_adata = ad.read_h5ad(SPEC_H5AD)
print(f"  Shape: {Z_adata.shape}  ({time.time()-t0:.1f}s)")

cond_to_idx = {n: i for i, n in enumerate(Z_adata.var_names)}
Z = np.array(Z_adata.X)  # (640830, 190)

# %% ── Figure 1: Distribution of max specificity z-scores ───────────────────
print("\nFig 1: Specificity distribution ...", flush=True)

max_z = Z.max(axis=1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(max_z, bins=100, color="#2c7bb6", alpha=0.8, edgecolor="none")
for thresh, color, label in [(2, "#fdae61", "z = 2  (moderate)"),
                              (4, "#d7191c", "z = 4  (specific)"),
                              (8, "#7b2d8b", "z = 8  (highly specific)")]:
    n = (max_z >= thresh).sum()
    ax.axvline(thresh, color=color, lw=1.5, linestyle="--", label=f"{label}  n={n:,}")
ax.set_xlabel("Max specificity z-score (best condition per peak)")
ax.set_ylabel("Number of peaks")
ax.set_title("Specificity of 640K peaks across 190 (celltype × timepoint) conditions")
ax.legend(fontsize=8)
ax.set_yscale("log")
plt.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"{FIG_DIR}/specificity_histogram.{ext}", bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved: {FIG_DIR}/specificity_histogram.{{pdf,png}}")

# %% ── Figure 2: Condition coverage tile (celltypes × timepoints) ────────────
print("\nFig 2: Condition coverage tile ...", flush=True)

_tp_re = re.compile(r'(\d+somites)$')
TIMEPOINTS = ["0somites", "5somites", "10somites", "15somites", "20somites", "30somites"]
CELLTYPES  = sorted(set(_tp_re.sub('', c).rstrip('_') for c in Z_adata.var_names))

# Build 32×6 matrix of n_specific_peaks (z ≥ 4)
tile_z4  = np.full((len(CELLTYPES), len(TIMEPOINTS)), np.nan)
tile_tot = np.full((len(CELLTYPES), len(TIMEPOINTS)), np.nan)

for i, ct in enumerate(CELLTYPES):
    for j, tp in enumerate(TIMEPOINTS):
        cond = f"{ct}_{tp}"
        if cond in cond_to_idx:
            col = Z[:, cond_to_idx[cond]]
            tile_z4[i, j]  = (col >= 4).sum()
            tile_tot[i, j] = (col >= 0).sum()  # always 640830

fig, ax = plt.subplots(figsize=(9, 14))
mask = np.isnan(tile_z4)
sns.heatmap(tile_z4, annot=False, fmt=".0f",
            xticklabels=[t.replace("somites", " s") for t in TIMEPOINTS],
            yticklabels=CELLTYPES,
            cmap="YlOrRd", mask=mask, linewidths=0.3,
            cbar_kws={"label": "Peaks with z ≥ 4"},
            ax=ax)
# Mark missing conditions
for i in range(len(CELLTYPES)):
    for j in range(len(TIMEPOINTS)):
        if mask[i, j]:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color="#dddddd", lw=0))
ax.set_title("Number of highly specific peaks (z ≥ 4) per condition", fontsize=11)
ax.set_xlabel("Timepoint")
ax.set_ylabel("")
plt.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"{FIG_DIR}/condition_coverage_tile.{ext}", bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved: {FIG_DIR}/condition_coverage_tile.{{pdf,png}}")

# %% ── Figure 3: UMAP highlights for each example query ─────────────────────
print("\nFig 3: UMAP highlights ...", flush=True)

umap_coords = Z_adata.obsm.get("X_umap_2D", Z_adata.obsm.get("X_umap"))
if umap_coords is None:
    print("  WARNING: no UMAP coordinates found, skipping Fig 3")
else:
    n_queries = len(EXAMPLE_QUERIES)
    fig, axes = plt.subplots(1, n_queries, figsize=(5 * n_queries, 5))
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(Z_adata.n_obs, size=min(Z_adata.n_obs, 80_000), replace=False)

    for ax, (ct, tp) in zip(axes, EXAMPLE_QUERIES):
        cond = f"{ct}_{tp}"
        if cond not in cond_to_idx:
            ax.set_title(f"{cond}\n(not found)")
            continue

        zscores = Z[:, cond_to_idx[cond]]
        top_mask = zscores >= 4.0
        n_top = top_mask.sum()

        ax.scatter(umap_coords[bg_idx, 0], umap_coords[bg_idx, 1],
                   s=0.3, c="#d5d5d5", alpha=0.3, rasterized=True, linewidths=0)
        if n_top > 0:
            sc = ax.scatter(umap_coords[top_mask, 0], umap_coords[top_mask, 1],
                            s=3, c=zscores[top_mask], cmap="YlOrRd",
                            vmin=4, vmax=min(zscores[top_mask].max(), 20),
                            alpha=0.8, zorder=3, linewidths=0)
            plt.colorbar(sc, ax=ax, label="z-score", shrink=0.7)

        ct_label = ct.replace("_", " ")
        ax.set_title(f"{ct_label}\n{tp}  (n={n_top:,})", fontsize=9)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_aspect("equal", "datalim")

    fig.suptitle("Highly specific peaks (z ≥ 4) on peak UMAP", fontsize=12, y=1.01)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"{FIG_DIR}/umap_query_highlights.{ext}", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {FIG_DIR}/umap_query_highlights.{{pdf,png}}")

# %% Load raw log_norm for heatmap figures
print("\nLoading raw log_norm from master h5ad ...", flush=True)
t0 = time.time()
adata_raw = ad.read_h5ad(MASTER_H5AD, backed="r")
print(f"  Shape: {adata_raw.shape}  ({time.time()-t0:.1f}s)")

# Condition labels for x-axis (sorted by celltype then timepoint)
cond_labels = adata_raw.var_names.tolist()

# %% ── Figures 4 & 5: Accessibility + specificity heatmaps ──────────────────
print("\nFigs 4 & 5: Accessibility + specificity heatmaps ...", flush=True)

TOP_N_HEAT = 5  # top peaks per query to show in heatmap

for ct, tp in EXAMPLE_QUERIES:
    cond = f"{ct}_{tp}"
    if cond not in cond_to_idx:
        print(f"  SKIP {cond}")
        continue

    col_idx = cond_to_idx[cond]
    zscores = Z[:, col_idx]
    top_idx = np.argsort(zscores)[-TOP_N_HEAT:][::-1]
    peak_ids = Z_adata.obs_names[top_idx].tolist()

    # Extract raw log_norm rows for top peaks
    raw_layer = adata_raw.layers["log_norm"]
    if hasattr(raw_layer, "toarray"):
        raw_rows = raw_layer[top_idx, :].toarray()
    else:
        raw_rows = np.array(raw_layer[top_idx, :])

    z_rows = Z[top_idx, :]

    # Truncated peak labels
    peak_labels = [f"{Z_adata.obs.loc[p, 'chrom']}:{Z_adata.obs.loc[p, 'start']}-"
                   f"{Z_adata.obs.loc[p, 'end']}\n{Z_adata.obs.loc[p, 'associated_gene']}"
                   for p in peak_ids]

    # Highlight column
    hl_col = col_idx

    for data, name, cmap, title_suffix in [
        (raw_rows, "accessibility", "Blues",   "log-norm accessibility"),
        (z_rows,   "specificity",   "RdYlBu_r","specificity z-score"),
    ]:
        fig, ax = plt.subplots(figsize=(max(12, len(cond_labels) * 0.07), 4))
        im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
        plt.colorbar(im, ax=ax, label=title_suffix, fraction=0.02, pad=0.01)

        # Highlight the queried condition
        ax.axvline(hl_col - 0.5, color="#e74c3c", lw=1.5)
        ax.axvline(hl_col + 0.5, color="#e74c3c", lw=1.5)

        ax.set_yticks(range(TOP_N_HEAT))
        ax.set_yticklabels(peak_labels, fontsize=7)
        ax.set_xticks([])
        ax.set_xlabel(f"Conditions (n={len(cond_labels)}) — red line = {cond}")
        ax.set_title(f"Top {TOP_N_HEAT} peaks for {ct} × {tp}  [{title_suffix}]", fontsize=10)
        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(f"{FIG_DIR}/{name}_heatmap_{cond}.{ext}", bbox_inches="tight", dpi=150)
        plt.close()
    print(f"  Saved heatmaps for {cond}")

adata_raw.file.close()
del adata_raw
gc.collect()

# %% ── Figure 6: Enriched motifs for top-100 query peaks ────────────────────
print("\nFig 6: Motif enrichment ...", flush=True)

if not os.path.exists(MOTIF_CSV):
    print(f"  WARNING: {MOTIF_CSV} not found, skipping Fig 6")
else:
    motif_df   = pd.read_csv(MOTIF_CSV, index_col=0)   # (36 clusters × 115 motifs)
    motif_info = pd.read_csv(MOTIF_INFO, index_col=0)  # motif → TF names

    # Build motif → TF label map
    def motif_label(motif_id):
        if motif_id in motif_info.index:
            tfs = str(motif_info.loc[motif_id, "indirect"])
            if tfs not in ("nan", ""):
                return tfs.split(",")[0].strip()
        return motif_id

    for ct, tp in EXAMPLE_QUERIES[:3]:  # first 3 queries
        cond = f"{ct}_{tp}"
        if cond not in cond_to_idx:
            continue

        col_idx = cond_to_idx[cond]
        zscores = Z[:, col_idx]
        top100_mask = zscores >= np.percentile(zscores[zscores > 0], 99)
        top100_clusters = Z_adata.obs.loc[top100_mask, "leiden_coarse"].astype(str).value_counts()

        # Weighted average motif enrichment across top clusters
        weighted_motifs = pd.Series(0.0, index=motif_df.columns)
        total_weight = 0
        for clust_id, count in top100_clusters.items():
            if clust_id in motif_df.index.astype(str):
                row = motif_df.loc[int(clust_id)] if int(clust_id) in motif_df.index else None
                if row is not None:
                    weighted_motifs += row * count
                    total_weight += count
        if total_weight > 0:
            weighted_motifs /= total_weight

        # Top 20 motifs
        top_motifs = weighted_motifs.nlargest(20)
        labels = [motif_label(m) for m in top_motifs.index]

        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ["#e74c3c" if s > 1.5 else "#3498db" if s > 0.5 else "#95a5a6"
                  for s in top_motifs.values]
        ax.barh(range(len(top_motifs)), top_motifs.values[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top_motifs)))
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Weighted motif enrichment z-score")
        ax.set_title(f"Top motifs: {ct.replace('_',' ')} × {tp}", fontsize=10)
        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(f"{FIG_DIR}/motif_enrichment_{cond}.{ext}", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  Saved: {FIG_DIR}/motif_enrichment_{cond}.{{pdf,png}}")

print("\nAll figures saved to:", FIG_DIR)
print("Done.")
