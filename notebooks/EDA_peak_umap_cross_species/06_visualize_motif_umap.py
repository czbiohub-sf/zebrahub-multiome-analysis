# %% [markdown]
# # Step 6: Visualize Cross-Species Motif UMAP
#
# **Input:** `cross_species_motif_embedded.h5ad`
#
# **Visualizations:**
#   3a. Species overlay (combined + per-species highlight panels)
#   3b. Specific TF motif overlays (16 key developmental TFs)
#   3c. Dominant TF family per peak
#   3d. Leiden × Motif enrichment heatmap
#   3e. Diagnostics (Leiden on UMAP, species composition per cluster, cell type/timepoint)
#
# **Env:** sc_rapids or single-cell-base (CPU is fine for plotting)

# %% Imports
import os
import re
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import scanpy as sc

print("Libraries loaded.")

# %% Paths
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
INPUT_H5AD = f"{SCRATCH}/cross_species_motif_embedded.h5ad"
INPUT_CSV_GZ = f"{SCRATCH}/cross_species_umap_coords.csv.gz"

FIG_DIR = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/cross_species_motif_umap"
os.makedirs(FIG_DIR, exist_ok=True)

# Point size and alpha for 1.87M points
PT_SIZE = 0.3
PT_ALPHA = 0.6
RASTERIZED = True

# %% Load data
print("Loading embedded h5ad ...")
adata = sc.read_h5ad(INPUT_H5AD)
print(f"  Shape: {adata.shape}")
print(f"  obsm keys: {list(adata.obsm.keys())}")
print(f"  obs cols: {list(adata.obs.columns)}")
print(f"  var cols: {list(adata.var.columns)}")

# Use binary layer for motif presence/absence visualization
if "binary" in adata.layers:
    X_binary = adata.layers["binary"]
else:
    X_binary = adata.X  # fallback

# UMAP coordinates
umap_coords = adata.obsm["X_umap"]
print(f"  UMAP range: x={umap_coords[:,0].min():.2f}–{umap_coords[:,0].max():.2f}  "
      f"y={umap_coords[:,1].min():.2f}–{umap_coords[:,1].max():.2f}")

# %% Helper: scatter with rasterization
def umap_scatter(ax, x, y, c, cmap=None, vmin=None, vmax=None, s=PT_SIZE,
                 alpha=PT_ALPHA, rasterized=RASTERIZED, **kwargs):
    sc_ = ax.scatter(x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                     s=s, alpha=alpha, rasterized=rasterized,
                     linewidths=0, **kwargs)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    return sc_

# %% ── 3a. Species Overlay ───────────────────────────────────────────────────
print("\n3a. Species overlay ...")

SPECIES_COLORS = {
    "zebrafish": "#1f77b4",   # blue
    "mouse":     "#ff7f0e",   # orange
    "human":     "#2ca02c",   # green
}
species_arr = adata.obs["species"].values
x = umap_coords[:, 0]
y = umap_coords[:, 1]

# Combined plot
fig, ax = plt.subplots(figsize=(8, 8))
for sp_name, col in SPECIES_COLORS.items():
    mask = species_arr == sp_name
    ax.scatter(x[mask], y[mask], c=col, s=PT_SIZE, alpha=PT_ALPHA,
               rasterized=RASTERIZED, linewidths=0, label=sp_name)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title("Cross-species motif UMAP — colored by species")
ax.legend(markerscale=10, framealpha=0.8)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/umap_species.pdf", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: umap_species.pdf")

# Per-species highlight panels (one colored, others gray)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (sp_name, col) in zip(axes, SPECIES_COLORS.items()):
    mask = species_arr == sp_name
    # Gray background (other species)
    ax.scatter(x[~mask], y[~mask], c="#cccccc", s=PT_SIZE * 0.5, alpha=0.3,
               rasterized=RASTERIZED, linewidths=0)
    # Highlighted species
    ax.scatter(x[mask], y[mask], c=col, s=PT_SIZE, alpha=PT_ALPHA,
               rasterized=RASTERIZED, linewidths=0, label=sp_name)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"{sp_name.capitalize()} peaks\n(n={mask.sum():,})")
plt.suptitle("Per-species UMAP highlight", fontsize=13)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/umap_species_per_panel.pdf", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: umap_species_per_panel.pdf")

# %% ── 3b. Specific TF Motif Overlays ────────────────────────────────────────
print("\n3b. TF motif overlays ...")

KEY_TFS = [
    # Hematopoietic
    "GATA1", "GATA2", "TAL1", "RUNX1", "SPI1",
    # Neural
    "SOX2", "SOX10", "PAX6", "NEUROD1", "ASCL1",
    # Cardiac / mesoderm
    "HAND2", "NKX2-5", "TBX5", "GATA4", "MEF2A",
    # General
    "CTCF", "FOXA2", "POU5F1", "MYOD1",
]

# Build a TF name → column index mapping from adata.var
# var index contains JASPAR IDs like "MA0037.3::GATA1"; also check var["tf_name"] if present
def find_motif_indices(adata, tf_names):
    """Return dict {tf_name: list_of_col_indices} matching any column."""
    tf_to_idx = {}
    var_names = adata.var_names  # JASPAR IDs e.g. MA0035.4::GATA1 or MA0035.4
    # Also check tf_name column if it exists
    has_tf_name_col = "tf_name" in adata.var.columns
    tf_name_col = adata.var["tf_name"].values if has_tf_name_col else np.array([""] * adata.n_vars)

    for tf in tf_names:
        pattern = re.compile(rf"(?i)(^|[:\._]){re.escape(tf)}($|[:\._]|\d)")
        idxs = []
        for i, (vn, tn) in enumerate(zip(var_names, tf_name_col)):
            if pattern.search(vn) or pattern.search(str(tn)):
                idxs.append(i)
        # Fallback: case-insensitive substring in var_names
        if not idxs:
            for i, vn in enumerate(var_names):
                if tf.upper() in vn.upper():
                    idxs.append(i)
        tf_to_idx[tf] = idxs

    return tf_to_idx

tf_to_idx = find_motif_indices(adata, KEY_TFS)
for tf, idxs in tf_to_idx.items():
    print(f"  {tf}: {len(idxs)} motif(s) → {[adata.var_names[i] for i in idxs[:3]]}")

# Plot 4×5 grid (one panel per TF)
n_tfs = len(KEY_TFS)
ncols = 4
nrows = int(np.ceil(n_tfs / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
axes = axes.flatten()

for i, tf in enumerate(KEY_TFS):
    ax = axes[i]
    idxs = tf_to_idx[tf]
    if idxs:
        # Union: peak has motif if any of the matched columns is 1
        if sp.issparse(X_binary):
            presence = np.asarray(X_binary[:, idxs].sum(axis=1)).ravel() > 0
        else:
            presence = X_binary[:, idxs].sum(axis=1) > 0
        # Gray for absent, red for present
        ax.scatter(x[~presence], y[~presence], c="#dddddd", s=PT_SIZE * 0.5,
                   alpha=0.3, rasterized=RASTERIZED, linewidths=0)
        ax.scatter(x[presence], y[presence], c="#d62728", s=PT_SIZE,
                   alpha=PT_ALPHA, rasterized=RASTERIZED, linewidths=0)
        frac = presence.mean() * 100
        ax.set_title(f"{tf}\n({frac:.1f}% peaks, n_motifs={len(idxs)})", fontsize=9)
    else:
        ax.set_title(f"{tf}\n(not found)", fontsize=9, color="gray")
        ax.scatter(x, y, c="#dddddd", s=PT_SIZE * 0.5, alpha=0.3,
                   rasterized=RASTERIZED, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])

# Hide unused axes
for j in range(n_tfs, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Key TF motif presence on cross-species UMAP", fontsize=13)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/umap_motif_key_tfs.pdf", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: umap_motif_key_tfs.pdf")

# Also save individual PDFs for each TF
for tf in KEY_TFS:
    idxs = tf_to_idx[tf]
    if not idxs:
        continue
    fig, ax = plt.subplots(figsize=(6, 6))
    if sp.issparse(X_binary):
        presence = np.asarray(X_binary[:, idxs].sum(axis=1)).ravel() > 0
    else:
        presence = X_binary[:, idxs].sum(axis=1) > 0
    ax.scatter(x[~presence], y[~presence], c="#dddddd", s=PT_SIZE * 0.5,
               alpha=0.3, rasterized=RASTERIZED, linewidths=0)
    ax.scatter(x[presence], y[presence], c="#d62728", s=PT_SIZE,
               alpha=PT_ALPHA, rasterized=RASTERIZED, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"{tf} motif presence ({presence.mean()*100:.1f}% peaks)")
    plt.tight_layout()
    tf_safe = tf.replace("/", "_").replace(":", "_")
    fig.savefig(f"{FIG_DIR}/umap_motif_{tf_safe}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
print(f"  Saved individual TF PDFs for {len(KEY_TFS)} TFs")

# %% ── 3c. TF Family Grouping ─────────────────────────────────────────────────
print("\n3c. TF family grouping ...")

# Regex-based family classification; ordered by priority (first match wins)
TF_FAMILY_RULES = [
    # Specific families first (before broad patterns)
    ("CTCF",            r"(?i)\bCTCF\b"),
    ("TEAD",            r"(?i)\bTEAD\b"),
    ("POU",             r"(?i)\bPOU[0-9A-Z]|\bOCT\b|\bPOU5F"),
    ("T-box",           r"(?i)\bT-box\b|\bTBR\b|\bTBX\b|\bEOMES\b|\bBRACHYURY\b|\bMESP\b"),
    ("GATA",            r"(?i)\bGATA[0-9]\b"),
    ("HMG/Sox",         r"(?i)\bSOX[0-9A-Z]|\bHMG\b|\bTCF[0-9A-Z]|\bLEF1\b"),
    ("ETS",             r"(?i)\bETS\b|\bELK\b|\bETV\b|\bERG\b|\bFLI\b|\bSPI\b|\bGABPA\b|\bELF\b|\bEHF\b|\bETS[12]\b"),
    ("Forkhead",        r"(?i)\bFOX[A-Z0-9]"),
    ("bHLH",            r"(?i)\bNGN\b|\bNEUROD\b|\bASCL\b|\bATOH\b|\bMYOD\b|\bMYF\b|\bHAND\b|\bTAL\b|\bSCL\b|\bHLF\b|\bE2[AF]\b|\bTCF[34]\b|\bHES[12]\b|\bHEY[12]\b|\bTWIST\b|\bDEC[12]\b|\bCLOCK\b|\bARNT\b|\bAHR\b|\bMAX\b|\bMYC\b|\bNHL\b"),
    ("Homeodomain/HOX", r"(?i)\bHOX[A-Z0-9]|\bCDX\b|\bMSX\b|\bDLX\b|\bNKX\b|\bPBX\b|\bPRDM\b|\bENX\b|\bPAX[0-9]"),
    ("KLF/SP",          r"(?i)\bKLF[0-9]\b|\bSP[1-9]\b|\bKRUPPEL\b"),
    ("C2H2 ZF",         r"(?i)\bZNF\b|\bZEB\b|\bWT1\b|\bSNAI\b|\bSNAP\b|\bEGR[0-9]\b|\bSP[1-9]\b|\bKLF\b"),
    ("Nuclear receptor",r"(?i)\bNR[0-9A-Z]|\bRAR[AB]|\bRXR[AB]|\bTHR\b|\bPPAR\b|\bGR\b|\bAR\b|\bER[AB]\b|\bERR\b|\bNOR1\b|\bROR\b|\bREV\b"),
    ("bZIP/AP-1",       r"(?i)\bAP-?1\b|\bJUN[AB]?\b|\bFOS[AB]?\b|\bCREB\b|\bATF[0-9]\b|\bNFE2\b|\bNRF[12]\b|\bMAF[AB FGKS]?"),
    ("MADS/MEF2",       r"(?i)\bMEF2\b|\bSRF\b|\bMCM\b"),
    ("RUNX",            r"(?i)\bRUNX[123]\b|\bCBF\b"),
    ("p53/p63/p73",     r"(?i)\bTP[56][0-9]\b|\bp53\b|\bp63\b|\bp73\b"),
    ("IRF",             r"(?i)\bIRF[0-9]\b|\bISRE\b"),
    ("RFX",             r"(?i)\bRFX[0-9A-Z]\b"),
]

def classify_tf_family(tf_name):
    """Return the family label for a TF name string."""
    # For heterodimers like 'GATA1::TAL1', use the first partner
    base = tf_name.split("::")[0]
    for family, pattern in TF_FAMILY_RULES:
        if re.search(pattern, base):
            return family
    return "Other"

# Apply to all motifs
adata.var["tf_family"] = [
    classify_tf_family(str(n)) for n in adata.var_names
]
family_counts = adata.var["tf_family"].value_counts()
print("  TF family distribution:")
print(family_counts.to_string())

# Compute dominant family per peak (family with most motif hits among present motifs)
# Efficient: group columns by family, sum binary, then argmax

family_labels = adata.var["tf_family"].values
unique_families = [f for f, _ in TF_FAMILY_RULES] + ["Other"]
unique_families = [f for f in unique_families if f in family_labels]

# Build (n_peaks, n_families) count matrix
family_sums = np.zeros((adata.n_obs, len(unique_families)), dtype=np.float32)
for j, fam in enumerate(unique_families):
    col_idxs = np.where(family_labels == fam)[0]
    if len(col_idxs) == 0:
        continue
    if sp.issparse(X_binary):
        family_sums[:, j] = np.asarray(X_binary[:, col_idxs].sum(axis=1)).ravel()
    else:
        family_sums[:, j] = X_binary[:, col_idxs].sum(axis=1)

# Dominant family: argmax (ties go to first family)
dominant_idx = family_sums.argmax(axis=1)
has_any_motif = family_sums.sum(axis=1) > 0
dominant_family = np.array([
    unique_families[i] if has_any_motif[k] else "No motif"
    for k, i in enumerate(dominant_idx)
])

adata.obs["dominant_tf_family"] = dominant_family
print(f"\n  Dominant TF family distribution:")
print(pd.Series(dominant_family).value_counts().to_string())

# Color palette for families
n_fam = len(unique_families) + 1  # +1 for "No motif"
cmap_fam = plt.cm.get_cmap("tab20", n_fam)
all_fam_labels = unique_families + ["No motif"]
fam_color_map = {fam: cmap_fam(i) for i, fam in enumerate(all_fam_labels)}

fig, ax = plt.subplots(figsize=(10, 9))
for fam in all_fam_labels:
    mask = dominant_family == fam
    if mask.sum() == 0:
        continue
    ax.scatter(x[mask], y[mask], c=[fam_color_map[fam]], s=PT_SIZE,
               alpha=PT_ALPHA, rasterized=RASTERIZED, linewidths=0, label=fam)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title("Cross-species motif UMAP — dominant TF family")
lgnd = ax.legend(markerscale=8, framealpha=0.8, fontsize=7,
                 loc="upper right", ncol=2)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/umap_dominant_tf_family.pdf", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: umap_dominant_tf_family.pdf")

# %% ── 3d. Leiden × Motif Enrichment Heatmap ─────────────────────────────────
print("\n3d. Leiden × motif enrichment heatmap ...")

leiden_key = "leiden_res1.0"
if leiden_key not in adata.obs.columns:
    # Fallback to first available resolution
    leiden_key = next(c for c in adata.obs.columns if c.startswith("leiden"))
    print(f"  Warning: leiden_res1.0 not found, using {leiden_key}")

leiden_labels = adata.obs[leiden_key].values
clusters = sorted(np.unique(leiden_labels), key=lambda c: int(c) if str(c).isdigit() else 0)
print(f"  Clusters: {len(clusters)}")

# Global motif frequency
if sp.issparse(X_binary):
    global_freq = np.asarray(X_binary.mean(axis=0)).ravel()
else:
    global_freq = X_binary.mean(axis=0)
global_freq = np.clip(global_freq, 1e-6, None)

# Per-cluster motif frequency
cluster_freq = np.zeros((len(clusters), adata.n_vars), dtype=np.float32)
for i, cl in enumerate(clusters):
    mask = leiden_labels == cl
    if sp.issparse(X_binary):
        cluster_freq[i] = np.asarray(X_binary[mask].mean(axis=0)).ravel()
    else:
        cluster_freq[i] = X_binary[mask].mean(axis=0)
cluster_freq = np.clip(cluster_freq, 1e-6, None)

# Log2 fold enrichment vs global
log2fc = np.log2(cluster_freq / global_freq[np.newaxis, :])  # (n_clusters, n_motifs)

# Select top 50 most variable motifs (max - min across clusters)
motif_variability = log2fc.max(axis=0) - log2fc.min(axis=0)
top50_idx = np.argsort(motif_variability)[-50:][::-1]

log2fc_top = log2fc[:, top50_idx]
top_motif_names = adata.var_names[top50_idx]
# Truncate long names for display
top_motif_names_short = [n.split("::")[-1][:20] for n in top_motif_names]

log2fc_df = pd.DataFrame(
    log2fc_top,
    index=[f"Cluster {c}" for c in clusters],
    columns=top_motif_names_short,
)

# Clustermap
vmax = min(np.abs(log2fc_top).max(), 3.0)
g = sns.clustermap(
    log2fc_df,
    cmap="RdBu_r",
    vmin=-vmax, vmax=vmax,
    center=0,
    figsize=(20, max(8, len(clusters) * 0.35)),
    xticklabels=True,
    yticklabels=True,
    linewidths=0,
    cbar_kws={"label": "log2 FC vs global"},
)
g.fig.suptitle(
    f"Leiden ({leiden_key}) × Top-50 variable motifs — log2 fold enrichment",
    fontsize=11, y=1.01,
)
g.fig.savefig(f"{FIG_DIR}/leiden_motif_enrichment_heatmap.pdf",
              dpi=150, bbox_inches="tight")
plt.close(g.fig)
print("  Saved: leiden_motif_enrichment_heatmap.pdf")

# %% ── 3e. Additional Diagnostics ────────────────────────────────────────────
print("\n3e. Diagnostics ...")

# Leiden clusters on UMAP
for res in [0.5, 1.0, 2.0, 5.0]:
    lkey = f"leiden_res{res}"
    if lkey not in adata.obs.columns:
        continue
    labels = adata.obs[lkey].values
    unique_labels = sorted(np.unique(labels), key=lambda c: int(c) if str(c).isdigit() else 0)
    n_cl = len(unique_labels)
    cmap_cl = plt.cm.get_cmap("tab20" if n_cl <= 20 else "nipy_spectral", n_cl)

    fig, ax = plt.subplots(figsize=(9, 8))
    for i, cl in enumerate(unique_labels):
        mask = labels == cl
        ax.scatter(x[mask], y[mask], c=[cmap_cl(i)], s=PT_SIZE,
                   alpha=PT_ALPHA, rasterized=RASTERIZED, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Leiden clusters (res={res}, n={n_cl})")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/umap_leiden_res{res}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
print("  Saved: Leiden UMAP plots")

# Species composition per Leiden cluster (res=1.0)
lkey = leiden_key
labels = adata.obs[lkey].values
comp_df = pd.crosstab(labels, adata.obs["species"].values, normalize="index")
comp_df = comp_df.reindex(sorted(comp_df.index, key=lambda c: int(c) if str(c).isdigit() else 0))

fig, ax = plt.subplots(figsize=(max(10, len(comp_df) * 0.35), 5))
comp_df.plot(
    kind="bar", stacked=True,
    color=[SPECIES_COLORS.get(c, "#888888") for c in comp_df.columns],
    ax=ax, width=0.8, edgecolor="none",
)
ax.set_xlabel("Leiden cluster")
ax.set_ylabel("Fraction of peaks")
ax.set_title(f"Species composition per Leiden cluster ({lkey})")
ax.legend(title="Species", loc="upper right", fontsize=8)
ax.tick_params(axis="x", labelrotation=90, labelsize=7)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/leiden_species_composition.pdf", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: leiden_species_composition.pdf")

# Cell type overlay (if available)
for ct_col in ["celltype", "top_celltype"]:
    if ct_col not in adata.obs.columns:
        continue
    ct_vals = adata.obs[ct_col].astype(str).values
    unique_cts = [c for c in pd.Series(ct_vals).value_counts().index if c not in ("nan", "None")][:30]
    n_ct = len(unique_cts)
    cmap_ct = plt.cm.get_cmap("tab20", min(n_ct, 20))

    fig, ax = plt.subplots(figsize=(10, 9))
    # Gray background for NaN / unlabeled
    nan_mask = ~np.isin(ct_vals, unique_cts)
    ax.scatter(x[nan_mask], y[nan_mask], c="#dddddd", s=PT_SIZE * 0.5,
               alpha=0.3, rasterized=RASTERIZED, linewidths=0)
    for i, ct in enumerate(unique_cts):
        mask = ct_vals == ct
        ax.scatter(x[mask], y[mask], c=[cmap_ct(i % 20)], s=PT_SIZE,
                   alpha=PT_ALPHA, rasterized=RASTERIZED, linewidths=0, label=ct)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Cell type overlay ({ct_col})")
    ax.legend(markerscale=8, fontsize=6, ncol=2, loc="upper right", framealpha=0.8)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/umap_{ct_col}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: umap_{ct_col}.pdf")
    break  # Only use first found column

# Timepoint overlay (zebrafish)
zf_mask = adata.obs["species"].values == "zebrafish"
for tp_col in ["timepoint", "top_timepoint"]:
    if tp_col not in adata.obs.columns:
        continue
    tp_vals = adata.obs[tp_col].astype(str).values
    # Plot only zebrafish timepoints
    tp_unique = [t for t in pd.Series(tp_vals[zf_mask]).value_counts().index
                 if t not in ("nan", "None")]

    if len(tp_unique) == 0:
        continue
    n_tp = len(tp_unique)
    cmap_tp = plt.cm.get_cmap("viridis", n_tp)

    fig, ax = plt.subplots(figsize=(9, 8))
    # Gray: non-zebrafish
    ax.scatter(x[~zf_mask], y[~zf_mask], c="#dddddd", s=PT_SIZE * 0.5,
               alpha=0.3, rasterized=RASTERIZED, linewidths=0)
    for i, tp in enumerate(tp_unique):
        mask = zf_mask & (tp_vals == tp)
        ax.scatter(x[mask], y[mask], c=[cmap_tp(i)], s=PT_SIZE,
                   alpha=PT_ALPHA, rasterized=RASTERIZED, linewidths=0, label=tp)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Zebrafish timepoints ({tp_col})")
    ax.legend(markerscale=8, fontsize=7, loc="upper right", framealpha=0.8, ncol=2)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/umap_zebrafish_{tp_col}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: umap_zebrafish_{tp_col}.pdf")
    break

# %% Save updated adata (with dominant_tf_family added)
print("\nSaving updated h5ad with new obs/var columns ...")
adata.write_h5ad(
    f"{SCRATCH}/cross_species_motif_embedded.h5ad",
    compression="gzip",
)
print("Done.")

# %% Summary
print("\n=== All outputs saved to ===")
print(f"  {FIG_DIR}/")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  - {f}")
