# %% Script 09g: Tau specificity index + UMAP visualizations
#
# Computes tau index and Gini coefficient from celltype-level mean accessibility
# for all 640K peaks, then generates 3 UMAP encodings:
#   1. Color (cividis)
#   2. Dot size
#   3. Dot transparency (low tau = transparent, high tau = opaque)
#
# Env: single-cell-base

import os, re, time
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Publication figure settings ──
import matplotlib as _mpl
_mpl.rcParams.update(_mpl.rcParamsDefault)
_mpl.rcParams['font.family'] = 'Arial'
_mpl.rcParams["pdf.fonttype"] = 42
_mpl.rcParams["ps.fonttype"]  = 42
import seaborn as _sns
_sns.set(style="whitegrid", context="paper")
_mpl.rcParams["savefig.dpi"]  = 300
# ────────────────────────────────────────────────────────────────────────────────

print("=== Script 09g: Tau Specificity Index + UMAP ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
FIG_DIR = f"{REPO}/figures/peak_parts_list/V3/specificity_overview"
os.makedirs(FIG_DIR, exist_ok=True)

MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"
MIN_CELLS = 20

TP_INT = {"0somites": 0, "5somites": 5, "10somites": 10,
          "15somites": 15, "20somites": 20, "30somites": 30}

# %% Load master h5ad
print("\nLoading master h5ad ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(MASTER_H5AD)
M = np.array(adata.X, dtype=np.float64)  # (640830, 190) log-norm
umap_coords = adata.obsm["X_umap_2D"]
print(f"  Shape: {adata.shape}  ({time.time()-t0:.1f}s)")

# %% Parse conditions → celltype mapping
def parse_condition(cond):
    m = re.search(r"(\d+somites)$", cond)
    if not m:
        return cond, ""
    tp = m.group(1)
    ct = cond[:-(len(tp)+1)]
    return ct, tp

cond_meta = pd.DataFrame(
    [parse_condition(c) for c in adata.var_names],
    columns=["celltype", "timepoint"],
    index=adata.var_names,
)
cond_meta["n_cells"] = adata.var["n_cells"].values
cond_meta["reliable"] = cond_meta["n_cells"] >= MIN_CELLS

reliable_groups = cond_meta[cond_meta["reliable"]].index.tolist()
reliable_celltypes = sorted(set(
    cond_meta.loc[col, "celltype"] for col in reliable_groups
))
# Exclude primordial_germ_cells (no reliable timepoints)
reliable_celltypes = [ct for ct in reliable_celltypes
                      if ct != "primordial_germ_cells"]
n_ct = len(reliable_celltypes)
print(f"  Reliable celltypes: {n_ct}")

# %% Compute celltype-level mean matrix
print("Computing celltype-level means ...", flush=True)
ct_mean = np.zeros((M.shape[0], n_ct), dtype=np.float64)

for ci, ct in enumerate(reliable_celltypes):
    ct_cols = [col for col in reliable_groups
               if cond_meta.loc[col, "celltype"] == ct]
    col_idx = [list(adata.var_names).index(c) for c in ct_cols]
    ct_mean[:, ci] = M[:, col_idx].mean(axis=1)

print(f"  Celltype-mean matrix: {ct_mean.shape}")

# %% Compute Tau index
# tau = sum(1 - x_hat_i) / (N - 1), where x_hat_i = x_i / max(x_i)
print("Computing Tau index ...", flush=True)
row_max = ct_mean.max(axis=1, keepdims=True)
# Avoid division by zero for peaks with zero accessibility everywhere
row_max_safe = np.maximum(row_max, 1e-10)
x_hat = ct_mean / row_max_safe
tau = (1.0 - x_hat).sum(axis=1) / (n_ct - 1)
# Peaks with zero everywhere → tau = 0 (not specific)
tau[row_max.ravel() < 1e-10] = 0.0

print(f"  Tau range: [{tau.min():.3f}, {tau.max():.3f}]")
print(f"  Tau > 0.8 (highly specific): {(tau > 0.8).sum():,} peaks")
print(f"  Tau > 0.6: {(tau > 0.6).sum():,} peaks")
print(f"  Tau < 0.2 (broadly accessible): {(tau < 0.2).sum():,} peaks")

# %% Compute Gini coefficient
print("Computing Gini coefficient ...", flush=True)

def gini_vectorized(X):
    """Gini coefficient per row of matrix X (n_peaks, n_celltypes)."""
    n = X.shape[1]
    X_sorted = np.sort(X, axis=1)
    index = np.arange(1, n + 1)
    row_sum = X_sorted.sum(axis=1)
    row_sum_safe = np.maximum(row_sum, 1e-10)
    gini = (2.0 * (X_sorted * index[None, :]).sum(axis=1) / (n * row_sum_safe)) - (n + 1) / n
    gini[row_sum < 1e-10] = 0.0
    return gini

gini = gini_vectorized(ct_mean)
print(f"  Gini range: [{gini.min():.3f}, {gini.max():.3f}]")
print(f"  Gini > 0.6: {(gini > 0.6).sum():,} peaks")

# %% Save metrics
metrics_df = pd.DataFrame({
    "tau": tau,
    "gini": gini,
    "max_accessibility": row_max.ravel(),
}, index=adata.obs_names)
metrics_csv = f"{OUTDIR}/V3_peak_specificity_metrics.csv"
metrics_df.to_csv(metrics_csv)
print(f"\nSaved: {metrics_csv}")

# %% Correlation between tau and gini
from scipy.stats import pearsonr
r, p = pearsonr(tau, gini)
print(f"  Tau vs Gini correlation: r={r:.3f}, p={p:.1e}")

# ══════════════════════════════════════════════════════════════════════════════
# UMAP VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n--- Generating UMAPs ---", flush=True)

# Subsample background for faster rendering
np.random.seed(42)
n_total = len(umap_coords)

# For 640K points, plot all but rasterize
print(f"  Plotting all {n_total:,} peaks (rasterized) ...", flush=True)

# ── Version 1: Color (cividis) ──
print("  V1: tau as color ...", flush=True)
fig, ax = plt.subplots(figsize=(9, 8))
order = np.argsort(tau)  # plot low tau first, high tau on top
sc = ax.scatter(umap_coords[order, 0], umap_coords[order, 1],
                c=tau[order], cmap="cividis", s=0.3, alpha=0.6,
                edgecolors="none", rasterized=True)
cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.01)
cbar.set_label("Tau specificity index", fontsize=10)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title(f"Peak specificity (Tau index)\n"
             f"0 = broadly accessible, 1 = celltype-specific\n"
             f"(n = {n_total:,} peaks)", fontsize=11)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/V3_peak_umap_tau_color.pdf", bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/V3_peak_umap_tau_color.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("    Saved: V3_peak_umap_tau_color.{pdf,png}")

# ── Version 2: Dot size ──
print("  V2: tau as dot size ...", flush=True)
fig, ax = plt.subplots(figsize=(9, 8))
# Scale: tau 0 → size 0.1, tau 1 → size 3.0
sizes = 0.1 + tau * 2.9
order = np.argsort(tau)  # small dots first
ax.scatter(umap_coords[order, 0], umap_coords[order, 1],
           s=sizes[order], c="#555555", alpha=0.4,
           edgecolors="none", rasterized=True)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title(f"Peak specificity (Tau index → dot size)\n"
             f"larger = more celltype-specific\n"
             f"(n = {n_total:,} peaks)", fontsize=11)
ax.set_aspect("equal")
# Size legend
for tau_val, label in [(0.2, "0.2"), (0.5, "0.5"), (0.8, "0.8"), (1.0, "1.0")]:
    ax.scatter([], [], s=0.1 + tau_val * 2.9, c="#555555", alpha=0.6,
               label=f"tau = {label}")
ax.legend(loc="upper right", fontsize=8, title="Tau", title_fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/V3_peak_umap_tau_size.pdf", bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/V3_peak_umap_tau_size.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("    Saved: V3_peak_umap_tau_size.{pdf,png}")

# ── Version 3: Dot transparency (low tau = transparent, high tau = opaque) ──
print("  V3: tau as transparency ...", flush=True)
fig, ax = plt.subplots(figsize=(9, 8))
# Build RGBA array: fixed color, alpha scales with tau
base_color = np.array(mcolors.to_rgba("#333333"))  # dark gray
rgba = np.tile(base_color, (n_total, 1))
# Alpha: tau 0 → 0.02 (nearly invisible), tau 1 → 0.9 (opaque)
rgba[:, 3] = 0.02 + tau * 0.88

order = np.argsort(tau)  # transparent dots first
ax.scatter(umap_coords[order, 0], umap_coords[order, 1],
           s=0.5, c=rgba[order], edgecolors="none", rasterized=True)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title(f"Peak specificity (Tau index → opacity)\n"
             f"visible = celltype-specific, faded = broadly accessible\n"
             f"(n = {n_total:,} peaks)", fontsize=11)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/V3_peak_umap_tau_alpha.pdf", bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/V3_peak_umap_tau_alpha.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("    Saved: V3_peak_umap_tau_alpha.{pdf,png}")

# ── Bonus: Tau distribution histogram ──
print("  Histogram ...", flush=True)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(tau, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
ax.axvline(0.8, color="red", ls="--", lw=1.2, label=f"tau=0.8 ({(tau>0.8).sum():,} peaks)")
ax.axvline(0.6, color="orange", ls="--", lw=1.2, label=f"tau=0.6 ({(tau>0.6).sum():,} peaks)")
ax.set_xlabel("Tau specificity index")
ax.set_ylabel("Number of peaks")
ax.set_title("Distribution of peak celltype specificity (Tau index)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/V3_tau_histogram.pdf", bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/V3_tau_histogram.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("    Saved: V3_tau_histogram.{pdf,png}")

print(f"\nDone. End: {time.strftime('%c')}")
