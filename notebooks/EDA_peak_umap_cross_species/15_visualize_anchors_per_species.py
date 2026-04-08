# %% [markdown]
# # Step 15: Visualize Anchor Peaks on Per-Species UMAPs
#
# Plots root and branch anchor peaks on each species' own pseudobulk UMAP
# to verify anchor quality before running the alignment.
#
# For zebrafish: uses peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad (X_umap)
# For mouse:     uses peaks_by_pb_celltype_stage_annotated_v2.h5ad (X_umap)
# For human:     uses peaks_by_pb_celltype_stage_annotated.h5ad (X_umap)
#
# Produces:
#   Fig 1: 3-panel — one per species, root=red, branch=blue, background=gray
#   Fig 2: 3-panel — branch anchors colored by lineage per species
#   Fig 3: Per-species panels with branch lineage breakdown (one row per lineage)

# %% Imports
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import anndata as ad

# %% Paths
BASE   = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
FIG_DIR = f"{BASE}/zebrahub-multiome-analysis/figures/cross_species_motif_umap/anchor_diagnostics"
os.makedirs(FIG_DIR, exist_ok=True)

ZF_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad"
MM_H5AD = f"{BASE}/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated_v2.h5ad"
HS_H5AD = f"{BASE}/data/public_data/human_domcke_2020/peaks_by_pb_celltype_stage_annotated.h5ad"
ROOT_CSV   = f"{SCRATCH}/anchors/root_anchors.csv"
BRANCH_CSV = f"{SCRATCH}/anchors/branch_anchors.csv"

UMAP_KEY = "X_umap"   # all three h5ads use this key

# %% Load anchors
print("Loading anchors ...")
root_df   = pd.read_csv(ROOT_CSV)
branch_df = pd.read_csv(BRANCH_CSV)
print(f"  Root anchors:   {len(root_df):,}")
print(f"  Branch anchors: {len(branch_df):,}")
print(f"  Branch lineages: {branch_df['lineage'].value_counts().to_dict()}")

# %% Helper: parse integer index from peak_id string
def parse_idx(val):
    """'zebrafish_peak_1234' -> 1234; NaN/empty -> -1"""
    if pd.isna(val) or str(val).strip() == "":
        return -1
    return int(str(val).rsplit("_", 1)[-1])

# %% Helper: load UMAP coords from h5ad (backed, cheap)
def load_umap(path, umap_key=UMAP_KEY):
    print(f"  Loading {os.path.basename(path)} ...", flush=True)
    import time; t0 = time.time()
    adata = ad.read_h5ad(path, backed="r")
    coords = np.array(adata.obsm[umap_key])   # load into memory
    print(f"    Shape: {adata.shape}  UMAP: {coords.shape}  ({time.time()-t0:.1f}s)")
    adata.file.close()
    return coords

# %% Load per-species UMAP coordinates
print("\nLoading per-species UMAPs ...")
umap_zf = load_umap(ZF_H5AD)
umap_mm = load_umap(MM_H5AD)
umap_hs = load_umap(HS_H5AD)

# %% Build anchor index arrays per species
def get_anchor_indices(df, col):
    """Return valid integer indices (exclude -1)."""
    return np.array([i for i in df[col].apply(parse_idx) if i >= 0])

# Root anchor indices
root_idx = {
    "zebrafish": get_anchor_indices(root_df, "peak_id_zf"),
    "mouse":     get_anchor_indices(root_df, "peak_id_mm"),
    "human":     get_anchor_indices(root_df, "peak_id_hs"),
}

# Branch anchor indices per lineage
LINEAGES = branch_df["lineage"].dropna().unique().tolist()
branch_idx = {}
for sp, col in [("zebrafish","peak_id_zf"), ("mouse","peak_id_mm"), ("human","peak_id_hs")]:
    branch_idx[sp] = {}
    for lin in LINEAGES:
        sub = branch_df[branch_df["lineage"] == lin]
        branch_idx[sp][lin] = get_anchor_indices(sub, col)

branch_all_idx = {
    sp: get_anchor_indices(branch_df, col)
    for sp, col in [("zebrafish","peak_id_zf"), ("mouse","peak_id_mm"), ("human","peak_id_hs")]
}

print("\nAnchor counts per species:")
for sp in ["zebrafish", "mouse", "human"]:
    print(f"  {sp}: root={len(root_idx[sp])}, branch={len(branch_all_idx[sp])}")

# %% Color maps
LINEAGE_COLORS = {
    "neural_cns":        "#e74c3c",
    "neural_crest":      "#9b59b6",
    "paraxial_mesoderm": "#3498db",
    "lateral_mesoderm":  "#2ecc71",
    "endoderm":          "#f39c12",
    "ectoderm":          "#1abc9c",
}

# %% ── Figure 1: root vs branch (3 panels, one per species) ──────────────────

SPECIES = [
    ("Zebrafish", "zebrafish", umap_zf),
    ("Mouse",     "mouse",     umap_mm),
    ("Human",     "human",     umap_hs),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Anchor peaks on per-species pseudobulk UMAPs", fontsize=14, y=1.02)

for ax, (label, sp, umap) in zip(axes, SPECIES):
    # background (subsample large datasets for speed)
    n = umap.shape[0]
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(n, size=min(n, 100_000), replace=False)
    ax.scatter(umap[bg_idx, 0], umap[bg_idx, 1],
               s=0.3, c="#d0d0d0", alpha=0.3, rasterized=True, linewidths=0)

    # branch anchors
    bidx = branch_all_idx[sp]
    if len(bidx):
        ax.scatter(umap[bidx, 0], umap[bidx, 1],
                   s=8, c="#3498db", alpha=0.8, zorder=3, linewidths=0, label=f"Branch ({len(bidx)})")

    # root anchors (on top)
    ridx = root_idx[sp]
    if len(ridx):
        ax.scatter(umap[ridx, 0], umap[ridx, 1],
                   s=12, c="#e74c3c", alpha=0.9, zorder=4, linewidths=0, label=f"Root ({len(ridx)})")

    ax.set_title(label, fontsize=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, fontsize=8, loc="upper right")
    ax.set_aspect("equal", "datalim")

plt.tight_layout()
out = f"{FIG_DIR}/anchors_root_branch_per_species"
fig.savefig(out + ".pdf", bbox_inches="tight", dpi=150)
fig.savefig(out + ".png", bbox_inches="tight", dpi=150)
plt.close()
print(f"\nSaved: {out}.{{pdf,png}}")

# %% ── Figure 2: branch anchors colored by lineage (3 panels) ────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Branch anchors by lineage on per-species UMAPs", fontsize=14, y=1.02)

for ax, (label, sp, umap) in zip(axes, SPECIES):
    n = umap.shape[0]
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(n, size=min(n, 100_000), replace=False)
    ax.scatter(umap[bg_idx, 0], umap[bg_idx, 1],
               s=0.3, c="#d0d0d0", alpha=0.3, rasterized=True, linewidths=0)

    # root anchors in gray outline
    ridx = root_idx[sp]
    if len(ridx):
        ax.scatter(umap[ridx, 0], umap[ridx, 1],
                   s=10, c="#888888", alpha=0.6, zorder=3, linewidths=0, label=f"Root ({len(ridx)})")

    # branch per lineage
    for lin in LINEAGES:
        idx = branch_idx[sp][lin]
        if len(idx) == 0:
            continue
        color = LINEAGE_COLORS.get(lin, "#aaaaaa")
        ax.scatter(umap[idx, 0], umap[idx, 1],
                   s=10, c=color, alpha=0.85, zorder=4, linewidths=0,
                   label=f"{lin} ({len(idx)})")

    ax.set_title(label, fontsize=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=2, fontsize=7, loc="upper right",
              framealpha=0.8, handlelength=1)
    ax.set_aspect("equal", "datalim")

plt.tight_layout()
out = f"{FIG_DIR}/anchors_branch_lineage_per_species"
fig.savefig(out + ".pdf", bbox_inches="tight", dpi=150)
fig.savefig(out + ".png", bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {out}.{{pdf,png}}")

# %% ── Figure 3: lineage rows × species columns ───────────────────────────────
# One row per lineage, one column per species

n_lin = len(LINEAGES)
fig, axes = plt.subplots(n_lin, 3, figsize=(18, 5 * n_lin))
if n_lin == 1:
    axes = axes[np.newaxis, :]
fig.suptitle("Branch anchors per lineage — per species", fontsize=14, y=1.01)

for row, lin in enumerate(LINEAGES):
    color = LINEAGE_COLORS.get(lin, "#aaaaaa")
    for col, (label, sp, umap) in enumerate(SPECIES):
        ax = axes[row, col]
        n = umap.shape[0]
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(n, size=min(n, 80_000), replace=False)
        ax.scatter(umap[bg_idx, 0], umap[bg_idx, 1],
                   s=0.2, c="#d0d0d0", alpha=0.3, rasterized=True, linewidths=0)

        idx = branch_idx[sp][lin]
        n_pts = len(idx)
        if n_pts:
            ax.scatter(umap[idx, 0], umap[idx, 1],
                       s=12, c=color, alpha=0.9, zorder=4, linewidths=0)

        if row == 0:
            ax.set_title(label, fontsize=11)
        ax.set_ylabel(lin.replace("_", "\n") if col == 0 else "")
        ax.set_xlabel("UMAP 1" if row == n_lin - 1 else "")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.text(0.97, 0.03, f"n={n_pts}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=color if n_pts else "#aaa")
        ax.set_aspect("equal", "datalim")

plt.tight_layout()
out = f"{FIG_DIR}/anchors_lineage_rows_per_species"
fig.savefig(out + ".pdf", bbox_inches="tight", dpi=150)
fig.savefig(out + ".png", bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {out}.{{pdf,png}}")

print("\nDone.")
print(f"All figures saved to: {FIG_DIR}")
