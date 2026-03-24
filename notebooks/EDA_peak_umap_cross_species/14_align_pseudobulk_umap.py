# %% [markdown]
# # Step 14: Align Per-Species Pseudobulk UMAPs Using Anchor Peaks
#
# Each species has its own pseudobulk peak UMAP (peaks × cell types):
#   zebrafish → X_umap_concord  (640K peaks × 190 pseudobulk columns)
#   mouse     → X_umap          (192K peaks × 145 pseudobulk columns)
#   human     → X_umap          (1M peaks   × 249 pseudobulk columns)
#
# This script uses biologically curated anchor peaks (root + branch) to
# Procrustes-align mouse and human 2D UMAP coordinates onto the zebrafish
# UMAP coordinate system, producing a single unified cross-species UMAP.
#
# Approach:
#   1. Load per-species h5ads; extract 2D UMAP coords
#   2. Extract anchor peak coordinates from each species using integer index
#      (zebrafish_peak_N → row N of ZF h5ad, etc.)
#   3. Procrustes: center → scale → rotate mouse/human UMAP → ZF UMAP
#   4. Apply transformation to all peaks in mouse/human
#   5. Combine into a single DataFrame and save
#   6. Visualize: species overlay, per-species lineage panels, anchor highlights
#
# Inputs:
#   ZF h5ad:     .../data/annotated_data/objects_v2/peaks_by_ct_tp_pseudobulked_all_peaks_pca_concord.h5ad
#   MM h5ad:     .../data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated_v2.h5ad
#   HS h5ad:     .../data/public_data/human_domcke_2020/peaks_by_pb_celltype_stage_annotated.h5ad
#   root_anchors.csv, branch_anchors.csv  (from scripts 10–11)
#
# Outputs (saved to SCRATCH/pseudobulk_umap_aligned/):
#   aligned_coords.csv.gz  — all peaks with (umap1_aligned, umap2_aligned, species, ...)
# Figures saved to figures/cross_species_motif_umap/pseudobulk_aligned_*/
#
# Env: single-cell-base | Resources: 4 CPUs, 64G, 1h

# %% Imports
import os, gc, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import orthogonal_procrustes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import anndata as ad
import scanpy as sc

print(f"anndata {ad.__version__}, scanpy {sc.__version__}")

# %% Paths
BASE   = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"

ZF_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_pseudobulked_all_peaks_pca_concord.h5ad"
MM_H5AD = f"{BASE}/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated_v2.h5ad"
HS_H5AD = f"{BASE}/data/public_data/human_domcke_2020/peaks_by_pb_celltype_stage_annotated.h5ad"

ROOT_CSV   = f"{SCRATCH}/anchors/root_anchors.csv"
BRANCH_CSV = f"{SCRATCH}/anchors/branch_anchors.csv"

OUT_DIR = f"{SCRATCH}/pseudobulk_umap_aligned"
os.makedirs(OUT_DIR, exist_ok=True)

FIG_DIR = f"{BASE}/zebrahub-multiome-analysis/figures/cross_species_motif_umap/pseudobulk_aligned"
os.makedirs(FIG_DIR, exist_ok=True)

# UMAP key to use per species
ZF_UMAP_KEY = "X_umap_concord"   # best ZF embedding
MM_UMAP_KEY = "X_umap"
HS_UMAP_KEY = "X_umap"

SPECIES_COLORS = {"zebrafish": "#1f77b4", "mouse": "#ff7f0e", "human": "#2ca02c"}
PT_SIZE = 0.3


# %% ── Load anchors ───────────────────────────────────────────────────────────────
print("Loading anchors ...")
root_anch   = pd.read_csv(ROOT_CSV)
branch_anch = pd.read_csv(BRANCH_CSV)

# Stack root + branch; keep all triplets that have at least ZF
cols = ["peak_id_zf", "peak_id_mm", "peak_id_hs"]
anchors = pd.concat([
    root_anch[cols].dropna(subset=["peak_id_zf"]),
    branch_anch[cols].dropna(subset=["peak_id_zf"]),
], ignore_index=True).drop_duplicates()
print(f"  Total anchor triplets: {len(anchors):,}  (root={len(root_anch)}, branch={len(branch_anch)})")


def parse_peak_idx(peak_id_series):
    """Extract integer index from 'species_peak_N' strings. Returns array of ints, -1 if None/nan."""
    result = np.full(len(peak_id_series), -1, dtype=int)
    for i, val in enumerate(peak_id_series):
        if pd.isna(val) or str(val).lower() in ("none", "nan", ""):
            continue
        try:
            result[i] = int(str(val).rsplit("_", 1)[-1])
        except ValueError:
            pass
    return result


zf_idx_anchor = parse_peak_idx(anchors["peak_id_zf"])
mm_idx_anchor = parse_peak_idx(anchors["peak_id_mm"])
hs_idx_anchor = parse_peak_idx(anchors["peak_id_hs"])


# %% ── Load per-species h5ads and extract UMAP coords ────────────────────────────
def load_umap(path, umap_key, species_label):
    """Load h5ad and return (obs_names, umap_coords_2d, obs_df)."""
    print(f"\nLoading {species_label}: {path.split('/')[-1]} ...")
    t0 = time.time()
    adata = sc.read_h5ad(path)
    print(f"  Shape: {adata.shape}  ({time.time()-t0:.1f}s)")
    print(f"  obsm keys: {list(adata.obsm.keys())}")

    if umap_key not in adata.obsm:
        # fallback
        for k in ["X_umap_concord", "X_umap", "X_umap_pca"]:
            if k in adata.obsm:
                umap_key = k
                break
    coords = adata.obsm[umap_key][:, :2].copy()
    print(f"  Using UMAP key: {umap_key}  ({coords.shape})")
    return adata.obs_names.astype(str), coords, adata.obs.copy()


zf_names, zf_umap, zf_obs = load_umap(ZF_H5AD, ZF_UMAP_KEY, "zebrafish")
mm_names, mm_umap, mm_obs = load_umap(MM_H5AD, MM_UMAP_KEY, "mouse")
hs_names, hs_umap, hs_obs = load_umap(HS_H5AD, HS_UMAP_KEY, "human")


# %% ── Build anchor coordinate matrices ──────────────────────────────────────────
print("\nBuilding anchor matrices ...")

def get_anchor_coords(umap_coords, idx_arr, ref_idx_arr=None):
    """
    Returns (src_mat, ref_mat, valid_mask) where valid_mask selects rows
    where src idx >= 0 and (if ref provided) ref idx >= 0 and < len.
    """
    n = len(umap_coords)
    valid = (idx_arr >= 0) & (idx_arr < n)
    if ref_idx_arr is not None:
        ref_n = len(ref_idx_arr)  # same length; ref_idx_arr are ZF indices
        valid = valid & (ref_idx_arr >= 0) & (ref_idx_arr < len(zf_umap))
        src_mat = umap_coords[idx_arr[valid]]
        ref_mat = zf_umap[ref_idx_arr[valid]]
        return src_mat, ref_mat, valid
    else:
        return umap_coords[idx_arr[valid]], None, valid


# MM → ZF
mm_src, zf_ref_mm, valid_mm = get_anchor_coords(mm_umap, mm_idx_anchor, zf_idx_anchor)
print(f"  MM anchor pairs: {valid_mm.sum()} / {len(anchors)}")

# HS → ZF
hs_src, zf_ref_hs, valid_hs = get_anchor_coords(hs_umap, hs_idx_anchor, zf_idx_anchor)
print(f"  HS anchor pairs: {valid_hs.sum()} / {len(anchors)}")


# %% ── Procrustes alignment ────────────────────────────────────────────────────────
def procrustes_align(X_all_src, X_src_anchors, X_ref_anchors):
    """
    Align all source points using anchor-derived rotation.
    Steps: center → scale → rotate → translate to reference centroid.
    Returns aligned X_all_src (same shape).
    """
    mu_src = X_src_anchors.mean(0)
    mu_ref = X_ref_anchors.mean(0)
    Xc_src = X_src_anchors - mu_src
    Xc_ref = X_ref_anchors - mu_ref

    # Scale: match Frobenius norms of anchor sets
    scale = np.linalg.norm(Xc_ref) / (np.linalg.norm(Xc_src) + 1e-12)

    # Rotation
    R, _ = orthogonal_procrustes(Xc_src * scale, Xc_ref)

    # Apply to all source points
    X_aligned = (X_all_src - mu_src) * scale @ R + mu_ref
    return X_aligned, R, scale, mu_src, mu_ref


print("\nProcrustes alignment ...")

mm_aligned, R_mm, scale_mm, mu_mm, mu_zf_mm = procrustes_align(mm_umap, mm_src, zf_ref_mm)
print(f"  MM→ZF  scale={scale_mm:.4f}")

hs_aligned, R_hs, scale_hs, mu_hs, mu_zf_hs = procrustes_align(hs_umap, hs_src, zf_ref_hs)
print(f"  HS→ZF  scale={scale_hs:.4f}")


# %% ── Validate: anchor co-localization ──────────────────────────────────────────
print("\nValidating anchor co-localization ...")

# MM→ZF after alignment
mm_anch_aligned = procrustes_align(mm_src, mm_src, zf_ref_mm)[0]
dists_mm = np.linalg.norm(mm_anch_aligned - zf_ref_mm, axis=1)
# Random baseline: shuffle ZF anchor positions
rng = np.random.default_rng(42)
zf_shuf = zf_umap[rng.choice(len(zf_umap), valid_mm.sum(), replace=False)]
dists_mm_rand = np.linalg.norm(mm_src - zf_shuf[:len(mm_src)], axis=1)
print(f"  MM anchors: mean dist after alignment = {dists_mm.mean():.4f}  |  random = {dists_mm_rand.mean():.4f}")

# HS→ZF after alignment
hs_anch_aligned = procrustes_align(hs_src, hs_src, zf_ref_hs)[0]
dists_hs = np.linalg.norm(hs_anch_aligned - zf_ref_hs, axis=1)
zf_shuf2 = zf_umap[rng.choice(len(zf_umap), valid_hs.sum(), replace=False)]
dists_hs_rand = np.linalg.norm(hs_src - zf_shuf2[:len(hs_src)], axis=1)
print(f"  HS anchors: mean dist after alignment = {dists_hs.mean():.4f}  |  random = {dists_hs_rand.mean():.4f}")


# %% ── Build combined coordinate table ───────────────────────────────────────────
print("\nBuilding combined coordinate table ...")

def make_coords_df(umap_coords, obs_names, obs_df, species_label, extra_cols=None):
    df = pd.DataFrame({
        "peak_id":  obs_names,
        "species":  species_label,
        "umap1":    umap_coords[:, 0],
        "umap2":    umap_coords[:, 1],
    })
    if extra_cols:
        for col in extra_cols:
            if col in obs_df.columns:
                df[col] = obs_df[col].astype(str).replace({"nan": "", "None": ""}).values
    return df


zf_df = make_coords_df(zf_umap, zf_names, zf_obs, "zebrafish",
                       extra_cols=["celltype", "timepoint", "lineage",
                                   "peak_type", "nearest_gene"])
mm_df = make_coords_df(mm_aligned, mm_names, mm_obs, "mouse",
                       extra_cols=["top_celltype", "top_timepoint",
                                   "peak_type", "nearest_gene"])
hs_df = make_coords_df(hs_aligned, hs_names, hs_obs, "human",
                       extra_cols=["top_celltype", "top_timepoint",
                                   "peak_type", "nearest_gene"])

combined = pd.concat([zf_df, mm_df, hs_df], ignore_index=True)
print(f"  Combined: {len(combined):,} peaks")

out_csv = f"{OUT_DIR}/aligned_coords.csv.gz"
combined.to_csv(out_csv, index=False)
print(f"  Saved: {out_csv}")


# %% ── Helpers ────────────────────────────────────────────────────────────────────
def scatter(ax, coords, colors, sizes=PT_SIZE, alpha=0.5, **kw):
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes,
               alpha=alpha, rasterized=True, linewidths=0, **kw)


def save_fig(fig, name):
    fig.savefig(f"{FIG_DIR}/{name}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.{{pdf,png}}")


# Combine coords as arrays for plotting
all_umap   = combined[["umap1", "umap2"]].values
all_species = combined["species"].values


# %% ── Fig 1: Species overlay (shuffled draw order) ─────────────────────────────
print("\nFig 1: species overlay ...")
fig, ax = plt.subplots(figsize=(8, 7))
order = np.random.permutation(len(all_umap))
c_arr = np.array([SPECIES_COLORS.get(s, "gray") for s in all_species])
scatter(ax, all_umap[order], c_arr[order])
handles = [mpatches.Patch(color=c, label=sp) for sp, c in SPECIES_COLORS.items()]
ax.legend(handles=handles, fontsize=10)
ax.set_title("Pseudobulk UMAP — anchor-aligned (all species)")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
plt.tight_layout()
save_fig(fig, "pseudobulk_aligned_species_overlay")


# %% ── Fig 2: Per-species highlight panels ───────────────────────────────────────
print("Fig 2: per-species highlight panels ...")
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
for ax, sp in zip(axes, ["zebrafish", "mouse", "human"]):
    mask = all_species == sp
    scatter(ax, all_umap[~mask], "lightgray", sizes=0.1, alpha=0.1)
    scatter(ax, all_umap[mask],  SPECIES_COLORS[sp], sizes=0.5, alpha=0.7)
    ax.set_title(sp); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
plt.suptitle("Pseudobulk UMAP — anchor-aligned", y=1.01)
plt.tight_layout()
save_fig(fig, "pseudobulk_aligned_per_species")


# %% ── Fig 3: Per-species panels colored by lineage/celltype ─────────────────────
# Zebrafish: "lineage" col; Mouse: "lineage" col; Human: "peak_lineage" col
print("Fig 3: per-species lineage panels ...")

LINEAGE_COL = {
    "zebrafish": "lineage",
    "mouse":     "lineage",
    "human":     "peak_lineage",
}

# Build shared color palette
all_lin_vals = set()
for sp_name, col in LINEAGE_COL.items():
    if col in combined.columns:
        vals = combined.loc[combined["species"] == sp_name, col]
        all_lin_vals.update(v for v in vals.unique() if v not in ("", "nan"))
    # human peak_lineage not in combined yet — add from hs_obs directly
    if col not in combined.columns and col in hs_obs.columns and sp_name == "human":
        all_lin_vals.update(v for v in hs_obs[col].astype(str).unique()
                            if v not in ("", "nan", "None"))

sorted_lins = sorted(all_lin_vals)
pal = plt.cm.tab20(np.linspace(0, 1, max(len(sorted_lins), 1)))
lin_colors = {l: pal[i] for i, l in enumerate(sorted_lins)}
GRAY = (0.85, 0.85, 0.85, 0.15)

# Add human peak_lineage to combined if needed
if "peak_lineage" not in combined.columns:
    hs_lin = hs_obs["peak_lineage"].astype(str).replace({"nan": "", "None": ""}).values \
             if "peak_lineage" in hs_obs.columns else np.full(len(hs_umap), "")
    lin_col_vals = np.concatenate([
        zf_obs["lineage"].astype(str).replace({"nan": "", "None": ""}).values
        if "lineage" in zf_obs.columns else np.full(len(zf_umap), ""),
        mm_obs["lineage"].astype(str).replace({"nan": "", "None": ""}).values
        if "lineage" in mm_obs.columns else np.full(len(mm_umap), ""),
        hs_lin,
    ])
    combined["peak_lineage"] = lin_col_vals

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for ax, sp_name in zip(axes, ["zebrafish", "mouse", "human"]):
    sp_mask  = all_species == sp_name
    col      = LINEAGE_COL[sp_name]
    sp_coords = all_umap[sp_mask]

    # gray background for other species
    scatter(ax, all_umap[~sp_mask], "lightgray", sizes=0.1, alpha=0.08)

    if col in combined.columns:
        sp_lin = combined.loc[combined["species"] == sp_name, col].values
        # unlabeled first
        empty = sp_lin == ""
        scatter(ax, sp_coords[empty], "lightgray", sizes=0.2, alpha=0.15)
        for lin in sorted_lins:
            m = sp_lin == lin
            if m.sum() > 0:
                scatter(ax, sp_coords[m], lin_colors[lin], sizes=1.5, alpha=0.75)
        present = [l for l in sorted_lins if (sp_lin == l).any()]
        handles = [mpatches.Patch(color=lin_colors[l], label=l) for l in present]
        ax.legend(handles=handles, fontsize=6, loc="upper right", ncol=1, framealpha=0.7)
    else:
        scatter(ax, sp_coords, SPECIES_COLORS[sp_name], sizes=0.5, alpha=0.5)

    ax.set_title(f"{sp_name}  [{col}]", fontsize=11)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

plt.suptitle("Pseudobulk UMAP — lineage annotations per species", y=1.01, fontsize=13)
plt.tight_layout()
save_fig(fig, "pseudobulk_aligned_per_species_lineage")


# %% ── Fig 4: Anchor peaks highlighted ───────────────────────────────────────────
print("Fig 4: anchor highlights ...")

# Build sets of per-species peak IDs for anchors
zf_root_ids = set(root_anch["peak_id_zf"].dropna().astype(str))
mm_root_ids = set(root_anch["peak_id_mm"].dropna().astype(str))
hs_root_ids = set(root_anch["peak_id_hs"].dropna().astype(str))
zf_br_ids   = set(branch_anch["peak_id_zf"].dropna().astype(str))
mm_br_ids   = set(branch_anch["peak_id_mm"].dropna().astype(str))
hs_br_ids   = set(branch_anch["peak_id_hs"].dropna().astype(str))

def get_anchor_type(peak_id, species):
    cs_id = f"{species}_peak_{peak_id}"   # cross-species ID format
    if species == "zebrafish":
        if cs_id in zf_root_ids: return "root"
        if cs_id in zf_br_ids:   return "branch"
    elif species == "mouse":
        if cs_id in mm_root_ids: return "root"
        if cs_id in mm_br_ids:   return "branch"
    elif species == "human":
        if cs_id in hs_root_ids: return "root"
        if cs_id in hs_br_ids:   return "branch"
    return "other"

# Build obs-integer index → anchor type
# Use the integer index from the combined df (positional)
# ZF block: 0..len(zf)-1; MM block: len(zf)..len(zf)+len(mm)-1; HS block: rest
n_zf, n_mm = len(zf_umap), len(mm_umap)

anchor_type = np.full(len(combined), "other", dtype=object)
# ZF
for peak_type_str, id_set in [("root", zf_root_ids), ("branch", zf_br_ids)]:
    indices = [int(s.split("_")[-1]) for s in id_set if s.startswith("zebrafish_peak_")]
    for idx in indices:
        if idx < n_zf:
            anchor_type[idx] = peak_type_str
# MM
for peak_type_str, id_set in [("root", mm_root_ids), ("branch", mm_br_ids)]:
    indices = [int(s.split("_")[-1]) for s in id_set if s.startswith("mouse_peak_")]
    for idx in indices:
        if idx < n_mm:
            anchor_type[n_zf + idx] = peak_type_str
# HS
n_hs = len(hs_umap)
for peak_type_str, id_set in [("root", hs_root_ids), ("branch", hs_br_ids)]:
    indices = [int(s.split("_")[-1]) for s in id_set if s.startswith("human_peak_")]
    for idx in indices:
        if idx < n_hs:
            anchor_type[n_zf + n_mm + idx] = peak_type_str

fig, ax = plt.subplots(figsize=(9, 7))
for color, atype, size, alpha in [
    ("lightgray", "other",  0.2, 0.15),
    ("steelblue", "branch", 4,   0.85),
    ("red",       "root",   5,   0.9),
]:
    m = anchor_type == atype
    if m.sum() > 0:
        scatter(ax, all_umap[m], color, sizes=size, alpha=alpha)
        ax.plot([], [], "o", color=color, label=f"{atype} (n={m.sum():,})", ms=5)
ax.legend(loc="upper right", fontsize=9)
ax.set_title("Anchor peaks on pseudobulk-aligned UMAP")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
plt.tight_layout()
save_fig(fig, "pseudobulk_aligned_anchors")


# %% ── Fig 5: Side-by-side unaligned vs aligned (per-species) ────────────────────
print("Fig 5: unaligned vs aligned per-species ...")
fig, axes = plt.subplots(2, 3, figsize=(21, 12))
unaligned_umaps = {
    "zebrafish": zf_umap,
    "mouse":     mm_umap,
    "human":     hs_umap,
}
aligned_umaps = {
    "zebrafish": zf_umap,
    "mouse":     mm_aligned,
    "human":     hs_aligned,
}

for col_idx, sp in enumerate(["zebrafish", "mouse", "human"]):
    c = SPECIES_COLORS[sp]
    sp_mask_all = all_species == sp
    for row_idx, (title, umap_d) in enumerate([("Unaligned", unaligned_umaps),
                                                ("Aligned",   aligned_umaps)]):
        ax = axes[row_idx][col_idx]
        sp_coords = umap_d[sp]
        scatter(ax, sp_coords, c, sizes=0.3, alpha=0.5)
        ax.set_title(f"{sp} — {title}")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

plt.suptitle("Pseudobulk UMAP: unaligned vs Procrustes-aligned", y=1.01, fontsize=13)
plt.tight_layout()
save_fig(fig, "pseudobulk_unaligned_vs_aligned")


# %% ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"Outputs saved to: {OUT_DIR}")
print(f"Figures saved to: {FIG_DIR}")
print("\nFigures generated:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith("pseudobulk"):
        print(f"  {f}")
print("Done.")
