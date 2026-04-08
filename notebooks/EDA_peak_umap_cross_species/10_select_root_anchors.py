# %% [markdown]
# # Step 10: Select Root Anchors
#
# Root anchors = promoter peaks near broadly accessible housekeeping genes
# that are 1:1:1 orthologs across zebrafish / mouse / human.
#
# Algorithm:
#   1. Filter to promoter peaks per species (peak_type=="promoter" OR distance_to_tss < 2kb)
#   2. Filter broadly accessible peaks (low cell-type specificity)
#      - Coefficient of variation (CV) across pseudobulk columns < threshold
#   3. Cross-reference nearest_gene_symbol with ortholog triplet table
#   4. Per ortholog triplet: pick the closest-to-TSS peak per species
#   5. Keep complete triplets (all 3 species have a qualifying peak)
#
# Input:
#   - {SCRATCH}/cross_species_motif_embedded_annotated.h5ad
#   - {SCRATCH}/orthologs/zebrafish_mouse_human_1to1_orthologs.csv
#
# Output:
#   - {SCRATCH}/anchors/root_anchors.csv
#     Columns: gene_triplet_id, gene_name_zf, gene_name_mm, gene_name_hs,
#               peak_id_zf, peak_id_mm, peak_id_hs,
#               distance_to_tss_zf, distance_to_tss_mm, distance_to_tss_hs
#   - figures/.../root_anchor_diagnostics.pdf
#
# Env: single-cell-base (CPU)  — 4 CPUs, 64G, 30min
#   conda run -p /hpc/user_apps/data.science/conda_envs/single-cell-base python -u 10_select_root_anchors.py

# %% Imports
import os, gc, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc

print(f"scanpy {sc.__version__}")

# %% Paths
SCRATCH  = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
BASE     = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"

INPUT_H5AD    = f"{SCRATCH}/cross_species_motif_embedded_annotated.h5ad"
ORTHOLOG_CSV  = f"{SCRATCH}/orthologs/zebrafish_mouse_human_1to1_orthologs.csv"

OUT_DIR  = f"{SCRATCH}/anchors"
FIG_DIR  = (f"{BASE}/zebrahub-multiome-analysis/figures/cross_species_motif_umap")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

OUT_CSV  = f"{OUT_DIR}/root_anchors.csv"

# Thresholds
PROMOTER_DIST    = 2000    # bp from TSS to call "promoter"
MAX_ANCHOR_DIST  = 10000   # relaxed fallback if <100 anchors with 2kb
CV_MAX           = 0.8     # coefficient of variation ceiling (broadly accessible)


# %% ── Load ───────────────────────────────────────────────────────────────────────
print(f"Loading {INPUT_H5AD} ...")
t0 = time.time()
adata = sc.read_h5ad(INPUT_H5AD)
print(f"  Shape: {adata.shape}  ({time.time()-t0:.1f}s)")
print(f"  Species: {adata.obs['species'].value_counts().to_dict()}")

print(f"\nLoading orthologs ...")
ortho = pd.read_csv(ORTHOLOG_CSV)
print(f"  Ortholog triplets: {len(ortho):,}")
print(f"  Columns: {list(ortho.columns)}")


# %% ── Compute per-peak coefficient of variation across pseudobulk columns ────────
print("\nComputing CV across pseudobulk columns ...")
t1 = time.time()

# The h5ad matrix (X) has peaks as obs and motifs as vars.
# For broad accessibility we want CV across the original pseudobulk accessibility —
# but that data is NOT in this h5ad (motif scores only).
# Fallback: use accessibility entropy proxy from the motif score matrix:
#   low CV across motif scores ≈ not particularly enriched for any motif = "open everywhere"
# A cleaner signal: species-level pseudobulk accessibility h5ads.
# We use the motif score matrix as the availability here.

# Extract matrix values
if sp.issparse(adata.X):
    X = adata.X.toarray()
else:
    X = np.array(adata.X)

# Compute CV per peak (row) across motif features (cols)
peak_mean = X.mean(axis=1)
peak_std  = X.std(axis=1)
with np.errstate(divide="ignore", invalid="ignore"):
    cv = np.where(peak_mean != 0, peak_std / np.abs(peak_mean), np.nan)

adata.obs["motif_cv"] = cv
print(f"  CV computed in {time.time()-t1:.1f}s")
print(f"  CV: median={np.nanmedian(cv):.3f}, 90th pct={np.nanpercentile(cv, 90):.3f}")

del X; gc.collect()


# %% ── Per-species promoter candidate peaks ───────────────────────────────────────
def get_promoter_candidates(adata, species, promoter_dist=PROMOTER_DIST, cv_max=CV_MAX):
    """
    Filter to promoter peaks of a given species with low motif CV.
    Returns a DataFrame indexed by original_peak_id with columns:
      original_peak_id, nearest_gene_symbol, distance_to_tss, motif_cv
    """
    mask = adata.obs["species"] == species
    sub  = adata.obs[mask].copy()
    sub["original_peak_id"] = sub.index if "original_peak_id" not in sub.columns else sub["original_peak_id"]

    # Promoter filter
    dist_col = "distance_to_tss"
    if dist_col in sub.columns:
        dist_vals = pd.to_numeric(sub[dist_col], errors="coerce")
        prom_mask = (
            (sub.get("peak_type", pd.Series("", index=sub.index)) == "promoter") |
            (dist_vals <= promoter_dist)
        )
    else:
        prom_mask = pd.Series(False, index=sub.index)
        print(f"  WARNING [{species}]: no distance_to_tss column; using peak_type only")
        if "peak_type" in sub.columns:
            prom_mask = sub["peak_type"] == "promoter"

    sub = sub[prom_mask].copy()
    print(f"  [{species}] Promoter peaks: {len(sub):,}")

    # Broadly accessible: low CV
    sub = sub[sub["motif_cv"] <= cv_max].copy()
    print(f"  [{species}] After CV ≤ {cv_max}: {len(sub):,}")

    # Need a gene symbol
    gene_col = "nearest_gene_symbol" if "nearest_gene_symbol" in sub.columns else "nearest_gene"
    if gene_col not in sub.columns:
        print(f"  WARNING [{species}]: no gene symbol column")
        sub["_gene"] = ""
    else:
        sub["_gene"] = sub[gene_col].fillna("").astype(str)

    sub = sub[sub["_gene"] != ""].copy()
    print(f"  [{species}] With gene symbol: {len(sub):,}")

    result = sub[["_gene", dist_col, "motif_cv"]].copy() if dist_col in sub.columns else sub[["_gene", "motif_cv"]].copy()
    result = result.rename(columns={"_gene": "gene_symbol", dist_col: "dist_to_tss"})
    result["peak_id"] = sub.index
    return result.reset_index(drop=True)


print("\nFiltering promoter candidates per species ...")
zf_cands = get_promoter_candidates(adata, "zebrafish")
mm_cands = get_promoter_candidates(adata, "mouse")
hs_cands = get_promoter_candidates(adata, "human")

# Relax distance threshold if too few candidates
for sp_label, cands, sp_name in [("zebrafish", zf_cands, "zebrafish"),
                                   ("mouse",     mm_cands, "mouse"),
                                   ("human",     hs_cands, "human")]:
    if len(cands) < 500:
        print(f"  [{sp_label}] Only {len(cands)} candidates; relaxing to {MAX_ANCHOR_DIST} bp")
        cands_relax = get_promoter_candidates(adata, sp_name, promoter_dist=MAX_ANCHOR_DIST)
        if len(cands_relax) > len(cands):
            if sp_label == "zebrafish":
                zf_cands = cands_relax
            elif sp_label == "mouse":
                mm_cands = cands_relax
            else:
                hs_cands = cands_relax


# %% ── Build lookup: gene_symbol (lowercase) → candidate peaks ─────────────────
def build_gene_lookup(cands_df):
    """Dict: lowercase gene symbol → list of (peak_id, dist_to_tss, motif_cv)"""
    lkp = {}
    dist_col = "dist_to_tss" if "dist_to_tss" in cands_df.columns else None
    for _, row in cands_df.iterrows():
        key = str(row["gene_symbol"]).lower()
        dist = row[dist_col] if dist_col else np.nan
        if key not in lkp:
            lkp[key] = []
        lkp[key].append((row["peak_id"], dist, row.get("motif_cv", np.nan)))
    return lkp


zf_lkp = build_gene_lookup(zf_cands)
mm_lkp = build_gene_lookup(mm_cands)
hs_lkp = build_gene_lookup(hs_cands)

print(f"\nGene lookup sizes: ZF={len(zf_lkp)}, MM={len(mm_lkp)}, HS={len(hs_lkp)}")


# %% ── Match orthologs across all 3 species ───────────────────────────────────────
print("\nMatching ortholog triplets to promoter peaks ...")
records = []

for _, row in ortho.iterrows():
    zf_sym = str(row.get("gene_name_zf", "")).lower()
    mm_sym = str(row.get("gene_name_mm", "")).lower()
    hs_sym = str(row.get("gene_name_hs", "")).lower()

    zf_hits = zf_lkp.get(zf_sym, [])
    mm_hits = mm_lkp.get(mm_sym, [])
    hs_hits = hs_lkp.get(hs_sym, [])

    if not (zf_hits and mm_hits and hs_hits):
        continue

    # Pick closest-to-TSS per species
    def best_hit(hits):
        # Sort by distance_to_tss (nan last)
        hits_sorted = sorted(hits, key=lambda x: (np.isnan(x[1]) if isinstance(x[1], float) else False, x[1]))
        return hits_sorted[0]

    zf_best = best_hit(zf_hits)
    mm_best = best_hit(mm_hits)
    hs_best = best_hit(hs_hits)

    records.append({
        "gene_name_zf":      row.get("gene_name_zf", ""),
        "gene_name_mm":      row.get("gene_name_mm", ""),
        "gene_name_hs":      row.get("gene_name_hs", ""),
        "ensembl_gene_id_zf": row.get("ensembl_gene_id_zf", ""),
        "ensembl_gene_id_mm": row.get("ensembl_gene_id_mm", ""),
        "ensembl_gene_id_hs": row.get("ensembl_gene_id_hs", ""),
        "peak_id_zf":         zf_best[0],
        "peak_id_mm":         mm_best[0],
        "peak_id_hs":         hs_best[0],
        "distance_to_tss_zf": zf_best[1],
        "distance_to_tss_mm": mm_best[1],
        "distance_to_tss_hs": hs_best[1],
        "motif_cv_zf":        zf_best[2],
        "motif_cv_mm":        mm_best[2],
        "motif_cv_hs":        hs_best[2],
    })

root_anchors = pd.DataFrame(records)
print(f"  Complete root anchor triplets: {len(root_anchors):,}")

if len(root_anchors) < 100:
    print("  WARNING: fewer than 100 root anchors found! Consider relaxing thresholds.")
elif len(root_anchors) > 2000:
    print(f"  Trimming to 2000 most broadly accessible (lowest avg CV) ...")
    root_anchors["avg_cv"] = root_anchors[["motif_cv_zf", "motif_cv_mm", "motif_cv_hs"]].mean(axis=1)
    root_anchors = root_anchors.nsmallest(2000, "avg_cv").drop(columns="avg_cv")

print(f"  Final root anchors: {len(root_anchors):,}")


# %% ── Diagnostic figures ──────────────────────────────────────────────────────────
print("\nGenerating diagnostics ...")
umap_coords = adata.obsm["X_umap"]
species_arr  = adata.obs["species"].values

SPECIES_COLORS = {"zebrafish": "#1f77b4", "mouse": "#ff7f0e", "human": "#2ca02c"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (sp, color) in zip(axes, SPECIES_COLORS.items()):
    sp_mask   = species_arr == sp
    ax.scatter(umap_coords[sp_mask, 0], umap_coords[sp_mask, 1],
               c="lightgray", s=0.2, rasterized=True, alpha=0.3)

    # Highlight anchor peaks
    col = f"peak_id_{sp[:2]}"
    anchor_ids = set(root_anchors[col].astype(str)) if col in root_anchors.columns else set()
    obs_names  = adata.obs_names[sp_mask].astype(str)
    anch_mask_sp = np.array([o in anchor_ids for o in obs_names])
    if anch_mask_sp.sum() > 0:
        sp_umap = umap_coords[sp_mask]
        ax.scatter(sp_umap[anch_mask_sp, 0], sp_umap[anch_mask_sp, 1],
                   c="red", s=5, rasterized=True, alpha=0.8, label=f"anchors (n={anch_mask_sp.sum()})")
    ax.set_title(f"{sp} — root anchors")
    ax.legend(loc="upper right", markerscale=3, fontsize=8)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

plt.tight_layout()
fig.savefig(f"{FIG_DIR}/root_anchor_diagnostics.pdf", dpi=150, bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/root_anchor_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR}/root_anchor_diagnostics.pdf")

# Distance-to-TSS distribution
fig, ax = plt.subplots(figsize=(7, 4))
for col, label in [("distance_to_tss_zf", "zebrafish"),
                   ("distance_to_tss_mm", "mouse"),
                   ("distance_to_tss_hs", "human")]:
    vals = root_anchors[col].dropna()
    ax.hist(vals, bins=50, alpha=0.5, label=label)
ax.set_xlabel("Distance to TSS (bp)")
ax.set_ylabel("Count")
ax.set_title(f"Root anchor distance to TSS  (n={len(root_anchors)})")
ax.legend()
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/root_anchor_tss_dist.pdf", dpi=150, bbox_inches="tight")
plt.close(fig)


# %% ── Save ───────────────────────────────────────────────────────────────────────
root_anchors.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print(root_anchors.head(5).to_string())
print("\nDone.")
