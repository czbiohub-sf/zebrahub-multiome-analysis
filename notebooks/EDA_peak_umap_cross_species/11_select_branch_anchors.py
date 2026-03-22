# %% [markdown]
# # Step 11: Select Branch Anchors
#
# Branch anchors = lineage-specific marker peaks near 1:1:1 orthologous
# lineage-marker genes, present in ≥2 of 3 species in matching lineages.
#
# Algorithm:
#   1. Define cross-species lineage mapping (manual dictionary)
#   2. Per species: identify lineage-specific peaks using accessibility score CV
#      (high CV = lineage-specific; low CV = broadly accessible)
#      + only peaks with dist_to_tss within 50kb
#   3. Cross-reference nearest_gene with ortholog triplet table
#   4. Keep genes that are lineage-specific in ≥2 species for matched lineages
#   5. Exclude root anchors
#
# Input:
#   - {SCRATCH}/cross_species_motif_embedded_annotated.h5ad
#   - {SCRATCH}/orthologs/zebrafish_mouse_human_1to1_orthologs.csv
#   - {SCRATCH}/anchors/root_anchors.csv
#
# Output:
#   - {SCRATCH}/anchors/branch_anchors.csv
#   - {SCRATCH}/anchors/lineage_mapping.csv
#   - figures/.../branch_anchor_diagnostics.pdf
#
# Env: single-cell-base (CPU)  — 4 CPUs, 64G, 1h
#   conda run -p /hpc/user_apps/data.science/conda_envs/single-cell-base python -u 11_select_branch_anchors.py

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
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"

INPUT_H5AD   = f"{SCRATCH}/cross_species_motif_embedded_annotated.h5ad"
ORTHOLOG_CSV = f"{SCRATCH}/orthologs/zebrafish_mouse_human_1to1_orthologs.csv"
ROOT_CSV     = f"{SCRATCH}/anchors/root_anchors.csv"

OUT_DIR  = f"{SCRATCH}/anchors"
FIG_DIR  = (f"{BASE}/zebrahub-multiome-analysis/figures/cross_species_motif_umap")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

OUT_BRANCH_CSV  = f"{OUT_DIR}/branch_anchors.csv"
OUT_LINEAGE_CSV = f"{OUT_DIR}/lineage_mapping.csv"

MAX_TSS_DIST   = 50_000    # bp
CV_LOW_CUTOFF  = 0.5       # peaks with CV > this are "lineage-specific"
MIN_SPECIES    = 2         # gene must be lineage-specific in ≥ this many species


# %% ── Cross-species lineage mapping ─────────────────────────────────────────────
# Manual mapping: cross-species lineage group → per-species cell type labels
# The cell type labels come from obs["top_celltype"] / obs["celltype"] columns.
# These are fuzzy — we do substring / case-insensitive matching.

LINEAGE_MAP = {
    "neural_cns": {
        "zebrafish": ["neural", "cns", "brain", "spinal", "retina", "optic"],
        "mouse":     ["neural", "ectoderm", "brain", "neuroectoderm", "spinal"],
        "human":     ["neural", "brain", "neuroectoderm", "CNS"],
    },
    "paraxial_mesoderm": {
        "zebrafish": ["paraxial", "somite", "muscle", "myotome"],
        "mouse":     ["mesoderm", "paraxial", "somite", "muscle"],
        "human":     ["mesoderm", "paraxial", "somite", "muscle"],
    },
    "lateral_mesoderm": {
        "zebrafish": ["lateral", "heart", "cardiac", "blood", "hematopoietic"],
        "mouse":     ["lateral", "heart", "cardiac", "blood", "hematopoietic"],
        "human":     ["lateral", "heart", "cardiac", "blood", "hematopoietic"],
    },
    "endoderm": {
        "zebrafish": ["endoderm", "gut", "liver", "pharyngeal"],
        "mouse":     ["endoderm", "gut", "liver", "primitive_gut"],
        "human":     ["endoderm", "gut", "liver"],
    },
    "ectoderm": {
        "zebrafish": ["ectoderm", "epidermis", "skin", "periderm"],
        "mouse":     ["ectoderm", "epidermis", "skin", "surface"],
        "human":     ["ectoderm", "epidermis", "skin"],
    },
    "neural_crest": {
        "zebrafish": ["neural crest", "crest", "craniofacial"],
        "mouse":     ["neural crest", "crest"],
        "human":     ["neural crest", "crest"],
    },
}

# Save lineage mapping
lineage_records = []
for lineage, sp_map in LINEAGE_MAP.items():
    for sp, terms in sp_map.items():
        lineage_records.append({"lineage": lineage, "species": sp, "keywords": "|".join(terms)})
lineage_df = pd.DataFrame(lineage_records)
lineage_df.to_csv(OUT_LINEAGE_CSV, index=False)
print(f"Lineage mapping saved: {OUT_LINEAGE_CSV}")
print(lineage_df.to_string())


# %% ── Load data ─────────────────────────────────────────────────────────────────
print(f"\nLoading {INPUT_H5AD} ...")
t0 = time.time()
adata = sc.read_h5ad(INPUT_H5AD)
print(f"  Shape: {adata.shape}  ({time.time()-t0:.1f}s)")
print(f"  obs columns: {list(adata.obs.columns)}")

ortho     = pd.read_csv(ORTHOLOG_CSV)
root_anch = pd.read_csv(ROOT_CSV)
root_peak_ids = set(
    root_anch["peak_id_zf"].astype(str).tolist() +
    root_anch["peak_id_mm"].astype(str).tolist() +
    root_anch["peak_id_hs"].astype(str).tolist()
)
print(f"  Orthologs: {len(ortho):,}, root anchors: {len(root_anch):,}")


# %% ── Compute motif CV (broadly accessible proxy) ───────────────────────────────
print("\nComputing motif CV ...")
t1 = time.time()
if sp.issparse(adata.X):
    X_arr = adata.X.toarray()
else:
    X_arr = np.array(adata.X)

peak_mean = X_arr.mean(axis=1)
peak_std  = X_arr.std(axis=1)
with np.errstate(divide="ignore", invalid="ignore"):
    cv = np.where(peak_mean != 0, peak_std / np.abs(peak_mean), np.nan)

adata.obs["motif_cv"] = cv
del X_arr; gc.collect()
print(f"  Done in {time.time()-t1:.1f}s")


# %% ── Helper: assign lineage label to each peak ─────────────────────────────────
def get_celltype_col(obs, species):
    for col in ["celltype", "top_celltype", "zf_celltype", "lineage"]:
        if col in obs.columns:
            return col
    return None


def assign_lineage(obs_sub, species, lineage_map):
    """
    Returns a Series (indexed like obs_sub) with cross-species lineage label.
    Peaks not matching any lineage get label 'other'.
    """
    ct_col = get_celltype_col(obs_sub, species)
    if ct_col is None:
        return pd.Series("other", index=obs_sub.index)

    ct_values = obs_sub[ct_col].fillna("").str.lower()
    lineage_labels = pd.Series("other", index=obs_sub.index)

    for lineage, sp_map in lineage_map.items():
        keywords = [k.lower() for k in sp_map.get(species, [])]
        match = ct_values.apply(lambda v: any(k in v for k in keywords))
        lineage_labels[match] = lineage

    return lineage_labels


# %% ── Per-species: get lineage-specific candidate peaks ─────────────────────────
def get_branch_candidates(adata, species, lineage_map,
                           max_tss_dist=MAX_TSS_DIST, cv_low=CV_LOW_CUTOFF):
    """
    Returns DataFrame: peak_id, gene_symbol, lineage, dist_to_tss, motif_cv
    filtered to:
      - within max_tss_dist of a TSS
      - lineage-specific (motif_cv > cv_low)
      - not 'other' lineage
      - has a gene symbol
    """
    mask = adata.obs["species"] == species
    sub  = adata.obs[mask].copy()

    dist_col = "distance_to_tss"
    if dist_col in sub.columns:
        dist_vals = pd.to_numeric(sub[dist_col], errors="coerce")
        dist_mask = dist_vals <= max_tss_dist
    else:
        dist_mask = pd.Series(True, index=sub.index)  # no filter if missing

    sub = sub[dist_mask].copy()
    print(f"  [{species}] Within {max_tss_dist} bp TSS: {len(sub):,}")

    # Lineage-specific: high CV
    sub = sub[sub["motif_cv"] > cv_low].copy()
    print(f"  [{species}] CV > {cv_low}: {len(sub):,}")

    # Assign lineage
    sub["lineage_label"] = assign_lineage(sub, species, lineage_map)
    sub = sub[sub["lineage_label"] != "other"].copy()
    print(f"  [{species}] With lineage label: {len(sub):,}")
    if len(sub) > 0:
        print(f"           Lineage distribution:\n{sub['lineage_label'].value_counts().to_dict()}")

    # Need gene symbol
    gene_col = "nearest_gene_symbol" if "nearest_gene_symbol" in sub.columns else "nearest_gene"
    if gene_col not in sub.columns:
        sub["_gene"] = ""
    else:
        sub["_gene"] = sub[gene_col].fillna("").astype(str)

    sub = sub[sub["_gene"] != ""].copy()
    print(f"  [{species}] With gene symbol: {len(sub):,}")

    result = pd.DataFrame({
        "peak_id":     sub.index,
        "gene_symbol": sub["_gene"].values,
        "lineage":     sub["lineage_label"].values,
        "dist_to_tss": pd.to_numeric(sub.get(dist_col, pd.Series(np.nan, index=sub.index)), errors="coerce").values,
        "motif_cv":    sub["motif_cv"].values,
    })
    return result.reset_index(drop=True)


print("\nGathering branch candidates per species ...")
zf_branch = get_branch_candidates(adata, "zebrafish", LINEAGE_MAP)
mm_branch = get_branch_candidates(adata, "mouse",     LINEAGE_MAP)
hs_branch = get_branch_candidates(adata, "human",     LINEAGE_MAP)


# %% ── Build lookup: (gene_symbol_lower, lineage) → (peak_id, dist, cv) ──────────
def build_branch_lookup(cands_df):
    lkp = {}
    for _, row in cands_df.iterrows():
        key = (str(row["gene_symbol"]).lower(), row["lineage"])
        entry = (row["peak_id"], row.get("dist_to_tss", np.nan), row.get("motif_cv", np.nan))
        lkp.setdefault(key, []).append(entry)
    return lkp


zf_lkp = build_branch_lookup(zf_branch)
mm_lkp = build_branch_lookup(mm_branch)
hs_lkp = build_branch_lookup(hs_branch)

print(f"\nBranch lookups: ZF={len(zf_lkp)}, MM={len(mm_lkp)}, HS={len(hs_lkp)}")


# %% ── Match ortholog triplets across lineages ────────────────────────────────────
def best_hit(hits):
    hits_sorted = sorted(hits, key=lambda x: (np.isnan(x[1]) if isinstance(x[1], float) else False, x[1]))
    return hits_sorted[0]


print("\nMatching branch ortholog triplets ...")
records = []

for lineage in LINEAGE_MAP.keys():
    n_before = len(records)
    for _, row in ortho.iterrows():
        zf_sym = str(row.get("gene_name_zf", "")).lower()
        mm_sym = str(row.get("gene_name_mm", "")).lower()
        hs_sym = str(row.get("gene_name_hs", "")).lower()

        zf_hits = zf_lkp.get((zf_sym, lineage), [])
        mm_hits = mm_lkp.get((mm_sym, lineage), [])
        hs_hits = hs_lkp.get((hs_sym, lineage), [])

        n_present = sum([len(zf_hits) > 0, len(mm_hits) > 0, len(hs_hits) > 0])
        if n_present < MIN_SPECIES:
            continue

        # Use available species; fill missing with None
        zf_best = best_hit(zf_hits) if zf_hits else (None, np.nan, np.nan)
        mm_best = best_hit(mm_hits) if mm_hits else (None, np.nan, np.nan)
        hs_best = best_hit(hs_hits) if hs_hits else (None, np.nan, np.nan)

        # Must have at least the dominant two species
        filled = [b[0] for b in [zf_best, mm_best, hs_best] if b[0] is not None]
        if len(filled) < MIN_SPECIES:
            continue

        # Exclude if any peak is a root anchor
        if any(str(b[0]) in root_peak_ids for b in [zf_best, mm_best, hs_best] if b[0]):
            continue

        records.append({
            "lineage":           lineage,
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
    print(f"  {lineage}: +{len(records)-n_before} triplets")

branch_anchors = pd.DataFrame(records)
print(f"\nTotal branch anchor triplets: {len(branch_anchors):,}")

if len(branch_anchors) < 200:
    print("  WARNING: fewer than 200 branch anchors.")
elif len(branch_anchors) > 2000:
    # Trim: keep highest-CV (most lineage-specific) per lineage
    print("  Trimming to 2000 most lineage-specific ...")
    branch_anchors["avg_cv"] = branch_anchors[["motif_cv_zf", "motif_cv_mm", "motif_cv_hs"]].mean(axis=1)
    branch_anchors = (branch_anchors
                      .sort_values("avg_cv", ascending=False)
                      .drop_duplicates(subset=["peak_id_zf", "peak_id_mm", "peak_id_hs"])
                      .head(2000)
                      .drop(columns="avg_cv"))

print(f"\nBranch anchors per lineage:\n{branch_anchors['lineage'].value_counts().to_string()}")


# %% ── Diagnostic figures ──────────────────────────────────────────────────────────
print("\nGenerating diagnostics ...")
umap_coords = adata.obsm["X_umap"]
species_arr  = adata.obs["species"].values

SPECIES_COLORS   = {"zebrafish": "#1f77b4", "mouse": "#ff7f0e", "human": "#2ca02c"}
LINEAGE_COLORS   = {
    "neural_cns":       "#9467bd",
    "paraxial_mesoderm":"#d62728",
    "lateral_mesoderm": "#e377c2",
    "endoderm":         "#8c564b",
    "ectoderm":         "#bcbd22",
    "neural_crest":     "#17becf",
}

for sp, sp_color in SPECIES_COLORS.items():
    sp_mask = species_arr == sp
    sp_umap = umap_coords[sp_mask]
    sp_names = adata.obs_names[sp_mask].astype(str)
    sp_name_set = set(sp_names)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sp_umap[:, 0], sp_umap[:, 1],
               c="lightgray", s=0.2, rasterized=True, alpha=0.2)

    col = f"peak_id_{sp[:2]}"
    if col in branch_anchors.columns:
        for lineage, l_color in LINEAGE_COLORS.items():
            sub = branch_anchors[branch_anchors["lineage"] == lineage]
            anchor_ids = set(sub[col].dropna().astype(str))
            anch_mask = np.array([n in anchor_ids for n in sp_names])
            if anch_mask.sum() > 0:
                ax.scatter(sp_umap[anch_mask, 0], sp_umap[anch_mask, 1],
                           c=l_color, s=8, rasterized=True, alpha=0.9,
                           label=f"{lineage} (n={anch_mask.sum()})")

    ax.set_title(f"{sp} — branch anchors")
    ax.legend(loc="upper right", markerscale=2, fontsize=7, ncol=1)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/branch_anchor_diagnostics_{sp}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/branch_anchor_diagnostics_{sp}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved diagnostics for {sp}")


# %% ── Save ───────────────────────────────────────────────────────────────────────
branch_anchors.to_csv(OUT_BRANCH_CSV, index=False)
print(f"\nSaved: {OUT_BRANCH_CSV}")
print(branch_anchors.head(5).to_string())
print("\nDone.")
