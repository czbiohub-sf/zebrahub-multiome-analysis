# %% [markdown]
# # Step 09: Backfill Annotations into Cross-Species h5ad
#
# Merges three annotation sources into the 17 GB cross-species h5ad:
#
#   1. Zebrafish: joins peak_type, leiden_coarse, nearest_gene, distance_to_tss, lineage
#      from all_peaks_annotated_ct_tp.csv using original_peak_id
#   2. Mouse/human: joins nearest_gene, distance_to_tss, peak_type
#      from scripts 07 annotation CSVs
#
# Input:
#   - {SCRATCH}/cross_species_motif_embedded_continuous.h5ad  (17 GB)
#   - {BASE}/annotated_data/objects_v2/all_peaks_annotated_ct_tp.csv  (777 MB, zebrafish)
#   - {SCRATCH}/gene_annotations/mouse_peaks_gene_annotated.csv
#   - {SCRATCH}/gene_annotations/human_peaks_gene_annotated.csv
#
# Output:
#   - {SCRATCH}/cross_species_motif_embedded_annotated.h5ad
#
# Env: single-cell-base (CPU)  — 4 CPUs, 64G, 1h
#   conda run -p /hpc/user_apps/data.science/conda_envs/single-cell-base python -u 09_backfill_annotations.py

# %% Imports
import os, gc, time
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

print(f"anndata {ad.__version__}, scanpy {sc.__version__}")

# %% Paths
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"

INPUT_H5AD    = f"{SCRATCH}/cross_species_motif_embedded_continuous.h5ad"
OUTPUT_H5AD   = f"{SCRATCH}/cross_species_motif_embedded_annotated.h5ad"

ZF_ANNOT_CSV  = f"{BASE}/data/annotated_data/objects_v2/all_peaks_annotated_ct_tp.csv"
MOUSE_ANN_CSV = f"{SCRATCH}/gene_annotations/mouse_peaks_gene_annotated.csv"
HUMAN_ANN_CSV = f"{SCRATCH}/gene_annotations/human_peaks_gene_annotated.csv"


# %% ── Load cross-species h5ad ───────────────────────────────────────────────────
print(f"Loading {INPUT_H5AD} ...")
t0 = time.time()
adata = sc.read_h5ad(INPUT_H5AD)
print(f"  Shape: {adata.shape}  ({time.time()-t0:.1f}s)")
print(f"  obs columns: {list(adata.obs.columns)}")
print(f"  Species counts:\n{adata.obs['species'].value_counts()}")

# Snapshot of current annotation state
for col in ["peak_type", "leiden_coarse", "nearest_gene", "distance_to_tss"]:
    if col in adata.obs.columns:
        n_filled = adata.obs[col].notna().sum()
        print(f"  {col}: {n_filled:,} / {adata.n_obs:,} non-null")
    else:
        print(f"  {col}: MISSING")

# Ensure 'original_peak_id' exists in obs
if "original_peak_id" not in adata.obs.columns:
    print("  Adding 'original_peak_id' from obs_names ...")
    adata.obs["original_peak_id"] = adata.obs_names.astype(str)


# %% ── Zebrafish backfill ─────────────────────────────────────────────────────────
print(f"\nLoading zebrafish annotation CSV: {ZF_ANNOT_CSV}")
t1 = time.time()

# Try to detect encoding and relevant columns
zf_ann = pd.read_csv(ZF_ANNOT_CSV, low_memory=False, nrows=5)
print(f"  ZF annotation columns: {list(zf_ann.columns)}")
del zf_ann

zf_ann = pd.read_csv(ZF_ANNOT_CSV, low_memory=False)
print(f"  ZF annotation rows: {len(zf_ann):,}  ({time.time()-t1:.1f}s)")

# ── Determine the join key ────────────────────────────────────────────────────
# The zebrafish obs index in the cross-species h5ad is typically "1-32-526"
# (no chr prefix). The annotation CSV may use the same format or have a 'peak_id' column.
zf_mask = adata.obs["species"] == "zebrafish"
sample_idx = adata.obs_names[zf_mask][:5].tolist()
print(f"  Sample ZF obs_names: {sample_idx}")
print(f"  ZF ann columns sample: {list(zf_ann.columns[:10])}")

# Try to find the correct key column in annotation CSV
candidate_keys = ["peak_id", "original_peak_id", "Unnamed: 0", "index"]
join_key = None
for ck in candidate_keys:
    if ck in zf_ann.columns:
        sample_vals = zf_ann[ck].head(3).tolist()
        print(f"  Candidate key '{ck}': sample = {sample_vals}")
        # Check if any cross over with sample obs_names
        test_set = set(zf_ann[ck].head(1000).astype(str))
        if any(s in test_set for s in sample_idx):
            join_key = ck
            print(f"  → Using join key: '{ck}'")
            break

if join_key is None:
    # Fallback: use first column
    join_key = zf_ann.columns[0]
    print(f"  WARNING: No matching key found; using first column '{join_key}' as fallback")

# ── Columns to backfill ────────────────────────────────────────────────────────
zf_backfill_cols = []
for c in ["peak_type", "leiden_coarse", "nearest_gene", "distance_to_tss", "lineage",
          "celltype", "timepoint"]:
    if c in zf_ann.columns:
        zf_backfill_cols.append(c)
print(f"  ZF backfill columns available: {zf_backfill_cols}")

# ── Build lookup dict keyed by peak_id ────────────────────────────────────────
zf_ann_indexed = zf_ann.set_index(join_key)
zf_obs = adata.obs[zf_mask].copy()
zf_obs_idx = zf_obs["original_peak_id"].astype(str)

matched = zf_obs_idx.isin(zf_ann_indexed.index)
print(f"  ZF peaks matching annotation: {matched.sum():,} / {len(zf_obs):,}")

NUMERIC_COLS = {"distance_to_tss"}
for col in zf_backfill_cols:
    new_col = f"zf_{col}" if col in ["celltype", "timepoint"] else col
    if new_col not in adata.obs.columns:
        # Initialize with correct dtype to avoid pandas incompatible-dtype error
        if col in NUMERIC_COLS:
            adata.obs[new_col] = np.nan
        else:
            adata.obs[new_col] = ""

    vals = zf_obs_idx.map(zf_ann_indexed[col].to_dict())
    if col in NUMERIC_COLS:
        adata.obs.loc[zf_mask, new_col] = pd.to_numeric(vals.values, errors="coerce")
    else:
        adata.obs.loc[zf_mask, new_col] = vals.fillna("").astype(str).values

print(f"  ZF backfill complete.")
del zf_ann, zf_obs; gc.collect()


# %% ── Mouse backfill ─────────────────────────────────────────────────────────────
print(f"\nLoading mouse annotation: {MOUSE_ANN_CSV}")
t2 = time.time()
mm_ann = pd.read_csv(MOUSE_ANN_CSV, low_memory=False)
print(f"  Mouse annotation rows: {len(mm_ann):,}  ({time.time()-t2:.1f}s)")
print(f"  Mouse ann columns: {list(mm_ann.columns)}")

mm_mask = adata.obs["species"] == "mouse"
mm_obs  = adata.obs[mm_mask].copy()

# The mouse obs index is "chr1-3035602-3036202"
# Mouse annotation CSV original_peak_id column should match
sample_mm_idx = adata.obs_names[mm_mask][:3].tolist()
print(f"  Sample mouse obs_names: {sample_mm_idx}")

# Check column alignment
if "original_peak_id" in mm_ann.columns:
    mm_ann_idx = mm_ann.set_index("original_peak_id")
else:
    # Build from chr/start/end columns
    mm_ann["original_peak_id"] = (mm_ann["chr"] + "-" +
                                   mm_ann["start"].astype(str) + "-" +
                                   mm_ann["end"].astype(str))
    mm_ann_idx = mm_ann.set_index("original_peak_id")

mm_obs_idx = mm_obs["original_peak_id"].astype(str)
matched_mm = mm_obs_idx.isin(mm_ann_idx.index)
print(f"  Mouse peaks matching annotation: {matched_mm.sum():,} / {len(mm_obs):,}")

for col in ["nearest_gene", "distance_to_tss", "peak_type", "gene_body_overlaps"]:
    if col in mm_ann_idx.columns:
        if col not in adata.obs.columns:
            adata.obs[col] = np.nan if col == "distance_to_tss" else ""
        vals = mm_obs_idx.map(mm_ann_idx[col].to_dict())
        if col == "distance_to_tss":
            adata.obs.loc[mm_mask, col] = pd.to_numeric(vals.values, errors="coerce")
        else:
            adata.obs.loc[mm_mask, col] = vals.fillna("").astype(str).values

print("  Mouse backfill complete.")
del mm_ann, mm_obs; gc.collect()


# %% ── Human backfill ─────────────────────────────────────────────────────────────
print(f"\nLoading human annotation: {HUMAN_ANN_CSV}")
t3 = time.time()
hs_ann = pd.read_csv(HUMAN_ANN_CSV, low_memory=False)
print(f"  Human annotation rows: {len(hs_ann):,}  ({time.time()-t3:.1f}s)")

hs_mask = adata.obs["species"] == "human"
hs_obs  = adata.obs[hs_mask].copy()

sample_hs_idx = adata.obs_names[hs_mask][:3].tolist()
print(f"  Sample human obs_names: {sample_hs_idx}")

if "original_peak_id" in hs_ann.columns:
    hs_ann_idx = hs_ann.set_index("original_peak_id")
else:
    hs_ann["original_peak_id"] = (hs_ann["chr"] + "-" +
                                   hs_ann["start"].astype(str) + "-" +
                                   hs_ann["end"].astype(str))
    hs_ann_idx = hs_ann.set_index("original_peak_id")

hs_obs_idx = hs_obs["original_peak_id"].astype(str)
matched_hs = hs_obs_idx.isin(hs_ann_idx.index)
print(f"  Human peaks matching annotation: {matched_hs.sum():,} / {len(hs_obs):,}")

for col in ["nearest_gene", "distance_to_tss", "peak_type", "gene_body_overlaps"]:
    if col in hs_ann_idx.columns:
        if col not in adata.obs.columns:
            adata.obs[col] = np.nan if col == "distance_to_tss" else ""
        vals = hs_obs_idx.map(hs_ann_idx[col].to_dict())
        if col == "distance_to_tss":
            adata.obs.loc[hs_mask, col] = pd.to_numeric(vals.values, errors="coerce")
        else:
            adata.obs.loc[hs_mask, col] = vals.fillna("").astype(str).values

print("  Human backfill complete.")
del hs_ann, hs_obs; gc.collect()


# %% ── Add unified nearest_gene_symbol column ────────────────────────────────────
# For ortholog matching downstream: lowercase for ZF, TitleCase for mouse,
# UPPERCASE for human — keep as-is; case-insensitive matching done in scripts 10/11.
print("\nAdding unified 'nearest_gene_symbol' column ...")
if "nearest_gene" in adata.obs.columns:
    adata.obs["nearest_gene_symbol"] = adata.obs["nearest_gene"].fillna("").astype(str)
    print(f"  Non-empty nearest_gene_symbol: {(adata.obs['nearest_gene_symbol'] != '').sum():,}")


# %% ── Summary ────────────────────────────────────────────────────────────────────
print("\nAnnotation summary after backfill:")
for col in ["peak_type", "nearest_gene", "distance_to_tss", "leiden_coarse", "lineage"]:
    if col in adata.obs.columns:
        n = adata.obs[col].notna().sum()
        n_empty = (adata.obs[col].astype(str) == "nan").sum()
        print(f"  {col}: {n:,} non-null  (empty string NaN: {n_empty:,})")

if "peak_type" in adata.obs.columns:
    print(f"\n  peak_type distribution:\n{adata.obs['peak_type'].value_counts()}")


# %% ── Save annotated h5ad ────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_H5AD} ...")
t_save = time.time()
adata.write_h5ad(OUTPUT_H5AD, compression="gzip")
print(f"  Saved in {time.time()-t_save:.1f}s")

print("\nDone.")
