#!/usr/bin/env python
"""
Compute 3D UMAP from CONCORD embeddings and export coordinates to CSV.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc

# CONCORD imports
from concord.utils.dim_reduction import run_umap

print("=" * 60)
print("CONCORD 3D UMAP Export")
print("=" * 60)

# =============================================================================
# 1. Load the data with CONCORD embeddings
# =============================================================================
print("\n[1/3] Loading data with CONCORD embeddings...")
peaks_pb = sc.read_h5ad(
    "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/"
    "objects_v2/peaks_by_ct_tp_pseudobulked_all_peaks_pca_concord.h5ad"
)
print(f"Loaded {peaks_pb.shape[0]} peaks x {peaks_pb.shape[1]} pseudobulk groups")

# Check available metadata
print("\nAvailable metadata columns in obs:")
print(peaks_pb.obs.columns.tolist())

# =============================================================================
# 2. Compute 3D UMAP from CONCORD embeddings
# =============================================================================
print("\n[2/3] Computing 3D UMAP from CONCORD embeddings...")

# Run 3D UMAP with cosine distance (important for CONCORD L2-normalized embeddings)
run_umap(
    peaks_pb,
    source_key="X_concord",
    result_key="X_umap_concord_3d",
    n_neighbors=15,
    min_dist=0.3,
    metric="cosine",
    n_components=3,
    random_state=42,
)

print("3D UMAP complete.")
print(f"Shape of 3D UMAP: {peaks_pb.obsm['X_umap_concord_3d'].shape}")

# =============================================================================
# 3. Export to CSV
# =============================================================================
print("\n[3/3] Exporting to CSV...")

# Get 3D coordinates
umap_3d = peaks_pb.obsm["X_umap_concord_3d"]

# Create DataFrame with coordinates
df = pd.DataFrame({
    "peak_id": peaks_pb.obs_names,
    "umap_x": umap_3d[:, 0],
    "umap_y": umap_3d[:, 1],
    "umap_z": umap_3d[:, 2],
})

# Add metadata columns
metadata_cols = ["celltype", "timepoint", "celltype_contrast", "timepoint_contrast"]
for col in metadata_cols:
    if col in peaks_pb.obs.columns:
        df[col] = peaks_pb.obs[col].values

# Add accessibility columns for timepoints
timepoint_cols = [c for c in peaks_pb.obs.columns if c.startswith("accessibility_") and "somites" in c]
for col in timepoint_cols:
    df[col] = peaks_pb.obs[col].values

# Save to CSV
output_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/data/"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "peaks_concord_3d_umap_coordinates.csv")
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")

# Also save the updated h5ad with 3D UMAP
peaks_pb.write_h5ad(
    "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/"
    "objects_v2/peaks_by_ct_tp_pseudobulked_all_peaks_pca_concord.h5ad"
)
print("Updated h5ad saved with 3D UMAP embeddings.")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
print(f"\nCSV columns: {df.columns.tolist()}")
print(f"Total rows: {len(df)}")
