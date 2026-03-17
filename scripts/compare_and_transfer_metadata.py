#!/usr/bin/env python
"""
Compare metadata between existing and new peak UMAP CSV files,
then transfer metadata from existing to new.
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("Metadata Comparison and Transfer Report")
print("=" * 70)

# =============================================================================
# 1. Load both CSV files
# =============================================================================
print("\n[1] Loading CSV files...")

existing_csv = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/data/peak_umap_3d_annotated_v6_Feb2026.csv"
new_csv = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/data/peaks_concord_3d_umap_coordinates.csv"

df_existing = pd.read_csv(existing_csv, index_col=0)
df_new = pd.read_csv(new_csv)

print(f"Existing CSV: {len(df_existing)} rows")
print(f"New CSV: {len(df_new)} rows")

print(f"\nExisting CSV columns: {df_existing.columns.tolist()}")
print(f"New CSV columns: {df_new.columns.tolist()}")

# =============================================================================
# 2. Compare peak IDs
# =============================================================================
print("\n" + "=" * 70)
print("[2] Comparing Peak IDs")
print("=" * 70)

existing_peaks = set(df_existing.index)
new_peaks = set(df_new['peak_id'])

common_peaks = existing_peaks & new_peaks
only_in_existing = existing_peaks - new_peaks
only_in_new = new_peaks - existing_peaks

print(f"Common peaks: {len(common_peaks)}")
print(f"Only in existing: {len(only_in_existing)}")
print(f"Only in new: {len(only_in_new)}")

if only_in_existing:
    print(f"  Examples only in existing: {list(only_in_existing)[:5]}")
if only_in_new:
    print(f"  Examples only in new: {list(only_in_new)[:5]}")

# =============================================================================
# 3. Compare metadata for common peaks
# =============================================================================
print("\n" + "=" * 70)
print("[3] Comparing Metadata (celltype, timepoint) for Common Peaks")
print("=" * 70)

# Set index for new df for easier comparison
df_new_indexed = df_new.set_index('peak_id')

# Get common peaks as list
common_peaks_list = list(common_peaks)

# Compare celltype
celltype_match = 0
celltype_mismatch = 0
celltype_mismatch_examples = []

for peak in common_peaks_list:
    existing_ct = df_existing.loc[peak, 'celltype']
    new_ct = df_new_indexed.loc[peak, 'celltype']
    if existing_ct == new_ct:
        celltype_match += 1
    else:
        celltype_mismatch += 1
        if len(celltype_mismatch_examples) < 10:
            celltype_mismatch_examples.append({
                'peak': peak,
                'existing': existing_ct,
                'new': new_ct
            })

print(f"\nCelltype comparison:")
print(f"  Matches: {celltype_match} ({100*celltype_match/len(common_peaks_list):.2f}%)")
print(f"  Mismatches: {celltype_mismatch} ({100*celltype_mismatch/len(common_peaks_list):.2f}%)")

if celltype_mismatch_examples:
    print(f"\n  Example mismatches (first 10):")
    for ex in celltype_mismatch_examples:
        print(f"    {ex['peak']}: existing='{ex['existing']}' vs new='{ex['new']}'")

# Compare timepoint
timepoint_match = 0
timepoint_mismatch = 0
timepoint_mismatch_examples = []

for peak in common_peaks_list:
    existing_tp = df_existing.loc[peak, 'timepoint']
    new_tp = df_new_indexed.loc[peak, 'timepoint']
    if existing_tp == new_tp:
        timepoint_match += 1
    else:
        timepoint_mismatch += 1
        if len(timepoint_mismatch_examples) < 10:
            timepoint_mismatch_examples.append({
                'peak': peak,
                'existing': existing_tp,
                'new': new_tp
            })

print(f"\nTimepoint comparison:")
print(f"  Matches: {timepoint_match} ({100*timepoint_match/len(common_peaks_list):.2f}%)")
print(f"  Mismatches: {timepoint_mismatch} ({100*timepoint_mismatch/len(common_peaks_list):.2f}%)")

if timepoint_mismatch_examples:
    print(f"\n  Example mismatches (first 10):")
    for ex in timepoint_mismatch_examples:
        print(f"    {ex['peak']}: existing='{ex['existing']}' vs new='{ex['new']}'")

# =============================================================================
# 4. Transfer metadata from existing to new
# =============================================================================
print("\n" + "=" * 70)
print("[4] Transferring Metadata from Existing to New CSV")
print("=" * 70)

# Columns to transfer from existing
transfer_cols = ['celltype', 'timepoint', 'lineage', 'peak_type', 'chromosome', 'leiden_coarse', 'leiden_fine']

print(f"Columns to transfer: {transfer_cols}")

# Create a mapping from existing df
metadata_map = df_existing[transfer_cols].to_dict('index')

# Apply to new df
for col in transfer_cols:
    df_new[col + '_from_existing'] = df_new['peak_id'].map(lambda x: metadata_map.get(x, {}).get(col, np.nan))

# Replace original columns with transferred ones, keeping NaN for peaks not in existing
for col in transfer_cols:
    df_new[col] = df_new[col + '_from_existing']
    df_new.drop(col + '_from_existing', axis=1, inplace=True)

# Check how many peaks got metadata transferred
peaks_with_metadata = df_new[transfer_cols[0]].notna().sum()
peaks_without_metadata = df_new[transfer_cols[0]].isna().sum()

print(f"\nPeaks with transferred metadata: {peaks_with_metadata}")
print(f"Peaks without metadata (not in existing): {peaks_without_metadata}")

# =============================================================================
# 5. Reorder columns and save
# =============================================================================
print("\n" + "=" * 70)
print("[5] Saving Updated CSV")
print("=" * 70)

# Desired column order
col_order = [
    'peak_id', 'umap_x', 'umap_y', 'umap_z',
    'celltype', 'timepoint', 'lineage', 'peak_type', 'chromosome',
    'leiden_coarse', 'leiden_fine',
    'celltype_contrast', 'timepoint_contrast',
    'accessibility_0somites', 'accessibility_5somites', 'accessibility_10somites',
    'accessibility_15somites', 'accessibility_20somites', 'accessibility_30somites',
    'accessibility_somites'
]

# Only include columns that exist
col_order = [c for c in col_order if c in df_new.columns]
# Add any remaining columns
remaining_cols = [c for c in df_new.columns if c not in col_order]
col_order = col_order + remaining_cols

df_new = df_new[col_order]

# Save
output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/data/peaks_concord_3d_umap_coordinates_with_metadata.csv"
df_new.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")

print(f"\nFinal columns: {df_new.columns.tolist()}")
print(f"Final shape: {df_new.shape}")

# Preview
print("\nPreview (first 3 rows):")
print(df_new.head(3).to_string())

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
