# Peak Parts List — Web Portal Handoff

## Overview

The "Peak Parts List" is an interactive page on the Zebrahub web portal allowing users to explore ~640K ATAC-seq peaks across 32 cell types × 6 developmental timepoints. Users can query celltype-specific regulatory elements, inspect their accessibility profiles, and download DNA sequences for synthetic enhancer design.

---

## User Flow

```
1. Landing page
   └─ Peak UMAP colored by most accessible celltype (640K peaks)
   └─ Tau specificity histogram (what fraction of peaks are celltype-specific?)

2. User selects a celltype (dropdown, 31 options excluding PGCs)
   └─ Shows top-200 peaks ranked by V3 z-score as a table:
      │  chr:start-end │ z-score │ linked_gene │ peak_type │ distance_to_tss │ ...
   └─ Peak UMAP highlights the 50 selected peaks

3. User clicks a peak row
   └─ Panel A: Celltype accessibility bar chart (31 celltypes)
   └─ Panel B: Temporal accessibility bar chart (6 timepoints within selected celltype)

4. User selects multiple peaks (checkbox)
   └─ Download as: BED file, FASTA sequences, or CSV with metadata
```

---

## Data Sources on HPC

All paths relative to:
```
BASE=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome
REPO=${BASE}/zebrahub-multiome-analysis
```

### Primary Data Files

| File | Path | Size | Description |
|------|------|------|-------------|
| **Master pseudobulk h5ad** | `${BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad` | 4.4 GB | 640,830 peaks × 190 conditions. This is THE source of truth. |
| **V3 z-score matrix** | `${REPO}/notebooks/EDA_peak_parts_list/outputs/V3/V3_specificity_matrix_celltype_level.h5ad` | 378 MB | 640,830 peaks × 31 celltypes. Celltype-level specificity z-scores. |
| **Tau specificity metrics** | `${REPO}/notebooks/EDA_peak_parts_list/outputs/V3/V3_peak_specificity_metrics.csv` | 48 MB | Tau index + Gini coefficient for all 640K peaks. |
| **Top peaks table (all 31)** | `${REPO}/notebooks/EDA_peak_parts_list/outputs/V3/V3_all_celltypes_top200_peaks.csv` | ~750 KB | Top-200 peaks for ALL 31 celltypes (6,200 rows) with metadata: coords, z-score, linked/nearest gene, peak_type, etc. **This is the primary table for the portal.** The portal can filter to top-50/100 dynamically using the `rank` column, or apply a `V3_zscore` threshold via a user-adjustable slider. |
| Top peaks table (focal 7) | `${REPO}/notebooks/EDA_peak_parts_list/outputs/V3/V3_celltype_level_top_peaks.csv` | 78 KB | Top-50 peaks for 7 focal celltypes only (legacy, used by motif enrichment scripts). |
| **danRer11 FASTA** | `/hpc/reference/sequencing_alignment/fasta_references/danRer11.primary.fa` | 1.4 GB | Reference genome for sequence extraction. |

### FIMO Motif Data (on scratch — portal-ready)

Precomputed FIMO scan of all 1,443 JASPAR H12CORE motifs against top-200 peaks for all 31 celltypes (6,200 peaks total). Run as a SLURM array (1 task per celltype), then merged with Fisher's exact test + FDR correction.

All files at: `/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/`

| File | Full Path | Description |
|------|-----------|-------------|
| **Portal table with motifs** | `/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/V3_all_celltypes_top200_peaks_with_motifs.csv` | Top-200 table (6,200 rows) with added columns: `n_tfs_with_hit`, `n_total_hits`, `top_tfs`, `has_motif_support` (bool, >=3 TFs). **Use this as the primary portal table.** |
| **Motif enrichment** | `/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/V3_top200_motif_enrichment_all31.csv` | Fisher's exact test results: enrichment z-score, FDR, hit rates, odds ratio for all 31 celltypes × TFs. Background = all other celltypes' 200 peaks pooled. |
| **Motif positions** | `/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/V3_top200_motif_positions.csv` | Every motif hit with exact position within the peak: `peak_id, tf, motif_accession, hit_start, hit_end, strand, score, pvalue`. Use to render motif position maps when user clicks a peak. |
| **Hit matrix** | `/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/V3_top200_motif_hit_matrix.csv` | Boolean matrix (6,200 peaks × ~949 unique TFs). Use for filtering peaks by TF presence. |
| **Per-peak summary** | `/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/V3_top200_peak_motif_summary.csv` | Per-peak: `n_tfs_with_hit`, `n_total_hits`, `top_tfs` (top 5 TFs with hit counts), `has_motif_support`. |

Per-celltype batch files (intermediate): `/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/batches/{celltype}_hits.csv` and `{celltype}_binary.npz`.

### Temporal Profiles (all 31 celltypes × top-200)

| File | Full Path | Description |
|------|-----------|-------------|
| **Temporal profiles (all 31)** | `${REPO}/notebooks/EDA_peak_parts_list/outputs/V3/V3_all_celltypes_top200_temporal_profiles.csv` | 37,200 rows (31 celltypes × 200 peaks × 6 timepoints). Columns: `celltype, peak_id, rank, V3_zscore, linked_gene, nearest_gene, timepoint, tp_int, accessibility, n_cells, reliable`. Use to render temporal bar charts when user clicks a peak. The `reliable` column (bool) flags timepoints with >=20 cells — grey out or hide unreliable timepoints in the UI. |
| Temporal profiles (focal 7, legacy) | `${REPO}/notebooks/EDA_peak_parts_list/outputs/V3/V3_celltype_level_temporal_profiles.csv` | 1,900 rows — 7 focal celltypes × top-50 only (superseded by the all-31 version above). |

### Precomputed Figures

| Directory | Full Path | Description |
|-----------|-----------|-------------|
| **Peak UMAPs (V2)** | `${REPO}/figures/peak_parts_list/V3/peak_umap/all_celltypes_V2/` | 248 files: all 31 celltypes × top-50/top-200 × labeled/nolabel × pdf/png. Flat color, no grid, no colorbar. Filename pattern: `{celltype}_top{50,200}_umap{,_nolabel}.{pdf,png}` |
| **Motif position maps (V2)** | `${REPO}/figures/peak_parts_list/V3/motif_position_maps_V2_top200/` | 139 maps across 28 celltypes (top 5 peaks each). Shows exact TF binding site positions within each peak, filtered to FDR < 0.05 enriched TFs only. Subdirectories per celltype. |
| **Motif enrichment heatmaps (top-200)** | `${REPO}/figures/peak_parts_list/V3/motif_enrichment/V3_interesting6_motif_*_top200.{pdf,png}` | Heatmaps for the 6 interesting celltypes computed from top-200 peaks (vs top-50 in the originals). Three versions: `zscore_hitrate_top200` (A+B panels), `log2fold_top200`, `heatmap_colorbar_top200`. More robust enrichment with 4× larger sample. |

### Supplementary Data Files

| File | Path | Size | Description |
|------|------|------|-------------|
| Motif enrichment (top-50, legacy) | `outputs/V3/V3_all31_motif_enrichment.csv` | 4.2 MB | Fisher's exact test from top-50 peaks (legacy, superseded by top-200 on scratch) |
| Motif enrichment (interesting 6) | `outputs/V3/V3_interesting6_motif_enrichment.csv` | 852 KB | Subset for 6 highlighted celltypes |
| RNA pseudobulk | `outputs/V3/rna_by_ct_tp_pseudobulked.h5ad` | 40 MB | 32,057 genes × 176 conditions |
| RNA z-scores | `outputs/V3/rna_specificity_matrix_celltype_level.h5ad` | 6.7 MB | 32,057 genes × 31 celltypes |

---

## Master h5ad Structure

### Shape: (640,830 peaks, 190 conditions)

### `.X` — log-normalized accessibility matrix
- Processing: raw counts → pseudobulk sum per (celltype × timepoint) → median-scale normalization → log1p
- Shape: (640830, 190), float64
- Value range: ~0–350 (log-norm scale)

### `.layers`
| Layer | Description |
|-------|-------------|
| `sum` | Raw pseudobulk counts (sum across cells per group) |
| `normalized` | Median-scaled counts (before log) |
| `log_norm` | Same as `.X` — log1p of normalized |

### `.obs` — Per-peak metadata (640,830 rows)

**Index format:** `{chrom}-{start}-{end}` (e.g., `1-32-526` — note: no `chr` prefix)

| Column | Type | Description | Portal use |
|--------|------|-------------|------------|
| `chrom` | Categorical | Chromosome number (no `chr` prefix) | Display as `chr{chrom}` |
| `start` | int | Peak start coordinate (danRer11) | BED export |
| `end` | int | Peak end coordinate | BED export |
| `peak_type` | Categorical | `Promoter`, `Exonic`, `Intronic`, `Intergenic` | Table column, filter |
| `nearest_gene` | Categorical | Nearest gene by TSS distance | Table column |
| `distance_to_tss` | float | Distance to nearest TSS (bp) | Table column |
| `linked_gene` | Categorical | Co-accessibility-linked gene (Cicero) | Table column (preferred over nearest) |
| `link_score` | float | Co-accessibility score | Tooltip |
| `associated_gene` | Categorical | Best gene: `linked_gene` if available, else `nearest_gene` | Primary display |
| `association_type` | Categorical | `linked` or `nearest` | Badge/icon |
| `celltype` | Categorical | Most accessible celltype (original annotation) | Filter |
| `celltype_contrast` | float | Original celltype contrast score | — |
| `timepoint` | Categorical | Most accessible timepoint | Filter |
| `leiden_coarse` | Categorical | Peak UMAP cluster assignment | Cluster view |
| `length` | int | Peak width in bp | Table column |

### `.var` — Per-condition metadata (190 columns)

| Column | Type | Description |
|--------|------|-------------|
| `annotation_ML_coarse` | str | Celltype name |
| `dev_stage` | str | Timepoint (e.g., `15somites`) |
| `n_cells` | int | Number of cells in this group |
| `scale_factor` | float | Median-scale normalization factor |

**Condition naming:** `{celltype}_{timepoint}` (e.g., `heart_myocardium_20somites`)

**Parsing:** `re.search(r'(\d+somites)$', condition_name)` to split celltype from timepoint.

**Reliable conditions:** `n_cells >= 20` (14 conditions are unreliable, all PGC timepoints)

### `.obsm` — Embeddings

| Key | Shape | Description |
|-----|-------|-------------|
| `X_umap_2D` | (640830, 2) | 2D UMAP coordinates for peak UMAP |
| `X_umap_3D` | (640830, 3) | 3D UMAP coordinates |
| `X_pca` | (640830, 50) | PCA embeddings |

---

## V3 Z-Score Matrix Structure

### Shape: (640,830 peaks, 31 celltypes)

### How it was computed
1. For each celltype, average `.X` (log-norm) across its reliable timepoints → (640K × 31) celltype-mean matrix
2. Leave-one-out z-score across celltypes: `z = (x_i - mean_other) / std_other`
3. Higher z = more specific to that celltype

### `.X` — z-score values
- Float32
- Range: approximately -3 to 150
- To get top-N for a celltype: `np.argsort(Z[:, ct_idx])[::-1][:N]`
- Suggested default: show top 200, let user filter by rank or min z-score

### `.var` — celltype names (31)
```
NMPs, PSM, differentiating_neurons, endocrine_pancreas, endoderm,
enteric_neurons, epidermis, fast_muscle, floor_plate, hatching_gland,
heart_myocardium, hemangioblasts, hematopoietic_vasculature, hindbrain,
lateral_plate_mesoderm, midbrain_hindbrain_boundary, muscle, neural,
neural_crest, neural_floor_plate, neural_optic, neural_posterior,
neural_telencephalon, neurons, notochord, optic_cup, pharyngeal_arches,
pronephros, somites, spinal_cord, tail_bud
```

Note: `primordial_germ_cells` is excluded (all timepoints have <20 cells).

---

## Key Operations for the Portal

### 1. Get peak UMAP coordinates + max celltype (for landing page)

```python
import anndata as ad, numpy as np

master = ad.read_h5ad("peaks_by_ct_tp_master_anno.h5ad")
umap = master.obsm["X_umap_2D"]  # (640830, 2)

z_adata = ad.read_h5ad("V3_specificity_matrix_celltype_level.h5ad")
Z = z_adata.X  # (640830, 31)
ct_names = list(z_adata.var_names)

max_ct_idx = np.argmax(Z, axis=1)
max_ct = [ct_names[i] for i in max_ct_idx]
max_z  = Z[np.arange(len(Z)), max_ct_idx]
```

### 2. Query top-50 peaks for a celltype

```python
celltype = "heart_myocardium"
ct_col = ct_names.index(celltype)
z_col = Z[:, ct_col]
top50_idx = np.argsort(z_col)[::-1][:50]

# Build table
table = master.obs.iloc[top50_idx][[
    "chrom", "start", "end", "peak_type",
    "nearest_gene", "linked_gene", "associated_gene",
    "distance_to_tss", "length",
]].copy()
table["V3_zscore"] = z_col[top50_idx]
table["coords"] = "chr" + table["chrom"].astype(str) + ":" + \
                  table["start"].astype(str) + "-" + table["end"].astype(str)
```

### 3. Get celltype accessibility profile for a peak

```python
import re

peak_iloc = 12345  # row index
row = master.X[peak_iloc]  # (190,) — all conditions

# Average across timepoints per celltype
ct_mean = {}
for ct in ct_names:
    col_mask = master.var["annotation_ML_coarse"].astype(str) == ct
    reliable = master.var["n_cells"].values >= 20
    cols = np.where(col_mask & reliable)[0]
    if len(cols) > 0:
        ct_mean[ct] = row[cols].mean()
```

### 4. Get temporal profile for a peak within a celltype

```python
celltype = "heart_myocardium"
timepoints = ["0somites", "5somites", "10somites", "15somites", "20somites", "30somites"]

tp_vals = {}
for tp in timepoints:
    cond_name = f"{celltype}_{tp}"
    if cond_name in master.var_names:
        col_idx = list(master.var_names).index(cond_name)
        n_cells = master.var.iloc[col_idx]["n_cells"]
        if n_cells >= 20:
            tp_vals[tp] = row[col_idx]
```

### 5. Extract DNA sequence for selected peaks

```python
import pysam

fa = pysam.FastaFile("danRer11.primary.fa")

# Peak obs index format: "1-32-526" (no chr prefix)
# FASTA uses "chr1" prefix
chrom, start, end = "1", 32, 526
seq = fa.fetch(f"chr{chrom}", start, end)

fa.close()
```

### 6. Export selected peaks as BED

```python
# selected_peaks: list of obs indices like ["1-32-526", "2-1000-1500"]
bed_rows = []
for peak_id in selected_peaks:
    row = master.obs.loc[peak_id]
    bed_rows.append(f"chr{row['chrom']}\t{row['start']}\t{row['end']}\t{peak_id}")

bed_content = "\n".join(bed_rows)
```

---

## Color Palette

Canonical celltype colors are defined in:
```
${REPO}/scripts/utils/module_dict_colors.py
```

Import: `from module_dict_colors import cell_type_color_dict`

This is a Python dict mapping celltype name → hex color (e.g., `"heart_myocardium": "#ccebc5"`).

---

## Precomputed Figures (for static fallback / thumbnails)

All at `${REPO}/figures/peak_parts_list/V3/`:

| Directory | Contents |
|-----------|----------|
| `specificity_overview/` | Tau UMAP (color/size/alpha), histogram, max-celltype UMAP |
| `peak_umap/all_celltypes/` | Individual peak UMAPs for all 31 celltypes |
| `peak_umap/per_celltype/` | Focal celltypes with gene labels |
| `peak_umap/` | Interesting-6 merged/panel, highlight views |
| `peak_profiles/per_celltype/` | Top-5 bar plots per focal celltype |
| `peak_profiles/marker_genes/` | Reverse-lookup profiles (myf5, myod1, myog, nkx2.5, hand2, gata4) |
| `temporal/heatmaps/` | Temporal heatmaps (absolute + row-normalized) |
| `gene_locus_views/per_gene/` | Genomic locus views for 12 marker genes |
| `motif_enrichment/` | TF enrichment heatmaps + 133 TF logos |
| `rna_atac_concordance/` | RNA-ATAC scatter plots |

---

## Important Caveats

1. **Peak index has no `chr` prefix**: obs index is `1-32-526`, not `chr1-32-526`. Add `chr` when querying FASTA or displaying coordinates.

2. **Unreliable conditions**: 14 out of 190 conditions have <20 cells (all PGC timepoints + a few others). Filter by `var["n_cells"] >= 20` before displaying temporal profiles.

3. **Gene annotations**: `linked_gene` (co-accessibility, more reliable) takes precedence over `nearest_gene` (TSS distance). Many peaks have neither — show genomic coordinates as fallback.

4. **Categorical dtypes**: h5ad obs columns are pandas Categorical. Use `.astype(str)` before string operations. `.fillna()` doesn't work on Categoricals — use `.astype(str).replace({"nan": ""})`.

5. **Memory**: The master h5ad is 4.4 GB. For the portal, precompute the celltype-mean matrix (640K × 31) and the UMAP coordinates as separate lightweight files (parquet/feather) rather than loading the full h5ad on every request.

---

## Suggested Precomputation for Portal

To avoid loading the 4.4 GB h5ad at request time, precompute these lightweight files:

| File | Shape | ~Size | Contents |
|------|-------|-------|----------|
| `peak_metadata.parquet` | 640K × ~15 cols | ~50 MB | chrom, start, end, peak_type, genes, UMAP coords, tau, max_celltype, max_z |
| `celltype_mean_matrix.parquet` | 640K × 31 | ~80 MB | Mean log-norm accessibility per celltype (for bar charts) |
| `condition_matrix.parquet` | 640K × 190 | ~500 MB | Full log-norm matrix (for temporal profiles). Or precompute per-celltype temporal vectors only. |
| `peak_sequences.fa` | ~640K seqs | ~400 MB | Pre-extracted FASTA (optional, avoids runtime pysam) |

A script to generate these can be created from the existing codebase.
