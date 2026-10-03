# Peak-level data objects (peak UMAP, manuscript Fig 3)

The peak UMAP work uses two main AnnData objects. Each object has one row per ATAC peak (640,830 peaks) and one column per pseudobulk group (190 groups). The two objects have the same peaks and the same UMAP coordinates. They group cells differently:

| | Master object (clustering and figures) | Celltype × timepoint object (interpretation) |
|---|---|---|
| File | `peaks_by_pb_annotated_master.h5ad` | `peaks_by_ct_tp_master.h5ad` |
| Columns (`var`) | RNA Leiden 0.4 merged cluster × stage, e.g. `0_0somites` (32 clusters × 6 stages) | `annotation_ML_coarse` × stage, e.g. `NMPs_0somites` (32 cell types × 6 stages) |
| Groups with < 20 / < 50 cells | 14 / 29 | 14 / 24 |
| `.X` | `layers['normalized']` | `layers['normalized']` |
| Layers | `normalized`, `sum` | `log_norm`, `normalized`, `sum` |
| `obsm` | `X_pca`, `X_umap` = `X_umap_2D`, `X_umap_3D` | the same arrays, copied from the master |
| `obs` | 81 columns | 70 columns (a subset of the master's) |
| Use it for | the peak UMAP, peak clusters, figure panels, the portal export | reading accessibility of a peak per cell type and stage |

Full paths on the Bruno HPC:

```
/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_annotated_master.h5ad   (3.9 GB, md5 302b94dd252f2d5c72a4559a9f6a4d39)
/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_master.h5ad          (4.7 GB)
```

The UMAP was computed on the Leiden grouping. The celltype × timepoint object holds a copy of those coordinates, so its matrix and its embedding come from different groupings. Do not recompute a UMAP on the celltype × timepoint matrix and expect the published layout.

## How the master object was built

1. `compute_leiden_clusters_cell_umap_wnn.py` clusters cells on the joint RNA+ATAC (WNN) graph and writes `leiden_0.4_merged` to `integrated_RNA_ATAC_counts_RNA_leiden(_filtered).h5ad` (lines 341, 416; `peaks_pb_leiden_clusters.py:123`).
2. `peaks_pb_leiden_clusters.py` pseudobulks the per-cell peak counts by `leiden_{res}_merged × dev_stage` for each resolution. For each group it sums the counts and scales them by (median group coverage / group coverage). Here coverage is the sum of the group's counts over all peaks, so `normalized` equals CPM times one global constant (`scripts/utils/pseudobulk_utils.py`). The notebook then z-scores each group column, computes 100 PCs, builds a 15-neighbor graph on the first 40 PCs, and runs UMAP (`min_dist=0.3`, `random_state=42`, rapids-singlecell on GPU). It writes `peaks_by_pb_leiden_{res}_merged.h5ad` (line 237).
3. `02_01_hierarchical_clustering_peak_umap.py` removes 4 peaks: 2 mitochondrial peaks and 2 peaks that extend past the chromosome end (lines 176-181). It adds Leiden peak clusters `leiden_0.2` … `leiden_5` (line 89) and the 36 `leiden_coarse` clusters, which it makes by merging `leiden_1.5` (line 286). It writes `peaks_by_pb_leiden_0.4_merged_annotated(_filtered).h5ad` (lines 233-234).
4. `01_EDA_annotate_peak_umap.py` and `04_associate_peaks_genes_peak_umap.py` add peak coordinates and lengths (`01_EDA` lines 152-176), `nearest_gene` and `distance_to_tss` (`01_EDA` lines 483-512), and Signac peak-gene links `linked_gene` / `link_*` (`04` lines 84-94).
5. `09_annotate_peak_umap_celltype_timepoints.py` reads the celltype × timepoint object and assigns each peak a `celltype`, `timepoint` and `lineage` label, each with a `_contrast` score. It writes `all_peaks_annotated_ct_tp.csv` (lines 435, 626). The recipe is described under [Label versions](#label-versions).
6. `export_3D_umap_coords.py` copies `peak_type_argelaguet` and the `09` labels into the master, exports `peak_umap_3d_annotated_v6.csv` for the portal, and writes the master (lines 85-132).

The fine clusters (`leiden_unified`) entered the master in an archived notebook, `notebooks/Fig3_GRN_dynamics/archive/EDA_extract_subGRN_reg_programs_old.ipynb` (cells near JSON line 453).

## Label versions

Two recipes for the per-peak `celltype` and `timepoint` labels exist. The two objects carry different versions:

| Recipe | Code | Carried by |
|---|---|---|
| **Old:** sum of `normalized` over all stages of a cell type, using all 190 groups, then argmax | `peaks_pb_preprocessing.py:686-740` (repeated in `peaks_pb_leiden_clusters.py:873-924`) | `peaks_by_ct_tp_master.h5ad` `obs['celltype']`, `obs['timepoint']`, all `obs['accessibility_*']` |
| **Current:** drop groups with < 20 cells, take the mean over the remaining stages, drop the primordial germ cell column, then argmax | `09_annotate_peak_umap_celltype_timepoints.py:205-322` | master `obs['celltype']`, `obs['timepoint']`, `obs['lineage']` and their `_contrast` columns |

A check on 20,000 random peaks: each object's `celltype` agrees with its own recipe for 98% of peaks and with the other recipe for 58%. The old recipe labels 14.7% of peaks as `primordial_germ_cells`, because those groups have very few cells. The current recipe labels none.

The master's `accessibility_*` columns are still the old sums. They are not the means that its `celltype` labels came from. To rank cell types for a peak, recompute from `layers['normalized']` of the celltype × timepoint object, or use `celltype` / `celltype_contrast` from the master.

## Other objects

All files are in `/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/` unless the table says otherwise.

| Object | Written by | Contents and status |
|---|---|---|
| `peaks_by_ct_tp_master_anno.h5ad` | only a commented-out write, `llm_annotation/litemind_prepare_query_deep_research.py:87` | Duplicate of `peaks_by_ct_tp_master.h5ad`. Matrices, `obs`, `var` and `obsm` are identical. Only `uns['umap']['params']` differs. The parts-list notebooks and `scripts/utils/build_peak_metadata_cache.py` read it. |
| `peaks_by_pb_leiden_{res}_merged.h5ad` | `peaks_pb_leiden_clusters.py:237` | One object per Leiden resolution (0.01-9); 0.4 is the figure resolution. 640,834 peaks. |
| `peaks_by_pb_leiden_0.4_merged_annotated(_filtered).h5ad` | `02_01…py:233-234`, later `01_EDA…py:562`, `08_subset_reg_programs.py:122` | Intermediate states of the master. `09`, `02_02`, `03_02`, `05` and `07` read `_filtered`. |
| `peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled_all_peaks.h5ad`, `…_wo_log_all_peaks.h5ad`, `…_wo_log_all_peaks_3d_umap.h5ad` | `peaks_pb_preprocessing.py:832, 880`; `peaks_pb_leiden_clusters.py:1013, 1061, 1095` | Early celltype × timepoint objects (640,834 peaks) with their own exploratory UMAPs. They carry the old labels. `.X` is `log_norm` in `…_all_peaks` and `normalized` in `…_wo_log`. |
| `peaks_by_ct_tp_pseudobulked_all_peaks_pca_concord.h5ad` | `peaks_pb_concord_umap.py:213`, `scripts/run_concord_peak_umap.py:193` | Celltype × timepoint object with PCA and CONCORD embeddings. The cross-species work uses it (`notebooks/EDA_peak_umap_cross_species/14_align_pseudobulk_umap.py:53`). |
| `pseudobulk_ct_tp_by_peaks.h5ad` (here and in `annotated_data/`) | archived `.ipynb` only (`archive/EDA_step1_peaks_pseudobulked.ipynb`) | The transposed matrix: 190 groups × 640,834 peaks. |
| `all_peaks_annotated.csv` (`annotated_data/` and `objects_v2/`) | `01_EDA…py:147, 624`; `04…py:132` | Peak type and gene annotation per peak. |
| `all_peaks_annotated_ct_tp.csv` | `09…py:435, 626` | Current `celltype` / `timepoint` / `lineage` labels per peak; input to `export_3D_umap_coords.py`. |
| `peak_umap_3d_annotated_v*.csv` | v6: `export_3D_umap_coords.py:124` (relative path) | 3D UMAP and labels for the portal. |
| `leiden_*_by_pseudobulk*.csv` | `llm_annotation/litemind_prepare_query_deep_research.py:121` and the paired `.ipynb` | Mean accessibility of each peak cluster in each of the 190 groups. |
| `peaks_by_motifs_master.h5ad`, `peaks_by_motifs_maelstrom_leiden_unified.h5ad` | archived `.ipynb` only (`Fig3_GRN_dynamics/archive/…_old.ipynb`) | Peaks × motif scores (GimmeMotifs maelstrom). |
| `notebooks/EDA_peak_parts_list/outputs/V3/peak_metadata_cache.parquet` | `scripts/utils/build_peak_metadata_cache.py:111` | Per-peak metadata cache for the gene-locus tools (gitignored). |

The per-cell inputs are in `/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/01_Signac_processed/`. `integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered.h5ad` holds the cells × peaks raw counts (94,562 cells). `integrated_RNA_ATAC_counts_RNA_leiden_filtered.h5ad` holds the per-cell `leiden_0.4_merged` labels.

## Known gaps

- **No writer for `peaks_by_ct_tp_master.h5ad`.** No `.py` or `.ipynb` in the repo writes it. Its `obs` includes `leiden_coarse`, so it was probably made by copying the Leiden-object `obs` and `obsm` onto a celltype × timepoint matrix. The commented recipe at `litemind_prepare_query_deep_research.py:64-87` follows that pattern.
- **Mismatched function signature in `09`.** `09_annotate_peak_umap_celltype_timepoints.py:225` calls `get_cell_count_for_group(col)` with one argument. The imported version in `scripts/peak_umap_utils/annotation_analysis.py:29` takes two. The paired `.ipynb` defines a one-argument version inline.
- **Reef CPM parquet.** The Reef file `pseudobulk_cpm.parquet` (640,832 × 166) does not come from this repo. It is a coarse cell type × stage matrix with groups below 50 cells removed, built as training targets for another project. It does not reproduce the Fig 3 UMAP. Use the master object, or rebuild the 190 Leiden groups from the per-cell `leiden_0.4_merged` labels.
