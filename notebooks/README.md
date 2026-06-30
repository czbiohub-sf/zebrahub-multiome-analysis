# Notebooks — directory → manuscript-figure map

Analysis notebooks, organized by subdirectory.

> ⚠️ **Some subdirectory names predate the manuscript's figure renumbering (V1 → V2),** so a directory's name does **not** always match its current figure number — e.g. `Fig3_GRN_dynamics/` produces manuscript **Figure 4**. Use the table below, and the panel-level **Figure → Notebook Map** in the [top-level README](../README.md), as the authoritative mapping. (Internal docstrings may likewise still say "Fig3/Fig4".)

Notebooks are jupytext-paired (`.ipynb` ↔ `.py:percent`) — see the top-level README.

| Directory | Manuscript figure(s) | Contents |
|---|---|---|
| `Fig1_atlas_QC/` | **Fig 1** (+ SI) | Integrated RNA+ATAC atlas, QC metrics, whole-embryo RNA/ATAC, marker-gene/peak dotplots, ATAC coverage tracks |
| `Fig2_ATAC_RNA_correlation_metacells/` | **Fig 2** (+ SI) | SEACells metacell aggregation; RNA–ATAC correlation / gene-UMAP dynamics across timepoints |
| `Fig_peak_umap/` | **Fig 3** (+ SI) | Genome-wide pseudobulk peak UMAP: clustering, motif enrichment, peak→gene association, trajectories |
| `EDA_peak_parts_list/` | **Fig 3** + SI (peaks "parts list"); **Fig 6** portal hand-offs | Top-peaks-per-celltype "parts list", multi-database motif rescan, per-celltype detail reports, portal hand-offs |
| `Fig3_GRN_dynamics/` | **Fig 4** (+ SI) &nbsp;⚠️ *name lags numbering* | Cell-type/timepoint-resolved GRNs, sub-GRN regulatory programs, network-similarity comparisons |
| `Fig4_in_silico_KO/` | **Fig 5** (+ SI) &nbsp;⚠️ *name lags numbering* | CellOracle in-silico perturbation (KO) simulations and trajectory/UMAP analyses |
| `Fig_GRN_zoom_in/` | Supplementary (GRN zoom-ins) | Focused sub-GRN subset visualizations |
| `Fig_cross_species_GRN/` | Supplementary (cross-species GRN) | Zebrafish ↔ mouse GRN comparison + mouse in-silico KO |
| `EDA_peak_umap_cross_species/` | Supplementary (cross-species peak UMAP) | Mouse/human peak-object preprocessing, motif-space embedding, Procrustes alignment to the zebrafish peak UMAP |
| `EDA_chromatin_velocity/` | Exploratory (not in manuscript) | Exploratory chromatin-velocity analysis |

**Quick reference — directories whose name ≠ current figure number:**
`Fig_peak_umap/` → **Fig 3** · `Fig3_GRN_dynamics/` → **Fig 4** · `Fig4_in_silico_KO/` → **Fig 5**.
