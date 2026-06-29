# Zebrahub-Multiome: A Time-Resolved Single-Cell Multi-Omic (RNA+ATAC) Atlas of Zebrafish Development

This repository contains the curated analysis code that produced the figures in the **Zebrahub-Multiome** manuscript — a time-resolved single-cell multiome (paired snRNA-seq + snATAC-seq) atlas of early zebrafish (*Danio rerio*, danRer11) embryogenesis across developmental timepoints. The atlas is used to chart how chromatin accessibility and gene expression co-vary, and to reconstruct the dynamics of gene regulatory networks (GRNs) over developmental time.

**This is a Resource, not a packaged software tool.** The repository is a reproducible record of the analyses behind the manuscript figures, organized so a reviewer can (a) see what is here, (b) recreate the compute environments, (c) trace every main/supplementary figure panel to the notebook that generated it, and (d) locate the underlying data. It is **not** intended as a stand-alone, installable, or general-purpose Python/R package; the scripts in `scripts/` are written specifically for this project's data layout.

**Getting the code.** A plain `git clone` is all you need — the repository is ~390 MB, uses **no Git LFS**, and every tracked notebook opens directly with its rendered outputs (no extra tooling required). Three exceptionally large notebooks are provided as their jupytext `.py` scripts only (noted under *Repository Structure*).

## Links

- **Interactive atlas / data portal:** https://zebrahub.ds.czbiohub.org/
- **Manuscript (preprint):** [Zebrahub-Multiome: Uncovering Gene Regulatory Network Dynamics During Zebrafish Embryogenesis. *bioRxiv* (2024).](https://www.biorxiv.org/content/10.1101/2024.10.18.618987v1) doi:10.1101/2024.10.18.618987
- **Peer-reviewed version:** _to be added upon publication._

### Analysis scope (by figure)

- **Figure 1 — Atlas & QC** (`notebooks/Fig1_atlas_QC/`): integrated RNA+ATAC atlas, QC metrics, whole-embryo RNA/ATAC analysis, marker-gene/peak dotplots, and ATAC coverage tracks.
- **Figure 2 — RNA–ATAC dynamics via metacells** (`notebooks/Fig2_ATAC_RNA_correlation_metacells/`): SEACells metacell aggregation and RNA–ATAC correlation / gene-UMAP dynamics across timepoints.
- **Figure 3 — Genome-wide peak UMAP** (`notebooks/Fig_peak_umap/`): pseudobulk peak UMAP, hierarchical clustering, motif enrichment, peak→gene association, and trajectories (the "peaks parts list").
- **Figure 4 — GRN dynamics** (`notebooks/Fig3_GRN_dynamics/`): cell-type/timepoint-resolved gene regulatory networks, sub-GRN regulatory programs, and network-similarity comparisons.
- **Figure 5 — In-silico knockouts** (`notebooks/Fig4_in_silico_KO/`): CellOracle-based in-silico perturbation (KO) simulations and trajectory/UMAP analyses of perturbation effects.
- **Figure 6 — Resource / web portal**: interactive exploration of the atlas plus a re-presentation of the peak, GRN, and KO results (composite figure; see the data-portal link above).
- **Supporting analyses:** cross-species GRN/peak comparisons (`notebooks/Fig_cross_species_GRN/`, `notebooks/EDA_peak_umap_cross_species/`), GRN zoom-ins (`notebooks/Fig_GRN_zoom_in/`), the peak "parts list" (`notebooks/EDA_peak_parts_list/`), and exploratory analyses under `notebooks/EDA_*`.

> **Note:** the `notebooks/` directory names predate the V2 figure renumbering — e.g. manuscript **Fig 3** (peak UMAP) lives in `notebooks/Fig_peak_umap/`, **Fig 4** (GRN) in `notebooks/Fig3_GRN_dynamics/`, and **Fig 5** (in-silico KO) in `notebooks/Fig4_in_silico_KO/`. See the **Figure → Notebook Map** for the exact mapping.

---

## System Requirements

### Operating system

All analyses were developed and run on **Linux (Red Hat Enterprise Linux 8, x86_64)** on an HPC cluster (kernel `4.18.0`-series). The conda environments below are exported from this platform and contain Linux-64 (`linux64` / `h5eee18b` / `linux-64`) binary builds, so they are expected to recreate cleanly only on **Linux x86_64**. macOS (incl. Apple Silicon) and Windows are **not** supported by the pinned environment files; on those platforms install loosely-pinned equivalents instead (see Installation note). No non-standard hardware is required; a workstation/node with **≥ 32 GB RAM** is recommended for the whole-embryo atlas objects.

### Software environments

The analysis is split across **three conda environments** (one per figure group). Exact pins are in `environments/*.yml`; key versions:

| Environment file | Used for | Python | Key packages (pinned) |
|---|---|---|---|
| `environments/single-cell-base.yml` | **Figs. 1 & 3** — atlas/QC + genome-wide peak UMAP | **3.10.13** | scanpy 1.9.8, anndata 0.10.5.post1, numpy 1.26.4, pandas 2.2.0, scikit-learn 1.3.0, scipy 1.12.0, leidenalg 0.10.2, umap-learn 0.5.3; (pip) scvelo 0.3.2, scanorama 1.7.4, scrublet 0.2.3, muon 0.1.6, pyslingshot 0.1.5 |
| `environments/seacells.yml` | **Fig. 2** — RNA–ATAC correlation via metacells | **3.8.18** | SEACells 0.3.3, scanpy 1.9.8, anndata 0.9.2, numpy 1.24.4, pandas 1.5.3, scikit-learn 1.3.1, scipy 1.10.1, palantir 1.3.2, ray 2.10.0, jax/jaxlib 0.4.13 |
| `environments/celloracle_env.yml` | **Figs. 4 & 5** — GRN dynamics + in-silico KO | **3.8.17** | CellOracle 0.18.0, cellrank 2.0.5, scanpy 1.9.3, anndata 0.9.2, numpy 1.24.4, pandas 1.5.3, gimmemotifs 0.17.0, genomepy 0.16.1, velocyto 0.17.17, xgboost 1.7.6, torch 2.0.1 |

**Conda channels** used (all three): `conda-forge`, `bioconda`, `defaults`; the `seacells` env additionally lists `anaconda`. Install via `conda` (or the faster, drop-in `mamba`).

> **Note on cross-environment version differences.** Package versions intentionally differ between the three environments (e.g. scanpy 1.9.8 / 1.9.8 / 1.9.3; anndata 0.10.5.post1 vs 0.9.2; numpy 1.26.4 vs 1.24.4; pandas 2.2.0 vs 1.5.3; pyslingshot 0.1.5 vs 0.0.2). This is expected — each figure group has its own self-consistent stack — and is not an error.

### R (preprocessing)

Several preprocessing steps run in **R** — Signac/Seurat object construction (`scripts/run_01_preprocess_*_signac.R`), Cicero co-accessibility (`scripts/run_02_*_cicero.R`), and Seurat→h5ad export (`scripts/export_seurat_*.R`). R preprocessing was run with **R 4.3** (`module load R/4.3`) using **Signac**, **Seurat**, and **Cicero/monocle3** (Bioconductor/CRAN). Exact package versions are recorded in [`environments/R_sessionInfo.txt`](environments/R_sessionInfo.txt); regenerate with `module load R/4.3 && Rscript environments/capture_R_sessioninfo.R > environments/R_sessionInfo.txt`.

### GPU (optional)

A GPU is **optional**. The `celloracle_env` environment ships CUDA 11 wheels (`torch==2.0.1`, `triton==2.0.0`, `nvidia-*-cu11`), and GPU-accelerated trajectory inference is available (`scripts/pyslingshot_gpu_accel.py`). All core figure analyses run on CPU; a GPU only speeds up trajectory inference / Torch-backed steps. No GPU is needed to reproduce the published figures.

---

## Installation

The three conda environments are recreated from the YAML files in `environments/`. Recommended: install [Miniforge/Mambaforge](https://github.com/conda-forge/miniforge) and use `mamba` for speed (substitute `conda` if you prefer).

### Environment files

The `environments/*.yml` files are portability-cleaned — plain `name:` and no machine-specific `prefix:` line — so `conda env create -f environments/<name>.yml` works directly on a Linux-64 machine. (Pins are Linux-64; on macOS/Windows install the headline packages by version instead — see *Platform note* below.)

### Create the environments

```bash
# Figs 1 & 3 — atlas/QC + peak UMAP
conda env create -f environments/single-cell-base.yml
# Fig. 2 — SEACells metacells
conda env create -f environments/seacells.yml
# Figs. 4 & 5 — CellOracle GRN + in-silico KO
conda env create -f environments/celloracle_env.yml
```

Activate the environment matching the figure you are reproducing:

```bash
conda activate single-cell-base   # Figs 1 & 3
conda activate seacells           # Fig. 2
conda activate celloracle_env     # Figs. 4 & 5
```

### Jupytext (notebooks)

Notebooks are version-controlled as paired `.py:percent` scripts via **Jupytext** (`jupytext.toml`, which sets `formats = "ipynb,py:percent"`). This pairs every `.ipynb` with a plain-text `.py` twin so that diffs and code review are clean and the notebook logic is readable without opening Jupyter. `jupytext` is included in the environments; to regenerate an `.ipynb` from its paired script: `jupytext --to notebook notebooks/<...>.py`. Editing either side keeps the pair in sync. **The repository uses no Git LFS**, so cloning needs no special tooling; three oversized notebooks (listed under *Repository Structure*) ship as `.py` only and can be rendered the same way.

### R preprocessing dependencies

Install **Signac**, **Seurat**, and **Cicero** (Bioconductor/CRAN) in **R 4.3** (`module load R/4.3`) for the `scripts/run_01_*`/`run_02_*`/`export_seurat_*` steps. Exact versions are in [`environments/R_sessionInfo.txt`](environments/R_sessionInfo.txt) (regenerate via `Rscript environments/capture_R_sessioninfo.R`).

### Typical install time

Roughly **~10–20 minutes per environment** on a normal broadband connection (dominated by solving and downloading the large pip dependency sets — `ray`, `jax`, `torch`+CUDA wheels). Using `mamba` instead of `conda` reduces solve time substantially. Total for all three environments: **~30–60 minutes**.

### Platform note

The pinned YAMLs are Linux-64 only (see System Requirements). On macOS/Windows, do **not** use these exact files; instead create fresh environments and `pip install` the key packages by their version pins (e.g. `scanpy`, `SEACells==0.3.3`, `celloracle==0.18.0`).

---

## Repository Structure

```text
zebrahub-multiome-analysis/
├── README.md
├── LICENSE                         # BSD 3-Clause
├── jupytext.toml                   # pairs each .ipynb with a .py:percent text file
│
├── environments/                   # conda environment specs (one per analysis stage)
│   ├── single-cell-base.yml        # Figs 1 & 3
│   ├── seacells.yml                # Fig 2
│   └── celloracle_env.yml          # Figs 4 & 5
│
├── notebooks/                      # analysis notebooks, organized by figure
│   ├── Fig1_atlas_QC/              # atlas integration, QC, marker genes/peaks, coverage plots
│   │   ├── Fig1_bcde_atlas_integrated.ipynb
│   │   ├── Fig1_efgh_dotplot_marker_genes_peaks.ipynb
│   │   ├── Fig1_f_R_coverage_plot_optimization.ipynb
│   │   ├── Fig1_coverage_plot_ATAC_integrated.ipynb
│   │   ├── Fig1_RNA_ATAC_pseudo_bulk_EDA.ipynb
│   │   ├── Fig1_SI_QC_metrics_part1.ipynb
│   │   ├── Fig1_SI_QC_metrics_R_part2.ipynb
│   │   ├── Fig1_SI_RNA_ATAC_whole_embryo.ipynb
│   │   ├── Fig1_SI_neighborhood.ipynb
│   │   └── post_processing/        # Seurat→h5ad export, peak↔gene linkage, cellxgene prep
│   │
│   ├── Fig2_ATAC_RNA_correlation_metacells/   # RNA–ATAC correlation via SEACells metacells
│   │   ├── Fig2_metacell_RNA_ATAC_dynamics_v2.py    # main Fig 2 — .py only (rendered nb too large)
│   │   ├── compute_metacells_n_cells.ipynb
│   │   ├── aggregate_metacell_counts.ipynb
│   │   └── FishEnrichR_query.ipynb
│   │
│   ├── Fig3_GRN_dynamics/          # GRN temporal dynamics, sub-GRN regulatory programs
│   │   ├── Fig3_GRN_dynamics.ipynb
│   │   ├── Fig3_SI_GRN_similarity_quant.py          # .py only (rendered nb too large)
│   │   ├── SI_QC_GRN_network_comparison_TDR118_TDR119_reseq_v2.ipynb
│   │   ├── EDA_extract_subGRN_reg_programs.ipynb
│   │   ├── viz_grn_heatmap_BenIovino.ipynb
│   │   └── utils/                  # module_extract_subGRN, grn_sim/overlap/temporal/viz, etc.
│   │
│   ├── Fig4_in_silico_KO/          # in-silico knockout simulation & quantification
│   │   ├── Fig4_in_silico_KO.ipynb
│   │   └── AlignUMAP_NMPs_timepoints.ipynb
│   │
│   ├── Fig_peak_umap/              # peak UMAP: clustering, motif enrichment, peak→gene, trajectories
│   │   ├── 01_EDA_annotate_peak_umap.ipynb
│   │   ├── 02_01_hierarchical_clustering_peak_umap.py   # .py only (rendered nb too large)
│   │   ├── 02_02_sub-clustering_peak_umap.ipynb
│   │   ├── 03_motif_enrichment_analysis_v3_coarse_fine.ipynb
│   │   ├── 04_associate_peaks_genes_peak_umap.ipynb
│   │   ├── 05_trajectory_inference_paga.ipynb
│   │   ├── llm_annotation/         # LiteMind LLM-based peak-cluster annotation
│   │   └── ... (peak-UMAP supporting notebooks)
│   │
│   ├── Fig_GRN_zoom_in/            # focused sub-GRN visualizations
│   │   └── GRN_subset.ipynb
│   │
│   ├── Fig_cross_species_GRN/      # zebrafish↔mouse GRN comparison + mouse in-silico KO
│   │   ├── 01_explore_mouse_data.py
│   │   ├── 02_mouse_NMP_GRN_analysis.py
│   │   ├── 03_mouse_insilico_KO.py
│   │   └── 04_zebrafish_mouse_comparison.py
│   │
│   ├── EDA_peak_parts_list/        # "parts list": top peaks per celltype + multi-DB motif rescan
│   ├── EDA_peak_umap_cross_species/# cross-species peak UMAP (mouse/human integration, Procrustes)
│   └── EDA_chromatin_velocity/     # exploratory chromatin-velocity analysis
│
└── scripts/                        # project-specific reusable Python/R modules
    ├── preprocessing/              # core pipeline (numbered run_0x_*)
    │   ├── run_01_preprocess_multiome_object_signac.R   # Signac preprocessing (R)
    │   ├── run_02_*_cicero*.R                            # Cicero co-accessibility / gene activity (R)
    │   ├── run_03_celloracle_filter_CCANS_map_to_genes.py
    │   ├── run_04_celloracle_compute_baseGRN.py
    │   ├── run_05_celloracle_compute_celltype_GRNs.py
    │   ├── run_06_*_compute_pseudotime.py               # Palantir / pySlingshot / CellOracle DPT
    │   ├── run_07_*_co_*KO*.py                           # in-silico KO simulation
    │   └── run_08_markovian_simulation.py
    ├── R_scripts/                  # Seurat/Signac integration, WNN, CHOIR, peak→gene linkage
    ├── SEACells_metacell/          # metacell computation + RNA–ATAC correlation (Fig 2)
    ├── fig1_utils/ … fig4_utils/   # per-figure plotting / analysis helpers
    ├── subGRN_utils/               # sub-GRN extraction & analysis (Fig 3)
    ├── gimmemotifs/                # de-novo motif analysis (GimmeMotifs / maelstrom)
    ├── jaspar-motif-scan/          # JASPAR motif scanning pipeline
    ├── litemind_peak_analysis/     # LLM-based biological interpretation of peak clusters
    ├── peak_umap_utils/, pyslingshot_gpu/, chrom_velo/  # supporting modules
    ├── utils/                      # gene-locus explorer, FIMO, synthetic-enhancer ranking, pub_fig_style
    ├── export_seurat_*.R, subset_zebrahub_references.R  # Seurat export utilities (R)
    ├── color_palettes.py, plot_2D_vector_flows_KOs.py   # visualization
    └── slurm_scripts/              # SLURM submission wrappers for the above
```

The public repository contains only the `main` branch; large intermediate files, rendered figure outputs, archival/pre-reorganization directories, and logs have been removed from history to keep the published Resource lightweight and reviewable. The repo uses **no Git LFS** — a plain `git clone` (~390 MB) yields every notebook with its rendered outputs. Three notebooks whose embedded outputs made them too large to store as regular files — `Fig2_ATAC_RNA_correlation_metacells/Fig2_metacell_RNA_ATAC_dynamics_v2`, `Fig3_GRN_dynamics/Fig3_SI_GRN_similarity_quant`, and `Fig_peak_umap/02_01_hierarchical_clustering_peak_umap` — are provided as their jupytext `.py` scripts only; regenerate a runnable notebook with `jupytext --to notebook <file>.py`.

---

## Figure → Notebook Map

Every manuscript figure (**V2 numbering**) mapped to the notebook(s) that produced it, verified against the notebook code. **Confidence:** `CONFIRMED` (code clearly produces it), `likely` (supported, exact panel unclear), `approx`, or *schematic* (hand-drawn, no notebook). Notebook paths point to the `.py:percent` jupytext files — open the paired `.ipynb` for rendered output. A few peak-UMAP / subGRN notebooks were run under a GPU env (`sc_rapids` / rapids-singlecell) rather than the figure's base conda env, noted inline.

### Main figures (Fig 1–6)

| Figure | Theme | Environment | Notebook(s) | Confidence |
|---|---|---|---|---|
| **Fig 1** | Multiome atlas integration & QC | `single-cell-base` (+ R for coverage) | `notebooks/Fig1_atlas_QC/Fig1_bcde_atlas_integrated.py` (a-bars, b, c, d, f-left); `notebooks/Fig1_atlas_QC/Fig1_efgh_dotplot_marker_genes_peaks.py` (f-right, g); `notebooks/Fig1_atlas_QC/Fig1_coverage_plot_ATAC_integrated.py` (h, R); `notebooks/Fig1_atlas_QC/Fig1_f_R_coverage_plot_optimization.py` (h support, R) | CONFIRMED — **panel e unresolved** (sequential RNA/ATAC gene-activation along NMP trajectories: final PDFs exist on disk but no committed producer notebook; flag for author). Panel a schematic is hand-drawn. |
| **Fig 2** | RNA–ATAC dynamics via SEACells metacells | `seacells` | `notebooks/Fig2_ATAC_RNA_correlation_metacells/Fig2_metacell_RNA_ATAC_dynamics_v2.py` (a, c, d, e, f); `.../FishEnrichR_query.py` (c labels); `.../aggregate_metacell_counts.py` (b prep); `.../compute_metacells_n_cells.py` (b prep) | CONFIRMED — panel b is a hand-drawn workflow schematic (prep notebooks listed). |
| **Fig 3** | Genome-wide peak UMAP (parts list) | `single-cell-base` *(actual: `sc_rapids` GPU for Fig_peak_umap)* | `notebooks/Fig_peak_umap/peaks_pb_preprocessing.py` (b); `.../peaks_pb_concord_umap.py` (c); `.../07_peak_type_stats.py` (d); `.../01_EDA_annotate_peak_umap.py` (d); `.../09_annotate_peak_umap_celltype_timepoints.py` (c, e, h); `.../02_01_hierarchical_clustering_peak_umap.py` (f, h); `.../02_02_sub-clustering_peak_umap.py` (i); `.../04_associate_peaks_genes_peak_umap.py` (g); `.../03_motif_enrichment_analysis_v3_coarse_fine.py` (j, g); `.../llm_annotation/litemind_generate_prompts_coarse_fine_clusters.py` (g); `notebooks/EDA_peak_parts_list/09l_hemangioblasts_combined_panels.py` (k); `.../09m_hemangioblasts_top5_umap.py` (k); `.../09i_motif_position_maps_v2.py` (k) | CONFIRMED — panels a (concept) and b (pipeline schematic) hand-drawn. |
| **Fig 4** | GRN dynamics | `celloracle_env` *(panels e/f/g/h subGRN+peak-UMAP run under `sc_rapids`)* | `notebooks/Fig3_GRN_dynamics/Fig3_GRN_dynamics.py` (c, d); `.../Fig3_SI_GRN_similarity_quant.py` (b); `.../EDA_extract_subGRN_reg_programs.py` (f, g, h backbone); `.../utils/visualize_hemangioblasts_shell_editable.py` (h render); `notebooks/Fig_peak_umap/02_01_hierarchical_clustering_peak_umap.py` (e coarse Leiden); `notebooks/Fig_peak_umap/peak_umap_leiden_clusts_viz_v1.py` (e optic_cup zoom) | CONFIRMED — panel a schematic; scheme halves of f/g hand-drawn. |
| **Fig 5** | In-silico TF knockout | `celloracle_env` (alignment step: `single-cell-base`) | `notebooks/Fig4_in_silico_KO/Fig4_in_silico_KO.py` (a-compute, b, c, d, e); `.../AlignUMAP_NMPs_timepoints.py` (d, e upstream `umap_aligned`) | CONFIRMED — panels a/b are schematic workflow diagrams (quantitative substance computed in main notebook). |
| **Fig 6** | Web portal / resource | `single-cell-base` (only Fig-6-specific code) | `notebooks/Fig1_atlas_QC/post_processing/prepare_h5ad_objects_cellxgene.py` (b); `notebooks/EDA_peak_parts_list/PORTAL_HANDOFF.md` (e); `.../09l_hemangioblasts_combined_panels.py` (e example); `.../PORTAL_GENE_LOCUS_HANDOFF.md` (e); `notebooks/Fig_peak_umap/` (e cluster level); `notebooks/Fig2_ATAC_RNA_correlation_metacells/` (c, re-presentation); `notebooks/Fig3_GRN_dynamics/` (d, re-presentation); `notebooks/Fig4_in_silico_KO/` (d, re-presentation) | **likely / partially unresolved** — composite portal montage + live web frontend (CZ CELLxGENE + custom dashboards) is **not** in this repo; no single notebook produces the figure. Panel a (architecture schematic) hand-drawn. Cross-figure dir pointers (c/d/e re-presentations) unverified at notebook level. |

### Supplementary figures

| SI figure | Environment | Notebook(s) | Confidence |
|---|---|---|---|
| **SFig data_processing** | (R) | `scripts/preprocessing/run_01_preprocess_multiome_object_signac.R` (a); `scripts/R_scripts/integrate_seurat_timepoints_RNA.R` (b); `scripts/R_scripts/integrate_seurat_timepoints_ATAC.R` (b); `scripts/R_scripts/integrated_seurat_RNA_ATAC_timepoints.R` (b); `scripts/R_scripts/compute_WNN_embeddings_RNA_ATAC.R` (a, b) | CONFIRMED — figure is a schematic; mapping is to the R pipeline it illustrates. |
| **SFig QC** | `single-cell-base` \| (R) | `notebooks/Fig1_atlas_QC/Fig1_SI_QC_metrics_part1.py` (a, b); `notebooks/Fig1_atlas_QC/Fig1_SI_QC_metrics_R_part2.py` (c-left, c-middle; R) | likely — panel **c-right (cells-passing-QC bar plot) unresolved** (no producer found in either QC notebook). |
| **sfig:neighborhood** | `single-cell-base` *(actual: `sc_rapids`)* | `notebooks/Fig1_atlas_QC/Fig1_SI_neighborhood.py` (a, c, d, e); `notebooks/Fig1_atlas_QC/post_processing/export_seurat_neighborhoods_R.py` (upstream data, R) | CONFIRMED — panel b schematic. Caption color labels for panel e are swapped vs code (author check). |
| **SFig peak_calling** | (R) | `notebooks/Fig1_atlas_QC/Fig1_coverage_plot_ATAC_integrated.ipynb` (a); `notebooks/Fig1_atlas_QC/archive/Fig1_SI_R_peak_calling_benchmark.ipynb` (b, c, d) | CONFIRMED — merged 4th overlap track in panel a assembled externally. |
| **SFig metacells** (sfig:metacells) | `seacells` | `notebooks/Fig2_ATAC_RNA_correlation_metacells/compute_metacells_n_cells.py` (a); `.../Fig2_metacell_RNA_ATAC_dynamics_v2.py` (b) | CONFIRMED |
| **SFig gsea_pathway** | `seacells` | `notebooks/Fig2_ATAC_RNA_correlation_metacells/Fig2_metacell_RNA_ATAC_dynamics_v2.py` (a + input to b); `.../FishEnrichR_query.py` (b) | CONFIRMED — caption "15 clusters" vs code's 14 (0–13); minor relabel. |
| **SFig chromatin_acc_dynamics** | R | `notebooks/Fig1_atlas_QC/Fig1_coverage_plot_ATAC_integrated.py` (b, c, d, e) | CONFIRMED — panel a schematic. |
| **SFig peak_umap_pseudobulk_scheme** | `single-cell-base` *(actual: `sc_rapids` for viz_v1)* | `notebooks/Fig_peak_umap/peak_umap_leiden_clusts_viz_v1.py` (c, d, e, f; supports b); `.../compute_leiden_clusters_cell_umap_wnn.py` (a, b-top; supports c); `.../peaks_pb_leiden_clusters.py` (b-bottom; supports c) | CONFIRMED |
| **SFig peak_umap_hyperparams** | `single-cell-base` *(actual: `sc_rapids`)* | `notebooks/Fig_peak_umap/10_umap_hyperparams.py` (a, b, c, d) | CONFIRMED |
| **SFig peak_2D_embeddings** | n/a *(actual: `sc_rapids`)* | `notebooks/Fig_peak_umap/10_umap_hyperparams.py` (a, b, c, d) | CONFIRMED |
| **SFig peak_composition** | `single-cell-base` *(actual: `sc_rapids`)* | `notebooks/Fig_peak_umap/01_EDA_annotate_peak_umap.py` (c, d, e); `.../09_annotate_peak_umap_celltype_timepoints.py` (b, f, g, h); `.../02_01_hierarchical_clustering_peak_umap.py` (a-left); `.../02_02_sub-clustering_peak_umap.py` (a-right); `.../peaks_pb_preprocessing.py` (b upstream) | CONFIRMED — panel-d total-accessibility histogram sub-element is **approx** (UMAP confirmed). |
| **SFig peak_umap_coarse_pathways** | n/a *(actual: `sc_rapids`)* | `notebooks/Fig_peak_umap/04_associate_peaks_genes_peak_umap.py` (a, b) | CONFIRMED — manuscript filename is a re-export of `all_clusters_pathways_labels_inside.pdf`. |
| **SFig peak_umap_motifs** | `celloracle_env` | `notebooks/Fig_peak_umap/03_motif_enrichment_analysis_v3_coarse_fine.py` (a, b, c, e, f); `.../03_02_motif_enrichment_leiden_fine.py` (d); `.../03_motif_enrichment_leiden_subclusters.py` (d, alt/earlier variant — `verified=false`) | CONFIRMED (panel d has two candidate producers). |
| **SFig peak_umap_subclusters** | `single-cell-base` *(actual: `sc_rapids`)* | `notebooks/Fig_peak_umap/02_02_sub-clustering_peak_umap.py` (a, b, c — coarse clusters 22, 13, 1) | CONFIRMED |
| **SFig peaks_parts_list** | `single-cell-base` *(panel-d FIMO needs `gReLu` env)* | `notebooks/EDA_peak_parts_list/09d_umap_interesting6.py` (a); `.../09c_peak_profile_examples.py` (b); `.../09c_marker_gene_profiles.py` (b co-cand); `.../09h_all_celltypes_top_peaks.py` (c, e data); `.../09d_umap_all_celltypes_v2.py` (c plot); `.../09d_advanced_visualizations.py` (c alt); `.../09f_interesting6_motif_heatmap.py` (d); `.../09l_hemangioblasts_combined_panels.py` (e, f-top); `.../09n_slc4a1a_tfr1a_motif_compare.py` (f-top); `.../09m_hemangioblasts_top5_umap.py` (e/f context); `scripts/utils/rank_synthetic_enhancers.py` (f-bottom method, `verified=false`) | likely — panel-a dashed outlines manual; panel-b `myh7l` peak **ad-hoc/uncommitted**; panel-e top20 table is an **uncommitted slice** (approx); panel-f synthetic-element boxes schematic. |
| **SFig peak_umap_cross_species** | `single-cell-base` *(actual: `sc_rapids`)* | `notebooks/EDA_peak_umap_cross_species/00_preprocess_peak_objects_Argelaguet2022.py` (a, b mouse); `.../01_preprocess_peak_objects_Domcke_Shendure_2020.py` (a, b human); `notebooks/Fig_peak_umap/09_annotate_peak_umap_celltype_timepoints.py` (b zebrafish); `notebooks/Fig_peak_umap/01_EDA_annotate_peak_umap.py` (a zebrafish) | CONFIRMED |
| **SFig coaccess** | `celloracle_env` (upstream: R) | `notebooks/Fig2_ATAC_RNA_correlation_metacells/archive/FigSI_chromatin_coaccess_dynamics.ipynb` (b); `scripts/preprocessing/run_02_compute_CCANs_cicero_parallelized.R` (upstream data for b) | CONFIRMED — panel a schematic. |
| **SFig celloracle_workflow** | n/a | *(none — hand-drawn schematic)* | CONFIRMED — **unresolved by design**: no producing notebook. Pipeline implemented (not drawn) by `scripts/preprocessing/run_03/04/05_celloracle_*.py`. |
| **SFig grn_whole_embryo** | `celloracle_env` | `notebooks/Fig3_GRN_dynamics/Fig3_GRN_dynamics.py` (b, c, d, e, f, g) | CONFIRMED — panel a schematic. |
| **SFig grn_comparison** | `celloracle_env` | `notebooks/Fig3_GRN_dynamics/SI_QC_GRN_network_comparison_TDR118_TDR119_reseq_v2.py` (c, d, e, f); `notebooks/Fig3_GRN_dynamics/Fig3_GRN_dynamics.py` (a, b data basis — `verified=false`) | likely — panels a/b data-supported but exact node-diagram/bar rendering unresolved. |
| **SFig subGRN** | `celloracle_env` *(EDA notebook kernel: `sc_rapids`)* | `notebooks/Fig3_GRN_dynamics/EDA_extract_subGRN_reg_programs.py` (d, e, f, g, h, i); `notebooks/Fig3_GRN_dynamics/utils/visualize_hemangioblasts_shell_editable.py` (j); `notebooks/Fig_peak_umap/02_02_sub-clustering_peak_umap.py` (a); `notebooks/Fig_peak_umap/08_subset_reg_programs.py` (a support); `notebooks/Fig3_GRN_dynamics/utils/module_plot_subGRN.py` (j helper) | CONFIRMED — panels b/c are method schematics. |

### Notes & items to confirm with the author

- **V2 renumbering (directory names lag the manuscript):** manuscript **Fig 3 = peak UMAP** lives in `notebooks/Fig_peak_umap/`; manuscript **Fig 4 = GRN dynamics** lives in `notebooks/Fig3_GRN_dynamics/`; manuscript **Fig 5 = in-silico KO** lives in `notebooks/Fig4_in_silico_KO/`. Internal docstrings still say "Fig3/Fig4" — expected, not a mismatch.
- **New resource figure:** **Fig 6** (web portal) is new; it is a composite montage of dashboard screenshots, not produced by any single analysis notebook.
- **Environment caveat:** the four documented conda envs (`single-cell-base`, `seacells`, `celloracle_env`, R) do not capture the GPU env. Most `Fig_peak_umap/` and subGRN-extraction notebooks actually run under **`sc_rapids`** (rapids_singlecell/cupy); reported as `single-cell-base` (closest scanpy enum) and flagged in-table. Panel-d FIMO in SFig peaks_parts_list needs a custom **`gReLu`** env.
- **Unresolved / flag-for-author items:**
  - **Fig 1 panel e** — RNA/ATAC gene-activation-along-NMP-trajectory plot has no committed producer (final PDFs exist on disk only).
  - **Fig 6** — portal montage + live CZ CELLxGENE/dashboard frontend is **not in this repo**; cross-figure re-presentation pointers (panels c/d) unverified at notebook level.
  - **SFig QC panel c-right** — cells-passing-QC bar plot has no found producer.
  - **SFig celloracle_workflow** — schematic with no producing notebook (by design).
  - **SFig grn_comparison panels a/b** — data computed in `Fig3_GRN_dynamics.py` but exact network/bar rendering not located (likely CellOracle built-in plot finalized in Illustrator).
- **Notable reassignments from the old draft:** Fig 3 panels c/e/f moved off `peak_umap_leiden_clusts_viz_v1.py` (which only emits clustering-QC scatters) onto `09_annotate_*` and `02_01_*`; SFig coaccess env corrected `seacells → celloracle_env` (notebook imports celloracle); SFig subGRN panel-a producer corrected `08_subset_reg_programs.py → 02_02_sub-clustering_peak_umap.py`, and panel-j pinned to `visualize_hemangioblasts_shell_editable.py` (cluster 26_11, not the EDA notebook's 26_8).

---

## Data Availability

The Zebrahub-Multiome atlas comprises **raw sequencing data**, **processed/annotated single-cell objects**, and a set of **derived analysis objects** that the notebooks in this repository load directly. Both raw and processed data must be downloadable by reviewers and readers; the table below maps each access point to what it provides.

> **Reviewer note:** the notebooks in this repo currently read processed objects from absolute lab paths under `/hpc/projects/.../zebrahub_multiome/data/...`. For the public release these objects are deposited at the archive(s) below; after download, update the path constants at the top of each notebook (or set the documented environment variables) to point at your local copy. See **Reproducing the Analysis → 2. Get the data**.

### Access points

| Resource | What it contains | Where to get it |
|---|---|---|
| **Interactive atlas (browse only)** | Cell-by-gene/peak views, cell-type and timepoint annotations for visual exploration. Not a substitute for downloadable files. | https://zebrahub.ds.czbiohub.org/ |
| **Raw sequencing data** | Demultiplexed snRNA-seq + snATAC-seq FASTQs / aligned reads for all multiome libraries and timepoints. | NCBI SRA BioProject **[PRJNA1164307](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1164307)** |
| **Processed & annotated single-cell objects** | Integrated multiome object(s) (cell-level RNA+ATAC) with cell-type, timepoint, and lineage annotations; the starting point for Fig. 1. | Processed `.h5ad` + CellOracle `.oracle` objects — [Google Drive folder](https://drive.google.com/drive/folders/1lk3VdHQuEiXTHZufMXp0C84T3pncqdCF?usp=share_link) |
| **Derived analysis objects** (loaded by the notebooks) | The pseudobulk / metacell / GRN objects listed below. | [Google Drive folder](https://drive.google.com/drive/folders/1lk3VdHQuEiXTHZufMXp0C84T3pncqdCF?usp=share_link) (same folder as above) |

### Derived objects the notebooks load (map portal → files)

The notebooks consume a small set of processed objects rather than re-deriving everything from raw reads. Reviewers should obtain these from the processed-data archive above. Representative objects (paths as referenced in the notebooks):

- **Peak pseudobulk matrices (snATAC), danRer11** — peaks × (cell-type × timepoint) pseudobulk `.h5ad` objects, e.g. `data/annotated_data/objects_v2/peaks_by_ct_tp_master*.h5ad`, `peaks_by_ct_tp_raw_counts_pseudobulked_median_scaled*_all_peaks*.h5ad`, `peaks_by_pb_*_leiden_*_merged_annotated*.h5ad` (peak-UMAP/clustering inputs for Fig. 1 & peak-UMAP panels), plus the leiden-cluster annotation CSVs (`leiden_*_by_pseudobulk.csv`, `num_cells_per_pseudobulk_group.csv`).
- **SEACells metacell aggregates (Fig. 2)** — per-sample metacell assignment/aggregate CSVs and objects, e.g. `objects_30cells_per_metacell/{sample}_seacells.csv`, `objects_75cells_per_metacell/{sample}_seacells.csv`, and metacell annotation CSVs (`{sample}_seacells_obs_annotation_ML_coarse.csv`).
- **CellOracle GRN exports (Figs. 4 & 5)** — per cell-type/timepoint GRN tables and perturbation outputs, e.g. `celloracle_grn_celltype_timepoint.csv` and `{data_id}/cosine_similarity_df_metacells_{data_id}.csv` (WT-vs-KO cosine-similarity outputs).
<!-- author note: confirm exact public filenames/URLs for each deposited object -->

### Reference genome

- Genome build: **GRCz11** (zebrafish; equivalent to UCSC **danRer11**). Read alignment and peak calling used a 10x **Cell Ranger ARC 2.0.2** reference built from the Ensembl **GRCz11** genome (`Danio.rerio.fa`) and annotation (`Danio.rerio.gtf`), primary chromosomes 1–25 + MT. TF-motif analyses used **CIS-BP v2 (*Danio rerio*)** and **JASPAR 2024**.

### Data-availability statement (template for the manuscript)

> Raw sequencing data have been deposited at NCBI SRA BioProject PRJNA1164307 and are publicly available as of the date of publication. Processed and annotated single-cell objects, and the derived pseudobulk/metacell/GRN objects required to reproduce the figures, are available at Google Drive (folder 1lk3VdHQuEiXTHZufMXp0C84T3pncqdCF) and through the Zebrahub data portal (https://zebrahub.ds.czbiohub.org/). All original analysis code is available in this repository (a citable Zenodo snapshot will be deposited upon acceptance). Any additional information required to reanalyze the data is available from the lead contact upon request.

---

## Reproducing the Analysis (Instructions for Use)

Reproducing a figure follows three steps: **(1) set up the environment, (2) get the data, (3) run the notebook for that figure.** Each figure block uses one of three conda environments; match the environment to the figure before launching the notebook.

### System requirements

- **OS:** Linux (developed and run on a CentOS/RHEL 8 HPC cluster; should work on any modern x86-64 Linux). macOS may work for the lighter `single-cell-base` analyses but is not tested.
- **Software:** `conda`/`mamba` (Miniconda/Miniforge); Python 3.x and R (R is used for the Signac–Cicero–Seurat preprocessing); `jupyter`/`jupyterlab`; `jupytext`.
- **Hardware:** A workstation with ≥ 32 GB RAM is sufficient for most figure notebooks against the provided processed objects. The full atlas and GRN/perturbation steps benefit from a larger memory node; some preprocessing/trajectory steps were run on GPU on the HPC. No specialized hardware is required to reproduce figures from the deposited processed objects.
- **Version control of notebooks:** notebooks are paired to `.py:percent` files via `jupytext.toml`; editing either side keeps the pair in sync.

### 1. Set up the environments

There are **three** environments (one per analysis stack):

| Environment | YAML | Used for | Key stack |
|---|---|---|---|
| `single-cell-base` | `environments/single-cell-base.yml` | **Figs 1 & 3** (atlas/QC, peak UMAP) | scanpy, anndata, numpy, pandas |
| `seacells` | `environments/seacells.yml` | **Figure 2** (RNA–ATAC correlation via metacells) | [SEACells](https://github.com/dpeerlab/SEACells) |
| `celloracle_env` | `environments/celloracle_env.yml` | **Figs 4 & 5** (GRN dynamics, in-silico KO) | [CellOracle](https://morris-lab.github.io/CellOracle.documentation/) |

> **Note.** The committed `environments/*.yml` are portability-cleaned (plain `name:`, no `prefix:`), so `conda env create -f environments/<name>.yml` works directly. Pins are Linux-64; on macOS/Windows install the headline packages (scanpy / SEACells / CellOracle) per their upstream docs.

Activate the environment that matches the figure you are reproducing, then register a Jupyter kernel, e.g.:

```bash
conda activate single-cell-base   # or seacells / celloracle_env
python -m ipykernel install --user --name single-cell-base
```

### 2. Get the data

Download the processed/derived objects from the archive(s) listed in **Data Availability** (raw data is only needed if you want to re-run preprocessing from FASTQs). Then point the notebooks at your local copy:

- Open the notebook (or its paired `.py`) and update the path constants near the top (e.g. the `data/annotated_data/objects_v2/...` and metacell/GRN paths) to your download location. <!-- author note: consider a single base-path env var instead of per-notebook path edits -->
- Optional preprocessing from raw data (R; Signac/Cicero/Seurat) lives in `scripts/preprocessing/` and `scripts/R_scripts/`, e.g. `scripts/preprocessing/run_01_preprocess_multiome_object_signac.R` and `scripts/preprocessing/run_02_compute_CCANs_cicero*.R`; Seurat→h5ad export via `scripts/export_seurat_*.R`. These regenerate the processed objects but are not required to reproduce figures from the deposited objects.

### 3. Run the notebook for each figure

Launch JupyterLab with the matching kernel and run the notebook top-to-bottom. See the **Figure → Notebook Map** above for the complete figure-to-notebook table; outputs (figure panels) are written to a local `figures/`-style directory (excluded from the public repo).

### Demo

A reviewer can verify the pipeline end-to-end with the lightest path: create the `single-cell-base` environment, download the peak-pseudobulk objects, and run `notebooks/Fig1_atlas_QC/Fig1_bcde_atlas_integrated.ipynb`. Expected runtime is on the order of minutes-to-tens-of-minutes on a standard workstation once the processed object is downloaded. <!-- author note: optionally add a small example object + expected output for a self-contained demo -->

---

## License

This project is licensed under the **BSD 3-Clause License** — see the [`LICENSE`](LICENSE) file for the full text.

> Copyright (c) 2023, Chan Zuckerberg Biohub

The BSD 3-Clause license applies to the code in this repository. Deposited datasets (raw and processed) may be released under their own data-use terms via the hosting archive and the Zebrahub portal; see **Data Availability**.

## Citation

If you use this code, the atlas, or the derived data, please cite the Zebrahub-Multiome manuscript.

**Preprint:**
> [Zebrahub-Multiome: Uncovering Gene Regulatory Network Dynamics During Zebrafish Embryogenesis. *bioRxiv* (2024).](https://www.biorxiv.org/content/10.1101/2024.10.18.618987v1) doi:10.1101/2024.10.18.618987

**Peer-reviewed version:**
> _Peer-reviewed citation to be added upon publication._

Please also consider citing the key tools this Resource builds on where relevant: **SEACells** (metacells, Fig. 2), **CellOracle** (GRN + in-silico KO, Figs. 4–5), and **Signac/Cicero** (ATAC preprocessing).

**Code archive:** a permanent, citable Zenodo snapshot of this repository will be deposited upon acceptance.