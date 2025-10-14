# Cross-Species GRN Analysis: Zebrafish, Mouse, and Human NMP Populations

This directory contains the workflow for cross-species comparison of gene regulatory networks (GRNs) in neuro-mesodermal progenitor (NMP) populations using CellOracle.

## Overview

The goal is to identify conserved transcription factors (TFs) regulating NMP differentiation across vertebrate species using scRNA-seq data and CellOracle's default base GRNs (TSS-based, avoiding the need for ATAC-seq data processing).

## Workflow

### Phase 1: Data Download and Exploration

**Script**: `scripts/cross_species/download_argelaguet_data.sh`
- Downloads pre-processed mouse organogenesis scRNA-seq data from Argelaguet et al., 2022
- Clones analysis code repository from GitHub

**Notebook**: `01_explore_mouse_data.py`
- Loads and explores mouse scRNA-seq data structure
- Identifies NMP populations and cell-type annotations
- Checks data quality and available embeddings

**Run**:
```bash
# Download data
bash ../../scripts/cross_species/download_argelaguet_data.sh

# Explore data (in Jupyter or convert to notebook)
jupytext --to notebook 01_explore_mouse_data.py
jupyter notebook 01_explore_mouse_data.ipynb
```

### Phase 2: GRN Computation with Default Base GRN

**Script**: `scripts/cross_species/compute_GRN_mouse_default_baseGRN.py`
- Reusable script for computing cell-type-specific GRNs
- Uses CellOracle's built-in mouse base GRN (no ATAC-seq required)
- Performs KNN imputation and GRN inference

**Notebook**: `02_mouse_NMP_GRN_analysis.py`
- Subsets NMP populations from full dataset
- Runs GRN computation pipeline
- Visualizes GRN structure and top TFs

**Run**:
```bash
# Run GRN computation (update paths in the script)
python ../../scripts/cross_species/compute_GRN_mouse_default_baseGRN.py \
    --output_path /path/to/output/ \
    --adata_path /path/to/mouse_nmp_subset.h5ad \
    --data_id mouse_argelaguet_nmp \
    --annotation celltype \
    --dim_reduce X_umap \
    --base_grn_type mouse_scATAC_atlas \
    --n_hvg 3000 \
    --alpha 10 \
    --n_jobs 8
```

### Phase 3: In Silico Knock-Out Analysis

**Script**: `scripts/cross_species/insilico_KO_cross_species.py`
- Reusable script for systematic TF perturbation analysis
- Simulates gene knock-outs and quantifies effects
- Ranks TFs by importance

**Notebook**: `03_mouse_insilico_KO.py`
- Defines candidate TFs for KO simulation
- Runs perturbation analysis
- Quantifies and visualizes perturbation effects
- Generates TF importance rankings

**Run**:
```bash
# Run in silico KO analysis
python ../../scripts/cross_species/insilico_KO_cross_species.py \
    --oracle_path /path/to/oracle.links \
    --adata_path /path/to/adata.h5ad \
    --output_path /path/to/output/ \
    --data_id mouse_argelaguet_nmp \
    --list_KO_genes "Tbx6,Sox2,Msgn1,T" \
    --annotation celltype \
    --basis X_umap \
    --n_propagation 3 \
    --n_jobs 4
```

### Phase 4: Cross-Species Comparison

**Script**: `scripts/cross_species/map_orthologs_utils.py`
- Utility functions for ortholog mapping
- Downloads ortholog database from Ensembl BioMart
- Maps TFs across zebrafish, mouse, and human

**Notebook**: `04_zebrafish_mouse_comparison.py`
- Maps orthologous TFs between zebrafish and mouse
- Compares TF importance scores across species
- Identifies conserved vs species-specific regulators
- Visualizes cross-species conservation

**Run**:
```bash
# Open notebook and run analysis
jupytext --to notebook 04_zebrafish_mouse_comparison.py
jupyter notebook 04_zebrafish_mouse_comparison.ipynb
```

## Directory Structure

```
notebooks/Fig_cross_species_GRN/
├── README.md                              # This file
├── 01_explore_mouse_data.py               # Data exploration
├── 02_mouse_NMP_GRN_analysis.py           # GRN computation
├── 03_mouse_insilico_KO.py                # In silico KO experiments
└── 04_zebrafish_mouse_comparison.py       # Cross-species comparison

scripts/cross_species/
├── download_argelaguet_data.sh            # Data download script
├── compute_GRN_mouse_default_baseGRN.py   # GRN computation (reusable)
├── insilico_KO_cross_species.py           # KO simulation (reusable)
└── map_orthologs_utils.py                 # Ortholog mapping utilities

data/public_data/
├── mouse_argelaguet_2022/
│   ├── anndata.h5ad                       # Mouse scRNA-seq data
│   ├── mouse_nmp_subset.h5ad              # NMP subset
│   └── celloracle_outputs/                # GRN and KO results
└── cross_species_comparison/              # Cross-species results
    ├── zebrafish_mouse_human_orthologs.csv
    ├── zebrafish_mouse_TF_comparison.csv
    └── conserved_TFs_zebrafish_mouse.csv

figures/cross_species_GRN/                  # Output figures
```

## Data Sources

### Mouse (Argelaguet et al., 2022)
- **Paper**: Mouse organogenesis cell atlas reveals temporal and spatial dynamics of chromatin accessibility
- **GEO**: GSE205117
- **Pre-processed data**: FTP server / Dropbox (see download script)
- **GitHub**: https://github.com/rargelaguet/mouse_organogenesis_10x_multiome_publication

### Human (Hamazaki et al., 2024) - Future
- **Paper**: Embryoid body culture of human pluripotent stem cells as a model of the embryo proper
- **GEO**: GSE208369
- **GitHub**: https://github.com/shendurelab/Human-RA-Gastruloid/
- **Status**: Awaiting processed data from authors

### Zebrafish (This study)
- **Data**: Zebrahub multiome atlas
- **Location**: `/data/processed_data/09_NMPs_subsetted_v2/` (update path)
- **In silico KO results**: From Fig4 analysis

## Key Dependencies

- **celloracle_env** conda environment (see `environments/celloracle_env.yml`)
- CellOracle (for GRN inference and perturbation analysis)
- scanpy (for scRNA-seq data processing)
- pybiomart (for ortholog mapping via Ensembl)
- matplotlib, seaborn (for visualization)

## Expected Outputs

1. **Cell-type-specific GRNs** for mouse NMP populations
2. **TF importance rankings** from in silico KO experiments
3. **Ortholog-mapped comparison** between zebrafish and mouse
4. **Conserved TF programs** across species
5. **Publication-ready figures** showing cross-species conservation

## Notes

- All notebooks are in **Jupytext .py format** for easy editing in IDE
- Convert to .ipynb using: `jupytext --to notebook file.py`
- Scripts are designed to be reusable for human data when available
- Uses CellOracle's **default base GRNs** (no ATAC-seq processing needed)

## Authors

- Yang-Joon Kim
- Last updated: 2025-10-09

## References

1. Argelaguet et al. (2022). Mouse organogenesis cell atlas. bioRxiv.
2. Hamazaki et al. (2024). Human embryoid body culture. Nature Cell Biology.
3. Kamimoto et al. (2023). CellOracle. Nature.
