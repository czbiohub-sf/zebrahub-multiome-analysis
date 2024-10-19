# Zebrahub-Multiome: a time-resolved single-cell multi-omic (RNA+ATAC) sequencing atlas of zebrafish development

## Summary

This repo contains scripts (Python/R) and Jupyter notebooks used for processing and analyzing single-cell Multiome datasets from early zebrafish embryos. 
The manuscript is published here: 
Data is accessible through the Zebrahub portal: https://zebrahub.ds.czbiohub.org/ 


## Organization
The structure of this repo is illustrated below.
```
├── figures 
├── notebooks                 
│   ├── 
├── scripts                 
│   ├──
├── environments
├── LICENSE
└── README.md
```

### figures/

This directory contains various plots to make the main and supplementary figures. Note that the figures/ directory saves the output from the Jupyter notebooks or scripts from notebooks/, or scripts/ directories.

### notebooks/

This directory includes a series of Jupyter notebooks that were used to create each figure or figure panel. We provide a structured tree diagram (below) representing the organization of the notebooks directory. This diagram delineates the specific notebooks responsible for generating each panel of the figures.

notebooks
```
|
├── Fig1_atlas_QC
│   └── Fig1_bcde_atlas_integrated.ipynb
│   ├── Fig1_f_R_coverage_plot_optimization.ipynb
│   ├── Fig1_gh_RNA_ATAC_whole_embryo.ipynb
│
├── Fig2_ATAC_RNA_correlation_metacells
│   ├── Fig2_compute_RNA_ATAC_corr_seacells.ipynb 
│   ├── FigSI_chromatin_coaccess_dynamics
│
├── Fig3_GRN_dynamics
│   ├── Fig3_GRN_dynamics.ipynb 
│   ├── SI_QC_GRN_network_comparison_TDR118_TDR119_reseq_v2.ipynb
│
├── Fig4
│   └── Fig4_in_silico_KO.ipynb
```
### scripts/

These are R/Python modules that contain the bulk of the code used for data processing. Please note that these scripts are explicitly written for, and specific to this project and/or the Zebrahub-Multiome project. They are not intended to form a stand-alone or general-purpose Python package.

## /environments: Setting up the conda environments

We used three conda environments listed below.
- 1) single-cell-base: basic scanpy, numpy, pandas environment for Figure 1
- 2) seacells: environment with SEACells for Figure 2 (see [SEACells](https://github.com/dpeerlab/SEACells) for troubleshooting)
- 3) celloracle_env:  environment for Figures 3 and 4 (see [CellOracle documentation](https://morris-lab.github.io/CellOracle.documentation/installation/index.html) for troubleshooting)

## License

This project is licensed under the BSD 3-Clause license - see the LICENSE file for details.

## Citation
[Zebrahub-Multiome: Uncovering Gene Regulatory Network Dynamics During Zebrafish Embryogenesis (biorxiv preprint)](https://www.biorxiv.org/content/10.1101/2024.10.18.618987v1)
