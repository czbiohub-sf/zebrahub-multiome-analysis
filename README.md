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
├── LICENSE
└── README.md
```

### figures/

This directory contains various plots to make the main and supplementary figures. Note that the figures/ directory saves the output from the Jupyter notebooks or scripts from notebooks/, or scripts/ directories.

### notebooks/

This directory includes a series of Jupyter notebooks that were used to create each figure or figure panel. We provide a structured tree diagram (below) representing the organization of the notebooks directory. This diagram delineates the specific notebooks responsible for generating each panel of the figures.

notebooks
|
├── 0.Set_timestamp.ipynb
├── enrichment
|   ├── 1.QC_filter_and_impute.ipynb
|   ├── 2.Batch_selection.ipynb
|   ├── 3.correlation_filter.ipynb
|   ├── 4.NOC_processing.ipynb
|   └── output
|       ├── correlation_tables
|       |   └── ..
|       ├── enrichment_and_volcano_tables
|       |   └── ..
|       └── preprocessing
|           └── ..
├── Fig1
│   └── panel_L
│       ├── Fig1_L_enrichment_heatmap.ipynb
│       └── output
│           └── ..
├── Fig2
│   ├── panel_B
│   │   ├── Fig2_B_heatmap.ipynb
│   │   └── output
│   │       └── ..
│   ├── panel_C
│   │   ├── Fig2_C_consensus_annotation.ipynb
│   │   └── output
│   │       └── ..
│   └── panel_D
│       ├── Fig2_D_umap.ipynb
│       └── output
│           └── ..
├── Fig3
│   ├── panels_A_B_F
│   │   ├── Fig3_A_B_F_local_k-NN_network.ipynb
│   │   └── output
│   |       └── ..
│   └── panels_C_D
│       ├── Fig3_C_D_cluster_connectivity_and_Jaccard_coeff.ipynb
│       └── output
│           └── ..
├── Fig4
│   └── panel_D
│       └── Please_read.txt
├── Fig5
│   ├── 0.Set_fig5_timestamp.ipynb
│   ├── panel_A
│   │   ├── 1.infected_enrichment
│   │   │   ├── 1.QC_filter_and_impute.ipynb
│   │   │   ├── 2.Batch_selection.ipynb
│   │   │   ├── 3.correlation_filter.ipynb
│   │   │   ├── 4.NOC_processing.ipynb
│   │   │   └── output
│   │   │       └── ..
│   │   ├── 2.control_enrichment
│   │   │   ├── 1.QC_filter_and_impute.ipynb
│   │   │   ├── 2.Batch_selection.ipynb
│   │   │   ├── 3.correlation_filter.ipynb
│   │   │   ├── 4.NOC_processing.ipynb
│   │   │   └── output
│   │   │       └── ..
│   │   └── 3.aligned_umap
│   │       ├── Fig5_A_aligned_umap.ipynb
│   │       └── output
│   │           └── ..
│   ├── panel_B
│   │   ├── Fig5_B_remodeling_score.ipynb
│   │   └── output
│   │       └── ..
│   ├── panel_C
│   │   ├── Fig5_C_umap_with_leiden_labels.ipynb
│   │   └── output
│   │       └── ..
│   ├── panel_D
│   │   ├── Fig5_D_trajectory.ipynb
│   │   └── output
│   │       └── ..
│   └── panel_E
│       ├── Fig5_E_Sankey_plot.ipynb
│       └── output
│           └── ..
└─── Supplementary_figures
    ├── Suppl_fig1
    │   ├── Suppl_fig1_marker_expression_in_cell_lines.ipynb
    │   └── output
    │       └── ..
    ├── Suppl_fig2
    │   └── README.md
    ├── Suppl_fig3
    │   ├── Suppl_fig3_faceted_volcano_plots.ipynb
    │   └── output
    │       └── ..
    ├── Suppl_fig4
    │   ├── Suppl_fig4_enrichment_heatmap_all_IPs.ipynb
    │   └── output
    │       └── ..
    ├── Suppl_fig5
    │   ├── panel_A
    │   │   ├── Suppl_fig5_A_IP_correlation_vs_interaction_stoi.ipynb
    │   │   └── output
    │   │       └── ..
    │   ├── panel_B
    │   │   ├── Suppl_fig5_B_enrichment_entropy.ipynb
    │   │   └── output
    │   │       └── ..
    │   └── panel_C
    │       ├── Suppl_fig5_C_tenary_plots.ipynb
    │       └── output
    │           └── ..
    ├── Suppl_fig6
    │   ├── panel_B
    │   │   └── README.md
    │   ├── panel_C
    │   │   ├── Suppl_fig6_C_XGBoost_classifier.ipynb
    │   │   ├── environment.yml
    │   │   └── output
    │   │       └── ..
    │   └── panel_F_G
    │       ├── Suppl_fig6_F_G_sankey_and_confusion.ipynb
    │       └── output
    │           └── ..
    ├── Suppl_fig7
    │   └── panel_C
    │       └── Suppl_fig7_C_upset_plot.ipynb

scripts/

These are Python modules that contain the bulk of the code used for data analysis and figure generation. They are used directly by the Jupyter notebooks discussed above. Please note that these scripts are explicitly written for, and specific to this project and/or the OpenCell project. They are not intended to form a stand-alone or general-purpose Python package.

License

This project is licensed under the BSD 3-Clause license - see the LICENSE file for details.
## Setting up the conda environments
TBD
## Citation
TBD
