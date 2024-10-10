# A collection of functions (modules) using network topology metrics
# NOTE: The network topology metrics is already computed by CellOracle (for the filtered GRN - 2000 edges)

# Example: The function here were used in the notebook below:
# zebrahub-multiome-analysis/02_GRN/04_Network_analysis/QC_GRN_network_comparison_TDR118_TDR119.ipynb

# Prerequisite
# ! module load anaconda
# ! conda activate celloracle_env

# 0. Import
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

import celloracle as co
co.__version__

# Load the GRN (Links object contains both filtered and unfiltered GRNs from all cell-types)
# GRN = co.load_hdf5(GRN_path)

# df_GRN = GRN.merged_score

def compute_corr_betwn_GRNs(df_GRN1, df_GRN2, celltype1, celltype2, network_metric):
    df1 = df_GRN1[df_GRN1.cluster==celltype1]
    df2 = df_GRN2[df_GRN2.cluster==celltype2]

    # Step 1. Get a union of gene_names
    gene_names = set(df1.index).union(df2.index)
    len(gene_names)

    # Step 2. Create a new dataframe with matching indices
    new_df1 = df1[df1.index.isin(gene_names)]
    new_df2 = df2[df2.index.isin(gene_names)]

    # Step 3. Fill missing values with 0
    new_df1 = new_df1.reindex(gene_names) #fill_value=0
    new_df2 = new_df2.reindex(gene_names)

    # Step 4. Create the zipped DataFrame
    zipped_df = pd.DataFrame({'metric_df1': new_df1[network_metric], 'metric_df2': new_df2[network_metric]})
    zipped_df

#     # Step 5. Generate scatter plots, with Pearson correlation coeff.
#     plt.scatter(x=zipped_df.metric_df1,
#                 y=zipped_df.metric_df2)
#     plt.xlabel("TDR118: "+ network_metric + "_" + celltype1)
#     plt.ylabel("TDR119: "+ network_metric + "_" + celltype2)
#     plt.title(network_metric)


    # Annotate the plot with the correlation coefficient
    corr = zipped_df.metric_df1.corr(zipped_df.metric_df2,
                                    method = "pearson")
    return corr



###### EXAMPLE CODE SNIPPET ######
# # define the cell-types
# celltypes = ['Adaxial_Cells', 'Differentiating_Neurons', 'Endoderm',
#        'Epidermal', 'Lateral_Mesoderm', 'Muscle', 'NMPs',
#        'Neural_Anterior', 'Neural_Crest', 'Neural_Posterior', 'Notochord',
#        'PSM', 'Somites', 'unassigned']

# # define the degree_metrics
# degree_metrics = ["degree_all","degree_in","degree_out",
#                   "degree_centrality_all","degree_centrality_in",
#                   "degree_centrality_out","eigenvector_centrality"]

# # define an empty dictionary to save the list of corr.coeff (per metric)
# dict_corr_coeff = {}
# # dict_corr_coeff["degree_centrality_all_same"] = corr_same_celltypes
# # dict_corr_coeff['degree_centrality_all_diff'] = corr_diff_celltypes

# # For loop to go over all degree metrics
# for metric in degree_metrics:
#     # define empty series to save the correlation coefficients
#     corr_same_celltypes =[]
#     corr_diff_celltypes = []

#     for ct1 in celltypes:
#         for ct2 in celltypes:
#             corr_coeff = compute_corr_betwn_GRNs(df_GRN1, df_GRN2, ct1, ct2, metric)

#             if ct1==ct2:
#                 corr_same_celltypes.append(corr_coeff)
#             else:
#                 corr_diff_celltypes.append(corr_coeff)
                
#     # define the keys for the dictionary
#     namekey_same = metric + "_same"
#     namekey_diff = metric + "_diff"
#     # save the corr.coeff. into the dictionary
#     dict_corr_coeff[namekey_same] = corr_same_celltypes
#     dict_corr_coeff[namekey_diff] = corr_diff_celltypes
    
#     # generate plots (optional)
#     # define the bin width
#     bin_width = 0.01

#     # Calculate the number of bins for each histogram
#     num_bins1 = int((max(corr_same_celltypes) - min(corr_same_celltypes)) / bin_width)
#     num_bins2 = int((max(corr_diff_celltypes) - min(corr_diff_celltypes)) / bin_width)

#     plt.figure()
#     sns.histplot(corr_same_celltypes, kde=True, bins=num_bins1, stat="density", label='same_celltypes')
#     sns.histplot(corr_diff_celltypes, kde=True, bins=num_bins2, stat="density", label='diff_celltypes')
#     plt.xlabel("Pearson correlation: GRN1/GRN2")
#     plt.ylabel("density")
#     plt.title(metric)
#     plt.legend()

#     #plt.savefig("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/network_plots_TDR118/QC_corr_centrality_TDR118_TDR119_allCelltypes.pdf")
#     plt.savefig(figpath + "QC_corr_" + metric + "_TDR118_TDR119_allCelltypes.pdf")
#     plt.savefig(figpath + "QC_corr_" + metric + "_TDR118_TDR119_allCelltypes.png")
#     plt.show()

# # A dictionary that has the distributions of corr.coeff., from all metrics
# data = dict_corr_coeff
# metric_names = degree_metrics

# # Extract metric names and categories
# # metric_names = list(set(key.split('_')[0] for key in data.keys()))
# categories = ['same', 'diff']

# # Compute means and standard errors for each metric and category
# means = {}
# std_errors = {}
# for metric in metric_names:
#     means[metric] = [np.mean(data[f'{metric}_{category}']) for category in categories]
#     std_errors[metric] = [np.std(data[f'{metric}_{category}']) / np.sqrt(len(data[f'{metric}_{category}'])) for category in categories]

# # Create x-values for each metric
# x_values = np.arange(len(metric_names))

# # Create a scatter plot with grouped data points and error bars
# fig, ax = plt.subplots()

# for i, category in enumerate(categories):
#     y_values = [np.mean(data[f'{metric}_{category}']) for metric in metric_names]
#     error_bar = [np.std(data[f'{metric}_{category}']) / np.sqrt(len(data[f'{metric}_{category}'])) for metric in metric_names]
#     ax.errorbar(x_values, y_values, yerr=error_bar, marker='o', linestyle='None', label=category)

# ax.set_xlabel('degree metric')
# ax.set_ylabel('correlation coefficients \n (across genes)')
# # ax.set_title('Mean Value with Standard Error (Grouped Scatter Plot with Error Bars)')
# ax.set_xticks(x_values)
# ax.set_yticks([0.7, 0.8, 0.9, 1])
# ax.set_ylim([0.7, 1.1])
# ax.set_xticklabels(metric_names, rotation = 45)
# ax.legend(loc="upper right")

# plt.tight_layout()
# plt.savefig(figpath + "corr_coeff_all_degree_metrics_scatter.pdf")
# plt.savefig(figpath + "corr_coeff_all_degree_metrics_scatter.png")
# plt.show()