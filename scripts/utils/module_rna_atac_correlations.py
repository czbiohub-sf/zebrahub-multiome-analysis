# NOTE. "seacells" conda environment is required to run this script.
import numpy as np
import pandas as pd
import scanpy as sc
import ray
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Ray
ray.init()

@ray.remote
def compute_correlation(gene, expr1, expr2):
    # if either exp1 or expr2 are "constant", then
    if np.all(expr1 == expr1[0]) or np.all(expr2 == expr2[0]):
        return gene, np.nan
    else:
        correlation, _ = pearsonr(expr1, expr2)
        return gene, correlation

def compute_gene_correlations(rna_ad, atac_ad, data_id):
    # shared_genes = np.intersect1d(rna_ad.var_names, atac_ad.var_names)
    # print(f"Number of shared genes: {len(shared_genes)}")

    # rna_ad = rna_ad[:, shared_genes]
    # atac_ad = atac_ad[:, shared_genes]
    # assert rna_ad.shape[1] == atac_ad.shape[1]
    print(f"Number of genes in RNA: {rna_ad.shape[1]}")
    print(f"Number of genes in ATAC: {atac_ad.shape[1]}")

    tasks = []
    for gene in rna_ad.var_names:
        expr1 = rna_ad[:, gene].X.toarray().flatten()
        expr2 = atac_ad[:, gene].X.toarray().flatten()
        tasks.append(compute_correlation.remote(gene, expr1, expr2))

    results = ray.get(tasks)

    correlation_dict = dict(results)
    correlation_df = pd.DataFrame(list(correlation_dict.items()), columns=['Gene', data_id])
    correlation_df = correlation_df.set_index("Gene")

    filtered_dict = {k: v for k, v in correlation_dict.items() if v > 0.75}
    print(filtered_dict)

    fig, ax = plt.subplots()
    ax.hist(correlation_dict.values(), bins=20)
    ax.set_xlabel('Pearson Correlation Coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Pearson Correlation Coefficients')

    return correlation_df, fig