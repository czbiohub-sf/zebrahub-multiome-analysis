import argparse
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from multiprocessing import Pool

def compute_correlation(args):
    rna_vector, atac_vector = args
    if np.std(rna_vector) == 0 or np.std(atac_vector) == 0:
        return np.nan  # Return NaN if either vector is constant
    else:
        return pearsonr(rna_vector, atac_vector)[0]

def main(rna_data_path, atac_data_path, chromosome, output_dir):
    # Load the subsetted data
    rna_data = sc.read_h5ad(rna_data_path)
    atac_data = sc.read_h5ad(atac_data_path)
    
    # Ensure data is in dense format for correlation computation
    rna_matrix = rna_data.X.toarray()
    atac_matrix = atac_data.X.toarray()
    
    # Prepare for parallel computation
    pool = Pool(processes=8)  # Adjust number of processes based on your environment
    num_genes = rna_matrix.shape[1]
    num_peaks = atac_matrix.shape[1]
    
    tasks = [(rna_matrix[:, i], atac_matrix[:, j]) for i in range(num_genes) for j in range(num_peaks)]
    correlations = pool.map(compute_correlation, tasks)
    
    # Reshape correlations to a matrix and create a DataFrame
    corr_matrix = np.array(correlations).reshape(num_genes, num_peaks)
    gene_peak_correlations = pd.DataFrame(corr_matrix, index=rna_data.var_names, columns=atac_data.var_names)
    
    # Save the correlation matrix
    output_path = f"{output_dir}/correlations_chr_{chromosome}.csv"
    gene_peak_correlations.to_csv(output_path)
    print(f"Saved correlation matrix to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute correlations between RNA and ATAC for a specific chromosome.")
    parser.add_argument("--rna_data_path", required=True, help="Path to the RNA data file for a specific chromosome.")
    parser.add_argument("--atac_data_path", required=True, help="Path to the ATAC data file for a specific chromosome.")
    parser.add_argument("--chromosome", required=True, help="Chromosome identifier.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output correlation matrix.")

    args = parser.parse_args()

    main(args.rna_data_path, args.atac_data_path, args.chromosome, args.output_dir)