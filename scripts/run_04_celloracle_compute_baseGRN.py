# This script is from 02_atac_peaks_to_TFinfo_window_20200801.ipynb notebook from the CellOracle github repo.
# NOTE: Run this script within the celloracle_env conda environment

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import time

import os, sys, shutil, importlib, glob
from tqdm.notebook import tqdm

import celloracle as co
from celloracle import motif_analysis as ma
from celloracle.utility import save_as_pickled_object
co.__version__

from genomepy import Genome

# Input Arguments:
# 1) filepath: filepath for the peaks_TSS_mapped file (below)
# 2) peaks_TSS_mapped: filename for the csv file with peaks mapped to the nearest TSS and filtered for high cicero co-accessibility scores.
# 3) data_id: identifier for the output dataframe file
# 4) ref_genome: reference genome (use the callname in CellOracle database)
# 5) motif_score_threshold: threshold (lower) for the motif score. 10 was used in CellOracle tutorials

# Parse command-line arguments
import argparse

# a syntax for running the python script as the main program (not in a module)
#if __name__ == "__main__":

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Filter and map data using CellOracle")

# Add command-line arguments
parser.add_argument('filepath', type=str, help="File path")
parser.add_argument('peaks_TSS_mapped', type=str, help="Peak file")
parser.add_argument('data_id', type=str, help="data_id")
parser.add_argument('ref_genome', type=str, help="reference genome")
parser.add_argument('motif_score_threshold', type=float, help="motif score threshold")

# Parse the command-line arguments
args = parser.parse_args()

# Access the arguments as attributes of the 'args' object
filepath = args.filepath
peaks_TSS_mapped = args.peaks_TSS_mapped
data_id = args.data_id
ref_genome = args.ref_genome
motif_score_threshold = args.motif_score_threshold

# define the main function here
def celloracle_compute_baseGRN(filepath, peaks_TSS_mapped, 
                                data_id, ref_genome, motif_score_threshold):
    """
    A function to compute a dataframe (base GRN) for peaks-by-TFs using transcription factor motif databases (CisBP v2.0 in most cases unless specified)
    It uses CellOracle-implemented gimmemotif package for motif scan.
    
    Parameters:
        filepath: filepath for the peaks_TSS_mapped file (below). We will save the output baseGRN here as well.
        peaks_TSS_mapped: filename for the csv file with peaks mapped to the nearest TSS and filtered for high cicero co-accessibility scores.
        data_id: identifier for the output dataframe file
        ref_genome: reference genome (use the callname in CellOracle database)
        motif_score_threshold: threshold (lower) for the motif score. 10 was used in CellOracle tutorials
    
    Returns: 
        df: a dataframe of peaks-by-TFs (base GRN)

    Examples:
        filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118_cicero_output/"
        peaks_TSS_mapped = "03_TDR118_processed_peak_file_danRer11.csv"
        data_id = "TDR118"
        ref_genome = "danRer11"
        motif_score_threshold = 10
    """
    # Step1. Load the reference genome
    # PLEASE make sure reference genome is correct.
    # ref_genome = "danRer11"
    print(ref_genome)
    # check if the genome is installed
    genome_installation = ma.is_genome_installed(ref_genome=ref_genome)
    print(ref_genome, "installation: ", genome_installation)

    # if the ref_genome was not installed, we can install it using the below command
    if genome_installation==False:
        import genomepy
        genomepy.install_genome(name="danRer11", provider="UCSC")
    else:
        pass


    # Step2. Load the processed peak data
    # Peaks were filtered for (1) mapping to the nearest TSS, and (2) cicero co-access score >=0.8 with the TSS peak
    peaks = pd.read_csv(filepath + peaks_TSS_mapped, index_col=0)
    peaks.head()

    # Step 3. Instantiate TFinfo object and search for TF binding motifs
    # The motif analysis module has a custom class, `TFinfo`. 
    # The TFinfo object executes the steps below.
    # 1) Converts a peak data into sequences of DNA elements.
    # 2) Scans the DNA sequences searching for TF binding motifs.
    # 3) Post-processes the motif scan results.
    # 4) Converts data into appropriate format. You can convert data into base-GRN. 
    # The base GRN data can be formatted as either a python dictionary or pandas dataframe. 
    # This output will be the final base GRN used in the GRN model construction step.

    # instantiate TFinfo object
    tfi = ma.TFinfo(peak_data_frame=peaks, 
                    ref_genome=ref_genome) 

    peaks = check_peak_format(peaks, ref_genome)


    # Initiate the motif scan
    # Start measuring time
    start_time = time.time()

    # Scan motifs. !!CAUTION!! This step may take several hours if you have many peaks!
    tfi.scan(fpr=0.02, 
            motifs=None,  # If you enter None, default motifs will be loaded.
            verbose=True, n_cpus=-1)

    # End measuring time
    end_time = time.time()
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time/60} minutes")

    # Save tfinfo object (unfiltered)
    tfi.to_hdf5(file_path=filepath + "04_" + data_id + "_peaks_motifs_TFs.celloracle.tfinfo")

    # Check motif scan results
    tfi.scanned_df.head()

    # Step 4. Filtering the motifs
    # Reset filtering 
    tfi.reset_filtering()

    # Do filtering
    tfi.filter_motifs_by_score(threshold=motif_score_threshold)

    # Format post-filtering results.
    tfi.make_TFinfo_dataframe_and_dictionary(verbose=True)

    # Step 5. Get final base GRN
    df = tfi.to_dataframe()
    # Save the result as a dataframe
    df.to_parquet(filepath + "05_" + data_id + "_base_GRN_dataframe" + ".parquet")
    
    return df





# Additional functioins (utilities)
def decompose_chrstr(peak_str):
    """
    Args:
        peak_str (str): peak_str. e.g. 'chr1_3094484_3095479'
        
    Returns:
        tuple: chromosome name, start position, end position
    """
    
    *chr_, start, end = peak_str.split("_")
    chr_ = "_".join(chr_)
    return chr_, start, end


def check_peak_format(peaks_df, ref_genome):
    """
    Check peak format. 
     (1) Check chromosome name. 
     (2) Check peak size (length) and remove sort DNA sequences (<5bp)
    
    """
    
    df = peaks_df.copy()
    
    n_peaks_before = df.shape[0]
    
    # Decompose peaks and make df
    decomposed = [decompose_chrstr(peak_str) for peak_str in df["peak_id"]]
    df_decomposed = pd.DataFrame(np.array(decomposed), index=peaks_df.index)
    df_decomposed.columns = ["chr", "start", "end"]
    df_decomposed["start"] = df_decomposed["start"].astype(int)
    df_decomposed["end"] = df_decomposed["end"].astype(int)
    
    # Load genome data
    genome_data = Genome(ref_genome)
    all_chr_list = list(genome_data.keys())
    
    
    # DNA length check
    lengths = np.abs(df_decomposed["end"] - df_decomposed["start"])
    
    
    # Filter peaks with invalid chromosome name
    n_threshold = 5
    df = df[(lengths >= n_threshold) & df_decomposed.chr.isin(all_chr_list)]
    
    # DNA length check
    lengths = np.abs(df_decomposed["end"] - df_decomposed["start"])
    
    # Data counting
    n_invalid_length = len(lengths[lengths < n_threshold])
    n_peaks_invalid_chr = n_peaks_before - df_decomposed.chr.isin(all_chr_list).sum()
    n_peaks_after = df.shape[0]
    
    
    #
    print("Peaks before filtering: ", n_peaks_before)
    print("Peaks with invalid chr_name: ", n_peaks_invalid_chr)
    print("Peaks with invalid length: ", n_invalid_length)
    print("Peaks after filtering: ", n_peaks_after)
    
    return df

##### LINUX TERMINAL COMMANDS #####
celloracle_compute_baseGRN(filepath, peaks_TSS_mapped, 
                            data_id, ref_genome, motif_score_threshold)
                
print("base GRN is computed")
