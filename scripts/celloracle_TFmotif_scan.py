# This script is from 02_atac_peaks_to_TFinfo_window_20200801.ipynb notebook from the CellOracle github repo.
# NOTE: Run this script within the celloracle_env conda environment

def celloracle_TFmotif_scan(peaks_TSS_mapped, output_path, data_id):
    """
    A function to compute a dataframe for peaks-by-TFs using transcription factor motif databases (CisBP v2.0 in most cases unless specified)
    It uses CellOracle-implemented gimmemotif package for motif scan
    
    Parameters:
        peaks_TSS_mapped: filepath for the csv file with peaks mapped to the nearest TSS and filtered for high cicero co-accessibility scores.
        output_path: filepath for the directory where the output dataframe will be saved
        data_id: identifier for the output dataframe file
    
    Returns: 
        df: a dataframe of peaks-by-TFs

    Example:
        peaks_TSS_mapped = ""
        output_path = ""
        data_id = "TDR118"
        
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt


    import seaborn as sns

    import os, sys, shutil, importlib, glob
    from tqdm.notebook import tqdm

    import celloracle as co
    from celloracle import motif_analysis as ma
    from celloracle.utility import save_as_pickled_object
    co.__version__

    # %config InlineBackend.figure_format = 'retina'
    # %matplotlib inline
    # plt.rcParams['figure.figsize'] = (15,7)
    # plt.rcParams["savefig.dpi"] = 600

    # PLEASE make sure reference genome is correct.
    ref_genome = "danRer11"

    genome_installation = ma.is_genome_installed(ref_genome=ref_genome)
    print(ref_genome, "installation: ", genome_installation)

    # Load the processed peak data
    # Peaks were filtered for (1) mapping to the nearest TSS, and (2) cicero co-access score >=0.8 with the TSS peak
    peaks = pd.read_csv(peaks_TSS_mapped, index_col=0)
    peaks.head()

    # Instantiate TFinfo object
    tfi = ma.TFinfo(peak_data_frame=peaks, 
                    ref_genome=ref_genome) 

    peaks = check_peak_format(peaks, ref_genome)


    # Initiate the motif scan
    %%time
    # Scan motifs. !!CAUTION!! This step may take several hours if you have many peaks!
    tfi.scan(fpr=0.02, 
            motifs=None,  # If you enter None, default motifs will be loaded.
            verbose=True, n_cpus=-1)

    # Save tfinfo object
    tfi.to_hdf5(file_path=output_path + data_id + ".celloracle.tfinfo")

    # Check motif scan results
    tfi.scanned_df.head()

    # Filtering the motifs
    # Reset filtering 
    tfi.reset_filtering()

    # Do filtering
    tfi.filter_motifs_by_score(threshold=10)

    # Format post-filtering results.
    tfi.make_TFinfo_dataframe_and_dictionary(verbose=True)

    # Get final base GRN
    # Save the result as a dataframe
    df = tfi.to_dataframe()
    df.to_parquet(output_path + "base_GRN_dataframe_" + data_id + ".parquet")
    
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

from genomepy import Genome

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


