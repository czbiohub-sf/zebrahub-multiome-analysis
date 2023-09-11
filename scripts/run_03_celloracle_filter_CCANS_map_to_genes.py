# From 02_preprocess_peak_data_CellOracle.ipynb notebook

# Overview (from the original notebooks by Kamimoto and Morris, Nature, 2023)
# Before building the base GRN, we need to annotate the coaccessible peaks 
# and filter our active promoter/enhancer elements. 
# First, we will identify the peaks around transcription starting sites (TSS). 
# We will then merge the Cicero data with the TSS peak information and 
# filter any peaks with weak connections to the TSS peaks. 
# As such, the filtered peak data will only include TSS peaks and peaks with strong TSS connections. 
# These will be our active promoter/enhancer elements for our base GRN. 

# Step 0. import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, shutil, importlib, glob
from tqdm.notebook import tqdm

from celloracle import motif_analysis as ma
import celloracle as co
co.__version__

#%config InlineBackend.figure_format = 'retina'

plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams["savefig.dpi"] = 300
#%matplotlib inline


# Input Arguments:
# filepath: input and output filepath. We will put both the inputs and outputs in the same directory.
# peak_file: name of the peak file. i.e."01_TDR118_CRG_arc_peaks.csv"
# CCAN_file: name of the cicero output file. i.e. "02_TDR118_cicero_connections_CRG_arc_peaks.csv"
# cicero_score_threshold: threshold for the cicero_score. CellOracle used cicero_score>=0.8
# filename: name of the file. i.e. "03_TDR118_processed_peak_file_danRer11.csv"
# save_figure: 
# figpath: path for the plots/figures

# Parse command-line arguments
import argparse

# a syntax for running the python script as the main program (not in a module)
#if __name__ == "__main__":


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Filter and map data using CellOracle")

# Add command-line arguments
parser.add_argument('filepath', type=str, help="File path")
parser.add_argument('peak_file', type=str, help="Peak file")
parser.add_argument('CCAN_file', type=str, help="CCAN file")
parser.add_argument('cicero_score_threshold', type=float, help="Cicero score threshold")
parser.add_argument('filename', type=str, help="Output filename")
parser.add_argument('save_figure', type=bool, help="Save figure (True/False)")
parser.add_argument('figpath', type=str, help="Figure path")

# Parse the command-line arguments
args = parser.parse_args()

# Access the arguments as attributes of the 'args' object
filepath = args.filepath
peak_file = args.peak_file
CCAN_file = args.CCAN_file
cicero_score_threshold = args.cicero_score_threshold
filename = args.filename
save_figure = args.save_figure
figpath = args.figpath

def process_CCANS(filepath, 
                        peak_file = "01_TDR118_CRG_arc_peaks.csv", 
                        CCAN_file = "02_TDR118_cicero_connections_CRG_arc_peaks.csv", 
                        cicero_score_threshold = 0.8,
                        filename="03_TDR118_processed_peak_file_danRer11.csv",
                        save_figure=False,
                        figpath=None):

    # Check if the figpath exists
    if save_figure==True:
        # figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/peaks_TDR118"
        os.makedirs(figpath, exist_ok=True)
    else:
        figpath = None

    # Step 1. Load scATAC peak data and peak connection data made with Cicero
    # Load scATAC-seq peak list.
    peaks = pd.read_csv(filepath+peak_file, index_col=0)
    peaks = peaks.x.values
    print("the number of peaks: " + str(len(peaks)))

    # add "chr" in front of each peak element for formatting (CellOracle genome reference)
    peaks = "chr" + peaks

    # reformat the peaks (replace the "-" with "_")
    peaks = [s.replace("-","_") for s in peaks]
    peaks = np.array(peaks)

    # Load Cicero coaccessibility scores.
    cicero_connections = pd.read_csv(filepath + CCAN_file, index_col=0)
    cicero_connections.head()
    print("number of CCANs: ", str(len(cicero_connections)))

    # Formatting 
    # add "chr" in front of each peak element for formatting (CellOracle genome reference)
    cicero_connections["Peak1"] = "chr" + cicero_connections["Peak1"]
    cicero_connections["Peak2"] = "chr" + cicero_connections["Peak2"]

    # replace the "-" with "_"
    cicero_connections["Peak1"] = [s.replace("-","_") for s in cicero_connections["Peak1"]]
    cicero_connections["Peak2"] = [s.replace("-","_") for s in cicero_connections["Peak2"]]

    cicero_connections.head()

    # Step 2. Annotate transcription start site (TSS)
    # IMPORTANT: Please make sure that you're setting the correct reference genome
    ma.SUPPORTED_REF_GENOME

    ##!! Please make sure to specify the correct reference genome here
    tss_annotated = ma.get_tss_info(peak_str_list=peaks, ref_genome="danRer11") 

    # Check results
    print("tss_annotated_tail:")
    tss_annotated.tail()

    # Step 3. Integrate TSS info and cicero connections
    # The output file after the integration process has three columns: 
    # `["peak_id", "gene_short_name", "coaccess"`].
    # "peak_id" is either the TSS peak or the peaks that have a connection to a TSS peak.
    # "gene_short_name" is the gene name that associated with the TSS site. 
    # "coaccess" is the coaccessibility score between the peak and a TSS peak. If the score is 1, it means that the peak is a TSS itself.
    # NOTES: 1) this step fitlers out the CCANs with negative cicero scores. 2) it also assigns cicero_score = 1 for the TSS-TSS peaks.

    integrated = ma.integrate_tss_peak_with_cicero(tss_peak=tss_annotated, 
                                               cicero_connections=cicero_connections)
    print(integrated.shape)
    integrated.head()

    # Step 4. Filter the peaks for high cicero scores (co-accessibility)
    # Here, we use the same threshold CellOracle paper used, 0.8
    peak = integrated[integrated.coaccess >= cicero_score_threshold]
    peak = peak[["peak_id", "gene_short_name"]].reset_index(drop=True)
    print(peak.shape)
    peak.head()

    # Step 5. Save the output file
    peak.to_csv(filepath+ filename)


    if save_figure==True:
        # plot1. distribution of cicero scores for CCANs
        cicero_connections.coaccess.hist(bins=30)
        plt.yscale("log")
        plt.xlabel("co-accessibility score")
        plt.ylabel("occurences")
        plt.savefig(figpath+"/cicero_scores_histogram.pdf")

        # plot2. number of CCANs per chromosome
        # NOTE: normalize the counts with the length of each chromosome
        # make an empty list 
        n_peaks_chrom = []

        for chromosome in np.arange(1,26):
            chrom_num = chromosome
            chrom_id = "chr"+str(chrom_num)
            n_peaks = len(cicero_connections[cicero_connections.Peak1.str.startswith(chrom_id)])
            n_peaks_chrom.append(n_peaks)

        sns.barplot(x=list(np.arange(1,26)), y=n_peaks_chrom, palette = "viridis")
        plt.xlabel("chromosomes")
        plt.ylabel("number of peaks")
        plt.savefig(figpath+"/n_CCANs_per_chromosome.pdf")

        # plot 3. a histogram to see the distribution of the co-accessibility scores
        plt.hist(integrated["coaccess"], bins=30)
        plt.yscale("log")
        plt.xlabel("co-accessibility score (cicero)")
        plt.ylabel("occurences")
        plt.savefig(figpath+"/CCAN_TSS_cicero_scores_histogram.pdf")


##### LINUX TERMINAL COMMANDS #####
print("input arguments:")
print([filepath, peak_file, 
                CCAN_file, cicero_score_threshold,
                filename, save_figure, figpath])

process_CCANS(filepath, peak_file, 
                CCAN_file, cicero_score_threshold,
                filename, save_figure, figpath)
                
print("CCANS processing completed")