# This script is from mFinder_tutorial_TF.ipynb notebook from the zebrahub_multiome_analysis github repo.
# NOTE: Run this script within the celloracle_env conda environment

# Import libraries
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

import celloracle as co
co.__version__

# Input Arguments:
# 1) filepath: filepath for the input (Links object)
# 2) Links: filename for the Links object
# 3) mfinder_path: filepath for the mfinder (installation)
# 4) output_path: filepath for the outputs (GRNs from all cell-types)

# Examples:
# 1) filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118_cicero_output/"
# 2) filename = "08_TDR118_celltype_GRNs.celloracle.links"
# 3) mfinder_path = "/hpc/projects/data.science/yangjoon.kim/github_repos/mfinder/mfinder1.21"
# 4) output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118_cicero_output/07_TDR118_celloracle_GRN/"


# Parse command-line arguments
import argparse

# a syntax for running the python script as the main program (not in a module)
#if __name__ == "__main__":

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Detect network motifs using mFinder")

# Add command-line arguments
# parser.add_argument('filepath', type=str, help="File path")
# parser.add_argument('filename', type=str, help="File name")
# parser.add_argument('mfinder_path', type=str, help="mfinder_path")
# parser.add_argument('output_path', type=str, help="output File path")
# Add command-line arguments with flags
parser.add_argument('-i', '--inputpath', dest='filepath', type=str, required=True, help="File path for the input")
parser.add_argument('-n', '--name', dest='filename', type=str, required=True, help="File name for the input")
parser.add_argument('-m', '--mfinder', dest='mfinder_path', type=str, required=True, help="Path to the mFinder executable")
parser.add_argument('-o', '--outputpath', dest='output_path', type=str, required=True, help="File path for the output")
#parser.add_argument('ref_genome', type=str, help="reference genome")
#parser.add_argument('motif_score_threshold', type=float, help="motif score threshold")

# Parse the command-line arguments
args = parser.parse_args()

# Access the arguments as attributes of the 'args' object
filepath = args.filepath
filename = args.filename
mfinder_path = args.mfinder_path
output_path = args.output_path
#motif_score_threshold = args.motif_score_threshold

# Print the arguments
# define the PATH for mfinder
cmd = "export PATH="+mfinder_path+":$PATH"
os.system(cmd)
#os.system("export PATH=/hpc/projects/data.science/yangjoon.kim/github_repos/mfinder/mfinder1.21:$PATH")

# create the output_path if it doesn't exist
if not os.path.exists(output_path):
   os.makedirs(output_path)

# move to the output filepath (This is not the best practice, so we'd need to find an alternative)
#os.chdir("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118_cicero_output/07_TDR118_celloracle_GRN/")

os.chdir(output_path)

def find_network_motifs_mfinder(filepath, filename, output_path):

    # set the filepath for the "filepath", where the GRN is saved.

    # import the Links object (cell-type specific GRNs for all cell-types)
    GRN_all = co.load_hdf5(filepath + filename)
    GRN_all = GRN_all.filtered_links

    for celltype in GRN_all.keys():

        # Step 1.
        GRN_celltype = GRN_all[celltype]

        # Step 2.
        list_genes_TFs = list(set(GRN_celltype.source).union(set(GRN_celltype.target)))
        # Create a dictionary mapping integers to gene names
        gene_dict = {index: gene_name for index, gene_name in enumerate(list_genes_TFs)}
        gene_dict

          # Now, we will reformat the GRN as described above
        # 1) grab the GRN, then extract the "source", "target", and create a dataframe
        # 2) add the "edge weight" as "1" for the third column
        df_mfinder = pd.DataFrame(columns =["source", "target", "edge_weight"])
        df_mfinder

        df_mfinder["source"] = GRN_celltype["source"]
        df_mfinder["target"] = GRN_celltype["target"]
        df_mfinder["edge_weight"] = 1

        # 3) convert the "source", "target" gene_names to "integers" using the gene_dict
        df_mfinder["source"] = df_mfinder["source"].map({v: k for k, v in gene_dict.items()})
        df_mfinder["target"] = df_mfinder["target"].map({v: k for k, v in gene_dict.items()})
        df_mfinder
        # save the reformatted GRN into a txt file
        # df_mfinder.to_csv(output_path + "filtered_GRN_"+celltype+"_mfinder_format.txt",
        #                     sep="\t", header=False, index=False)
        output_file_name = "filtered_GRN_" + celltype + "_mfinder_format.txt"
        output_file_path = os.path.join(output_path, output_file_name)

        df_mfinder.to_csv(output_file_path, sep="\t", header=False, index=False)

        # Step 3. run mFinder
        # default setting (network_size=3, num_random_)
        # input filename
        input_file = "filtered_GRN_"+celltype+"_mfinder_format.txt"
        # output filename
        output_file = "motifs_"+celltype+ "_OUT.txt"

        # define the mFinder command
        cmd = "mfinder "+input_file + " -f "+output_file
        # directly use the mfinder_path, instead of adding $PATH
        #cmd = mfinder_path + " " + input_file + " -o " + output_file
        cmd
        # run mFinder
        os.system(cmd)

# run the function
find_network_motifs_mfinder(filepath, filename, output_path)