# extract the motif information as a dataframe from mFinder outout text files
# Background: mFinder OUTPUT file is formatted for human eyes, so not computational friendly for parsing.
# This function takes the mFinder OUTPUT file, processes and returns a dataframe.
# Import libraries
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc

def extract_motif_info(mFinder_output):

    # Step 1. read out the file, and filter for the relevant text block.
    # Open and read the text file
    filepath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/baseGRN_CisBP_RNA_zebrahub/09_network_motifs/"
    with open(filepath + 'motifs_0budstage_Somites_OUT.txt', 'r') as file:
        lines = file.readlines()

    # lines
    # extract the significant motifs (Z_score > 2)

    # Initialize variables to store extracted lines
    extracted_lines = []
    # start_pattern - note that the start_pattern appears a couple of times (for insignificant motifs as well)
    start_pattern = 'MOTIF\tNREAL\tNRAND\t\tNREAL\tNREAL\tUNIQ\tCREAL'

    # Flag to indicate when to start and stop extraction
    start_extraction = False
    end_pattern_count = 0

    # Append the start pattern to the extracted lines
    extracted_lines.append(start_pattern)

    # Iterate through the lines
    for line in lines:
        if start_extraction:
            extracted_lines.append(line.strip())  # Add the stripped line to the result
            if not line.strip():
                end_pattern_count += 1
                if end_pattern_count == 2:
                    break  # Stop extraction when two consecutive empty lines are found
            else:
                end_pattern_count = 0  # Reset the count if a non-empty line is encountered
        elif line.strip() == start_pattern:
            start_extraction = True  # Start extraction when the start pattern is found

    # Print the extracted lines
    for extracted_line in extracted_lines:
        print(extracted_line)

    # Step 2. extract the motif information from the text block, and save as a dataframe
    # the first two lines of the extracted text block are the column names
    str1 = extracted_lines[0]
    str2 = extracted_lines[1]

    # concatenate the first and the second strings to create the column names
    col_names = [em1 + ("_" + em2 if em2 else "") for em1, em2 in zip(str1.split("\t"), str2.split("\t"))]

    # filter out the element that is an empty string (mFinder's mistake in formatting)
    col_names = [element for element in col_names if element]
    col_names

    # create a dataframe to save the motif ID and scores
    df = pd.DataFrame(columns=col_names)

    # line indices for the motifs
    motif_line_indices = [index for index,line in enumerate(extracted_lines) if len(line.split('\t')) == 7]

    motifs_list = []

    for index, line_index in enumerate(motif_line_indices):
        line = extracted_lines[line_index]
        df.loc[index] = line.split("\t")
        
        # extract the motif
        motif_matrice = extracted_lines[line_index+2:line_index+5]
        # Split into rows
        rows = [list(map(int, row.split())) for row in motif_matrice]
        # Convert values to a NumPy array
        motif = np.array(rows)
        
        motifs_list.append(motif)
        
    df["motifs"] = motifs_list
    df
    return df