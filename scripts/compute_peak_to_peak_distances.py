# A python script to compute the peak-peak distances for all peak-peak pairs from cicero output.
# Last updated: 10/18/2023
# Author: Yang-Joon Kim


import pandas as pd
import numpy as np

# Step 1. import the cicero output file
cicero_output_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118_cicero_output/"
# Load Cicero coaccessibility scores.
cicero_connections = pd.read_csv(cicero_output_path + "02_TDR118_cicero_connections_CRG_arc_peaks.csv", index_col=0)
#cicero_connections.head()

# # compute the distances between the two peaks, and register them in the dataframe along with co-access scores.
cicero_connections["distance"] = [abs(np.average([int(str2.split("-")[2]),int(str2.split("-")[1])]) - np.average([int(str1.split("-")[2]),int(str1.split("-")[1])])) \
                                      for str1, str2 in zip(cicero_connections['Peak1'], cicero_connections['Peak2'])]
#cicero_connections.head()

print("distances between all peak pairs are computed")

# save the result into a csv file
cicero_connections.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/TDR118_cicero_output/02_TDR118_cicero_connections_CRG_arc_peaks_distances.csv")

print("updated dataframe is saved as a csv file")