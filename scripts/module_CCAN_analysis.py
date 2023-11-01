# Module for CCAN analyses (cicero result, temporal dynamics of cis-regulatory networks)

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# a function to plot the genomic regions
import matplotlib.patches as patches


# input:
# output:

# functions
def compute_start_end(df):
    """Function to compute start and end coordinates for a dataframe."""
    df['start'] = df['peak_id'].str.split('_').str[1].astype(int)
    df['end'] = df['peak_id'].str.split('_').str[2].astype(int)
    return df

# plot_peaks with y_base as an input argument to stagger different plots on top of each other (along the y-axis)
def plot_peaks(df, ax, y_base, color, label=None):
    """Function to plot peaks at a specified y-axis basis."""
    for _, row in df.iterrows():
        rect = patches.Rectangle((row['start'], y_base), row['end'] - row['start'], 0.1, linewidth=1, edgecolor=color, facecolor=color, label=label)
        ax.add_patch(rect)
        label = None  # Set to None to prevent repeating the label
        

def plot_peaks_with_alpha(df, ax, color, label=None):
    """Function to plot peaks with alpha for overlapping peaks."""
    for i, (_, row) in enumerate(df.iterrows()):
        # Only add label for the first rectangle of each group
        lbl = label if i == 0 else None
        rect = patches.Rectangle((row['start'], 0.9), row['end'] - row['start'], 0.2, linewidth=1, edgecolor=color, facecolor=color, alpha=0.5, label=lbl)
        ax.add_patch(rect)

def plot_CCANs_genomic_loci(CCANs, timepoints, gene_name, colordict, 
                            save_fig=False, figpath = None):
    """
    Description: 
    This function takes a list of CCANs (dataframes for each timepoint), 
    plots the genomic region with CCANs for each timepoint for "gene_name".
    
    Parameters:
    1) gene_name: Name of the gene, i.e. "myf5", "her1"
    2) CCANs: A list of dataframes (one dataframe for each timepoint), i.e. [df1, df2, df3]
    3) timepoints: A list of timepoint labels corresponding to each dataframe in CCANs
    4) colordict: A dictionary of {timepoints:colors (viridis)}
    """
    
    if len(CCANs) != len(timepoints):
        raise ValueError("The number of CCANs dataframes and timepoints labels must be equal")

    # colormap - just define at the beginning (this can be replaced)
    # Define the timepoints
    all_timepoints = ["0budstage", "5somites", "10somites", "15somites", "20somites", "30somites"]

    # Load the "viridis" colormap
    viridis = plt.cm.get_cmap('viridis', 256)

    # Select a subset of the colormap to ensure that "30 somites" is yellow
    # You can adjust the start and stop indices to shift the colors
    start = 50
    stop = 256
    colors = viridis(np.linspace(start/256, stop/256, len(all_timepoints)))

    # Create a dictionary to map timepoints to colors
    color_dict = dict(zip(all_timepoints, colors))
    color_dict
    
    # generate a figure object    
    fig, ax = plt.subplots(figsize=(10, 2))
    
    genomic_start, genomic_end = float('inf'), 0

    for index, (df, stage) in enumerate(zip(CCANs, timepoints)):
        df = df[df.gene_short_name == gene_name]
        df = compute_start_end(df)
        
        genomic_start = min(genomic_start, df['start'].min())
        genomic_end = max(genomic_end, df['end'].max())
        
        plot_peaks(df, ax, 1 + index*0.1, colordict[stage], stage)
    
    ax.plot([genomic_start, genomic_end], [1, 1], color='grey', linewidth=2)
    
    ax.set_ylim(0.7, 1 + len(CCANs)*0.1 + 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Genomic Coordinate')
    ax.set_title(f'Genomic Region Plot for {gene_name}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_fig==True:
        plt.savefig(figpath + "coverage_plot_CCANs_" + gene_name + ".png")
        plt.savefig(figpath + "coverage_plot_CCANs_" + gene_name + ".pdf")
    
    plt.show()
        
        
# plot_CCANs_genomic_loci([mapped_peaks_15somites, mapped_peaks_20somites, mapped_peaks_30somites],
#                         gene_name="myf5", timepoints=["15somites", "20somites","30somites"], 
#                         colordict=color_dict, save_fig=False, figpath=None)
    

