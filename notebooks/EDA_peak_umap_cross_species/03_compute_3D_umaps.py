# Jupytext notebook to compute 3D UMAP coordinates with metadata (annotations) for the peak UMAPs from mouse and human, respectively.

# %% import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
import scanpy as sc

import cupy as cp
import rapids_singlecell as rsc
 
# %% figure parameter setting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')
# Set default DPI for saved figures
mpl.rcParams['savefig.dpi'] = 600

# %% plot the 3D UMAP
import plotly.express as px
import plotly.graph_objects as go
def plot_3d_umap(umap_array, 
                 color_array=None,
                 color_label='cluster',
                 title='3D UMAP',
                 point_size=3,
                 opacity=0.7,
                 height=800,
                 width=1000):
    """
    Create an interactive 3D UMAP visualization using plotly.
    
    Parameters:
    -----------
    umap_array : np.array
        Array of shape (n_cells, 3) containing 3D UMAP coordinates
    color_array : array-like, optional
        Array of values/categories to color the points by
    color_label : str, optional
        Label for the color legend
    title : str, optional
        Title of the plot
    point_size : int, optional
        Size of the scatter points
    opacity : float, optional
        Opacity of the points (0-1)
    height : int, optional
        Height of the plot in pixels
    width : int, optional
        Width of the plot in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    
    # Create a DataFrame with UMAP coordinates
    df = pd.DataFrame(
        umap_array,
        columns=['UMAP1', 'UMAP2', 'UMAP3']
    )
    
    if color_array is not None:
        df[color_label] = color_array.values  # aligns by position, not index
        
        # Create figure with color
        fig = px.scatter_3d(
            df,
            x='UMAP1',
            y='UMAP2',
            z='UMAP3',
            color=color_label,
            title=title,
            opacity=opacity,
            height=height,
            width=width
        )
    else:
        # Create figure without color
        fig = px.scatter_3d(
            df,
            x='UMAP1',
            y='UMAP2',
            z='UMAP3',
            title=title,
            opacity=opacity,
            height=height,
            width=width
        )
    
    # Update marker size
    fig.update_traces(marker_size=point_size)
    
    # Update layout for better visualization
    fig.update_layout(
        scene = dict(
            xaxis_title='UMAP1',
            yaxis_title='UMAP2',
            zaxis_title='UMAP3',
            aspectmode='cube'  # This ensures equal aspect ratio
        ),
        showlegend=True
    )
    
    return fig

# %% mouse (Argelaguet 2022)
# load the adata object with annotation
peaks_by_pb_mouse = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated_v2.h5ad")
peaks_by_pb_mouse

# %% compute the 3D UMAP
# %%
# first, copy over the 2D UMAP
peaks_by_pb_mouse.obsm["X_umap_2D"] = peaks_by_pb_mouse.obsm["X_umap"]

# compute the 3D UMAP
rsc.tl.umap(peaks_by_pb_mouse, min_dist=0.4, random_state=42, n_components=3)

# save the 3D UMAP coordinates
umap_3d_array = peaks_by_pb_mouse.obsm["X_umap"]

# add the 3D UMAP explicitly into .obsm
peaks_by_pb_mouse.obsm["X_umap_3D"] = umap_3d_array
peaks_by_pb_mouse

# %% 

# %% plot the 3D UMAP (check)
# Example usage with your data:
umap_coords = peaks_by_pb_mouse.obsm['X_umap_3D']  # Your UMAP coordinates
# If you have clusters or other metadata to color by:
lineage = peaks_by_pb_mouse.obs['peak_lineage']  # or whatever your metadata column is
timepoint = peaks_by_pb_mouse.obs['peak_top_timepoint'] 
# Create the 3D plot
plot_3d_umap(umap_coords, color_array=lineage, color_label='lineage', 
             point_size=2, opacity=0.5, title='3D UMAP')

# %% create a dataframe for 3D UMAP coordinates and metadata
df = pd.DataFrame(index=peaks_by_pb_mouse.obs_names)
# Add 3D UMAP coordinates
umap_3d = peaks_by_pb_mouse.obsm["X_umap_3D"]
df["UMAP_1"] = umap_3d[:, 0]
df["UMAP_2"] = umap_3d[:, 1]
df["UMAP_3"] = umap_3d[:, 2]

df["celltype"] = peaks_by_pb_mouse.obs["peak_top_celltype"]

df["timepoint"] = peaks_by_pb_mouse.obs["peak_top_timepoint"]
df["lineage"] = peaks_by_pb_mouse.obs["peak_lineage"]
df["peak_type"] = peaks_by_pb_mouse.obs["peak_type"]
df["chromosome"] = peaks_by_pb_mouse.obs["chr"]
df["leiden_coarse"] = peaks_by_pb_mouse.obs["leiden_coarse"]
#df["leiden_fine"] = peaks_by_pb_mouse.obs["leiden_unified"]
# export the dataframe
df.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/3d_umap_coords_mouse.csv")
# save the master object with the coordinates
# peaks_by_pb_mouse.write_h5ad("")

# %% human (Domcke 2020)
# load the adata object with annotation
peaks_by_pb_human = sc.read_h5ad("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/human_domcke_2020/peaks_by_pb_celltype_stage_annotated.h5ad")
peaks_by_pb_human

# %% compute the 3D UMAP
# %%
# first, copy over the 2D UMAP
peaks_by_pb_human.obsm["X_umap_2D"] = peaks_by_pb_human.obsm["X_umap"]

# compute the 3D UMAP
rsc.tl.umap(peaks_by_pb_human, min_dist=0.5, random_state=42, n_components=3)

# save the 3D UMAP coordinates
umap_3d_array = peaks_by_pb_human.obsm["X_umap"]

# add the 3D UMAP explicitly into .obsm
peaks_by_pb_human.obsm["X_umap_3D"] = umap_3d_array
peaks_by_pb_human

# %% plot the 3D UMAP (check)
# Example usage with your data:
umap_coords = peaks_by_pb_human.obsm['X_umap_3D']  # Your UMAP coordinates
# If you have clusters or other metadata to color by:
# if "peak_lineage" does not exist in .obs
# if "peak_lineage" in peaks_by_pb_human.obs.columns:
#     print("peak_lineage already computed")
# else:
#     # compute the peak_lineage
    
lineage = peaks_by_pb_human.obs['peak_lineage']  # or whatever your metadata column is
timepoint = peaks_by_pb_human.obs['peak_top_timepoint'] 
# Create the 3D plot
plot_3d_umap(umap_coords, color_array=lineage, color_label='lineage', 
             point_size=2, opacity=0.5, title='3D UMAP')

# %% create a dataframe for 3D UMAP coordinates and metadata
df = pd.DataFrame(index=peaks_by_pb_human.obs_names)
# Add 3D UMAP coordinates
umap_3d = peaks_by_pb_human.obsm["X_umap_3D"]
df["UMAP_1"] = umap_3d[:, 0]
df["UMAP_2"] = umap_3d[:, 1]
df["UMAP_3"] = umap_3d[:, 2]

df["celltype"] = peaks_by_pb_human.obs["peak_top_celltype"]

df["timepoint"] = peaks_by_pb_human.obs["peak_top_timepoint"]
df["lineage"] = peaks_by_pb_human.obs["peak_lineage"]
df["peak_type"] = peaks_by_pb_human.obs["peak_type"]
df["chromosome"] = peaks_by_pb_human.obs["chr"]
df["leiden_coarse"] = peaks_by_pb_human.obs["leiden_coarse"]
#df["leiden_fine"] = peaks_by_pb_mouse.obs["leiden_unified"]
# export the dataframe
df.to_csv("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/human_domcke_2020/3d_umap_coords_human.csv")