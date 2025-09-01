# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chromatin Velocity Analysis - Zebrahub Multiome
# 
# This notebook tests the chromatin velocity implementation with actual zebrahub-multiome datasets.
# 
# **Data inputs:**
# - Peaks-by-pseudobulk accessibility: `/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_annotated_master.h5ad`
# - Peak-peak co-accessibility: `/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/cicero_integrated_ATAC_v2/02_integrated_ATAC_v2_cicero_connections_peaks_integrated_peaks.csv`
# 
# **Analysis workflow:**
# 1. Load and examine input data structure
# 2. Initialize chromatin velocity computation
# 3. Run velocity estimation pipeline
# 4. Generate visualizations and validation plots
# 5. Test temporal consistency

# %%
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import chromatin velocity modules
import sys
sys.path.append('../scripts')

from chromatin_velocity import ChromatinVelocity
from chromatin_velocity_scvelo import ChromatinVelocityAnalysis  
from chromatin_velocity_viz import ChromatinVelocityVisualizer
from chromatin_velocity_validation import ChromatinVelocityValidator

# Set up plotting
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 100
sc.settings.verbosity = 1

print("✓ All modules imported successfully")

# %% [markdown]
# ## 1. Load and Examine Input Data

# %%
# Data paths
peaks_pb_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_annotated_master.h5ad"
coaccess_path = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/cicero_integrated_ATAC_v2/02_integrated_ATAC_v2_cicero_connections_peaks_integrated_peaks.csv"

print("Loading peaks-by-pseudobulk data...")
adata_peaks = sc.read_h5ad(peaks_pb_path)

print(f"Peaks data shape: {adata_peaks.shape}")
print(f"Observations (pseudobulks): {adata_peaks.n_obs}")
print(f"Variables (peaks): {adata_peaks.n_vars}")
print(f"\nFirst few pseudobulk names:")
print(adata_peaks.obs_names[:10].tolist())

# %%
print("Loading co-accessibility data...")
coaccess_df = pd.read_csv(coaccess_path)

print(f"Co-accessibility data shape: {coaccess_df.shape}")
print(f"Columns: {list(coaccess_df.columns)}")
print(f"\nFirst few rows:")
print(coaccess_df.head())

# Check co-accessibility score distribution
if 'coaccess' in coaccess_df.columns:
    coaccess_col = 'coaccess'
elif 'score' in coaccess_df.columns:
    coaccess_col = 'score'
else:
    coaccess_col = coaccess_df.columns[-1]  # Assume last column is score
    
print(f"\nCo-accessibility score statistics (column: {coaccess_col}):")
print(coaccess_df[coaccess_col].describe())

# %%
# Examine temporal structure in pseudobulk names
print("Analyzing temporal structure in pseudobulk names:")
pb_names = adata_peaks.obs_names.tolist()

# Extract timepoint information (assuming format like celltype_timepoint or celltype.timepoint)
timepoints = []
celltypes = []

for name in pb_names[:20]:  # Sample first 20
    print(f"Pseudobulk: {name}")
    # Try to parse timepoint information
    if '_' in name:
        parts = name.split('_')
        if len(parts) >= 2:
            potential_timepoint = parts[-1]
            celltype = '_'.join(parts[:-1])
            timepoints.append(potential_timepoint)
            celltypes.append(celltype)

print(f"\nUnique timepoints detected: {set(timepoints[:20])}")
print(f"Unique celltypes detected: {set(celltypes[:20])}")

# %% [markdown]
# ## 2. Initialize Chromatin Velocity Analysis

# %%
print("Initializing ChromatinVelocity object...")

# Initialize the chromatin velocity analyzer
cv = ChromatinVelocity(
    accessibility_threshold=0.01,  # Minimum accessibility to consider
    coaccessibility_threshold=0.1,  # Minimum co-accessibility to consider  
    max_connections_per_peak=100,   # Limit connections per peak
    normalization_method='log1p',   # Normalization approach
    verbose=True
)

print("✓ ChromatinVelocity initialized")

# %%
print("Processing co-accessibility data...")

# Convert co-accessibility dataframe to the expected format
# The chromatin velocity module expects columns: ['Peak1', 'Peak2', 'score']
coaccess_formatted = coaccess_df.copy()

# Standardize column names
col_mapping = {}
for col in coaccess_formatted.columns:
    if 'peak1' in col.lower() or col.lower().startswith('peak') and '1' in col:
        col_mapping[col] = 'Peak1'
    elif 'peak2' in col.lower() or col.lower().startswith('peak') and '2' in col:
        col_mapping[col] = 'Peak2'  
    elif 'score' in col.lower() or 'coaccess' in col.lower():
        col_mapping[col] = 'score'

coaccess_formatted = coaccess_formatted.rename(columns=col_mapping)

print(f"Formatted co-accessibility columns: {list(coaccess_formatted.columns)}")
print(f"Required columns present: {'Peak1' in coaccess_formatted.columns}, {'Peak2' in coaccess_formatted.columns}, {'score' in coaccess_formatted.columns}")

# %%
# Filter co-accessibility data for peaks present in our dataset
peak_names = adata_peaks.var_names.tolist()
print(f"Total peaks in accessibility data: {len(peak_names)}")

if 'Peak1' in coaccess_formatted.columns and 'Peak2' in coaccess_formatted.columns:
    # Filter for peaks that exist in our dataset
    peak_set = set(peak_names)
    filtered_coaccess = coaccess_formatted[
        coaccess_formatted['Peak1'].isin(peak_set) & 
        coaccess_formatted['Peak2'].isin(peak_set)
    ].copy()
    
    print(f"Co-accessibility pairs after filtering: {len(filtered_coaccess)}")
    print(f"Unique peaks in co-accessibility: {len(set(filtered_coaccess['Peak1'].tolist() + filtered_coaccess['Peak2'].tolist()))}")
    
    # Show score distribution after filtering
    print(f"\nFiltered co-accessibility score distribution:")
    print(filtered_coaccess['score'].describe())

# %% [markdown]
# ## 3. Run Chromatin Velocity Computation

# %%
print("Computing chromatin velocity...")

try:
    # Run the chromatin velocity computation
    velocity_results = cv.compute_chromatin_velocity(
        accessibility_data=adata_peaks.X.toarray() if hasattr(adata_peaks.X, 'toarray') else adata_peaks.X,
        peak_names=adata_peaks.var_names.tolist(),
        sample_names=adata_peaks.obs_names.tolist(),
        coaccessibility_data=filtered_coaccess if 'filtered_coaccess' in locals() else coaccess_formatted
    )
    
    print("✓ Chromatin velocity computation completed!")
    print(f"Velocity matrix shape: {velocity_results['velocity'].shape}")
    print(f"Unspliced matrix shape: {velocity_results['unspliced'].shape}")
    
except Exception as e:
    print(f"Error in chromatin velocity computation: {str(e)}")
    print("This might be due to data format issues. Let's investigate...")
    
    # Debug data formats
    print(f"Accessibility data type: {type(adata_peaks.X)}")
    print(f"Accessibility data shape: {adata_peaks.X.shape}")
    print(f"Peak names type: {type(adata_peaks.var_names)}")
    print(f"Sample names type: {type(adata_peaks.obs_names)}")

# %% [markdown]
# ## 4. Visualization and Analysis (if computation successful)

# %%
# Only proceed if velocity computation was successful
if 'velocity_results' in locals():
    print("Creating velocity-enhanced AnnData object...")
    
    # Create new AnnData object with velocity information
    adata_velocity = ad.AnnData(
        X=velocity_results['spliced'],  # Use spliced (observed) as main data
        obs=adata_peaks.obs.copy(),
        var=adata_peaks.var.copy()
    )
    
    # Add velocity layers
    adata_velocity.layers['spliced'] = velocity_results['spliced']
    adata_velocity.layers['unspliced'] = velocity_results['unspliced'] 
    adata_velocity.layers['velocity'] = velocity_results['velocity']
    
    print(f"✓ Velocity AnnData created with shape: {adata_velocity.shape}")
    print(f"Available layers: {list(adata_velocity.layers.keys())}")

# %%
# Visualize velocity statistics
if 'velocity_results' in locals():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Velocity distribution
    velocity_flat = velocity_results['velocity'].flatten()
    axes[0,0].hist(velocity_flat[~np.isnan(velocity_flat)], bins=50, alpha=0.7)
    axes[0,0].set_title('Chromatin Velocity Distribution')
    axes[0,0].set_xlabel('Velocity')
    axes[0,0].set_ylabel('Frequency')
    
    # Spliced vs Unspliced correlation
    spliced_mean = np.mean(velocity_results['spliced'], axis=0)
    unspliced_mean = np.mean(velocity_results['unspliced'], axis=0)
    axes[0,1].scatter(spliced_mean, unspliced_mean, alpha=0.5, s=1)
    axes[0,1].set_xlabel('Mean Spliced (Observed)')
    axes[0,1].set_ylabel('Mean Unspliced (Predicted)')  
    axes[0,1].set_title('Spliced vs Unspliced')
    
    # Velocity magnitude per peak
    velocity_magnitude = np.sqrt(np.sum(velocity_results['velocity']**2, axis=0))
    axes[1,0].hist(velocity_magnitude, bins=50, alpha=0.7)
    axes[1,0].set_title('Velocity Magnitude per Peak')
    axes[1,0].set_xlabel('Velocity Magnitude')
    axes[1,0].set_ylabel('Frequency')
    
    # Accessibility vs Velocity correlation
    accessibility_mean = np.mean(velocity_results['spliced'], axis=0)
    velocity_mean = np.mean(velocity_results['velocity'], axis=0)
    axes[1,1].scatter(accessibility_mean, velocity_mean, alpha=0.5, s=1)
    axes[1,1].set_xlabel('Mean Accessibility')
    axes[1,1].set_ylabel('Mean Velocity')
    axes[1,1].set_title('Accessibility vs Velocity')
    
    plt.tight_layout()
    plt.show()

# %% [markdown] 
# ## 5. Temporal Consistency Analysis

# %%
# Analyze temporal patterns if velocity computation was successful
if 'velocity_results' in locals():
    print("Analyzing temporal consistency...")
    
    # Parse timepoint information from pseudobulk names
    pb_info = []
    for pb_name in adata_peaks.obs_names:
        # Extract celltype and timepoint information
        # Adjust parsing based on actual naming convention
        parts = pb_name.split('_') if '_' in pb_name else pb_name.split('.')
        if len(parts) >= 2:
            celltype = '_'.join(parts[:-1]) if len(parts) > 2 else parts[0]
            timepoint = parts[-1]
            pb_info.append({'pseudobulk': pb_name, 'celltype': celltype, 'timepoint': timepoint})
        else:
            pb_info.append({'pseudobulk': pb_name, 'celltype': pb_name, 'timepoint': 'unknown'})
    
    pb_df = pd.DataFrame(pb_info)
    print(f"Parsed {len(pb_df)} pseudobulk samples")
    print(f"Unique timepoints: {sorted(pb_df['timepoint'].unique())}")
    print(f"Unique celltypes: {len(pb_df['celltype'].unique())}")
    
    # Add temporal information to velocity object
    adata_velocity.obs['celltype'] = pb_df['celltype'].values
    adata_velocity.obs['timepoint'] = pb_df['timepoint'].values

# %%
# Plot velocity patterns by timepoint and celltype
if 'velocity_results' in locals():
    # Calculate velocity statistics per group
    velocity_stats = []
    
    for celltype in pb_df['celltype'].unique()[:10]:  # Limit to first 10 celltypes for visualization
        for timepoint in pb_df['timepoint'].unique():
            mask = (pb_df['celltype'] == celltype) & (pb_df['timepoint'] == timepoint)
            if mask.sum() > 0:
                indices = np.where(mask)[0]
                velocity_subset = velocity_results['velocity'][indices, :]
                velocity_stats.append({
                    'celltype': celltype,
                    'timepoint': timepoint, 
                    'mean_velocity': np.nanmean(velocity_subset),
                    'velocity_magnitude': np.nanmean(np.sqrt(np.nansum(velocity_subset**2, axis=1))),
                    'n_samples': len(indices)
                })
    
    stats_df = pd.DataFrame(velocity_stats)
    
    if len(stats_df) > 0:
        # Plot velocity magnitude by timepoint
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Velocity magnitude by timepoint
        sns.boxplot(data=stats_df, x='timepoint', y='velocity_magnitude', ax=axes[0])
        axes[0].set_title('Velocity Magnitude by Timepoint')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Velocity magnitude by celltype (top celltypes only)
        top_celltypes = stats_df.groupby('celltype')['n_samples'].sum().nlargest(10).index
        stats_subset = stats_df[stats_df['celltype'].isin(top_celltypes)]
        sns.boxplot(data=stats_subset, x='celltype', y='velocity_magnitude', ax=axes[1])
        axes[1].set_title('Velocity Magnitude by Cell Type (Top 10)')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 6. Advanced Analysis with scVelo Integration (if applicable)

# %%
# Test scVelo integration if velocity computation was successful
if 'velocity_results' in locals():
    print("Testing scVelo integration...")
    
    try:
        # Initialize scVelo analysis
        scvelo_analyzer = ChromatinVelocityAnalysis(
            min_shared_counts=20,
            n_pcs=30,
            n_neighbors=30
        )
        
        # Prepare data for scVelo (requires some preprocessing)
        print("Preparing data for scVelo analysis...")
        
        # This would typically require:
        # 1. Embedding computation (PCA/UMAP)  
        # 2. Neighborhood graph construction
        # 3. Velocity estimation with different modes
        
        print("Note: Full scVelo analysis requires embedding and neighborhood computation")
        print("This would be the next step for comprehensive velocity analysis")
        
    except Exception as e:
        print(f"scVelo integration test: {str(e)}")

# %% [markdown]
# ## 7. Summary and Next Steps

# %%
print("=== CHROMATIN VELOCITY ANALYSIS SUMMARY ===")
print()

if 'velocity_results' in locals():
    print("✅ SUCCESS: Chromatin velocity computation completed")
    print(f"   - Computed velocity for {velocity_results['velocity'].shape[1]} peaks")
    print(f"   - Across {velocity_results['velocity'].shape[0]} pseudobulk samples")
    print(f"   - Mean velocity magnitude: {np.nanmean(np.sqrt(np.nansum(velocity_results['velocity']**2, axis=1))):.4f}")
else:
    print("❌ Chromatin velocity computation encountered issues")
    print("   - Check data format compatibility")
    print("   - Verify co-accessibility file structure")

print()
print("📊 DATA SUMMARY:")
print(f"   - Peak accessibility data: {adata_peaks.shape[0]} pseudobulks × {adata_peaks.shape[1]} peaks")
print(f"   - Co-accessibility pairs: {len(coaccess_df)} connections")
if 'filtered_coaccess' in locals():
    print(f"   - Filtered co-accessibility pairs: {len(filtered_coaccess)} connections")

print()
print("🔬 NEXT STEPS:")
print("   1. Optimize co-accessibility thresholds based on biological relevance")
print("   2. Integrate with existing peak UMAP clustering analysis")  
print("   3. Validate predictions against known regulatory dynamics")
print("   4. Generate publication-quality visualizations")
print("   5. Compare with RNA velocity if available")

# %%
# Save results if computation was successful
if 'velocity_results' in locals():
    print("Saving velocity results...")
    
    # Create output directory
    output_dir = Path("../figures/chromatin_velocity_test/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save velocity AnnData object
    adata_velocity.write_h5ad(output_dir / "chromatin_velocity_results.h5ad")
    
    # Save summary statistics
    summary_stats = {
        'n_peaks': velocity_results['velocity'].shape[1],
        'n_pseudobulks': velocity_results['velocity'].shape[0],
        'mean_velocity_magnitude': float(np.nanmean(np.sqrt(np.nansum(velocity_results['velocity']**2, axis=1)))),
        'velocity_range': [float(np.nanmin(velocity_results['velocity'])), float(np.nanmax(velocity_results['velocity']))],
        'n_coaccessibility_connections': len(filtered_coaccess) if 'filtered_coaccess' in locals() else len(coaccess_df)
    }
    
    import json
    with open(output_dir / "velocity_summary_stats.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"✅ Results saved to {output_dir}")

print("\n=== ANALYSIS COMPLETE ===")