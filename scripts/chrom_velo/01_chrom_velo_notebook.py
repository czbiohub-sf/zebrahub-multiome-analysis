# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chromatin Velocity Analysis - New Hybrid Approach
#
# This notebook implements a new chromatin velocity computation method that addresses
# the limitations of the previous approach.
#
# ## Key Improvements
#
# 1. **Temporal Tracking**: Track the SAME peaks across timepoints (not different peak sets)
# 2. **Co-accessibility Regularization**: Use regulatory context for smoothing
# 3. **Local Projection**: Project velocity to 2D using local linear transformations
#
# ## Comparison to Old Approach
#
# | Aspect | Old Method | New Method |
# |--------|------------|------------|
# | Peak tracking | Average across DIFFERENT peaks at each timepoint | Track SAME peaks over time |
# | Velocity assignment | All peaks at time t get identical velocity | Each peak has unique velocity |
# | Dimensionality reduction | Global SVD (190D → 2D) | Local PCA-based projection |
# | Co-accessibility | Not used | Used for regularization |
# | Works well for | Promoters (global signal) | All peak types (local dynamics) |

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import custom modules
import sys
sys.path.append('../chrom_velo/')
from chrom_velo_core import ChromatinVelocityComputer
from chrom_velo_viz import ChromatinVelocityVisualizer

# Set plotting style
plt.rcParams['figure.dpi'] = 100
sns.set_style('whitegrid')

print("Setup complete!")

# %% [markdown]
# ## Configuration

# %%
# Data paths
PEAKS_DATA = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_pb_annotated_master.h5ad"
COACCESS_DATA = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/13_peak_umap_analysis/cicero_integrated_ATAC_v2/02_integrated_ATAC_v2_cicero_connections_peaks_integrated_peaks.csv"

# Output paths
OUTPUT_DIR = Path("../../figures/chrom_velocity/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GPU Acceleration
USE_GPU = True  # Set to True to use GPU acceleration (CuPy/cuML required)

# Velocity Computation Parameters
ALPHA = 0.7  # Weight for temporal vs co-accessibility (0.7 = 70% temporal, 30% coaccess)
MIN_COACCESS_SCORE = 0.5  # Minimum co-accessibility threshold
MAX_CONNECTIONS = 100  # Maximum connections per peak
SMOOTHING_FACTOR = 0.5  # Spline smoothing parameter
VELOCITY_SCALE = 0.1  # Scaling for visualization

# Testing Parameters
SUBSET_PEAKS = None  # Use None for full dataset, or e.g. 10000 for quick testing
RUN_COMPARISON = True  # Compare temporal-only vs regularized velocity

# Checkpoint Parameters
LOAD_FROM_CHECKPOINT = True  # Set to True to load from saved checkpoint
CHECKPOINT_PATH = "../../data/processed_data/15_chrom_velo_processed/chromatin_velocity_computed.h5ad"

print(f"Configuration:")
print(f"  GPU acceleration: {USE_GPU}")
print(f"  Alpha (temporal weight): {ALPHA}")
print(f"  Min co-accessibility: {MIN_COACCESS_SCORE}")
print(f"  Max connections: {MAX_CONNECTIONS}")
print(f"  Subset peaks: {SUBSET_PEAKS or 'Full dataset'}")
print(f"  Run comparison: {RUN_COMPARISON}")
print(f"  Load from checkpoint: {LOAD_FROM_CHECKPOINT}")

# %% [markdown]
# ## Load Data

# %%
print("Loading peaks-by-pseudobulk data...")
adata = sc.read_h5ad(PEAKS_DATA)
print(f"  Shape: {adata.shape} (peaks × pseudobulks)")
print(f"  Pseudobulk names: {adata.var_names[:5].tolist()} ... {adata.var_names[-3:].tolist()}")
print(f"  Peak annotations: {list(adata.obs.columns)}")

# Check if UMAP coordinates exist
if 'X_umap' in adata.obsm:
    print(f"  UMAP coordinates: {adata.obsm['X_umap'].shape}")
else:
    print("  UMAP coordinates not found - will compute during analysis")

# %%
print("\nLoading co-accessibility data...")
coaccess_df = pd.read_csv(COACCESS_DATA)
print(f"  Shape: {coaccess_df.shape}")
print(f"  Columns: {list(coaccess_df.columns)}")
print(f"  Co-accessibility range: {coaccess_df['coaccess'].min():.3f} - {coaccess_df['coaccess'].max():.3f}")
print(f"  Connections with score >= {MIN_COACCESS_SCORE}: {(coaccess_df['coaccess'] >= MIN_COACCESS_SCORE).sum():,}")

# %% [markdown]
# ## Optional: Subset Data for Quick Testing

# %%
if SUBSET_PEAKS is not None:
    print(f"\nSubsetting to {SUBSET_PEAKS} peaks for quick testing...")
    # Stratified sampling by peak type
    if 'peak_type' in adata.obs.columns:
        subset_idx = []
        for peak_type in adata.obs['peak_type'].cat.categories:
            type_mask = adata.obs['peak_type'] == peak_type
            n_type = type_mask.sum()
            n_sample = int(SUBSET_PEAKS * n_type / len(adata))

            type_idx = np.where(type_mask)[0]
            sampled_idx = np.random.choice(type_idx, size=min(n_sample, len(type_idx)), replace=False)
            subset_idx.extend(sampled_idx)

        subset_idx = np.array(subset_idx)
        adata_subset = adata[subset_idx, :].copy()
        print(f"  Subset shape: {adata_subset.shape}")
        print(f"  Peak type distribution:")
        print(adata_subset.obs['peak_type'].value_counts())

        adata = adata_subset
    else:
        # Random sampling
        subset_idx = np.random.choice(len(adata), size=SUBSET_PEAKS, replace=False)
        adata = adata[subset_idx, :].copy()
        print(f"  Subset shape: {adata.shape}")

# %% [markdown]
# ## Check for Checkpoint

# %%
if LOAD_FROM_CHECKPOINT and Path(CHECKPOINT_PATH).exists():
    print("\n" + "="*60)
    print("LOADING FROM CHECKPOINT")
    print("="*60)
    print(f"Loading precomputed velocities from: {CHECKPOINT_PATH}")

    adata_checkpoint = sc.read_h5ad(CHECKPOINT_PATH)

    # Reconstruct ChromatinVelocityComputer with loaded results
    cv_computer = ChromatinVelocityComputer(
        adata=adata_checkpoint,
        alpha=ALPHA,
        use_gpu=USE_GPU,
        verbose=True
    )
    cv_computer.temporal_velocity = adata_checkpoint.layers['temporal_velocity']
    cv_computer.regularized_velocity = adata_checkpoint.layers['regularized_velocity']
    velocity_2d_reg = adata_checkpoint.obsm['velocity_umap']
    umap_coords = adata_checkpoint.obsm['X_umap']

    # Also load temporal 2D velocity if available
    if 'velocity_umap_temporal' in adata_checkpoint.obsm.keys():
        velocity_2d_temp = adata_checkpoint.obsm['velocity_umap_temporal']
    else:
        # Need to recompute temporal 2D projection
        print("\nProjecting temporal velocity to 2D...")
        temp_reg = cv_computer.regularized_velocity
        cv_computer.regularized_velocity = cv_computer.temporal_velocity
        velocity_2d_temp, _ = cv_computer.project_to_2d(
            use_umap_coords=True,
            velocity_key='regularized'
        )
        cv_computer.regularized_velocity = temp_reg

    # Update adata reference
    adata = adata_checkpoint

    print("✓ Loaded from checkpoint. Skipping computation, proceeding to visualization.")
    print(f"  Temporal velocity shape: {cv_computer.temporal_velocity.shape}")
    print(f"  Regularized velocity shape: {cv_computer.regularized_velocity.shape}")
    print(f"  2D velocity shape: {velocity_2d_reg.shape}")

    # Skip to visualization (we'll use a flag)
    SKIP_COMPUTATION = True
else:
    SKIP_COMPUTATION = False

# %% [markdown]
# ## Initialize Chromatin Velocity Computer

# %%
if not SKIP_COMPUTATION:
    print("\n" + "="*60)
    print("Initializing ChromatinVelocityComputer...")
    print("="*60)

    cv_computer = ChromatinVelocityComputer(
        adata=adata,
        coaccess_df=coaccess_df,
    alpha=ALPHA,
    use_gpu=USE_GPU,
    verbose=True
)

    print(f"\nConfiguration:")
    print(f"  Unique timepoints: {cv_computer.unique_timepoints}")
    print(f"  Number of pseudobulks: {len(cv_computer.pseudobulk_timepoints)}")
    print(f"  Alpha (temporal weight): {cv_computer.alpha}")
    print(f"  GPU acceleration: {cv_computer.use_gpu}")

# %% [markdown]
# ## Step 1: Compute Temporal Velocity
#
# This step tracks the same peaks across timepoints and computes temporal derivatives.

# %%
if not SKIP_COMPUTATION:
    print("\n" + "="*60)
    print("STEP 1: Computing Temporal Velocity")
    print("="*60)

    temporal_velocity = cv_computer.compute_temporal_velocity(
        smoothing_factor=SMOOTHING_FACTOR
    )

    print(f"\nTemporal velocity statistics:")
    print(f"  Shape: {temporal_velocity.shape}")
    print(f"  Mean magnitude: {np.abs(temporal_velocity).mean():.4f}")
    print(f"  Std magnitude: {np.abs(temporal_velocity).std():.4f}")
    print(f"  Min: {temporal_velocity.min():.4f}, Max: {temporal_velocity.max():.4f}")

# %% [markdown]
# ## Step 2: Regularize Using Co-accessibility
#
# This step smooths the velocity using co-accessible peaks.

# %%
if not SKIP_COMPUTATION:
    print("\n" + "="*60)
    print("STEP 2: Regularizing Velocity with Co-accessibility")
    print("="*60)

    regularized_velocity = cv_computer.compute_coaccessibility_regularization(
        min_coaccess_score=MIN_COACCESS_SCORE,
        max_connections=MAX_CONNECTIONS
    )

    print(f"\nRegularized velocity statistics:")
    print(f"  Shape: {regularized_velocity.shape}")
    print(f"  Mean magnitude: {np.abs(regularized_velocity).mean():.4f}")
    print(f"  Std magnitude: {np.abs(regularized_velocity).std():.4f}")
    print(f"  Min: {regularized_velocity.min():.4f}, Max: {regularized_velocity.max():.4f}")

    # Compare to temporal velocity
    change = np.abs(regularized_velocity - temporal_velocity).mean()
    print(f"\nMean absolute change from temporal: {change:.4f}")

# %% [markdown]
# ## Step 3: Project to 2D UMAP Space

# %%
if not SKIP_COMPUTATION:
    print("\n" + "="*60)
    print("STEP 3: Projecting Velocity to 2D")
    print("="*60)

    # Project regularized velocity
    velocity_2d_reg, umap_coords = cv_computer.project_to_2d(
        use_umap_coords=True,
        velocity_key='regularized'
    )

    # Also project temporal velocity for comparison
    print("\nProjecting temporal velocity for comparison...")
    # Temporarily store regularized velocity
    temp_reg = cv_computer.regularized_velocity
    cv_computer.regularized_velocity = cv_computer.temporal_velocity  # Trick to project temporal
    velocity_2d_temp, _ = cv_computer.project_to_2d(
        use_umap_coords=True,
        velocity_key='regularized'
    )
    cv_computer.regularized_velocity = temp_reg  # Restore

    print(f"\n2D velocity statistics (regularized):")
    print(f"  Shape: {velocity_2d_reg.shape}")
    vel_mag_2d = np.linalg.norm(velocity_2d_reg, axis=1)
    print(f"  Mean magnitude: {vel_mag_2d.mean():.4f}")
    print(f"  Std magnitude: {vel_mag_2d.std():.4f}")

    # %% [markdown]
    # ## Save Checkpoint (Intermediate Results)

    # %%
    print("\n" + "="*60)
    print("SAVING CHECKPOINT")
    print("="*60)

    from pathlib import Path
    checkpoint_dir = Path("../../data/processed_data/15_chrom_velo_processed/")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "chromatin_velocity_computed.h5ad"

    # Store 2D velocity in the computer object for saving
    cv_computer.velocity_2d = velocity_2d_reg

    # Also store temporal 2D velocity in adata for future use
    adata.obsm['velocity_umap_temporal'] = velocity_2d_temp

    # Save results
    cv_computer.save_results(str(checkpoint_path))

    print(f"\n✓ Checkpoint saved to: {checkpoint_path}")
    print(f"  Contains: temporal_velocity, regularized_velocity, velocity_umap")
    print(f"  This allows recovery if visualization fails.")

# %% [markdown]
# ## Step 4: Visualize Results

# %%
print("\n" + "="*60)
print("STEP 4: Creating Visualizations")
print("="*60)

# Initialize visualizer
viz = ChromatinVelocityVisualizer(
    adata=adata,
    velocity_2d=velocity_2d_reg,
    umap_coords=umap_coords
)

# %%
# Plot 1: Streamplot colored by peak type
print("\nCreating streamplot...")
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
viz.plot_streamplot(
    color_by='peak_type_argelaguet',
    velocity_scale=VELOCITY_SCALE,
    density=1.5,
    ax=ax
)
output_path = OUTPUT_DIR / "01_streamplot_peak_type.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  Saved to {output_path}")
plt.show()

# %%
# Plot 2: Arrow plot
print("\nCreating arrow plot...")
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
viz.plot_arrows(
    color_by='peak_type_argelaguet',
    velocity_scale=VELOCITY_SCALE,
    subsample=3000,
    min_velocity=0.05,
    ax=ax
)
output_path = OUTPUT_DIR / "02_arrows_peak_type.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  Saved to {output_path}")
plt.show()

# %%
# Plot 3: Velocity magnitude distribution
print("\nCreating magnitude distribution plot...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
viz.plot_velocity_magnitude_distribution(groupby='peak_type_argelaguet', ax=ax)
output_path = OUTPUT_DIR / "03_magnitude_distribution.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  Saved to {output_path}")
plt.show()

# %%
# Plot 4: Temporal vs Regularized comparison
print("\nCreating temporal vs regularized comparison...")
fig = viz.plot_velocity_comparison(
    temporal_velocity_2d=velocity_2d_temp,
    regularized_velocity_2d=velocity_2d_reg,
    subsample=5000
)
output_path = OUTPUT_DIR / "04_temporal_vs_regularized.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  Saved to {output_path}")
plt.show()

# %%
# Plot 5: Comprehensive summary
print("\nCreating comprehensive summary figure...")
fig = viz.plot_comprehensive_summary(
    color_by='peak_type_argelaguet',
    velocity_scale=VELOCITY_SCALE,
    output_path=OUTPUT_DIR / "05_comprehensive_summary.png"
)
plt.show()

# %% [markdown]
# ## Step 5: Save Results

# %%
print("\n" + "="*60)
print("STEP 5: Saving Results")
print("="*60)

output_h5ad = OUTPUT_DIR / "chromatin_velocity_results.h5ad"
cv_computer.save_results(str(output_h5ad))

print(f"\nResults saved to: {output_h5ad}")
print(f"  Layers added:")
print(f"    - 'temporal_velocity': Raw temporal velocity")
print(f"    - 'regularized_velocity': Co-accessibility regularized velocity")
print(f"  Obsm added:")
print(f"    - 'velocity_umap': 2D velocity vectors")

# %% [markdown]
# ## Summary and Next Steps

# %%
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

print(f"\nResults summary:")
print(f"  Total peaks analyzed: {len(adata):,}")
print(f"  Timepoints: {list(cv_computer.unique_timepoints)}")
if not SKIP_COMPUTATION:
    print(f"  Mean temporal velocity magnitude: {np.abs(temporal_velocity).mean():.4f}")
    print(f"  Mean regularized velocity magnitude: {np.abs(regularized_velocity).mean():.4f}")
    print(f"  Mean 2D velocity magnitude: {vel_mag_2d.mean():.4f}")
else:
    print(f"  Mean temporal velocity magnitude: {np.abs(cv_computer.temporal_velocity).mean():.4f}")
    print(f"  Mean regularized velocity magnitude: {np.abs(cv_computer.regularized_velocity).mean():.4f}")
    vel_mag_2d = np.linalg.norm(velocity_2d_reg, axis=1)
    print(f"  Mean 2D velocity magnitude: {vel_mag_2d.mean():.4f}")

print(f"\nOutput files:")
print(f"  Figures: {OUTPUT_DIR}/")
print(f"  Results: {output_h5ad}")

print("\nNext steps:")
print("  1. Validate results on promoters (should maintain good performance)")
print("  2. Check enhancers/intergenic regions (expect improvement)")
print("  3. Compare to old implementation side-by-side")
print("  4. Optimize parameters (alpha, min_coaccess_score, etc.)")
print("  5. Apply to full dataset if using subset")

# %% [markdown]
# ## Timing Summary

# %%
# Print timing statistics
cv_computer.print_timing_summary()

# %% [markdown]
# ## Comparison: Temporal-Only vs Regularized (Optional)
#
# This section compares velocity computed with ONLY temporal derivatives (alpha=1.0)
# versus velocity with co-accessibility regularization (alpha=0.7).

# %%
if RUN_COMPARISON:
    print("\n" + "="*60)
    print("COMPARISON: Temporal-Only vs Regularized Velocity")
    print("="*60)

    # Compute temporal-only velocity (alpha=1.0, no co-accessibility)
    print("\nComputing temporal-only velocity (alpha=1.0)...")
    cv_temporal_only = ChromatinVelocityComputer(
        adata=adata,
        coaccess_df=None,  # No co-accessibility
        alpha=1.0,
        use_gpu=USE_GPU,
        verbose=False
    )

    # Reuse temporal velocity from main computation
    cv_temporal_only.temporal_velocity = cv_computer.temporal_velocity.copy()
    cv_temporal_only.regularized_velocity = cv_computer.temporal_velocity.copy()  # Same as temporal

    # Project to 2D
    velocity_2d_temporal_only, _ = cv_temporal_only.project_to_2d(
        use_umap_coords=True,
        velocity_key='temporal'
    )

    # Create comparison visualization
    print("\nCreating temporal vs regularized comparison plot...")
    viz_temp = ChromatinVelocityVisualizer(
        adata=adata,
        velocity_2d=velocity_2d_temporal_only,
        umap_coords=umap_coords
    )

    fig = viz.plot_velocity_comparison(
        temporal_velocity_2d=velocity_2d_temporal_only,
        regularized_velocity_2d=velocity_2d_reg,
        subsample=5000
    )
    output_path = OUTPUT_DIR / "06_temporal_vs_regularized_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.show()

    # Statistical comparison
    print("\nStatistical Comparison:")
    print(f"  Temporal-only mean magnitude: {np.linalg.norm(velocity_2d_temporal_only, axis=1).mean():.4f}")
    print(f"  Regularized mean magnitude: {np.linalg.norm(velocity_2d_reg, axis=1).mean():.4f}")

    correlation = np.corrcoef(
        np.linalg.norm(velocity_2d_temporal_only, axis=1),
        np.linalg.norm(velocity_2d_reg, axis=1)
    )[0, 1]
    print(f"  Magnitude correlation: {correlation:.3f}")

    # Peak type comparison
    if 'peak_type' in adata.obs.columns:
        print("\nMean velocity by peak type:")
        print("  Peak Type          | Temporal-only | Regularized | Difference")
        print("  " + "-"*60)
        for peak_type in adata.obs['peak_type'].cat.categories:
            mask = adata.obs['peak_type'] == peak_type
            temp_mag = np.linalg.norm(velocity_2d_temporal_only[mask], axis=1).mean()
            reg_mag = np.linalg.norm(velocity_2d_reg[mask], axis=1).mean()
            diff = reg_mag - temp_mag
            print(f"  {peak_type:18s} | {temp_mag:13.4f} | {reg_mag:11.4f} | {diff:+10.4f}")

    print("\n✓ Comparison complete!")

# %% [markdown]
# ## Parameter Exploration (Optional)

# %%
# Uncomment to explore different alpha values
"""
print("\n" + "="*60)
print("PARAMETER EXPLORATION: Alpha Values")
print("="*60)

alphas = [0.5, 0.7, 0.9]
results = {}

for alpha in alphas:
    print(f"\nTesting alpha = {alpha}...")
    cv_temp = ChromatinVelocityComputer(adata, coaccess_df, alpha=alpha)
    cv_temp.temporal_velocity = temporal_velocity.copy()  # Reuse temporal
    reg_vel = cv_temp.compute_coaccessibility_regularization(MIN_COACCESS_SCORE, MAX_CONNECTIONS)
    results[alpha] = reg_vel
    print(f"  Mean magnitude: {np.abs(reg_vel).mean():.4f}")

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, alpha in enumerate(alphas):
    axes[i].hist(np.abs(results[alpha]).flatten(), bins=50, alpha=0.7)
    axes[i].set_title(f'Alpha = {alpha}')
    axes[i].set_xlabel('Velocity magnitude')
    axes[i].set_ylabel('Count')
plt.tight_layout()
plt.show()
"""

# %%
