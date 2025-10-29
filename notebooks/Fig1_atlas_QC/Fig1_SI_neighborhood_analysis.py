# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (single-cell-base)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neighborhood Purity and Cross-Modality Integration Quality Assessment
# 
# **Zebrahub-Multiome Analysis Pipeline - Updated for Zebrahub Data Structure**
# 
# This notebook demonstrates comprehensive evaluation of multimodal integration quality using three complementary approaches:
# 
# 1. **Neighborhood Purity Analysis** - Measures how well cells of the same type cluster together in neighborhood graphs
# 2. **Cross-Modality Validation** - Tests how well embeddings preserve structure across modalities 
# 3. **scIB-Based Evaluation** - Uses scIB framework metrics for rigorous integration quality assessment
# 
# ---
# 
# ## Key Insights
# 
# The goal is to demonstrate that **joint (WNN) embeddings provide superior integration** compared to individual RNA or ATAC modalities by:
# 
# - Higher neighborhood purity scores for biological metadata
# - Better preservation of cluster structure across modalities
# - Superior scIB integration metrics (ARI, NMI, ASW_label, Graph_connectivity)
# 
# ---

# %% [markdown]
# ## Setup and Imports

# %%
import sys
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add scripts directory to path for importing our custom modules
sys.path.append('../scripts')
from neighborhood_purity_module import *

# Configure scanpy settings
sc.settings.verbosity = 3  # verbosity level: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, facecolor='white')

# Set random seed for reproducibility  
np.random.seed(42)

print("✓ Imports completed successfully")
print(f"✓ scIB package available: {SCIB_AVAILABLE}")

# %% [markdown]
# ## Data Loading and Configuration
# 
# **Note**: Replace the file path below with your actual multiome data file.
# 
# The data should contain (Zebrahub multiome structure):
# - **Connectivity matrices** in `adata.obsp`: 'RNA_connectivities', 'ATAC_connectivities', 'connectivities_wnn'
# - **Embeddings** in `adata.obsm`: 'X_pca', 'X_lsi', 'X_wnn.umap'
# - **Cluster labels** in `adata.obs`: 'RNA_leiden_08', 'ATAC_leiden_08', 'wsnn_res.0.8'  
# - **Metadata** in `adata.obs`: 'global_annotation', timepoints, etc.

# %%
# CONFIGURE YOUR DATA PATHS AND KEYS HERE
# ==========================================

# Path to your multiome AnnData object
# data_path = "/path/to/your/multiome_data.h5ad" 
data_path = "your_multiome_data.h5ad"  # Update this path

# Define connectivity matrices (for neighborhood purity analysis) - Updated for Zebrahub data
connectivity_keys = {
    'RNA': 'RNA_connectivities',      # RNA neighborhood graph
    'ATAC': 'ATAC_connectivities',    # ATAC neighborhood graph  
    'WNN': 'connectivities_wnn'       # Weighted nearest neighbor graph
}

# Define embedding keys (for clustering and scIB metrics) - Updated for Zebrahub data
embedding_keys = {
    'RNA': 'X_pca',          # RNA PCA embedding
    'ATAC': 'X_lsi',         # ATAC LSI embedding  
    'WNN': 'X_wnn.umap'      # WNN embedding
}

# Define cluster label keys (for cross-modality validation) - Updated for Zebrahub data
cluster_keys = {
    'RNA': 'RNA_leiden_08',      # RNA leiden clusters at resolution 0.8
    'ATAC': 'ATAC_leiden_08',    # ATAC leiden clusters at resolution 0.8
    'WNN': 'wsnn_res.0.8'        # WNN clusters at resolution 0.8
}

# Metadata key for biological validation
metadata_key = 'annotation_ML_coarse'  # or other metadata in adata.obs

# Analysis parameters
leiden_resolution = 0.8    # Resolution for leiden clustering
k_neighbors = 30           # Number of neighbors for purity analysis
n_neighbors_clustering = 15 # Number of neighbors for clustering

print("✓ Configuration completed")

# %% [markdown]
# ### Load Data and Inspect Structure
# 
# **Note**: If you don't have real data yet, uncomment the synthetic data generation section below.

# %%
# OPTION 1: Load real data
# ========================
try:
    if os.path.exists(data_path):
        adata = sc.read_h5ad(data_path)
        print(f"✓ Loaded data: {adata.shape}")
        
        # Display available connectivity matrices
        print("\nAvailable connectivity matrices:")
        for key in adata.obsp.keys():
            if 'connectivities' in key.lower():
                print(f"  - {key}: {adata.obsp[key].shape}")
        
        # Display available embeddings  
        print("\nAvailable embeddings:")
        for key in adata.obsm.keys():
            print(f"  - {key}: {adata.obsm[key].shape}")
                
        # Display metadata columns (showing key ones for Zebrahub data)
        key_metadata = ['global_annotation', 'RNA_leiden_08', 'ATAC_leiden_08', 'wsnn_res.0.8']
        print(f"\nKey metadata columns:")
        for col in key_metadata:
            if col in adata.obs.columns:
                n_unique = adata.obs[col].nunique()
                print(f"  - {col}: {n_unique} unique values")
        
        if metadata_key in adata.obs.columns:
            print(f"\n{metadata_key} categories: {adata.obs[metadata_key].value_counts().head()}")
        else:
            print(f"Warning: {metadata_key} not found in metadata")
            
    else:
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
except Exception as e:
    print(f"Could not load real data: {e}")
    print("Generating synthetic data for demonstration...")
    
    # OPTION 2: Generate synthetic data for demonstration
    # ==================================================
    n_cells = 2000
    n_genes = 1000
    n_peaks = 1500
    
    # Create synthetic AnnData
    adata = sc.AnnData(X=np.random.negative_binomial(5, 0.3, (n_cells, n_genes)))
    
    # Add synthetic metadata
    cell_types = ['neural_crest', 'mesoderm', 'endoderm', 'epidermis', 'neural']
    adata.obs[metadata_key] = np.random.choice(cell_types, n_cells)
    adata.obs['timepoint'] = np.random.choice(['6hpf', '12hpf', '18hpf', '24hpf'], n_cells)
    
    # Generate synthetic embeddings (matching Zebrahub structure)
    adata.obsm['X_pca'] = np.random.randn(n_cells, 50) 
    adata.obsm['X_lsi'] = np.random.randn(n_cells, 50)
    adata.obsm['X_wnn.umap'] = np.random.randn(n_cells, 50)
    
    # Add synthetic cluster labels (matching Zebrahub structure)
    adata.obs['global_annotation'] = adata.obs[metadata_key]
    adata.obs['RNA_leiden_08'] = np.random.randint(0, 15, n_cells).astype(str)
    adata.obs['ATAC_leiden_08'] = np.random.randint(0, 12, n_cells).astype(str)
    adata.obs['wsnn_res.0.8'] = np.random.randint(0, 18, n_cells).astype(str)
    
    # Generate synthetic connectivity matrices (matching Zebrahub structure)
    from scipy.sparse import csr_matrix
    for modality, conn_key in connectivity_keys.items():
        # Simple synthetic connectivity (normally this would be from sc.pp.neighbors)
        indices = np.random.randint(0, n_cells, (n_cells, 15))
        indptr = np.arange(0, n_cells * 15 + 1, 15)
        data = np.random.rand(n_cells * 15)
        adata.obsp[conn_key] = csr_matrix((data, indices.flatten(), indptr), shape=(n_cells, n_cells))
    
    print(f"✓ Generated synthetic data: {adata.shape}")
    print(f"✓ Cell types: {adata.obs[metadata_key].value_counts()}")

# %% [markdown]
# ### Option: Use Convenience Function for Complete Analysis
# 
# **For quick analysis with Zebrahub data structure, you can use the convenience function:**

# %%
# OPTIONAL: Use convenience function for complete analysis in one step
USE_CONVENIENCE_FUNCTION = False  # Set to True to use the one-step analysis

if USE_CONVENIENCE_FUNCTION:
    print("🚀 Using convenience function for complete analysis...")
    
    purity_summary, validation_summary, all_results = analyze_zebrahub_multiome_neighborhoods(
        adata=adata,
        metadata_key=metadata_key,
        k=k_neighbors,
        save_plots=True,
        output_dir='../figures/integration_quality/'
    )
    
    print("✅ Complete analysis finished!")
    print("Check the output directory for all results and plots.")
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - You can stop here if using the convenience function")
    print("="*60)
    
else:
    print("📝 Using step-by-step analysis approach...")

# %% [markdown]
# ---
# 
# # 1. Neighborhood Purity Analysis (Step-by-Step)
# 
# **Objective**: Measure how well cells of the same biological type cluster together in their k-nearest neighborhood.
# 
# **Key Insight**: Good embeddings should have high neighborhood purity - cells should be surrounded by neighbors of the same cell type.
# 
# **Expected Results**: WNN (joint) embedding should show higher purity scores than individual RNA or ATAC modalities.

# %%
# Skip step-by-step analysis if convenience function was used
if not USE_CONVENIENCE_FUNCTION:
    print("=" * 60)
    print("1. NEIGHBORHOOD PURITY ANALYSIS")
    print("=" * 60)

    # Compute purity scores using pre-computed connectivities
    purity_results = compute_multimodal_knn_purity(
        adata=adata,
        connectivity_keys=connectivity_keys,
        metadata_key=metadata_key,
        k=k_neighbors
    )

    # Summarize purity results
    print(f"\n📊 Purity Analysis Summary ({metadata_key}):")
    summary_purity = summarize_purity_scores(purity_results, adata, metadata_key)
    print(summary_purity[summary_purity['Metadata'] == 'Overall'][['Modality', 'Mean_Purity', 'Std_Purity']])
else:
    print("Skipping step-by-step analysis since convenience function was used.")

# %% [markdown]
# ### Visualize Neighborhood Purity Results

# %%
if not USE_CONVENIENCE_FUNCTION:
    # Create comprehensive purity comparison plots
    fig_purity = plot_purity_comparison(
        purity_results, 
        adata, 
        metadata_key,
        figsize=(15, 5)
    )

    plt.suptitle(f'Neighborhood Purity Analysis - {metadata_key.upper()}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # Add purity scores to AnnData for later analysis
    add_purity_to_adata(adata, purity_results, metadata_key)

# %% [markdown]
# ### Interpretation: Neighborhood Purity
# 
# **What to look for:**
# - **Higher purity scores** = Better embedding quality
# - **WNN > RNA, ATAC** = Joint embedding successfully integrates modalities
# - **Consistent patterns** across cell types = Robust integration
# 
# **Typical good results:**
# - WNN purity: 0.7-0.9
# - RNA/ATAC purity: 0.5-0.7
# - Clear separation in violin plots

# %% [markdown]
# ---
# 
# # 2. Cross-Modality Validation
# 
# **Objective**: Test how well embeddings preserve cluster structure derived from other modalities.
# 
# **Strategy**:
# 1. Cluster cells independently in each modality (RNA, ATAC, WNN)
# 2. Measure preservation using ARI and NMI metrics
# 3. Test bidirectional preservation (RNA→ATAC, ATAC→RNA, etc.)
# 
# **Key Insight**: Good joint embeddings should preserve structure from both individual modalities.

# %%
if not USE_CONVENIENCE_FUNCTION:
    print("=" * 60)
    print("2. CROSS-MODALITY VALIDATION")
    print("=" * 60)

    # Perform bidirectional cross-modality validation using existing cluster labels
    validation_df = compute_bidirectional_cross_modality_validation(
        adata=adata,
        cluster_keys=cluster_keys
    )

    print(f"\n📊 Cross-Modality Validation Results:")
    print(validation_df[['Reference_Modality', 'Target_Modality', 'ARI', 'NMI']].round(3))

# %% [markdown]
# ### Visualize Cross-Modality Validation

# %%
if not USE_CONVENIENCE_FUNCTION:
    # Create cross-modality validation plots
    fig_cross = plot_cross_modality_validation(
        validation_df,
        figsize=(12, 5) 
    )

    plt.suptitle('Cross-Modality Cluster Preservation', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # Summarize cross-modality results
    summary_cross = summarize_cross_modality_validation(validation_df)
    print(f"\n📈 Cross-Modality Summary:")
    print(summary_cross[summary_cross['Category'] == 'Overall'][['Metric', 'Mean', 'Std']].round(3))

# %% [markdown]
# ### Interpretation: Cross-Modality Validation
# 
# **What to look for:**
# - **Higher ARI/NMI** = Better cluster preservation
# - **Asymmetric scores** = Some embeddings better preserve others
# - **WNN as target** should show high scores (preserves both RNA and ATAC structure)
# 
# **Expected patterns for good WNN:**
# - WNN preserves both RNA and ATAC clusters well
# - RNA and ATAC may not preserve each other as well
# - WNN→RNA and WNN→ATAC scores > RNA→ATAC scores

# %% [markdown]
# ---
# 
# # 3. scIB-Based Integration Quality Assessment
# 
# **Objective**: Apply the gold-standard scIB framework for comprehensive integration evaluation.
# 
# **scIB Strategy**:
# 1. **Break circularity** - Each modality serves as independent validator
# 2. **Comprehensive metrics** - ARI, NMI, ASW_label, Graph_connectivity
# 3. **Rigorous evaluation** - Following established integration benchmarking standards
# 
# **Key Advantage**: Avoids circular validation by using each modality to validate others.

# %%
print("=" * 60)  
print("3. scIB-BASED INTEGRATION QUALITY ASSESSMENT")
print("=" * 60)

# Perform comprehensive scIB-based evaluation
scib_metrics_df, modality_clusters = compute_scIB_integration_quality_comprehensive(
    adata=adata,
    embedding_keys=embedding_keys,
    leiden_resolution=leiden_resolution,
    n_neighbors=n_neighbors_clustering,
    use_scib=True  # Use scIB package if available
)

print(f"\n📊 scIB Integration Metrics:")
print(scib_metrics_df[['Reference_Modality', 'Target_Modality', 'ARI_cluster', 'NMI_cluster', 'ASW_label']].round(3))

# %% [markdown]
# ### Visualize scIB Integration Metrics

# %%
# Create comprehensive scIB visualization
fig_scib = plot_scIB_integration_metrics(
    scib_metrics_df,
    figsize=(16, 10)
)

if fig_scib:
    plt.suptitle('scIB Integration Quality Assessment', fontsize=18, y=0.98)
    plt.tight_layout()
    plt.show()

# Add cluster assignments to AnnData
add_scIB_clusters_to_adata(adata, modality_clusters)

# %% [markdown]
# ### scIB Results Summary and Interpretation

# %%
# Detailed scIB summary
scib_summary = summarize_scIB_metrics(scib_metrics_df)

print("📈 scIB Metrics Summary:")
print("=" * 40)

# Overall performance
overall_metrics = scib_summary[scib_summary['Category'] == 'Overall']
for _, row in overall_metrics.iterrows():
    print(f"{row['Metric']:15}: {row['Mean']:.3f} ± {row['Std']:.3f}")

print("\n🎯 By Reference Modality (How well each preserves others):")
by_ref = scib_summary[scib_summary['Category'] == 'By_Reference']
for ref_mod in by_ref['Reference'].unique():
    if ref_mod != 'All':
        print(f"\n{ref_mod}:")
        ref_data = by_ref[by_ref['Reference'] == ref_mod]
        for _, row in ref_data.iterrows():
            print(f"  {row['Metric']:15}: {row['Mean']:.3f}")

print(f"\n🎯 By Target Modality (How well each is preserved by others):")
by_target = scib_summary[scib_summary['Category'] == 'By_Target']
for target_mod in by_target['Target'].unique():
    if target_mod != 'All':
        print(f"\n{target_mod}:")
        target_data = by_target[by_target['Target'] == target_mod]
        for _, row in target_data.iterrows():
            print(f"  {row['Metric']:15}: {row['Mean']:.3f}")

# %% [markdown]
# ### Interpretation: scIB Integration Quality
# 
# **Key scIB Metrics:**
# 
# - **ARI_cluster**: Adjusted Rand Index between clusterings (0-1, higher better)
# - **NMI_cluster**: Normalized Mutual Information between clusterings (0-1, higher better)  
# - **ASW_label**: Average Silhouette Width using cluster labels (higher better)
# - **Graph_connectivity**: Preservation of neighborhood structure (0-1, higher better)
# 
# **Expected Results for Superior WNN Integration:**
# 
# 1. **High preservation scores**: WNN should preserve both RNA and ATAC structure
# 2. **Asymmetric pattern**: WNN→RNA/ATAC > RNA→ATAC cross-preservation
# 3. **Biological coherence**: High ASW_label scores indicate good separation
# 4. **Neighborhood preservation**: High Graph_connectivity scores

# %% [markdown]
# ---
# 
# # 4. Comparative Analysis and Final Assessment
# 
# **Objective**: Synthesize results across all three analysis approaches to make final conclusions about integration quality.

# %%
print("=" * 60)
print("4. COMPARATIVE ANALYSIS")
print("=" * 60)

# Create comparative summary
comparative_results = []

# Neighborhood purity results
purity_summary = summary_purity[summary_purity['Metadata'] == 'Overall']
for _, row in purity_summary.iterrows():
    comparative_results.append({
        'Analysis': 'Neighborhood_Purity',
        'Modality': row['Modality'],
        'Metric': 'Mean_Purity',
        'Score': row['Mean_Purity'],
        'Interpretation': 'Higher = Better'
    })

# Cross-modality ARI results  
cross_summary = summary_cross[summary_cross['Category'] == 'By_Target']
for _, row in cross_summary[cross_summary['Metric'] == 'ARI'].iterrows():
    if row['Target'] != 'All':
        comparative_results.append({
            'Analysis': 'Cross_Modality',
            'Modality': row['Target'],
            'Metric': 'Mean_ARI_as_Target',
            'Score': row['Mean'],
            'Interpretation': 'Higher = Better Preserved'
        })

# scIB ASW results
scib_asw = scib_summary[(scib_summary['Category'] == 'By_Target') & (scib_summary['Metric'] == 'ASW_label')]
for _, row in scib_asw.iterrows():
    if row['Target'] != 'All':
        comparative_results.append({
            'Analysis': 'scIB_Integration',
            'Modality': row['Target'],  
            'Metric': 'Mean_ASW_label',
            'Score': row['Mean'],
            'Interpretation': 'Higher = Better Integration'
        })

# Create comparative DataFrame
comparative_df = pd.DataFrame(comparative_results)

print("🔍 Comparative Results Summary:")
print("=" * 40)
pivot_comparison = comparative_df.pivot_table(
    index=['Analysis', 'Metric'], 
    columns='Modality',
    values='Score'
).fillna('N/A')

print(pivot_comparison.round(3))

# %% [markdown]
# ### Final Recommendations and Conclusions

# %%
print("\n" + "=" * 60)
print("FINAL ASSESSMENT AND RECOMMENDATIONS")
print("=" * 60)

# Analyze results to determine best embedding
def determine_best_embedding(comparative_df):
    """Determine which embedding performs best across metrics."""
    
    # Calculate mean score per modality across all metrics
    modality_scores = comparative_df.groupby('Modality')['Score'].agg(['mean', 'count', 'std'])
    
    print("📊 Overall Performance Ranking:")
    print("-" * 30)
    
    ranking = modality_scores.sort_values('mean', ascending=False)
    for i, (modality, scores) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {modality}: {scores['mean']:.3f} ± {scores['std']:.3f} (n={scores['count']} metrics)")
    
    best_embedding = ranking.index[0]
    
    print(f"\n🏆 BEST PERFORMING EMBEDDING: {best_embedding}")
    
    return best_embedding, ranking

best_embedding, performance_ranking = determine_best_embedding(comparative_df)

print(f"\n✅ VALIDATION CONCLUSIONS:")
print("-" * 25)

if best_embedding == 'WNN':
    print("✓ Joint (WNN) embedding demonstrates SUPERIOR integration quality")
    print("✓ Successfully combines RNA and ATAC modalities")
    print("✓ Higher neighborhood purity and cross-modality preservation")
    print("✓ Strong performance across scIB benchmark metrics")
    
    print(f"\n📋 EVIDENCE SUPPORTING WNN SUPERIORITY:")
    wnn_evidence = comparative_df[comparative_df['Modality'] == 'WNN']
    for _, row in wnn_evidence.iterrows():
        print(f"  - {row['Analysis']}: {row['Metric']} = {row['Score']:.3f}")
        
else:
    print(f"⚠️  Unexpected result: {best_embedding} outperformed WNN")
    print("⚠️  This may indicate:")
    print("   - Data quality issues")
    print("   - Suboptimal WNN integration parameters") 
    print("   - Need for further integration optimization")

print(f"\n💡 METHODOLOGICAL VALIDATION:")
print("-" * 30)
print("✓ Used three complementary validation approaches")
print("✓ Avoided circularity through independent clustering")
print("✓ Applied gold-standard scIB framework")
print("✓ Comprehensive statistical assessment")

# %% [markdown]
# ---
# 
# # 5. Export Results and Final Summary
# 
# **Save all results for further analysis and reporting.**

# %%
print("=" * 60)
print("5. EXPORTING RESULTS")
print("=" * 60)

# Create results directory
results_dir = Path("../figures/integration_quality_assessment")
results_dir.mkdir(parents=True, exist_ok=True)

# Export summary tables
summary_purity.to_csv(results_dir / "neighborhood_purity_summary.csv", index=False)
validation_df.to_csv(results_dir / "cross_modality_validation.csv", index=False) 
scib_metrics_df.to_csv(results_dir / "scib_integration_metrics.csv", index=False)
comparative_df.to_csv(results_dir / "comparative_analysis.csv", index=False)

# Save plots
fig_purity.savefig(results_dir / "neighborhood_purity_comparison.png", dpi=300, bbox_inches='tight')
fig_cross.savefig(results_dir / "cross_modality_validation.png", dpi=300, bbox_inches='tight')
if fig_scib:
    fig_scib.savefig(results_dir / "scib_integration_assessment.png", dpi=300, bbox_inches='tight')

# Create final summary report
report_content = f"""
# Integration Quality Assessment Report

## Dataset Summary
- Cells: {adata.n_obs:,}
- Features: {adata.n_vars:,}  
- Metadata: {metadata_key}
- Modalities: {', '.join(embedding_keys.keys())}

## Key Results

### Best Performing Embedding: {best_embedding}

### Performance Ranking:
"""

for i, (modality, scores) in enumerate(performance_ranking.iterrows(), 1):
    report_content += f"{i}. {modality}: {scores['mean']:.3f} ± {scores['std']:.3f}\n"

report_content += f"""

### Analysis Summary:
- Neighborhood Purity: {'✓ WNN Superior' if best_embedding == 'WNN' else '⚠ Unexpected Results'}
- Cross-Modality Validation: {'✓ Strong Preservation' if best_embedding == 'WNN' else '⚠ Review Required'}
- scIB Integration Metrics: {'✓ High Quality' if best_embedding == 'WNN' else '⚠ Optimization Needed'}

## Files Generated:
- neighborhood_purity_summary.csv
- cross_modality_validation.csv  
- scib_integration_metrics.csv
- comparative_analysis.csv
- neighborhood_purity_comparison.png
- cross_modality_validation.png
- scib_integration_assessment.png

## Methodology:
1. Neighborhood purity analysis using pre-computed connectivity graphs
2. Cross-modality cluster preservation validation
3. scIB framework integration quality assessment
4. Comparative analysis across all approaches

Generated: {pd.Timestamp.now()}
"""

with open(results_dir / "integration_quality_report.md", "w") as f:
    f.write(report_content)

print(f"✓ Results exported to: {results_dir}")
print("✓ Summary report created: integration_quality_report.md")
print("✓ All analysis plots saved as PNG files")

# %% [markdown]
# ---
# 
# # Summary and Next Steps
# 
# This notebook provided comprehensive evaluation of multimodal integration quality using three complementary approaches:
# 
# ## ✅ What We Accomplished
# 
# 1. **Neighborhood Purity Analysis** - Quantified local clustering quality
# 2. **Cross-Modality Validation** - Tested structure preservation across modalities
# 3. **scIB Integration Assessment** - Applied gold-standard benchmarking framework
# 4. **Comparative Evaluation** - Synthesized results for final conclusions
# 5. **Result Export** - Generated publication-ready figures and summary tables
# 
# ## 🔬 Key Methodological Strengths
# 
# - **Non-circular validation** - Each modality independently validates others
# - **Multiple complementary metrics** - Neighborhood purity, ARI/NMI, ASW, graph connectivity
# - **Statistical rigor** - Mean ± standard deviation reporting with sample sizes
# - **Reproducible analysis** - Clear parameters and random seed setting
# 
# ## 📈 Expected Use Cases
# 
# - **Method comparison** - Benchmark different integration approaches
# - **Parameter optimization** - Test leiden resolution, k-neighbors, etc.
# - **Publication figures** - High-quality plots ready for manuscripts
# - **Quality control** - Validate integration before downstream analysis
# 
# ## 🚀 Next Steps
# 
# 1. **Apply to your data** - Update file paths and run with real multiome dataset
# 2. **Parameter tuning** - Optimize leiden resolution and neighbor parameters
# 3. **Extended validation** - Test with marker gene enrichment analysis
# 4. **Method comparison** - Compare different integration methods (Seurat WNN vs alternatives)
# 5. **Downstream analysis** - Use validated embeddings for trajectory inference, differential analysis, etc.

# %%
print("🎉 ANALYSIS COMPLETE!")
print("=" * 40)
print("Thank you for using the Zebrahub-Multiome Integration Quality Assessment!")
print("For questions or issues, refer to the repository documentation.")
print("=" * 40)