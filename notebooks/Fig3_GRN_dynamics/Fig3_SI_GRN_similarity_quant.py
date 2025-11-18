# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: celloracle_env
#     language: python
#     name: celloracle_env
# ---

# %% [markdown]
# ## Fig3_SI_GRN similarity quantification
#
# - last updated: 9/11/2025
# - 

# %%
# 0. Import
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

import scipy.sparse as sp
from itertools import combinations

# %%
import celloracle as co
co.__version__

# %%
# visualization settings
# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams["savefig.dpi"] = 600

# %%
import logging

# Set the logging level to WARN, filtering out informational messages
logging.getLogger().setLevel(logging.WARNING)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault) #Reset rcParams to default
# Set the default font to Arial
mpl.rcParams['font.family'] = 'Avenir'

# Editable text and proper LaTeX fonts in illustrator
# matplotlib.rcParams['ps.useafm'] = True
# Editable fonts. 42 is the magic number for editable text in PDFs
mpl.rcParams['pdf.fonttype'] = 42
sns.set(style='whitegrid', context='paper')

# %%
# define the figure path
figpath = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/figures/viz_GRN_sim_quant/"
os.makedirs(figpath, exist_ok=True)

# %% [markdown]
# ## Step 0. Import the GRNs (Links object)

# %%
# define the master directory for all Links objects (GRN objects from CellOracle)
oracle_base_dir = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/processed_data/04_celloracle_celltype_GRNs_ML_annotation/"

# We're using "TDR118" as the representative for "15-somites", and drop the "TDR119" for now.
# We'll use the "TDR119" for benchmark/comparison of biological replicates later on.
list_files = ['TDR126', 'TDR127', 'TDR128',
              'TDR118', 'TDR125', 'TDR124']

os.listdir(oracle_base_dir)

# %%
# define an empty dictionary
dict_links = {}

# for loop to import all Links objects
for dataset in list_files:
    file_name = f"{dataset}/08_{dataset}_celltype_GRNs.celloracle.links"
    file_path = os.path.join(oracle_base_dir, file_name)
    dict_links[dataset] = co.load_hdf5(file_path)
    
    print("importing ", dataset)
    
dict_links

# %% [markdown]
# ## Step 1. Further filtering of weak edges within the GRNs
# - By default, we keep 2000 edges for each GRN [celltype, time]. 
# - We'd like to filter out the weak edges by (1) edge strength, and (2) p-values

# %%
# define a new dict to save the "pruned" links
n_edges = 2000

# define an empty dict
dict_links_pruned = {}

for dataset in dict_links.keys():
    # filter for n_edges
    links = dict_links[dataset]
    links.filter_links(thread_number=n_edges)
    dict_links_pruned[dataset] = links
    
dict_links_pruned

# %%
# import the filtered_links from each GRN, and save them into another dictionary
dict_filtered_GRNs = {}

# for loop to import all filtered_GRN dataframes
for dataset in dict_links_pruned.keys():
    # extract the filtered links
    dict_filtered_GRNs[dataset] = dict_links_pruned[dataset].filtered_links
    
    print("importing filtered GRNs", dataset)
    
# dict_filtered_GRNs

# %%
# import the merged_scores from each GRN, and save them into another dictionary
dict_merged_score = {}

# for loop to import all filtered_GRN dataframes
for dataset in dict_links_pruned.keys():
    # extract the filtered links
    dict_merged_score[dataset] = dict_links_pruned[dataset].merged_score
    
    print("importing ", dataset)
    
# dict_merged_score

# %% [markdown]
# ### NOTES:
#
# - For any testing for n_edges, we'll have to re-run the above 3-cells.
#

# %% [markdown]
# ### NOTES: we have imported three dictionaries (nested with the dataset as the primary key, and the celltype as the secondary key).
#
# - dict_links (all CellOracle objects, called Links)
# - dict_filtered_GRNs (all filterd GRNs, 2000 edges per celltype, for all timepoints)
# - dict_merged_score (all network topology metrics from the filtered GRN above)
#

# %%
dict_filtered_GRNs["TDR118"].keys()

# %%
dict_filtered_GRNs["TDR118"]

# %%
dict_filtered_GRNs["TDR118"]["NMPs"]

# %%
dict_merged_score["TDR118"].head()

# %%
dict_merged_score["TDR118"][dict_merged_score["TDR118"].cluster=="NMPs"].sort_values("degree_all", ascending=False)

# %%
dict_merged_score["TDR118"][dict_merged_score["TDR118"].cluster=="primordial_germ_cells"].sort_values("degree_centrality_all", ascending=False)

# %%
dict_merged_score["TDR126"][dict_merged_score["TDR126"].cluster=="primordial_germ_cells"].sort_values("degree_centrality_all", ascending=False)

# %% [markdown]
# ## compute the GRN similarity metrics

# %%
# # import the unfiltered_links from each GRN, and save them into another dictionary
# dict_unfiltered_GRNs = {}

# # for loop to import all filtered_GRN dataframes
# for dataset in dict_links.keys():
#     # extract the filtered links
#     dict_unfiltered_GRNs[dataset] = dict_links[dataset].links_dict
    
#     print("importing unfiltered GRNs", dataset)
    
# # dict_filtered_GRNs

# %%

# %%
# import the module to compute the overlap between the two GRNs
from module_grn_overlap_quant import *


# %%
# check the overlap between the GRNs (TF:gene pairs) - "filtered GRNs"
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from collections import defaultdict, Counter

# Extract basic info from your data
timepoints = list(dict_filtered_GRNs.keys())
celltypes = list(dict_filtered_GRNs[timepoints[0]].keys())

print(f"Found {len(timepoints)} timepoints: {timepoints}")
print(f"Found {len(celltypes)} celltypes")

# ============================================================================
# 1. COMPUTE PRESENCE FRACTIONS FOR ALL TF-GENE PAIRS
# ============================================================================

def extract_tf_gene_pairs(grn_matrix):
    """Extract TF-gene pairs from a GRN matrix"""
    if grn_matrix is None or grn_matrix.empty:
        return set()
    
    # Stack the matrix to get (target, TF) pairs with non-zero values
    grn_stacked = grn_matrix.stack()
    pairs = set(f"{target}_{tf}" for (target, tf), val in grn_stacked.items() if val != 0)
    return pairs

# Count TF-gene pair occurrences across different scenarios
print("\n" + "="*60)
print("COMPUTING PRESENCE FRACTIONS...")
print("="*60)

# Scenario 1: Across all celltypes at each timepoint (for timepoint analysis)
timepoint_presence = {}
for tp in timepoints:
    print(f"\nAnalyzing timepoint: {tp}")
    
    # Get all valid GRNs for this timepoint
    tp_grns = []
    valid_celltypes = []
    
    for ct in celltypes:
        grn = dict_filtered_GRNs[tp].get(ct)
        if grn is not None and not grn.empty:
            tp_grns.append(grn)
            valid_celltypes.append(ct)
    
    print(f"  Valid celltypes: {len(valid_celltypes)}/{len(celltypes)}")
    
    if len(tp_grns) < 2:
        print(f"  Skipping {tp} - insufficient data")
        continue
    
    # Count pair occurrences
    pair_counts = Counter()
    total_grns = len(tp_grns)
    
    for grn in tp_grns:
        pairs = extract_tf_gene_pairs(grn)
        for pair in pairs:
            pair_counts[pair] += 1
    
    # Calculate presence fractions
    presence_fractions = {pair: count/total_grns for pair, count in pair_counts.items()}
    timepoint_presence[tp] = {
        'presence_fractions': presence_fractions,
        'total_pairs': len(pair_counts),
        'total_grns': total_grns,
        'valid_celltypes': valid_celltypes
    }
    
    print(f"  Total unique pairs: {len(pair_counts)}")
    print(f"  Mean presence fraction: {np.mean(list(presence_fractions.values())):.3f}")

# Scenario 2: Across all timepoints for each celltype (for celltype analysis)
celltype_presence = {}
for ct in celltypes:
    print(f"\nAnalyzing celltype: {ct}")
    
    # Get all valid GRNs for this celltype
    ct_grns = []
    valid_timepoints = []
    
    for tp in timepoints:
        grn = dict_filtered_GRNs[tp].get(ct)
        if grn is not None and not grn.empty:
            ct_grns.append(grn)
            valid_timepoints.append(tp)
    
    print(f"  Valid timepoints: {len(valid_timepoints)}/{len(timepoints)}")
    
    if len(ct_grns) < 2:
        print(f"  Skipping {ct} - insufficient data")
        continue
    
    # Count pair occurrences
    pair_counts = Counter()
    total_grns = len(ct_grns)
    
    for grn in ct_grns:
        pairs = extract_tf_gene_pairs(grn)
        for pair in pairs:
            pair_counts[pair] += 1
    
    # Calculate presence fractions
    presence_fractions = {pair: count/total_grns for pair, count in pair_counts.items()}
    celltype_presence[ct] = {
        'presence_fractions': presence_fractions,
        'total_pairs': len(pair_counts),
        'total_grns': total_grns,
        'valid_timepoints': valid_timepoints
    }
    
    print(f"  Total unique pairs: {len(pair_counts)}")
    print(f"  Mean presence fraction: {np.mean(list(presence_fractions.values())):.3f}")

# ============================================================================
# 2. ANALYZE PRESENCE FRACTION DISTRIBUTIONS
# ============================================================================

print("\n" + "="*60)
print("ANALYZING PRESENCE FRACTION DISTRIBUTIONS...")
print("="*60)

# Collect all presence fractions for analysis
all_tp_fractions = []
tp_labels = []
for tp, data in timepoint_presence.items():
    fractions = list(data['presence_fractions'].values())
    all_tp_fractions.extend(fractions)
    tp_labels.extend([tp] * len(fractions))

all_ct_fractions = []
ct_labels = []
for ct, data in celltype_presence.items():
    fractions = list(data['presence_fractions'].values())
    all_ct_fractions.extend(fractions)
    ct_labels.extend([ct] * len(fractions))

# Create summary statistics
tp_summary = []
for tp, data in timepoint_presence.items():
    fractions = list(data['presence_fractions'].values())
    if fractions:
        tp_summary.append({
            'timepoint': tp,
            'mean_presence': np.mean(fractions),
            'median_presence': np.median(fractions),
            'std_presence': np.std(fractions),
            'total_pairs': len(fractions),
            'pairs_50pct': sum(1 for f in fractions if f >= 0.5),
            'pairs_80pct': sum(1 for f in fractions if f >= 0.8),
            'pairs_90pct': sum(1 for f in fractions if f >= 0.9)
        })

ct_summary = []
for ct, data in celltype_presence.items():
    fractions = list(data['presence_fractions'].values())
    if fractions:
        ct_summary.append({
            'celltype': ct,
            'mean_presence': np.mean(fractions),
            'median_presence': np.median(fractions),
            'std_presence': np.std(fractions),
            'total_pairs': len(fractions),
            'pairs_50pct': sum(1 for f in fractions if f >= 0.5),
            'pairs_80pct': sum(1 for f in fractions if f >= 0.8),
            'pairs_90pct': sum(1 for f in fractions if f >= 0.9)
        })

tp_summary_df = pd.DataFrame(tp_summary)
ct_summary_df = pd.DataFrame(ct_summary)

print("\nTIMEPOINT ANALYSIS SUMMARY:")
print(tp_summary_df.round(3))

print("\nCELLTYPE ANALYSIS SUMMARY:")
print(ct_summary_df.round(3))

# ============================================================================
# 3. CREATE COMPREHENSIVE VISUALIZATIONS
# ============================================================================

print("\n" + "="*60)
print("CREATING VISUALIZATIONS...")
print("="*60)

# Create subplot figure
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Presence Fraction Distribution (Timepoint Analysis)',
        'Presence Fraction Distribution (Celltype Analysis)',
        'Data Retention at Different Thresholds (Timepoints)',
        'Data Retention at Different Thresholds (Celltypes)',
        'Mean Presence Fraction by Timepoint',
        'Mean Presence Fraction by Celltype'
    ),
    specs=[[{"type": "histogram"}, {"type": "histogram"}],
           [{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "bar"}, {"type": "bar"}]]
)

# Plot 1: Histogram of presence fractions for timepoint analysis
fig.add_trace(
    go.Histogram(
        x=all_tp_fractions,
        nbinsx=20,
        name='Timepoint Analysis',
        opacity=0.7,
        marker_color='blue'
    ),
    row=1, col=1
)

# Plot 2: Histogram of presence fractions for celltype analysis
fig.add_trace(
    go.Histogram(
        x=all_ct_fractions,
        nbinsx=20,
        name='Celltype Analysis',
        opacity=0.7,
        marker_color='red'
    ),
    row=1, col=2
)

# Plot 3: Data retention at different thresholds (timepoints)
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
tp_retention = []
for threshold in thresholds:
    retained = sum(1 for f in all_tp_fractions if f >= threshold)
    tp_retention.append(retained / len(all_tp_fractions) * 100)

fig.add_trace(
    go.Scatter(
        x=thresholds,
        y=tp_retention,
        mode='lines+markers',
        name='Timepoint Retention',
        marker=dict(size=10, color='blue'),
        line=dict(width=3)
    ),
    row=2, col=1
)

# Plot 4: Data retention at different thresholds (celltypes)
ct_retention = []
for threshold in thresholds:
    retained = sum(1 for f in all_ct_fractions if f >= threshold)
    ct_retention.append(retained / len(all_ct_fractions) * 100)

fig.add_trace(
    go.Scatter(
        x=thresholds,
        y=ct_retention,
        mode='lines+markers',
        name='Celltype Retention',
        marker=dict(size=10, color='red'),
        line=dict(width=3)
    ),
    row=2, col=2
)

# Plot 5: Mean presence fraction by timepoint
fig.add_trace(
    go.Bar(
        x=tp_summary_df['timepoint'],
        y=tp_summary_df['mean_presence'],
        name='Timepoint Means',
        marker_color='lightblue',
        error_y=dict(type='data', array=tp_summary_df['std_presence'])
    ),
    row=3, col=1
)

# Plot 6: Mean presence fraction by celltype
fig.add_trace(
    go.Bar(
        x=list(range(len(ct_summary_df))),
        y=ct_summary_df['mean_presence'],
        name='Celltype Means',
        marker_color='lightcoral',
        error_y=dict(type='data', array=ct_summary_df['std_presence']),
        text=ct_summary_df['celltype'],
        hovertemplate='%{text}<br>Mean: %{y:.3f}<extra></extra>'
    ),
    row=3, col=2
)

# Update layout
fig.update_xaxes(title_text="Presence Fraction", row=1, col=1)
fig.update_xaxes(title_text="Presence Fraction", row=1, col=2)
fig.update_xaxes(title_text="Min Presence Threshold", row=2, col=1)
fig.update_xaxes(title_text="Min Presence Threshold", row=2, col=2)
fig.update_xaxes(title_text="Timepoints", row=3, col=1, tickangle=45)
fig.update_xaxes(title_text="Celltypes", row=3, col=2, tickangle=45)

fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=2)
fig.update_yaxes(title_text="% Pairs Retained", row=2, col=1)
fig.update_yaxes(title_text="% Pairs Retained", row=2, col=2)
fig.update_yaxes(title_text="Mean Presence Fraction", row=3, col=1)
fig.update_yaxes(title_text="Mean Presence Fraction", row=3, col=2)

fig.update_layout(
    height=1200,
    width=1400,
    title_text="TF-Gene Pair Presence Fraction Analysis",
    showlegend=False
)

fig.show()

# ============================================================================
# 4. GENERATE RECOMMENDATIONS
# ============================================================================

print("\n" + "="*60)
print("RECOMMENDATIONS BASED ON YOUR DATA:")
print("="*60)

# Calculate overall statistics
overall_tp_mean = np.mean(all_tp_fractions)
overall_ct_mean = np.mean(all_ct_fractions)

print(f"\nOverall Statistics:")
print(f"  Timepoint analysis - Mean presence fraction: {overall_tp_mean:.3f}")
print(f"  Celltype analysis - Mean presence fraction: {overall_ct_mean:.3f}")

# Recommend thresholds based on data distribution
def recommend_threshold(fractions, analysis_type):
    percentiles = [50, 70, 80, 90]
    threshold_retention = []
    
    print(f"\n{analysis_type} Analysis - Data retention at different thresholds:")
    for pct in [50, 60, 70, 80, 90]:
        threshold = pct / 100
        retained = sum(1 for f in fractions if f >= threshold)
        retention_pct = retained / len(fractions) * 100
        threshold_retention.append(retention_pct)
        print(f"  {threshold:.1f} threshold: {retention_pct:.1f}% of pairs retained ({retained:,} pairs)")
    
    # Recommend based on retention
    if threshold_retention[3] >= 20:  # 80% threshold retains ≥20% of data
        return 0.8, "Conservative (recommended)"
    elif threshold_retention[2] >= 30:  # 70% threshold retains ≥30% of data  
        return 0.7, "Moderate (recommended)"
    else:
        return 0.6, "Liberal (recommended)"

tp_rec_threshold, tp_rec_reason = recommend_threshold(all_tp_fractions, "Timepoint")
ct_rec_threshold, ct_rec_reason = recommend_threshold(all_ct_fractions, "Celltype")

print(f"\nRECOMMENDED THRESHOLDS:")
print(f"  Timepoint analysis: {tp_rec_threshold} - {tp_rec_reason}")
print(f"  Celltype analysis: {ct_rec_threshold} - {ct_rec_reason}")

# Check for potential issues
print(f"\nPOTENTIAL ISSUES TO WATCH:")
if overall_tp_mean < 0.3:
    print("  ⚠️  Low overall presence fractions in timepoint analysis - data may be very sparse")
if overall_ct_mean < 0.3:
    print("  ⚠️  Low overall presence fractions in celltype analysis - data may be very sparse")
if len(all_tp_fractions) < 100:
    print("  ⚠️  Very few TF-gene pairs in timepoint analysis - limited statistical power")
if len(all_ct_fractions) < 100:
    print("  ⚠️  Very few TF-gene pairs in celltype analysis - limited statistical power")

print(f"\nSUGGESTED NEXT STEPS:")
print(f"1. Use the recommended thresholds above for your main analysis")
print(f"2. Run sensitivity analysis with ±0.1 threshold variation")
print(f"3. Report the number of pairs retained at your chosen threshold")
print(f"4. Consider biological relevance of excluded pairs (are they important regulators?)")

# Save summary to CSV for reference
tp_summary_df.to_csv('timepoint_presence_summary.csv', index=False)
ct_summary_df.to_csv('celltype_presence_summary.csv', index=False)
print(f"\nSummary tables saved to CSV files for reference.")

# %%
# Jupyter Notebook Usage Example

# Import the module
import module_grn_overlap_quant as grn_overlap

# ============================================================================
# OPTION 1: Complete workflow (easiest)
# ============================================================================

# Run complete analysis with all default options
results = grn_overlap.complete_grn_overlap_analysis(
    dict_unfiltered_GRNs,  # or dict_filtered_GRNs
    save_prefix="unfiltered_grn",
    save_plot_path="unfiltered_grn_overlap_analysis.html",
    show_plot=True,
    save_csv=True,
    verbose=True
)

# Access results
print("Recommended thresholds:")
print(f"Timepoint analysis: {results['recommendations']['tp_rec_threshold']}")
print(f"Celltype analysis: {results['recommendations']['ct_rec_threshold']}")

# View summary dataframes
display(results['tp_summary_df'])
display(results['ct_summary_df'])

# ============================================================================
# OPTION 2: Step-by-step analysis (more control)
# ============================================================================

# Step 1: Run basic analysis
analysis_results = grn_overlap.analyze_grn_overlap(dict_unfiltered_GRNs, verbose=True)

# Step 2: Get threshold recommendations
recommendations = grn_overlap.recommend_thresholds(
    analysis_results['all_tp_fractions'], 
    analysis_results['all_ct_fractions'], 
    verbose=True
)

# Step 3: Create custom visualization
fig = grn_overlap.plot_grn_overlap_analysis(
    analysis_results, 
    save_path="custom_grn_analysis.html",
    show_plot=True
)

# Step 4: Save results to CSV
file_paths = grn_overlap.save_results_to_csv(analysis_results, prefix="custom_grn")

# ============================================================================
# OPTION 3: Individual function usage (maximum control)
# ============================================================================

# Just compute timepoint presence fractions
tp_presence = grn_overlap.compute_timepoint_presence_fractions(dict_unfiltered_GRNs)

# Just compute celltype presence fractions  
ct_presence = grn_overlap.compute_celltype_presence_fractions(dict_unfiltered_GRNs)

# Create summary statistics
tp_summary = grn_overlap.create_summary_statistics(tp_presence, 'timepoint')
ct_summary = grn_overlap.create_summary_statistics(ct_presence, 'celltype')

# Display specific results
print("Timepoint Analysis Summary:")
display(tp_summary)

# ============================================================================
# COMPARE FILTERED vs UNFILTERED
# ============================================================================

# Analyze both datasets
unfiltered_results = grn_overlap.analyze_grn_overlap(dict_unfiltered_GRNs, verbose=False)
filtered_results = grn_overlap.analyze_grn_overlap(dict_filtered_GRNs, verbose=False)

# Compare mean presence fractions
print("Comparison of filtered vs unfiltered GRNs:")
print(f"Unfiltered - TP mean: {np.mean(unfiltered_results['all_tp_fractions']):.3f}")
print(f"Filtered - TP mean: {np.mean(filtered_results['all_tp_fractions']):.3f}")
print(f"Unfiltered - CT mean: {np.mean(unfiltered_results['all_ct_fractions']):.3f}")
print(f"Filtered - CT mean: {np.mean(filtered_results['all_ct_fractions']):.3f}")

# ============================================================================
# EXTRACT SPECIFIC INFORMATION
# ============================================================================

# Get specific recommendations for your similarity analysis
tp_threshold = results['recommendations']['tp_rec_threshold']
ct_threshold = results['recommendations']['ct_rec_threshold']

print(f"Use threshold {tp_threshold} for timepoint similarity analysis")
print(f"Use threshold {ct_threshold} for celltype similarity analysis")

# Check data sparsity
overall_tp_sparsity = 1 - results['recommendations']['overall_tp_mean'] 
overall_ct_sparsity = 1 - results['recommendations']['overall_ct_mean']

print(f"Overall data sparsity:")
print(f"  Timepoint analysis: {overall_tp_sparsity:.1%} of pairs are context-specific")
print(f"  Celltype analysis: {overall_ct_sparsity:.1%} of pairs are context-specific")

if overall_tp_sparsity > 0.8:
    print("💡 Recommendation: Use superset approach for similarity analysis!")
else:
    print("💡 Recommendation: Standard intersection approach should work well.")

# %%

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jaccard
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class GRNSimilarityAnalyzer:
    def __init__(self, dict_filtered_GRNs):
        """
        Initialize with the GRN dictionary
        dict_filtered_GRNs: {timepoint: {celltype: GRN_matrix}}
        """
        self.grns = dict_filtered_GRNs
        self.timepoints = list(dict_filtered_GRNs.keys())
        self.celltypes = list(dict_filtered_GRNs[self.timepoints[0]].keys())
        print(f"Found {len(self.timepoints)} timepoints: {self.timepoints}")
        print(f"Found {len(self.celltypes)} celltypes")
        
    def extract_tf_gene_pairs(self, grn_matrix):
        """Extract TF-gene pairs from a GRN matrix (handles both wide and long formats)"""
        if grn_matrix is None or grn_matrix.empty:
            return set()
        
        pairs = set()
        
        # Check if this is long format (has 'source', 'target' columns)
        if 'source' in grn_matrix.columns and 'target' in grn_matrix.columns:
            # Long format: source, target, coef_mean (or similar)
            coef_col = 'coef_mean' if 'coef_mean' in grn_matrix.columns else grn_matrix.columns[-1]
            
            for _, row in grn_matrix.iterrows():
                source = row['source']  # This is the TF
                target = row['target']  # This is the target gene
                val = row[coef_col]
                
                # Check if value is non-zero
                try:
                    if pd.notna(val) and float(val) != 0:
                        pairs.add(f"{target}_{source}")  # target_TF format
                except (ValueError, TypeError):
                    continue
                    
        else:
            # Wide format: matrix with targets as rows, TFs as columns
            grn_stacked = grn_matrix.stack()
            for (target, tf), val in grn_stacked.items():
                try:
                    if pd.notna(val) and float(val) != 0:
                        pairs.add(f"{target}_{tf}")
                except (ValueError, TypeError):
                    continue
        
        return pairs
    
    def get_superset_pairs(self, grn_dict):
        """
        Get union (superset) of all TF-gene pairs from a dictionary of GRNs
        grn_dict: {identifier: GRN_matrix}
        """
        all_pairs = set()
        valid_grns = 0
        
        for identifier, grn in grn_dict.items():
            if grn is not None and not grn.empty:
                pairs = self.extract_tf_gene_pairs(grn)
                all_pairs.update(pairs)
                valid_grns += 1
        
        print(f"  Superset contains {len(all_pairs):,} unique TF-gene pairs from {valid_grns} valid GRNs")
        return sorted(list(all_pairs))  # Sort for consistent ordering
    
    def grn_to_vector_superset(self, grn, superset_pairs):
        """
        Convert GRN matrix to feature vector using superset of pairs
        Missing pairs are filled with 0 (no regulation)
        Handles both wide and long format GRN matrices
        """
        if grn is None or grn.empty:
            return np.zeros(len(superset_pairs))
        
        feature_dict = {}
        
        # Check if this is long format (has 'source', 'target' columns)
        if 'source' in grn.columns and 'target' in grn.columns:
            # Long format: source, target, coef_mean (or similar)
            coef_col = 'coef_mean' if 'coef_mean' in grn.columns else grn.columns[-1]
            
            for _, row in grn.iterrows():
                source = row['source']  # This is the TF
                target = row['target']  # This is the target gene
                val = row[coef_col]
                
                # Convert to numeric
                try:
                    val = float(val) if pd.notna(val) else 0.0
                except (ValueError, TypeError):
                    val = 0.0
                
                feature_dict[f"{target}_{source}"] = val
                
        else:
            # Wide format: matrix with targets as rows, TFs as columns
            grn_stacked = grn.stack()
            for (target, tf), val in grn_stacked.items():
                # Convert to numeric
                try:
                    val = float(val) if pd.notna(val) else 0.0
                except (ValueError, TypeError):
                    val = 0.0
                
                feature_dict[f"{target}_{tf}"] = val
        
        # Create vector for superset: existing pairs get their values, missing pairs get 0
        vector = np.array([feature_dict.get(pair, 0.0) for pair in superset_pairs], dtype=np.float64)
        return vector
    
    def compute_similarity_matrix(self, vectors_dict, metric='pearson'):
        """
        Compute similarity matrix from feature vectors dictionary
        vectors_dict: {identifier: vector}
        """
        identifiers = list(vectors_dict.keys())
        vectors = list(vectors_dict.values())
        n = len(vectors)
        
        similarity_matrix = np.full((n, n), np.nan)
        
        for i in range(n):
            for j in range(i, n):
                if np.all(np.isnan(vectors[i])) or np.all(np.isnan(vectors[j])):
                    continue
                    
                if metric == 'pearson':
                    # Handle constant vectors
                    if np.std(vectors[i]) == 0 or np.std(vectors[j]) == 0:
                        corr = 1.0 if np.allclose(vectors[i], vectors[j]) else 0.0
                    else:
                        corr, _ = pearsonr(vectors[i], vectors[j])
                        if np.isnan(corr):
                            corr = 0.0
                            
                elif metric == 'cosine':
                    corr = cosine_similarity([vectors[i]], [vectors[j]])[0, 0]
                    
                elif metric == 'spearman':
                    # Handle constant vectors
                    if np.std(vectors[i]) == 0 or np.std(vectors[j]) == 0:
                        corr = 1.0 if np.allclose(vectors[i], vectors[j]) else 0.0
                    else:
                        corr, _ = spearmanr(vectors[i], vectors[j])
                        if np.isnan(corr):
                            corr = 0.0
                
                elif metric == 'jaccard':
                    # Jaccard on binary vectors (non-zero elements)
                    binary_i = (vectors[i] != 0).astype(int)
                    binary_j = (vectors[j] != 0).astype(int)
                    
                    intersection = np.sum(binary_i & binary_j)
                    union = np.sum(binary_i | binary_j)
                    
                    corr = intersection / union if union > 0 else 0.0
                
                similarity_matrix[i, j] = similarity_matrix[j, i] = corr
        
        return similarity_matrix, identifiers
    
    def compute_data_characteristics(self, vectors_dict, context=""):
        """Compute and report data characteristics"""
        vectors = list(vectors_dict.values())
        identifiers = list(vectors_dict.keys())
        
        if not vectors:
            return {}
        
        # Ensure all vectors are numeric
        numeric_vectors = []
        for v in vectors:
            if v.dtype.kind not in 'biufc':  # not a numeric type
                print(f"⚠️ Warning: Non-numeric vector detected, converting to float64")
                v = np.array([float(x) if x != '' else 0.0 for x in v], dtype=np.float64)
            numeric_vectors.append(v)
        
        vectors = numeric_vectors
        
        # Basic statistics
        total_pairs = len(vectors[0]) if vectors else 0
        sparsity_per_grn = [np.mean(v == 0) for v in vectors]
        nonzero_per_grn = [np.sum(v != 0) for v in vectors]
        
        # Compute mean values only for non-zero elements
        mean_values = []
        for v in vectors:
            nonzero_vals = v[v != 0]
            if len(nonzero_vals) > 0:
                mean_values.append(np.mean(nonzero_vals))
            else:
                mean_values.append(0.0)
        
        stats = {
            'context': context,
            'total_pairs': total_pairs,
            'n_grns': len(vectors),
            'mean_sparsity': np.mean(sparsity_per_grn),
            'std_sparsity': np.std(sparsity_per_grn),
            'mean_nonzero_pairs': np.mean(nonzero_per_grn),
            'std_nonzero_pairs': np.std(nonzero_per_grn),
            'mean_interaction_strength': np.mean(mean_values),
            'sparsity_per_grn': dict(zip(identifiers, sparsity_per_grn)),
            'nonzero_per_grn': dict(zip(identifiers, nonzero_per_grn))
        }
        
        # Print summary
        print(f"\n📊 Data characteristics for {context}:")
        print(f"  Total TF-gene pairs in superset: {total_pairs:,}")
        print(f"  Number of GRNs: {len(vectors)}")
        print(f"  Mean sparsity (% zeros): {stats['mean_sparsity']:.3f} ± {stats['std_sparsity']:.3f}")
        print(f"  Mean active pairs per GRN: {stats['mean_nonzero_pairs']:,.0f} ± {stats['std_nonzero_pairs']:,.0f}")
        print(f"  Mean interaction strength: {stats['mean_interaction_strength']:.3f}")
        
        # Warnings
        if stats['mean_sparsity'] > 0.95:
            print("  ⚠️  Very sparse data (>95% zeros) - consider Jaccard similarity")
        if stats['mean_nonzero_pairs'] < 100:
            print("  ⚠️  Very few active pairs per GRN - limited statistical power")
            
        return stats
    
    def analyze_timepoint_similarity(self, metrics=['pearson', 'cosine', 'spearman', 'jaccard']):
        """
        Analyze how similar celltypes are within each timepoint using superset approach
        """
        results = {
            'timepoints': [],
            'data_stats': {}
        }
        
        # Initialize results for each metric
        for metric in metrics:
            results[f'{metric}_mean'] = []
            results[f'{metric}_std'] = []
        
        for tp in self.timepoints:
            print(f"\n🔍 Analyzing timepoint: {tp}")
            
            # Get all valid GRNs for this timepoint
            tp_grns = {}
            for ct in self.celltypes:
                grn = self.grns[tp].get(ct)
                if grn is not None and not grn.empty:
                    tp_grns[ct] = grn
            
            if len(tp_grns) < 2:
                print(f"  ❌ Skipping {tp} - insufficient data ({len(tp_grns)} valid GRNs)")
                continue
            
            print(f"  Valid celltypes: {len(tp_grns)}/{len(self.celltypes)}")
            
            # Get superset of all TF-gene pairs for this timepoint
            superset_pairs = self.get_superset_pairs(tp_grns)
            
            # Convert each GRN to vector using superset (missing = 0)
            vectors_dict = {}
            for ct, grn in tp_grns.items():
                vectors_dict[ct] = self.grn_to_vector_superset(grn, superset_pairs)
            
            # Compute data characteristics
            data_stats = self.compute_data_characteristics(vectors_dict, f"timepoint {tp}")
            results['data_stats'][tp] = data_stats
            
            # Compute similarities for each requested metric
            metric_results = {}
            for metric in metrics:
                sim_matrix, identifiers = self.compute_similarity_matrix(vectors_dict, metric)
                
                # Extract upper triangular values (unique pairs)
                mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
                sim_values = sim_matrix[mask]
                sim_values = sim_values[~np.isnan(sim_values)]
                
                if len(sim_values) > 0:
                    metric_results[metric] = {
                        'mean': np.mean(sim_values),
                        'std': np.std(sim_values),
                        'values': sim_values,
                        'n_comparisons': len(sim_values)
                    }
                    print(f"  {metric.capitalize()} similarity: {np.mean(sim_values):.3f} ± {np.std(sim_values):.3f} ({len(sim_values)} pairs)")
                else:
                    metric_results[metric] = {'mean': np.nan, 'std': np.nan, 'values': [], 'n_comparisons': 0}
                    print(f"  {metric.capitalize()} similarity: No valid comparisons")
            
            # Store results
            results['timepoints'].append(tp)
            for metric in metrics:
                results[f'{metric}_mean'].append(metric_results[metric]['mean'])
                results[f'{metric}_std'].append(metric_results[metric]['std'])
                # Calculate SEM (standard error of the mean)
                n_comparisons = metric_results[metric]['n_comparisons']
                sem = metric_results[metric]['std'] / np.sqrt(n_comparisons) if n_comparisons > 0 else 0
                results[f'{metric}_sem'] = results.get(f'{metric}_sem', [])
                results[f'{metric}_sem'].append(sem)
        
        return pd.DataFrame({k: v for k, v in results.items() if k != 'data_stats'}), results['data_stats']
    
    def analyze_celltype_similarity(self, metrics=['pearson', 'cosine', 'spearman', 'jaccard']):
        """
        Analyze how similar timepoints are within each celltype using superset approach
        """
        results = {
            'celltypes': [],
            'data_stats': {}
        }
        
        # Initialize results for each metric
        for metric in metrics:
            results[f'{metric}_mean'] = []
            results[f'{metric}_std'] = []
        
        for ct in self.celltypes:
            print(f"\n🔍 Analyzing celltype: {ct}")
            
            # Get all valid GRNs for this celltype across timepoints
            ct_grns = {}
            for tp in self.timepoints:
                grn = self.grns[tp].get(ct)
                if grn is not None and not grn.empty:
                    ct_grns[tp] = grn
            
            if len(ct_grns) < 2:
                print(f"  ❌ Skipping {ct} - insufficient data ({len(ct_grns)} valid GRNs)")
                continue
            
            print(f"  Valid timepoints: {len(ct_grns)}/{len(self.timepoints)}")
            
            # Get superset of all TF-gene pairs for this celltype
            superset_pairs = self.get_superset_pairs(ct_grns)
            
            # Convert each GRN to vector using superset (missing = 0)
            vectors_dict = {}
            for tp, grn in ct_grns.items():
                vectors_dict[tp] = self.grn_to_vector_superset(grn, superset_pairs)
            
            # Compute data characteristics
            data_stats = self.compute_data_characteristics(vectors_dict, f"celltype {ct}")
            results['data_stats'][ct] = data_stats
            
            # Compute similarities for each requested metric
            metric_results = {}
            for metric in metrics:
                sim_matrix, identifiers = self.compute_similarity_matrix(vectors_dict, metric)
                
                # Extract upper triangular values
                mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
                sim_values = sim_matrix[mask]
                sim_values = sim_values[~np.isnan(sim_values)]
                
                if len(sim_values) > 0:
                    metric_results[metric] = {
                        'mean': np.mean(sim_values),
                        'std': np.std(sim_values),
                        'values': sim_values,
                        'n_comparisons': len(sim_values)
                    }
                    print(f"  {metric.capitalize()} similarity: {np.mean(sim_values):.3f} ± {np.std(sim_values):.3f} ({len(sim_values)} pairs)")
                else:
                    metric_results[metric] = {'mean': np.nan, 'std': np.nan, 'values': [], 'n_comparisons': 0}
                    print(f"  {metric.capitalize()} similarity: No valid comparisons")
            
            # Store results
            results['celltypes'].append(ct)
            for metric in metrics:
                results[f'{metric}_mean'].append(metric_results[metric]['mean'])
                results[f'{metric}_std'].append(metric_results[metric]['std'])
                # Calculate SEM (standard error of the mean)
                n_comparisons = metric_results[metric]['n_comparisons']
                sem = metric_results[metric]['std'] / np.sqrt(n_comparisons) if n_comparisons > 0 else 0
                results[f'{metric}_sem'] = results.get(f'{metric}_sem', [])
                results[f'{metric}_sem'].append(sem)
        
        return pd.DataFrame({k: v for k, v in results.items() if k != 'data_stats'}), results['data_stats']
    
    def plot_similarity_trends(self, tp_results, ct_results, tp_stats=None, ct_stats=None, 
                              metrics=['pearson', 'cosine', 'spearman'], save_path=None, 
                              sort_celltypes=True, include_violins=True):
        """
        Create comprehensive similarity plots with multiple metrics
        
        Parameters:
        -----------
        sort_celltypes : bool
            Whether to sort celltypes by similarity magnitude (high to low)
        include_violins : bool
            Whether to include violin plots showing distributions
        """
        n_metrics = len(metrics)
        n_rows = 4 if include_violins else 3
        
        # Create subplot layout
        subplot_titles = []
        subplot_titles.extend([f'{m.capitalize()} - Celltype Similarity Within Timepoints' for m in metrics])
        subplot_titles.extend([f'{m.capitalize()} - Timepoint Similarity Within Celltypes' for m in metrics])
        subplot_titles.extend([f'{m.capitalize()} - Data Characteristics' for m in metrics])
        if include_violins:
            subplot_titles.extend([f'{m.capitalize()} - Similarity Distributions' for m in metrics])
        
        fig = make_subplots(
            rows=n_rows, cols=n_metrics,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": False} for _ in range(n_metrics)] for _ in range(n_rows)],
            vertical_spacing=0.08
        )
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        
        # Row 1: Timepoint analysis (celltype similarity over time)
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in tp_results.columns and f'{metric}_sem' in tp_results.columns:
                fig.add_trace(
                    go.Scatter(
                        x=tp_results['timepoints'],
                        y=tp_results[f'{metric}_mean'],
                        error_y=dict(type='data', array=tp_results[f'{metric}_sem'], visible=True),
                        mode='lines+markers',
                        name=f'{metric.capitalize()} (TP)',
                        line=dict(color=colors[i % len(colors)], width=3),
                        marker=dict(size=8)
                    ),
                    row=1, col=i+1
                )
        
        # Row 2: Celltype analysis (timepoint similarity over celltypes) - SORTED
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in ct_results.columns and f'{metric}_sem' in ct_results.columns:
                
                # Sort celltypes by similarity magnitude if requested
                if sort_celltypes:
                    # Sort by mean similarity (descending) 
                    ct_sorted = ct_results.copy().sort_values(f'{metric}_mean', ascending=False)
                    x_values = list(range(len(ct_sorted)))
                    x_labels = ct_sorted['celltypes'].tolist()
                    y_values = ct_sorted[f'{metric}_mean']
                    error_values = ct_sorted[f'{metric}_sem']
                else:
                    x_values = list(range(len(ct_results)))
                    x_labels = ct_results['celltypes'].tolist()
                    y_values = ct_results[f'{metric}_mean']
                    error_values = ct_results[f'{metric}_sem']
                
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        error_y=dict(type='data', array=error_values, visible=True),
                        mode='markers',
                        name=f'{metric.capitalize()} (CT)',
                        marker=dict(color=colors[i % len(colors)], size=8, opacity=0.7),
                        text=x_labels,
                        hovertemplate='%{text}<br>' + f'{metric.capitalize()}: %{{y:.3f}} ± %{{error_y.array:.4f}}<extra></extra>',
                        showlegend=False
                    ),
                    row=2, col=i+1
                )
                
                # Update x-axis to show celltype names
                fig.update_xaxes(
                    ticktext=x_labels,
                    tickvals=x_values,
                    tickangle=45,
                    row=2, col=i+1
                )
        
        # Row 3: Data characteristics (if provided)
        if tp_stats:
            for i, metric in enumerate(metrics):
                # Plot sparsity over timepoints
                sparsity_data = [tp_stats[tp]['mean_sparsity'] for tp in tp_results['timepoints'] 
                               if tp in tp_stats]
                
                fig.add_trace(
                    go.Scatter(
                        x=tp_results['timepoints'],
                        y=sparsity_data,
                        mode='lines+markers',
                        name='Sparsity',
                        line=dict(color='gray', dash='dash'),
                        showlegend=i == 0
                    ),
                    row=3, col=i+1
                )
        
        # Row 4: Violin plots (if requested and data available)
        if include_violins and hasattr(self, '_last_detailed_results'):
            tp_raw_data = self._last_detailed_results['timepoint_analysis']['raw_similarities']
            
            if not tp_raw_data.empty:
                for i, metric in enumerate(metrics):
                    metric_data = tp_raw_data[tp_raw_data['metric'] == metric]
                    
                    if not metric_data.empty:
                        # Create violin plot data for this metric
                        timepoint_order = ['TDR126', 'TDR127', 'TDR128', 'TDR118', 'TDR125', 'TDR124']
                        hpf_labels = {'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
                                     'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'}
                        
                        for j, tp in enumerate(timepoint_order):
                            if tp in metric_data['timepoint'].values:
                                tp_metric_data = metric_data[metric_data['timepoint'] == tp]
                                
                                fig.add_trace(
                                    go.Violin(
                                        y=tp_metric_data['similarity'],
                                        x=[hpf_labels[tp]] * len(tp_metric_data),
                                        name=hpf_labels[tp],
                                        box_visible=True,
                                        meanline_visible=True,
                                        fillcolor=colors[j % len(colors)],
                                        opacity=0.5,
                                        showlegend=False,
                                        side='positive' if j % 2 == 0 else 'negative'
                                    ),
                                    row=4, col=i+1
                                )
        
        # Update layout
        for i in range(n_metrics):
            fig.update_xaxes(title_text="Timepoints", tickangle=45, row=1, col=i+1)
            fig.update_xaxes(title_text="Celltypes (Sorted by Similarity)", tickangle=45, row=2, col=i+1)
            fig.update_xaxes(title_text="Timepoints", tickangle=45, row=3, col=i+1)
            if include_violins:
                fig.update_xaxes(title_text="Timepoints", tickangle=45, row=4, col=i+1)
            
            fig.update_yaxes(title_text="Similarity", row=1, col=i+1)
            fig.update_yaxes(title_text="Similarity", row=2, col=i+1)
            fig.update_yaxes(title_text="Sparsity", row=3, col=i+1)
            if include_violins:
                fig.update_yaxes(title_text="Similarity Distribution", row=4, col=i+1)
        
        fig.update_layout(
            height=300 * n_rows,
            width=400 * n_metrics,
            title_text="GRN Similarity Analysis: Superset Approach with Multiple Metrics<br><sub>Error bars show Standard Error of the Mean (SEM)</sub>",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_similarity_violins(self, detailed_results=None, metrics=['pearson', 'cosine', 'spearman'], 
                               save_path=None, lineage_groups=None):
        """
        Create violin plots showing similarity distributions
        
        Parameters:
        -----------
        detailed_results : dict, optional
            Results from compute_detailed_similarities(). If None, will compute automatically
        metrics : list
            Metrics to plot
        save_path : str, optional
            Path to save HTML file
        lineage_groups : dict, optional
            Lineage groupings for coloring
        """
        
        # If no detailed results provided, compute them
        if detailed_results is None:
            print("🔍 Computing detailed similarities for violin plots...")
            detailed_results = self.compute_detailed_similarities(metrics=metrics, lineage_groups=lineage_groups)
        
        tp_raw_data = detailed_results['timepoint_analysis']['raw_similarities']
        
        if tp_raw_data.empty:
            print("❌ No timepoint data available for violin plots")
            return None
        
        # Create violin plots
        n_metrics = len(metrics)
        fig = make_subplots(
            rows=2, cols=n_metrics,
            subplot_titles=[f'{m.capitalize()} - Similarity Distributions by Timepoint' for m in metrics] +
                          [f'{m.capitalize()} - Similarity by Lineage Interactions' for m in metrics],
            specs=[[{"secondary_y": False} for _ in range(n_metrics)],
                   [{"secondary_y": False} for _ in range(n_metrics)]],
            vertical_spacing=0.12
        )
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        timepoint_order = ['TDR126', 'TDR127', 'TDR128', 'TDR118', 'TDR125', 'TDR124']
        hpf_labels = {'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
                     'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'}
        
        # Row 1: Violin plots by timepoint
        for i, metric in enumerate(metrics):
            metric_data = tp_raw_data[tp_raw_data['metric'] == metric]
            
            if not metric_data.empty:
                for j, tp in enumerate(timepoint_order):
                    if tp in metric_data['timepoint'].values:
                        tp_metric_data = metric_data[metric_data['timepoint'] == tp]
                        
                        fig.add_trace(
                            go.Violin(
                                y=tp_metric_data['similarity'],
                                x=[hpf_labels[tp]] * len(tp_metric_data),
                                name=hpf_labels[tp],
                                box_visible=True,
                                meanline_visible=True,
                                fillcolor=colors[j % len(colors)],
                                opacity=0.7,
                                showlegend=(i == 0),
                                points='outliers'  # Show outlier points
                            ),
                            row=1, col=i+1
                        )
        
        # Row 2: Violin plots by lineage interactions (if lineage data available)
        if 'lineage_pair_type' in tp_raw_data.columns:
            for i, metric in enumerate(metrics):
                metric_data = tp_raw_data[tp_raw_data['metric'] == metric]
                
                if not metric_data.empty:
                    # Get unique lineage pair types
                    lineage_types = sorted(metric_data['lineage_pair_type'].unique())
                    lineage_colors = px.colors.qualitative.Set3
                    
                    for j, lineage_type in enumerate(lineage_types):
                        lineage_data = metric_data[metric_data['lineage_pair_type'] == lineage_type]
                        
                        fig.add_trace(
                            go.Violin(
                                y=lineage_data['similarity'],
                                x=[lineage_type] * len(lineage_data),
                                name=lineage_type,
                                box_visible=True,
                                meanline_visible=True,
                                fillcolor=lineage_colors[j % len(lineage_colors)],
                                opacity=0.7,
                                showlegend=(i == 0),
                                points='outliers'
                            ),
                            row=2, col=i+1
                        )
        
        # Update layout
        for i in range(n_metrics):
            fig.update_xaxes(title_text="Developmental Stage", tickangle=45, row=1, col=i+1)
            fig.update_xaxes(title_text="Lineage Interaction Type", tickangle=45, row=2, col=i+1)
            
            fig.update_yaxes(title_text="Similarity Score", row=1, col=i+1)
            fig.update_yaxes(title_text="Similarity Score", row=2, col=i+1)
        
        fig.update_layout(
            height=800,
            width=400 * n_metrics,
            title_text="GRN Similarity Distributions: Violin Plot Analysis<br><sub>Showing full distributions and outliers</sub>",
            showlegend=True
        )
        
        if save_path:
            plt.savefig(save_path)
        
        fig.show()
        return fig

# Usage example with superset approach:
"""
# Initialize the analyzer
analyzer = GRNSimilarityAnalyzer(dict_filtered_GRNs)

# Analyze with multiple metrics
metrics = ['pearson', 'cosine', 'jaccard']

# Analyze celltype similarity within timepoints (your original analysis)
tp_results, tp_stats = analyzer.analyze_timepoint_similarity(metrics=metrics)

# Analyze timepoint similarity within celltypes (temporal dynamics)
ct_results, ct_stats = analyzer.analyze_celltype_similarity(metrics=metrics)

# Create comprehensive plots
fig = analyzer.plot_similarity_trends(tp_results, ct_results, tp_stats, ct_stats, 
                                     metrics=metrics, save_path='grn_similarity_superset.html')
fig.show()

# Display results
print("\n" + "="*60)
print("TIMEPOINT ANALYSIS RESULTS:")
print("="*60)
print(tp_results.round(3))

print("\n" + "="*60) 
print("CELLTYPE ANALYSIS RESULTS:")
print("="*60)
print(ct_results.round(3))
"""

# %%
# Check a sample GRN matrix
sample_grn = dict_filtered_GRNs['TDR126']['NMPs']
print(f"Matrix dtype: {sample_grn.dtypes}")
print(f"Sample values: {sample_grn.iloc[:3, :3]}")
print(f"Value types: {[type(x) for x in sample_grn.iloc[0, :3]]}")

# %%
# Initialize the analyzer
analyzer = GRNSimilarityAnalyzer(dict_filtered_GRNs)

# Analyze with multiple metrics
metrics = ['pearson', 'cosine', 'jaccard']

# Analyze celltype similarity within timepoints (your original analysis)
tp_results, tp_stats = analyzer.analyze_timepoint_similarity(metrics=metrics)

# Analyze timepoint similarity within celltypes (temporal dynamics)
ct_results, ct_stats = analyzer.analyze_celltype_similarity(metrics=metrics)

# Create comprehensive plots
fig = analyzer.plot_similarity_trends(tp_results, ct_results, tp_stats, ct_stats, 
                                     metrics=metrics, save_path='grn_similarity_superset.html')
fig.show()

# Display results
print("\n" + "="*60)
print("TIMEPOINT ANALYSIS RESULTS:")
print("="*60)
print(tp_results.round(3))

print("\n" + "="*60) 
print("CELLTYPE ANALYSIS RESULTS:")
print("="*60)
print(ct_results.round(3))

# %%
# Initialize the analyzer
analyzer = GRNSimilarityAnalyzer(dict_filtered_GRNs)

# Analyze with multiple metrics
metrics = ['pearson', 'cosine', 'spearman']

# Analyze celltype similarity within timepoints (your original analysis)
tp_results, tp_stats = analyzer.analyze_timepoint_similarity(metrics=metrics)

# Analyze timepoint similarity within celltypes (temporal dynamics)
ct_results, ct_stats = analyzer.analyze_celltype_similarity(metrics=metrics)

# Create comprehensive plots
fig = analyzer.plot_similarity_trends(tp_results, ct_results, tp_stats, ct_stats, 
                                     metrics=metrics, save_path=figpath + 'grn_similarity_superset.pdf')
fig.show()

# # Display results
# print("\n" + "="*60)
# print("TIMEPOINT ANALYSIS RESULTS:")
# print("="*60)
# print(tp_results.round(3))

# print("\n" + "="*60) 
# print("CELLTYPE ANALYSIS RESULTS:")
# print("="*60)
# print(ct_results.round(3))


# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jaccard
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

class GRNSimilarityAnalyzer:
    def __init__(self, dict_filtered_GRNs):
        """
        Initialize with the GRN dictionary
        dict_filtered_GRNs: {timepoint: {celltype: GRN_matrix}}
        """
        self.grns = dict_filtered_GRNs
        self.timepoints = list(dict_filtered_GRNs.keys())
        self.celltypes = list(dict_filtered_GRNs[self.timepoints[0]].keys())
        print(f"Found {len(self.timepoints)} timepoints: {self.timepoints}")
        print(f"Found {len(self.celltypes)} celltypes")
        
    def extract_tf_gene_pairs(self, grn_matrix):
        """Extract TF-gene pairs from a GRN matrix (handles both wide and long formats)"""
        if grn_matrix is None or grn_matrix.empty:
            return set()
        
        pairs = set()
        
        # Check if this is long format (has 'source', 'target' columns)
        if 'source' in grn_matrix.columns and 'target' in grn_matrix.columns:
            # Long format: source, target, coef_mean (or similar)
            coef_col = 'coef_mean' if 'coef_mean' in grn_matrix.columns else grn_matrix.columns[-1]
            
            for _, row in grn_matrix.iterrows():
                source = row['source']  # This is the TF
                target = row['target']  # This is the target gene
                val = row[coef_col]
                
                # Check if value is non-zero
                try:
                    if pd.notna(val) and float(val) != 0:
                        pairs.add(f"{target}_{source}")  # target_TF format
                except (ValueError, TypeError):
                    continue
                    
        else:
            # Wide format: matrix with targets as rows, TFs as columns
            grn_stacked = grn_matrix.stack()
            for (target, tf), val in grn_stacked.items():
                try:
                    if pd.notna(val) and float(val) != 0:
                        pairs.add(f"{target}_{tf}")
                except (ValueError, TypeError):
                    continue
        
        return pairs
    
    def get_superset_pairs(self, grn_dict):
        """
        Get union (superset) of all TF-gene pairs from a dictionary of GRNs
        grn_dict: {identifier: GRN_matrix}
        """
        all_pairs = set()
        valid_grns = 0
        
        for identifier, grn in grn_dict.items():
            if grn is not None and not grn.empty:
                pairs = self.extract_tf_gene_pairs(grn)
                all_pairs.update(pairs)
                valid_grns += 1
        
        print(f"  Superset contains {len(all_pairs):,} unique TF-gene pairs from {valid_grns} valid GRNs")
        return sorted(list(all_pairs))  # Sort for consistent ordering
    
    def grn_to_vector_superset(self, grn, superset_pairs):
        """
        Convert GRN matrix to feature vector using superset of pairs
        Missing pairs are filled with 0 (no regulation)
        Handles both wide and long format GRN matrices
        """
        if grn is None or grn.empty:
            return np.zeros(len(superset_pairs))
        
        feature_dict = {}
        
        # Check if this is long format (has 'source', 'target' columns)
        if 'source' in grn.columns and 'target' in grn.columns:
            # Long format: source, target, coef_mean (or similar)
            coef_col = 'coef_mean' if 'coef_mean' in grn.columns else grn.columns[-1]
            
            for _, row in grn.iterrows():
                source = row['source']  # This is the TF
                target = row['target']  # This is the target gene
                val = row[coef_col]
                
                # Convert to numeric
                try:
                    val = float(val) if pd.notna(val) else 0.0
                except (ValueError, TypeError):
                    val = 0.0
                
                feature_dict[f"{target}_{source}"] = val
                
        else:
            # Wide format: matrix with targets as rows, TFs as columns
            grn_stacked = grn.stack()
            for (target, tf), val in grn_stacked.items():
                # Convert to numeric
                try:
                    val = float(val) if pd.notna(val) else 0.0
                except (ValueError, TypeError):
                    val = 0.0
                
                feature_dict[f"{target}_{tf}"] = val
        
        # Create vector for superset: existing pairs get their values, missing pairs get 0
        vector = np.array([feature_dict.get(pair, 0.0) for pair in superset_pairs], dtype=np.float64)
        return vector
    
    def compute_similarity_matrix(self, vectors_dict, metric='pearson'):
        """
        Compute similarity matrix from feature vectors dictionary
        vectors_dict: {identifier: vector}
        """
        identifiers = list(vectors_dict.keys())
        vectors = list(vectors_dict.values())
        n = len(vectors)
        
        similarity_matrix = np.full((n, n), np.nan)
        
        for i in range(n):
            for j in range(i, n):
                if np.all(np.isnan(vectors[i])) or np.all(np.isnan(vectors[j])):
                    continue
                    
                if metric == 'pearson':
                    # Handle constant vectors
                    if np.std(vectors[i]) == 0 or np.std(vectors[j]) == 0:
                        corr = 1.0 if np.allclose(vectors[i], vectors[j]) else 0.0
                    else:
                        corr, _ = pearsonr(vectors[i], vectors[j])
                        if np.isnan(corr):
                            corr = 0.0
                            
                elif metric == 'cosine':
                    corr = cosine_similarity([vectors[i]], [vectors[j]])[0, 0]
                    
                elif metric == 'spearman':
                    # Handle constant vectors
                    if np.std(vectors[i]) == 0 or np.std(vectors[j]) == 0:
                        corr = 1.0 if np.allclose(vectors[i], vectors[j]) else 0.0
                    else:
                        corr, _ = spearmanr(vectors[i], vectors[j])
                        if np.isnan(corr):
                            corr = 0.0
                
                elif metric == 'jaccard':
                    # Jaccard on binary vectors (non-zero elements)
                    binary_i = (vectors[i] != 0).astype(int)
                    binary_j = (vectors[j] != 0).astype(int)
                    
                    intersection = np.sum(binary_i & binary_j)
                    union = np.sum(binary_i | binary_j)
                    
                    corr = intersection / union if union > 0 else 0.0
                
                similarity_matrix[i, j] = similarity_matrix[j, i] = corr
        
        return similarity_matrix, identifiers
    
    def compute_data_characteristics(self, vectors_dict, context=""):
        """Compute and report data characteristics"""
        vectors = list(vectors_dict.values())
        identifiers = list(vectors_dict.keys())
        
        if not vectors:
            return {}
        
        # Ensure all vectors are numeric
        numeric_vectors = []
        for v in vectors:
            if v.dtype.kind not in 'biufc':  # not a numeric type
                print(f"Warning: Non-numeric vector detected, converting to float64")
                v = np.array([float(x) if x != '' else 0.0 for x in v], dtype=np.float64)
            numeric_vectors.append(v)
        
        vectors = numeric_vectors
        
        # Basic statistics
        total_pairs = len(vectors[0]) if vectors else 0
        sparsity_per_grn = [np.mean(v == 0) for v in vectors]
        nonzero_per_grn = [np.sum(v != 0) for v in vectors]
        
        # Compute mean values only for non-zero elements
        mean_values = []
        for v in vectors:
            nonzero_vals = v[v != 0]
            if len(nonzero_vals) > 0:
                mean_values.append(np.mean(nonzero_vals))
            else:
                mean_values.append(0.0)
        
        stats = {
            'context': context,
            'total_pairs': total_pairs,
            'n_grns': len(vectors),
            'mean_sparsity': np.mean(sparsity_per_grn),
            'std_sparsity': np.std(sparsity_per_grn),
            'mean_nonzero_pairs': np.mean(nonzero_per_grn),
            'std_nonzero_pairs': np.std(nonzero_per_grn),
            'mean_interaction_strength': np.mean(mean_values),
            'sparsity_per_grn': dict(zip(identifiers, sparsity_per_grn)),
            'nonzero_per_grn': dict(zip(identifiers, nonzero_per_grn))
        }
        
        # Print summary
        print(f"\n📊 Data characteristics for {context}:")
        print(f"  Total TF-gene pairs in superset: {total_pairs:,}")
        print(f"  Number of GRNs: {len(vectors)}")
        print(f"  Mean sparsity (% zeros): {stats['mean_sparsity']:.3f} ± {stats['std_sparsity']:.3f}")
        print(f"  Mean active pairs per GRN: {stats['mean_nonzero_pairs']:,.0f} ± {stats['std_nonzero_pairs']:,.0f}")
        print(f"  Mean interaction strength: {stats['mean_interaction_strength']:.3f}")
        
        # Warnings
        if stats['mean_sparsity'] > 0.95:
            print("  Very sparse data (>95% zeros) - consider Jaccard similarity")
        if stats['mean_nonzero_pairs'] < 100:
            print("  Very few active pairs per GRN - limited statistical power")
            
        return stats
    
    def analyze_timepoint_similarity(self, metrics=['pearson', 'cosine', 'spearman', 'jaccard'],
                                   comparison_method='superset', min_overlap_threshold=0.2):
        """
        Analyze how similar celltypes are within each timepoint using superset approach
        
        Parameters:
        -----------
        metrics : list
            Similarity metrics to compute
        comparison_method : str
            'superset' - use union of all pairs, fill missing with 0
            'intersection' - use only common pairs
            'adaptive' - use intersection if overlap > threshold, else superset
        min_overlap_threshold : float
            Minimum overlap fraction required for intersection method (0.0-1.0)
        """
        results = {
            'timepoints': [],
            'data_stats': {},
            'comparison_method_used': []
        }
        
        # Initialize results for each metric
        for metric in metrics:
            results[f'{metric}_mean'] = []
            results[f'{metric}_std'] = []
            results[f'{metric}_sem'] = []
        
        for tp in self.timepoints:
            print(f"\n🔍 Analyzing timepoint: {tp}")
            
            # Get all valid GRNs for this timepoint
            tp_grns = {}
            for ct in self.celltypes:
                grn = self.grns[tp].get(ct)
                if grn is not None and not grn.empty:
                    tp_grns[ct] = grn
            
            if len(tp_grns) < 2:
                print(f"  ❌ Skipping {tp} - insufficient data ({len(tp_grns)} valid GRNs)")
                continue
            
            print(f"  Valid celltypes: {len(tp_grns)}/{len(self.celltypes)}")
            
            # Determine which comparison method to use
            if comparison_method == 'adaptive':
                # Check overlap between GRNs to decide method
                all_pair_sets = []
                for grn in tp_grns.values():
                    pairs = self.extract_tf_gene_pairs(grn)
                    all_pair_sets.append(pairs)
                
                # Calculate average pairwise overlap
                total_overlaps = 0
                total_comparisons = 0
                for i in range(len(all_pair_sets)):
                    for j in range(i+1, len(all_pair_sets)):
                        intersection_size = len(all_pair_sets[i] & all_pair_sets[j])
                        union_size = len(all_pair_sets[i] | all_pair_sets[j])
                        if union_size > 0:
                            total_overlaps += intersection_size / union_size
                            total_comparisons += 1
                
                avg_overlap = total_overlaps / total_comparisons if total_comparisons > 0 else 0
                method_used = 'intersection' if avg_overlap >= min_overlap_threshold else 'superset'
                print(f"  Average overlap: {avg_overlap:.3f}, using {method_used} method")
            else:
                method_used = comparison_method
                print(f"  Using {method_used} method")
            
            results['comparison_method_used'].append(method_used)
            
            # Apply the chosen method
            if method_used == 'intersection':
                # Find common pairs across all GRNs
                common_pairs = None
                for grn in tp_grns.values():
                    pairs = self.extract_tf_gene_pairs(grn)
                    if common_pairs is None:
                        common_pairs = pairs
                    else:
                        common_pairs = common_pairs.intersection(pairs)
                
                common_pairs = sorted(list(common_pairs))
                print(f"  Using {len(common_pairs)} common pairs")
                
                if len(common_pairs) < 10:
                    print(f"  ⚠️ Very few common pairs, results may not be reliable")
                
                # Convert to vectors using only common pairs
                vectors_dict = {}
                for ct, grn in tp_grns.items():
                    vector = np.array([self._get_pair_value(grn, pair) for pair in common_pairs])
                    vectors_dict[ct] = vector
                    
            else:  # superset method
                # Get superset of all pairs
                superset_pairs = self.get_superset_pairs(tp_grns)
                print(f"  Using {len(superset_pairs)} total pairs (superset)")
                
                # Convert to vectors using superset (missing = 0)
                vectors_dict = {}
                for ct, grn in tp_grns.items():
                    vectors_dict[ct] = self.grn_to_vector_superset(grn, superset_pairs)
            
            # Compute data characteristics
            data_stats = self.compute_data_characteristics(vectors_dict, f"timepoint {tp}")
            results['data_stats'][tp] = data_stats
            
            # Compute similarities for each requested metric
            metric_results = {}
            for metric in metrics:
                sim_matrix, identifiers = self.compute_similarity_matrix(vectors_dict, metric)
                
                # Extract upper triangular values (unique pairs)
                mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
                sim_values = sim_matrix[mask]
                sim_values = sim_values[~np.isnan(sim_values)]
                
                if len(sim_values) > 0:
                    metric_results[metric] = {
                        'mean': np.mean(sim_values),
                        'std': np.std(sim_values),
                        'values': sim_values,
                        'n_comparisons': len(sim_values)
                    }
                    print(f"  {metric.capitalize()} similarity: {np.mean(sim_values):.3f} ± {np.std(sim_values):.3f} ({len(sim_values)} pairs)")
                else:
                    metric_results[metric] = {'mean': np.nan, 'std': np.nan, 'values': [], 'n_comparisons': 0}
                    print(f"  {metric.capitalize()} similarity: No valid comparisons")
            
            # Store results
            results['timepoints'].append(tp)
            for metric in metrics:
                results[f'{metric}_mean'].append(metric_results[metric]['mean'])
                results[f'{metric}_std'].append(metric_results[metric]['std'])
                # Calculate SEM (standard error of the mean)
                n_comparisons = metric_results[metric]['n_comparisons']
                sem = metric_results[metric]['std'] / np.sqrt(n_comparisons) if n_comparisons > 0 else 0
                results[f'{metric}_sem'].append(sem)
        
        return pd.DataFrame({k: v for k, v in results.items() if k not in ['data_stats', 'comparison_method_used']}), results['data_stats']
    
    def _get_pair_value(self, grn, pair):
        """Helper function to get value for a TF-gene pair from GRN"""
        target, source = pair.split('_', 1)
        
        # Handle long format (source, target, coef_mean columns)
        if 'source' in grn.columns and 'target' in grn.columns:
            coef_col = 'coef_mean' if 'coef_mean' in grn.columns else grn.columns[-1]
            match = grn[(grn['source'] == source) & (grn['target'] == target)]
            if not match.empty:
                try:
                    return float(match.iloc[0][coef_col])
                except (ValueError, TypeError):
                    return 0.0
            return 0.0
        else:
            # Handle wide format
            try:
                if target in grn.index and source in grn.columns:
                    val = grn.loc[target, source]
                    return float(val) if pd.notna(val) else 0.0
            except (KeyError, ValueError, TypeError):
                pass
            return 0.0
        """
        Analyze how similar celltypes are within each timepoint using superset approach
        """
        results = {
            'timepoints': [],
            'data_stats': {}
        }
        
        # Initialize results for each metric
        for metric in metrics:
            results[f'{metric}_mean'] = []
            results[f'{metric}_std'] = []
        
        for tp in self.timepoints:
            print(f"\n🔍 Analyzing timepoint: {tp}")
            
            # Get all valid GRNs for this timepoint
            tp_grns = {}
            for ct in self.celltypes:
                grn = self.grns[tp].get(ct)
                if grn is not None and not grn.empty:
                    tp_grns[ct] = grn
            
            if len(tp_grns) < 2:
                print(f"  ❌ Skipping {tp} - insufficient data ({len(tp_grns)} valid GRNs)")
                continue
            
            print(f"  Valid celltypes: {len(tp_grns)}/{len(self.celltypes)}")
            
            # Get superset of all TF-gene pairs for this timepoint
            superset_pairs = self.get_superset_pairs(tp_grns)
            
            # Convert each GRN to vector using superset (missing = 0)
            vectors_dict = {}
            for ct, grn in tp_grns.items():
                vectors_dict[ct] = self.grn_to_vector_superset(grn, superset_pairs)
            
            # Compute data characteristics
            data_stats = self.compute_data_characteristics(vectors_dict, f"timepoint {tp}")
            results['data_stats'][tp] = data_stats
            
            # Compute similarities for each requested metric
            metric_results = {}
            for metric in metrics:
                sim_matrix, identifiers = self.compute_similarity_matrix(vectors_dict, metric)
                
                # Extract upper triangular values (unique pairs)
                mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
                sim_values = sim_matrix[mask]
                sim_values = sim_values[~np.isnan(sim_values)]
                
                if len(sim_values) > 0:
                    metric_results[metric] = {
                        'mean': np.mean(sim_values),
                        'std': np.std(sim_values),
                        'values': sim_values,
                        'n_comparisons': len(sim_values)
                    }
                    print(f"  {metric.capitalize()} similarity: {np.mean(sim_values):.3f} ± {np.std(sim_values):.3f} ({len(sim_values)} pairs)")
                else:
                    metric_results[metric] = {'mean': np.nan, 'std': np.nan, 'values': [], 'n_comparisons': 0}
                    print(f"  {metric.capitalize()} similarity: No valid comparisons")
            
            # Store results
            results['timepoints'].append(tp)
            for metric in metrics:
                results[f'{metric}_mean'].append(metric_results[metric]['mean'])
                results[f'{metric}_std'].append(metric_results[metric]['std'])
                # Calculate SEM (standard error of the mean)
                n_comparisons = metric_results[metric]['n_comparisons']
                sem = metric_results[metric]['std'] / np.sqrt(n_comparisons) if n_comparisons > 0 else 0
                results[f'{metric}_sem'] = results.get(f'{metric}_sem', [])
                results[f'{metric}_sem'].append(sem)
        
    def analyze_celltype_similarity(self, metrics=['pearson', 'cosine', 'spearman', 'jaccard'],
                                  comparison_method='superset', min_overlap_threshold=0.2):
        """
        Analyze how similar timepoints are within each celltype using superset approach
        
        Parameters:
        -----------
        metrics : list
            Similarity metrics to compute  
        comparison_method : str
            'superset' - use union of all pairs, fill missing with 0
            'intersection' - use only common pairs
            'adaptive' - use intersection if overlap > threshold, else superset
        min_overlap_threshold : float
            Minimum overlap fraction required for intersection method (0.0-1.0)
        """
        results = {
            'celltypes': [],
            'data_stats': {},
            'comparison_method_used': []
        }
        
        # Initialize results for each metric
        for metric in metrics:
            results[f'{metric}_mean'] = []
            results[f'{metric}_std'] = []
            results[f'{metric}_sem'] = []
        
        for ct in self.celltypes:
            print(f"\n🔍 Analyzing celltype: {ct}")
            
            # Get all valid GRNs for this celltype across timepoints
            ct_grns = {}
            for tp in self.timepoints:
                grn = self.grns[tp].get(ct)
                if grn is not None and not grn.empty:
                    ct_grns[tp] = grn
            
            if len(ct_grns) < 2:
                print(f"  ❌ Skipping {ct} - insufficient data ({len(ct_grns)} valid GRNs)")
                continue
            
            print(f"  Valid timepoints: {len(ct_grns)}/{len(self.timepoints)}")
            
            # Determine which comparison method to use
            if comparison_method == 'adaptive':
                # Check overlap between GRNs
                all_pair_sets = []
                for grn in ct_grns.values():
                    pairs = self.extract_tf_gene_pairs(grn)
                    all_pair_sets.append(pairs)
                
                # Calculate average pairwise overlap
                total_overlaps = 0
                total_comparisons = 0
                for i in range(len(all_pair_sets)):
                    for j in range(i+1, len(all_pair_sets)):
                        intersection_size = len(all_pair_sets[i] & all_pair_sets[j])
                        union_size = len(all_pair_sets[i] | all_pair_sets[j])
                        if union_size > 0:
                            total_overlaps += intersection_size / union_size
                            total_comparisons += 1
                
                avg_overlap = total_overlaps / total_comparisons if total_comparisons > 0 else 0
                method_used = 'intersection' if avg_overlap >= min_overlap_threshold else 'superset'
                print(f"  Average overlap: {avg_overlap:.3f}, using {method_used} method")
            else:
                method_used = comparison_method
                print(f"  Using {method_used} method")
            
            results['comparison_method_used'].append(method_used)
            
            # Apply the chosen method
            if method_used == 'intersection':
                # Find common pairs across all timepoints for this celltype
                common_pairs = None
                for grn in ct_grns.values():
                    pairs = self.extract_tf_gene_pairs(grn)
                    if common_pairs is None:
                        common_pairs = pairs
                    else:
                        common_pairs = common_pairs.intersection(pairs)
                
                common_pairs = sorted(list(common_pairs))
                print(f"  Using {len(common_pairs)} common pairs")
                
                if len(common_pairs) < 10:
                    print(f"  ⚠️ Very few common pairs, results may not be reliable")
                
                # Convert to vectors using only common pairs
                vectors_dict = {}
                for tp, grn in ct_grns.items():
                    vector = np.array([self._get_pair_value(grn, pair) for pair in common_pairs])
                    vectors_dict[tp] = vector
                    
            else:  # superset method
                # Get superset of all pairs
                superset_pairs = self.get_superset_pairs(ct_grns)
                print(f"  Using {len(superset_pairs)} total pairs (superset)")
                
                # Convert to vectors using superset (missing = 0)
                vectors_dict = {}
                for tp, grn in ct_grns.items():
                    vectors_dict[tp] = self.grn_to_vector_superset(grn, superset_pairs)
            
            # Compute data characteristics
            data_stats = self.compute_data_characteristics(vectors_dict, f"celltype {ct}")
            results['data_stats'][ct] = data_stats
            
            # Compute similarities for each requested metric
            metric_results = {}
            for metric in metrics:
                sim_matrix, identifiers = self.compute_similarity_matrix(vectors_dict, metric)
                
                # Extract upper triangular values
                mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
                sim_values = sim_matrix[mask]
                sim_values = sim_values[~np.isnan(sim_values)]
                
                if len(sim_values) > 0:
                    metric_results[metric] = {
                        'mean': np.mean(sim_values),
                        'std': np.std(sim_values),
                        'values': sim_values,
                        'n_comparisons': len(sim_values)
                    }
                    print(f"  {metric.capitalize()} similarity: {np.mean(sim_values):.3f} ± {np.std(sim_values):.3f} ({len(sim_values)} pairs)")
                else:
                    metric_results[metric] = {'mean': np.nan, 'std': np.nan, 'values': [], 'n_comparisons': 0}
                    print(f"  {metric.capitalize()} similarity: No valid comparisons")
            
            # Store results
            results['celltypes'].append(ct)
            for metric in metrics:
                results[f'{metric}_mean'].append(metric_results[metric]['mean'])
                results[f'{metric}_std'].append(metric_results[metric]['std'])
                # Calculate SEM (standard error of the mean)
                n_comparisons = metric_results[metric]['n_comparisons']
                sem = metric_results[metric]['std'] / np.sqrt(n_comparisons) if n_comparisons > 0 else 0
                results[f'{metric}_sem'].append(sem)
        
        return pd.DataFrame({k: v for k, v in results.items() if k not in ['data_stats', 'comparison_method_used']}), results['data_stats']
    
   
    def plot_similarity_trends(self, tp_results, ct_results, tp_stats=None, ct_stats=None, 
                              metrics=['pearson', 'cosine', 'spearman'], save_path=None, 
                              sort_celltypes=True, include_violins=True, save_individual_panels=False):
        """
        Create comprehensive similarity plots with multiple metrics
        
        Parameters:
        -----------
        sort_celltypes : bool
            Whether to sort celltypes by similarity magnitude (high to low)
        include_violins : bool
            Whether to include violin plots showing distributions
        save_path : str or bool
            If string: save HTML with this path
            If True: save individual panels as PDF
            If None: don't save
        save_individual_panels : bool
            Whether to save each panel as separate PDF (requires matplotlib)
        """
        n_metrics = len(metrics)
        n_rows = 4 if include_violins else 3
        
        # Create subplot layout
        subplot_titles = []
        subplot_titles.extend([f'{m.capitalize()} - Celltype Similarity Within Timepoints' for m in metrics])
        subplot_titles.extend([f'{m.capitalize()} - Timepoint Similarity Within Celltypes' for m in metrics])
        subplot_titles.extend([f'{m.capitalize()} - Data Characteristics' for m in metrics])
        if include_violins:
            subplot_titles.extend([f'{m.capitalize()} - Similarity Distributions' for m in metrics])
        
        fig = make_subplots(
            rows=n_rows, cols=n_metrics,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": False} for _ in range(n_metrics)] for _ in range(n_rows)],
            vertical_spacing=0.08
        )
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        
        # Row 1: Timepoint analysis (celltype similarity over time)
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in tp_results.columns and f'{metric}_sem' in tp_results.columns:
                fig.add_trace(
                    go.Scatter(
                        x=tp_results['timepoints'],
                        y=tp_results[f'{metric}_mean'],
                        error_y=dict(type='data', array=tp_results[f'{metric}_sem'], visible=True),
                        mode='lines+markers',
                        name=f'{metric.capitalize()} (TP)',
                        line=dict(color=colors[i % len(colors)], width=3),
                        marker=dict(size=8)
                    ),
                    row=1, col=i+1
                )
        
        # Row 2: Celltype analysis (timepoint similarity over celltypes) - SORTED
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in ct_results.columns and f'{metric}_sem' in ct_results.columns:
                
                # Sort celltypes by similarity magnitude if requested
                if sort_celltypes:
                    # Sort by mean similarity (descending) 
                    ct_sorted = ct_results.copy().sort_values(f'{metric}_mean', ascending=False)
                    x_values = list(range(len(ct_sorted)))
                    x_labels = ct_sorted['celltypes'].tolist()
                    y_values = ct_sorted[f'{metric}_mean']
                    error_values = ct_sorted[f'{metric}_sem']
                else:
                    x_values = list(range(len(ct_results)))
                    x_labels = ct_results['celltypes'].tolist()
                    y_values = ct_results[f'{metric}_mean']
                    error_values = ct_results[f'{metric}_sem']
                
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        error_y=dict(type='data', array=error_values, visible=True),
                        mode='markers',
                        name=f'{metric.capitalize()} (CT)',
                        marker=dict(color=colors[i % len(colors)], size=8, opacity=0.7),
                        text=x_labels,
                        hovertemplate='%{text}<br>' + f'{metric.capitalize()}: %{{y:.3f}} ± %{{error_y.array:.4f}}<extra></extra>',
                        showlegend=False
                    ),
                    row=2, col=i+1
                )
                
                # Update x-axis to show celltype names
                fig.update_xaxes(
                    ticktext=x_labels,
                    tickvals=x_values,
                    tickangle=45,
                    row=2, col=i+1
                )
        
        # Row 3: Data characteristics (if provided)
        if tp_stats:
            for i, metric in enumerate(metrics):
                # Plot sparsity over timepoints
                sparsity_data = [tp_stats[tp]['mean_sparsity'] for tp in tp_results['timepoints'] 
                               if tp in tp_stats]
                
                fig.add_trace(
                    go.Scatter(
                        x=tp_results['timepoints'],
                        y=sparsity_data,
                        mode='lines+markers',
                        name='Sparsity',
                        line=dict(color='gray', dash='dash'),
                        showlegend=i == 0
                    ),
                    row=3, col=i+1
                )
        
        # Row 4: Violin plots (if requested and data available)
        if include_violins and hasattr(self, '_last_detailed_results'):
            tp_raw_data = self._last_detailed_results['timepoint_analysis']['raw_similarities']
            
            if not tp_raw_data.empty:
                for i, metric in enumerate(metrics):
                    metric_data = tp_raw_data[tp_raw_data['metric'] == metric]
                    
                    if not metric_data.empty:
                        # Create violin plot data for this metric
                        timepoint_order = ['TDR126', 'TDR127', 'TDR128', 'TDR118', 'TDR125', 'TDR124']
                        hpf_labels = {'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
                                     'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'}
                        
                        for j, tp in enumerate(timepoint_order):
                            if tp in metric_data['timepoint'].values:
                                tp_metric_data = metric_data[metric_data['timepoint'] == tp]
                                
                                fig.add_trace(
                                    go.Violin(
                                        y=tp_metric_data['similarity'],
                                        x=[hpf_labels[tp]] * len(tp_metric_data),
                                        name=hpf_labels[tp],
                                        box_visible=True,
                                        meanline_visible=True,
                                        fillcolor=colors[j % len(colors)],
                                        opacity=0.5,
                                        showlegend=False,
                                        side='positive' if j % 2 == 0 else 'negative'
                                    ),
                                    row=4, col=i+1
                                )
        
        # Update layout
        for i in range(n_metrics):
            fig.update_xaxes(title_text="Timepoints", tickangle=45, row=1, col=i+1)
            fig.update_xaxes(title_text="Celltypes (Sorted by Similarity)", tickangle=45, row=2, col=i+1)
            fig.update_xaxes(title_text="Timepoints", tickangle=45, row=3, col=i+1)
            if include_violins:
                fig.update_xaxes(title_text="Timepoints", tickangle=45, row=4, col=i+1)
            
            fig.update_yaxes(title_text="Similarity", row=1, col=i+1)
            fig.update_yaxes(title_text="Similarity", row=2, col=i+1)
            fig.update_yaxes(title_text="Sparsity", row=3, col=i+1)
            if include_violins:
                fig.update_yaxes(title_text="Similarity Distribution", row=4, col=i+1)
        
        fig.update_layout(
            height=300 * n_rows,
            width=400 * n_metrics,
            title_text="GRN Similarity Analysis: Superset Approach with Multiple Metrics<br><sub>Error bars show Standard Error of the Mean (SEM)</sub>",
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        
        # Save files based on save_path parameter
        if save_path == True or save_individual_panels:
            self._save_individual_panels(fig, tp_results, ct_results, metrics, sort_celltypes)
        elif isinstance(save_path, str):
            fig.write_html(save_path)
        
        return fig
    
    def _save_individual_panels(self, main_fig, tp_results, ct_results, metrics, sort_celltypes):
        """
        Save individual panels as separate PDF files using matplotlib
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set publication-quality style
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        hpf_labels = {'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
                     'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'}
        
        print("💾 Saving individual panels as PDF files...")
        
        # Panel 1: Timepoint trends (all metrics together)
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in tp_results.columns:
                x_vals = [hpf_labels.get(tp, tp) for tp in tp_results['timepoints']]
                y_vals = tp_results[f'{metric}_mean']
                err_vals = tp_results[f'{metric}_sem'] if f'{metric}_sem' in tp_results.columns else None
                
                ax1.errorbar(x_vals, y_vals, yerr=err_vals, 
                           marker='o', linewidth=2, markersize=6,
                           label=metric.capitalize(), color=colors[i % len(colors)])
        
        ax1.set_xlabel('Developmental Stage')
        ax1.set_ylabel('Mean Similarity Score')
        ax1.set_title('GRN Similarity Trends Across Development\n(Celltype Similarity Within Timepoints)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('grn_timepoint_trends.pdf', bbox_inches='tight')
        plt.close()
        
        # Panel 2: Individual metric timepoint trends
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in tp_results.columns:
                fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
                
                x_vals = [hpf_labels.get(tp, tp) for tp in tp_results['timepoints']]
                y_vals = tp_results[f'{metric}_mean']
                err_vals = tp_results[f'{metric}_sem'] if f'{metric}_sem' in tp_results.columns else None
                
                ax2.errorbar(x_vals, y_vals, yerr=err_vals,
                           marker='o', linewidth=3, markersize=8,
                           color=colors[i % len(colors)], capsize=5)
                
                ax2.set_xlabel('Developmental Stage')
                ax2.set_ylabel(f'{metric.capitalize()} Similarity')
                ax2.set_title(f'{metric.capitalize()} Similarity Across Development')
                ax2.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'grn_{metric}_timepoint_trend.pdf', bbox_inches='tight')
                plt.close()
        
    def _save_individual_panels(self, main_fig, tp_results, ct_results, metrics, sort_celltypes):
        """
        Save individual panels as separate PDF files using matplotlib
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set publication-quality style - clean white background
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': False,  # No grids
            'axes.spines.top': False,
            'axes.spines.right': False
        })
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        hpf_labels = {'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
                     'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'}
        
        print("💾 Saving individual panels as PDF files...")
        
        # Panel 1: Timepoint trends (all metrics together)
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6), facecolor='white')
        ax1.set_facecolor('white')
        
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in tp_results.columns:
                x_vals = [hpf_labels.get(tp, tp) for tp in tp_results['timepoints']]
                y_vals = tp_results[f'{metric}_mean']
                err_vals = tp_results[f'{metric}_sem'] if f'{metric}_sem' in tp_results.columns else None
                
                ax1.errorbar(x_vals, y_vals, yerr=err_vals, 
                           marker='o', linewidth=2, markersize=6,
                           label=metric.capitalize(), color=colors[i % len(colors)],
                           capsize=4, capthick=1.5)
        
        ax1.set_xlabel('Developmental Stage')
        ax1.set_ylabel('Mean Similarity Score')
        ax1.set_title('GRN Similarity Trends Across Development\n(Celltype Similarity Within Timepoints)')
        ax1.legend(frameon=False)
        ax1.grid(False)  # No grid
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('grn_timepoint_trends.pdf', bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Panel 2: Individual metric timepoint trends
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in tp_results.columns:
                fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6), facecolor='white')
                ax2.set_facecolor('white')
                
                x_vals = [hpf_labels.get(tp, tp) for tp in tp_results['timepoints']]
                y_vals = tp_results[f'{metric}_mean']
                err_vals = tp_results[f'{metric}_sem'] if f'{metric}_sem' in tp_results.columns else None
                
                ax2.errorbar(x_vals, y_vals, yerr=err_vals,
                           marker='o', linewidth=3, markersize=8,
                           color=colors[i % len(colors)], capsize=5, capthick=2)
                
                ax2.set_xlabel('Developmental Stage')
                ax2.set_ylabel(f'{metric.capitalize()} Similarity')
                ax2.set_title(f'{metric.capitalize()} Similarity Across Development')
                ax2.grid(False)  # No grid
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'grn_{metric}_timepoint_trend.pdf', bbox_inches='tight', facecolor='white')
                plt.close()
        
        # Panel 3: Celltype similarity (sorted) with SEM
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in ct_results.columns:
                fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6), facecolor='white')
                ax3.set_facecolor('white')
                
                # Sort celltypes by similarity if requested
                if sort_celltypes:
                    ct_sorted = ct_results.copy().sort_values(f'{metric}_mean', ascending=False)
                else:
                    ct_sorted = ct_results.copy()
                
                x_vals = range(len(ct_sorted))
                y_vals = ct_sorted[f'{metric}_mean']
                # Use SEM for error bars
                if f'{metric}_sem' in ct_sorted.columns:
                    err_vals = ct_sorted[f'{metric}_sem']
                elif f'{metric}_std' in ct_sorted.columns:
                    # Calculate SEM from std if not available
                    counts = ct_sorted.get(f'{metric}_count', pd.Series([1]*len(ct_sorted)))
                    err_vals = ct_sorted[f'{metric}_std'] / np.sqrt(counts)
                else:
                    err_vals = None
                
                celltype_names = ct_sorted['celltypes']
                
                ax3.errorbar(x_vals, y_vals, yerr=err_vals,
                           marker='o', linewidth=0, markersize=6,
                           color=colors[i % len(colors)], capsize=3,
                           linestyle='none')
                
                ax3.set_xlabel('Cell Types (Sorted by Temporal Variability)')
                ax3.set_ylabel(f'{metric.capitalize()} Similarity')
                ax3.set_title(f'{metric.capitalize()} Temporal Dynamics by Cell Type\n(High = Stable, Low = Dynamic)')
                ax3.set_xticks(x_vals)
                ax3.set_xticklabels(celltype_names, rotation=45, ha='right')
                ax3.grid(False)  # No grid
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)
                plt.tight_layout()
                plt.savefig(f'grn_{metric}_celltype_sorted.pdf', bbox_inches='tight', facecolor='white')
                plt.close()
        
        # Panel 4: Violin plots (if detailed results available)
        if hasattr(self, '_last_detailed_results'):
            tp_raw_data = self._last_detailed_results['timepoint_analysis']['raw_similarities']
            
            if not tp_raw_data.empty:
                for i, metric in enumerate(metrics):
                    metric_data = tp_raw_data[tp_raw_data['metric'] == metric]
                    
                    if not metric_data.empty:
                        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
                        ax4.set_facecolor('white')
                        
                        # Prepare data for seaborn
                        plot_data = metric_data.copy()
                        plot_data['hpf'] = plot_data['timepoint'].map(hpf_labels)
                        
                        sns.violinplot(data=plot_data, x='hpf', y='similarity', 
                                     palette='viridis', inner='box', ax=ax4)
                        
                        ax4.set_xlabel('Developmental Stage')
                        ax4.set_ylabel(f'{metric.capitalize()} Similarity')
                        ax4.set_title(f'{metric.capitalize()} Similarity Distributions\n(All Celltype Pairs Within Each Timepoint)')
                        ax4.grid(False)  # No grid
                        ax4.spines['top'].set_visible(False)
                        ax4.spines['right'].set_visible(False)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig(f'grn_{metric}_violin_distributions.pdf', bbox_inches='tight', facecolor='white')
                        plt.close()
        
        print("✅ Individual panels saved as PDF files:")
        print("  - grn_timepoint_trends.pdf (all metrics combined)")
        for metric in metrics:
            print(f"  - grn_{metric}_timepoint_trend.pdf (individual metric)")
            print(f"  - grn_{metric}_celltype_sorted.pdf (sorted celltypes)")
            if hasattr(self, '_last_detailed_results'):
                print(f"  - grn_{metric}_violin_distributions.pdf (violin plots)")
    
    def plot_similarity_violins(self, detailed_results=None, metrics=['pearson', 'cosine', 'spearman'], 
                               save_path=None, lineage_groups=None):
        """
        Create violin plots showing similarity distributions
        
        Parameters:
        -----------
        detailed_results : dict, optional
            Results from compute_detailed_similarities(). If None, will compute automatically
        metrics : list
            Metrics to plot
        save_path : str, optional
            Path to save HTML file
        lineage_groups : dict, optional
            Lineage groupings for coloring
        """
        
        # If no detailed results provided, compute them
        if detailed_results is None:
            print("🔍 Computing detailed similarities for violin plots...")
            detailed_results = self.compute_detailed_similarities(metrics=metrics, lineage_groups=lineage_groups)
        
        tp_raw_data = detailed_results['timepoint_analysis']['raw_similarities']
        
        if tp_raw_data.empty:
            print("❌ No timepoint data available for violin plots")
            return None
        
        # Create violin plots
        n_metrics = len(metrics)
        fig = make_subplots(
            rows=2, cols=n_metrics,
            subplot_titles=[f'{m.capitalize()} - Similarity Distributions by Timepoint' for m in metrics] +
                          [f'{m.capitalize()} - Similarity by Lineage Interactions' for m in metrics],
            specs=[[{"secondary_y": False} for _ in range(n_metrics)],
                   [{"secondary_y": False} for _ in range(n_metrics)]],
            vertical_spacing=0.12
        )
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        timepoint_order = ['TDR126', 'TDR127', 'TDR128', 'TDR118', 'TDR125', 'TDR124']
        hpf_labels = {'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
                     'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'}
        
        # Row 1: Violin plots by timepoint
        for i, metric in enumerate(metrics):
            metric_data = tp_raw_data[tp_raw_data['metric'] == metric]
            
            if not metric_data.empty:
                for j, tp in enumerate(timepoint_order):
                    if tp in metric_data['timepoint'].values:
                        tp_metric_data = metric_data[metric_data['timepoint'] == tp]
                        
                        fig.add_trace(
                            go.Violin(
                                y=tp_metric_data['similarity'],
                                x=[hpf_labels[tp]] * len(tp_metric_data),
                                name=hpf_labels[tp],
                                box_visible=True,
                                meanline_visible=True,
                                fillcolor=colors[j % len(colors)],
                                opacity=0.7,
                                showlegend=(i == 0),
                                points='outliers'  # Show outlier points
                            ),
                            row=1, col=i+1
                        )
        
        # Row 2: Violin plots by lineage interactions (if lineage data available)
        if 'lineage_pair_type' in tp_raw_data.columns:
            for i, metric in enumerate(metrics):
                metric_data = tp_raw_data[tp_raw_data['metric'] == metric]
                
                if not metric_data.empty:
                    # Get unique lineage pair types
                    lineage_types = sorted(metric_data['lineage_pair_type'].unique())
                    lineage_colors = px.colors.qualitative.Set3
                    
                    for j, lineage_type in enumerate(lineage_types):
                        lineage_data = metric_data[metric_data['lineage_pair_type'] == lineage_type]
                        
                        fig.add_trace(
                            go.Violin(
                                y=lineage_data['similarity'],
                                x=[lineage_type] * len(lineage_data),
                                name=lineage_type,
                                box_visible=True,
                                meanline_visible=True,
                                fillcolor=lineage_colors[j % len(lineage_colors)],
                                opacity=0.7,
                                showlegend=(i == 0),
                                points='outliers'
                            ),
                            row=2, col=i+1
                        )
        
        # Update layout
        for i in range(n_metrics):
            fig.update_xaxes(title_text="Developmental Stage", tickangle=45, row=1, col=i+1)
            fig.update_xaxes(title_text="Lineage Interaction Type", tickangle=45, row=2, col=i+1)
            
            fig.update_yaxes(title_text="Similarity Score", row=1, col=i+1)
            fig.update_yaxes(title_text="Similarity Score", row=2, col=i+1)
        
        fig.update_layout(
            height=800,
            width=400 * n_metrics,
            title_text="GRN Similarity Distributions: Violin Plot Analysis<br><sub>Showing full distributions and outliers</sub>",
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        return fig

# Usage example with superset approach:
"""
# Initialize the analyzer
analyzer = GRNSimilarityAnalyzer(dict_filtered_GRNs)

# Analyze with multiple metrics
metrics = ['pearson', 'cosine', 'jaccard']

# Analyze celltype similarity within timepoints (your original analysis)
tp_results, tp_stats = analyzer.analyze_timepoint_similarity(metrics=metrics)

# Analyze timepoint similarity within celltypes (temporal dynamics)
ct_results, ct_stats = analyzer.analyze_celltype_similarity(metrics=metrics)

# Create comprehensive plots
fig = analyzer.plot_similarity_trends(tp_results, ct_results, tp_stats, ct_stats, 
                                     metrics=metrics, save_path='grn_similarity_superset.html')
fig.show()

# Display results
print("\n" + "="*60)
print("TIMEPOINT ANALYSIS RESULTS:")
print("="*60)
print(tp_results.round(3))

print("\n" + "="*60) 
print("CELLTYPE ANALYSIS RESULTS:")
print("="*60)
print(ct_results.round(3))
"""

# %%

# %%
# Initialize the analyzer
analyzer = GRNSimilarityAnalyzer(dict_filtered_GRNs)

# Analyze with multiple metrics
metrics = ['pearson', 'cosine', 'spearman', 'jaccard']

# Analyze celltype similarity within timepoints (your original analysis)
tp_results, tp_stats = analyzer.analyze_timepoint_similarity(metrics=metrics)

# Analyze timepoint similarity within celltypes (temporal dynamics)
ct_results, ct_stats = analyzer.analyze_celltype_similarity(metrics=metrics)

# Create comprehensive plots
fig = analyzer.plot_similarity_trends(tp_results, ct_results, tp_stats, ct_stats, 
                                     metrics=metrics, save_path=figpath + 'grn_similarity_superset.pdf')
fig.show()


# %%
tp_results

# %%
ct_results

# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_grn_scatter_plots(tp_results, ct_results, metric='cosine', save_pdf=False):
    """
    Create two scatter plots: timepoint trends and celltype variability
    
    Parameters:
    -----------
    tp_results : pd.DataFrame
        Timepoint analysis results with columns: timepoints, {metric}_mean, {metric}_sem
    ct_results : pd.DataFrame  
        Celltype analysis results with columns: celltypes, {metric}_mean, {metric}_sem
    metric : str
        Which metric to plot ('cosine', 'pearson', 'spearman', 'jaccard')
    save_pdf : bool
        Whether to save as PDF files
    """
    
    # Set clean matplotlib style
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Timepoint mapping for better labels
    hpf_labels = {
        'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
        'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'
    }
    
    # ========================================================================
    # PLOT 1: Timepoint Analysis (6 timepoints)
    # ========================================================================
    
    # Calculate figure width based on number of timepoints
    n_timepoints = len(tp_results)
    tp_width = max(6, n_timepoints * 1.2)  # Minimum 6 inches, scale with data
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(tp_width, 5), facecolor='white')
    ax1.set_facecolor('white')
    
    # Prepare data
    x_labels = [hpf_labels.get(tp, tp) for tp in tp_results['timepoints']]
    x_positions = range(len(x_labels))
    y_values = tp_results[f'{metric}_mean']
    error_values = tp_results[f'{metric}_sem'] if f'{metric}_sem' in tp_results.columns else None
    
    # Create scatter plot with error bars
    ax1.errorbar(x_positions, y_values, yerr=error_values,
                marker='o', linewidth=3, markersize=8,
                color='#2E86C1', capsize=5, capthick=2,
                markerfacecolor='#2E86C1', markeredgecolor='white', markeredgewidth=1)
    
    # Formatting
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set_xlabel('Developmental Stage', fontweight='bold')
    ax1.set_ylabel(f'{metric.capitalize()} Similarity', fontweight='bold')
    ax1.set_title(f'{metric.capitalize()} Similarity Across Development\n(Celltype Similarity Within Timepoints)', 
                 fontweight='bold', pad=20)
    
    # Clean styling
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.grid(False)
    
    plt.tight_layout()
    
    if save_pdf:
        plt.savefig(f'grn_{metric}_timepoint_scatter.pdf', 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ Saved: grn_{metric}_timepoint_scatter.pdf")
    
    plt.show()
    
def plot_grn_scatter_plots(tp_results, ct_results, metric='cosine', save_pdf=False,
                          celltype_colors=None, celltype_to_lineage=None):
    """
    Create two scatter plots: timepoint trends and celltype variability
    
    Parameters:
    -----------
    tp_results : pd.DataFrame
        Timepoint analysis results with columns: timepoints, {metric}_mean, {metric}_sem
    ct_results : pd.DataFrame  
        Celltype analysis results with columns: celltypes, {metric}_mean, {metric}_sem
    metric : str
        Which metric to plot ('cosine', 'pearson', 'spearman', 'jaccard')
    save_pdf : bool
        Whether to save as PDF files
    celltype_colors : dict, optional
        Color mapping for lineages
    celltype_to_lineage : dict, optional
        Mapping from lineage to list of celltypes
    """
    
def plot_grn_scatter_plots(tp_results, ct_results, metric='cosine', save_pdf=False,
                          lineage_colors=None, celltype_to_lineage=None, 
                          celltype_colors=None):  # Keep backward compatibility
    """
    Create two scatter plots: timepoint trends and celltype variability
    
    Parameters:
    -----------
    tp_results : pd.DataFrame
        Timepoint analysis results with columns: timepoints, {metric}_mean, {metric}_sem
    ct_results : pd.DataFrame  
        Celltype analysis results with columns: celltypes, {metric}_mean, {metric}_sem
    metric : str
        Which metric to plot ('cosine', 'pearson', 'spearman', 'jaccard')
    save_pdf : bool
        Whether to save as PDF files
    lineage_colors : dict, optional
        Color mapping for lineages
    celltype_to_lineage : dict, optional
        Mapping from lineage to list of celltypes
    """
    from difflib import get_close_matches
    
    # Handle backward compatibility
    if celltype_colors is not None and lineage_colors is None:
        lineage_colors = celltype_colors
    
    # Default color scheme if not provided
    if lineage_colors is None:
        lineage_colors = {
            'CNS': '#DAA520',                    # Golden/orange
            'Endoderm': '#6A5ACD',              # Blue/purple  
            'Epiderm': '#DC143C',               # Red
            'Germline': '#DA70D6',              # Magenta/orchid
            'Lateral Mesoderm': '#228B22',      # Forest green
            'Neural Crest': '#20B2AA',          # Light sea green/teal
            'Paraxial Mesoderm': '#4169E1'      # Royal blue
        }
    
    if celltype_to_lineage is None:
        celltype_to_lineage = {
            "CNS": [
                "neural", "neural_optic", "neural_optic2", "neural_posterior", 
                "neural_telencephalon", "neurons", "hindbrain", "midbrain_hindbrain_boundary",
                "midbrain_hindbrain_boundary2", "optic_cup", "spinal_cord", 
                "differentiating_neurons", "floor_plate", "neural_floor_plate", "enteric_neurons"
            ],
            "Neural Crest": ["neural_crest", "neural_crest2"],
            "Paraxial Mesoderm": [
                "somites", "fast_muscle", "muscle", "PSM", "floor_plate2", 
                "NMPs", "tail_bud", "notochord"
            ],
            "Lateral Mesoderm": [
                "lateral_plate_mesoderm", "heart_myocardium", "hematopoietic_vasculature",
                "pharyngeal_arches", "pronephros", "pronephros2", "hemangioblasts", "hatching_gland"
            ],
            "Endoderm": ["endoderm", "endocrine_pancreas"],
            "Epiderm": ["epidermis", "epidermis2", "epidermis3", "epidermis4"],
            "Germline": ["primordial_germ_cells"]
        }
    
    # Create comprehensive mapping: celltype -> lineage using fuzzy matching
    def find_celltype_lineage(celltype_name, celltype_to_lineage):
        """Find the best matching lineage for a celltype using fuzzy matching"""
        
        # First try exact match
        for lineage, celltypes in celltype_to_lineage.items():
            if celltype_name in celltypes:
                return lineage
        
        # If no exact match, try fuzzy matching
        all_reference_celltypes = []
        lineage_lookup = {}
        
        for lineage, celltypes in celltype_to_lineage.items():
            for ct in celltypes:
                all_reference_celltypes.append(ct)
                lineage_lookup[ct] = lineage
        
        # Find closest matches (top 3, minimum 60% similarity)
        closest_matches = get_close_matches(celltype_name, all_reference_celltypes, 
                                          n=3, cutoff=0.6)
        
        if closest_matches:
            best_match = closest_matches[0]
            matched_lineage = lineage_lookup[best_match]
            print(f"  Fuzzy match: '{celltype_name}' → '{best_match}' ({matched_lineage})")
            return matched_lineage
        
        # If still no match, try partial string matching
        for lineage, celltypes in celltype_to_lineage.items():
            for ref_ct in celltypes:
                if ref_ct in celltype_name or celltype_name in ref_ct:
                    print(f"  Partial match: '{celltype_name}' → '{ref_ct}' ({lineage})")
                    return lineage
        
        print(f"  ⚠️ No match found for: '{celltype_name}' → using 'Other'")
        return 'Other'
    
    def get_celltype_color(celltype, lineage_colors, celltype_to_lineage):
        """Get color for a celltype based on its lineage"""
        lineage = find_celltype_lineage(celltype, celltype_to_lineage)
        return lineage_colors.get(lineage, '#808080')  # Gray for unmapped celltypes
    
    # Set clean matplotlib style
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Timepoint mapping for better labels
    hpf_labels = {
        'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
        'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'
    }
    
    print("🎨 Mapping celltypes to lineages...")
    
    # ========================================================================
    # PLOT 1: Timepoint Analysis (6 timepoints)
    # ========================================================================
    
    # Calculate figure width based on number of timepoints
    n_timepoints = len(tp_results)
    tp_width = max(6, n_timepoints * 1.2)  # Minimum 6 inches, scale with data
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(tp_width, 5), facecolor='white')
    ax1.set_facecolor('white')
    
    # Prepare data
    x_labels = [hpf_labels.get(tp, tp) for tp in tp_results['timepoints']]
    x_positions = range(len(x_labels))
    y_values = tp_results[f'{metric}_mean']
    error_values = tp_results[f'{metric}_sem'] if f'{metric}_sem' in tp_results.columns else None
    
    # Create scatter plot with error bars
    ax1.errorbar(x_positions, y_values, yerr=error_values,
                marker='o', linewidth=3, markersize=8,
                color='#2E86C1', capsize=5, capthick=2,
                markerfacecolor='#2E86C1', markeredgecolor='white', markeredgewidth=1)
    
    # Formatting
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set_xlabel('Developmental Stage', fontweight='bold')
    ax1.set_ylabel(f'{metric.capitalize()} Similarity', fontweight='bold')
    ax1.set_title(f'{metric.capitalize()} Similarity Across Development\n(Celltype Similarity Within Timepoints)', 
                 fontweight='bold', pad=20)
    
    # Clean styling
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.grid(False)
    
    plt.tight_layout()
    
    if save_pdf:
        plt.savefig(f'grn_{metric}_timepoint_scatter.pdf', 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ Saved: grn_{metric}_timepoint_scatter.pdf")
    
    plt.show()
    
    # ========================================================================
    # PLOT 2: Celltype Analysis (31 celltypes) - SORTED with LINEAGE COLORS
    # ========================================================================
    
    # Calculate figure width based on number of celltypes
    n_celltypes = len(ct_results)
    ct_width = max(12, n_celltypes * 0.4)  # Minimum 12 inches, scale with data
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(ct_width, 6), facecolor='white')
    ax2.set_facecolor('white')
    
    # Sort celltypes by similarity (high to low) for better visualization
    ct_sorted = ct_results.copy().sort_values(f'{metric}_mean', ascending=False)
    
    # Prepare data
    x_positions = range(len(ct_sorted))
    y_values = ct_sorted[f'{metric}_mean']
    error_values = ct_sorted[f'{metric}_sem'] if f'{metric}_sem' in ct_sorted.columns else None
    celltype_labels = ct_sorted['celltypes']
    
    # Get colors for each celltype based on lineage (with fuzzy matching)
    point_colors = []
    lineage_assignments = {}
    
    for ct in celltype_labels:
        lineage = find_celltype_lineage(ct, celltype_to_lineage)
        color = lineage_colors.get(lineage, '#808080')
        point_colors.append(color)
        lineage_assignments[ct] = lineage
    
    # Create scatter plot with lineage-based colors
    for i, (x, y, color, ct) in enumerate(zip(x_positions, y_values, point_colors, celltype_labels)):
        # Plot error bar in same color as point
        if error_values is not None:
            ax2.errorbar(x, y, yerr=error_values.iloc[i],
                        marker='o', markersize=6, color=color,
                        markerfacecolor=color, markeredgecolor='white', markeredgewidth=1,
                        capsize=3, capthick=1.5, linestyle='none')
        else:
            ax2.scatter(x, y, color=color, s=40, 
                       edgecolors='white', linewidth=1)
    
    # Formatting
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(celltype_labels, rotation=45, ha='right')
    ax2.set_xlabel('Cell Types (Sorted: High→Low Temporal Similarity)', fontweight='bold')
    ax2.set_ylabel(f'{metric.capitalize()} Similarity', fontweight='bold')
    ax2.set_title(f'{metric.capitalize()} Temporal Dynamics by Cell Type\n(High = Stable Over Time, Low = Dynamic Over Time)', 
                 fontweight='bold', pad=20)
    
    # Clean styling
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.grid(False)
    
    # Add legend for lineage colors (only for lineages that appear in the data)
    from matplotlib.patches import Patch
    unique_lineages = set(lineage_assignments.values())
    legend_elements = [Patch(facecolor=lineage_colors.get(lineage, '#808080'), label=lineage) 
                      for lineage in sorted(unique_lineages) if lineage != 'Other']
    if legend_elements:
        legend = ax2.legend(handles=legend_elements, loc='upper right', frameon=False,
                           title='Lineages')
        legend.get_title().set_fontweight('bold')  # Set title fontweight separately
    
    # Set clean matplotlib style
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Timepoint mapping for better labels
    hpf_labels = {
        'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
        'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'
    }
    
    # ========================================================================
    # PLOT 1: Timepoint Analysis (6 timepoints)
    # ========================================================================
    
    # Calculate figure width based on number of timepoints
    n_timepoints = len(tp_results)
    tp_width = max(6, n_timepoints * 1.2)  # Minimum 6 inches, scale with data
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(tp_width, 5), facecolor='white')
    ax1.set_facecolor('white')
    
    # Prepare data
    x_labels = [hpf_labels.get(tp, tp) for tp in tp_results['timepoints']]
    x_positions = range(len(x_labels))
    y_values = tp_results[f'{metric}_mean']
    error_values = tp_results[f'{metric}_sem'] if f'{metric}_sem' in tp_results.columns else None
    
    # Create scatter plot with error bars
    ax1.errorbar(x_positions, y_values, yerr=error_values,
                marker='o', linewidth=3, markersize=8,
                color='#2E86C1', capsize=5, capthick=2,
                markerfacecolor='#2E86C1', markeredgecolor='white', markeredgewidth=1)
    
    # Formatting
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set_xlabel('Developmental Stage', fontweight='bold')
    ax1.set_ylabel(f'{metric.capitalize()} Similarity', fontweight='bold')
    ax1.set_title(f'{metric.capitalize()} Similarity Across Development\n(Celltype Similarity Within Timepoints)', 
                 fontweight='bold', pad=20)
    
    # Clean styling
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.grid(False)
    
    plt.tight_layout()
    
    if save_pdf:
        plt.savefig(f'grn_{metric}_timepoint_scatter.pdf', 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ Saved: grn_{metric}_timepoint_scatter.pdf")
    
    plt.show()
    
    # ========================================================================
    # PLOT 2: Celltype Analysis (31 celltypes) - SORTED with LINEAGE COLORS
    # ========================================================================
    
    # Calculate figure width based on number of celltypes
    n_celltypes = len(ct_results)
    ct_width = max(12, n_celltypes * 0.4)  # Minimum 12 inches, scale with data
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(ct_width, 6), facecolor='white')
    ax2.set_facecolor('white')
    
    # Sort celltypes by similarity (high to low) for better visualization
    ct_sorted = ct_results.copy().sort_values(f'{metric}_mean', ascending=False)
    
    # Prepare data
    x_positions = range(len(ct_sorted))
    y_values = ct_sorted[f'{metric}_mean']
    error_values = ct_sorted[f'{metric}_sem'] if f'{metric}_sem' in ct_sorted.columns else None
    celltype_labels = ct_sorted['celltypes']
    
    # Get colors for each celltype based on lineage (with fuzzy matching)
    point_colors = []
    lineage_assignments = {}
    
    for ct in celltype_labels:
        lineage = find_celltype_lineage(ct, celltype_to_lineage)
        color = lineage_colors.get(lineage, '#808080')
        point_colors.append(color)
        lineage_assignments[ct] = lineage
    
    # Create scatter plot with lineage-based colors
    for i, (x, y, color, ct) in enumerate(zip(x_positions, y_values, point_colors, celltype_labels)):
        # Plot error bar in same color as point
        if error_values is not None:
            ax2.errorbar(x, y, yerr=error_values.iloc[i],
                        marker='o', markersize=6, color=color,
                        markerfacecolor=color, markeredgecolor='white', markeredgewidth=1,
                        capsize=3, capthick=1.5, linestyle='none')
        else:
            ax2.scatter(x, y, color=color, s=40, 
                       edgecolors='white', linewidth=1)
    
    # Formatting
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(celltype_labels, rotation=45, ha='right')
    ax2.set_xlabel('Cell Types (Sorted: High→Low Temporal Similarity)', fontweight='bold')
    ax2.set_ylabel(f'{metric.capitalize()} Similarity', fontweight='bold')
    ax2.set_title(f'{metric.capitalize()} Temporal Dynamics by Cell Type\n(High = Stable Over Time, Low = Dynamic Over Time)', 
                 fontweight='bold', pad=20)
    
    # Clean styling
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.grid(False)
    
    # Add legend for lineage colors (only for lineages that appear in the data)
    from matplotlib.patches import Patch
    unique_lineages = set(lineage_assignments.values())
    legend_elements = [Patch(facecolor=lineage_colors.get(lineage, '#808080'), label=lineage) 
                      for lineage in sorted(unique_lineages) if lineage != 'Other']
    if legend_elements:
        ax2.legend(handles=legend_elements, loc='upper right', frameon=False,
                  title='Lineages', title_fontweight='bold')
    
    plt.tight_layout()
    
    if save_pdf:
        plt.savefig(f'grn_{metric}_celltype_scatter.pdf', 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ Saved: grn_{metric}_celltype_scatter.pdf")
    
    plt.show()
    
    # Print lineage assignment summary
    print(f"\n🎨 LINEAGE ASSIGNMENT SUMMARY:")
    print("="*50)
    lineage_counts = {}
    for lineage in lineage_assignments.values():
        lineage_counts[lineage] = lineage_counts.get(lineage, 0) + 1
    
    for lineage, count in sorted(lineage_counts.items()):
        color = lineage_colors.get(lineage, '#808080')
        print(f"  {lineage}: {count} celltypes (color: {color})")
    
    # Print most/least stable celltypes by lineage
    print(f"\n📊 STABILITY BY LINEAGE:")
    print("="*50)
    for lineage in sorted(unique_lineages):
        if lineage != 'Other':
            lineage_cts = [ct for ct, lin in lineage_assignments.items() if lin == lineage]
            if lineage_cts:
                lineage_similarities = ct_sorted[ct_sorted['celltypes'].isin(lineage_cts)][f'{metric}_mean']
                if not lineage_similarities.empty:
                    print(f"  {lineage}: {lineage_similarities.mean():.3f} ± {lineage_similarities.std():.3f}")
                    most_stable = ct_sorted[ct_sorted['celltypes'].isin(lineage_cts)].iloc[0]['celltypes']
                    most_dynamic = ct_sorted[ct_sorted['celltypes'].isin(lineage_cts)].iloc[-1]['celltypes']
                    print(f"    Most stable: {most_stable}")
                    print(f"    Most dynamic: {most_dynamic}")
    
    plt.tight_layout()
    
    if save_pdf:
        plt.savefig(f'grn_{metric}_celltype_scatter.pdf', 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ Saved: grn_{metric}_celltype_scatter.pdf")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 {metric.upper()} SIMILARITY SUMMARY:")
    print("="*50)
    print(f"Timepoint Analysis:")
    print(f"  Range: {tp_results[f'{metric}_mean'].min():.3f} - {tp_results[f'{metric}_mean'].max():.3f}")
    print(f"  14hpf (TDR128): {tp_results[tp_results['timepoints']=='TDR128'][f'{metric}_mean'].iloc[0]:.3f}")
    print(f"  Trend: {'Decreasing' if tp_results[f'{metric}_mean'].iloc[-1] < tp_results[f'{metric}_mean'].iloc[0] else 'Increasing'}")
    
    print(f"\nCelltype Analysis:")
    print(f"  Most stable (high similarity): {ct_sorted['celltypes'].iloc[0]} ({ct_sorted[f'{metric}_mean'].iloc[0]:.3f})")
    print(f"  Most dynamic (low similarity): {ct_sorted['celltypes'].iloc[-1]} ({ct_sorted[f'{metric}_mean'].iloc[-1]:.3f})")
    print(f"  Range: {ct_results[f'{metric}_mean'].min():.3f} - {ct_results[f'{metric}_mean'].max():.3f}")
    
    return fig1, fig2

def plot_all_metrics_comparison(tp_results, ct_results, metrics=['pearson', 'cosine', 'spearman'], 
                               save_pdf=False):
    """
    Create comparison plots for all metrics side by side
    """
    # Set up figure with subplots
    n_metrics = len(metrics)
    
    # Timepoint comparison
    fig1, axes1 = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5), facecolor='white')
    if n_metrics == 1:
        axes1 = [axes1]
    
    # Celltype comparison  
    n_celltypes = len(ct_results)
    ct_width = max(12, n_celltypes * 0.3)
    fig2, axes2 = plt.subplots(1, n_metrics, figsize=(ct_width, 6), facecolor='white')
    if n_metrics == 1:
        axes2 = [axes2]
    
    colors = ['#2E86C1', '#E74C3C', '#28B463', '#8E44AD']
    hpf_labels = {
        'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
        'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'
    }
    
    # Plot timepoint analysis for each metric
    for i, metric in enumerate(metrics):
        if f'{metric}_mean' in tp_results.columns:
            ax = axes1[i]
            ax.set_facecolor('white')
            
            x_labels = [hpf_labels.get(tp, tp) for tp in tp_results['timepoints']]
            x_positions = range(len(x_labels))
            y_values = tp_results[f'{metric}_mean']
            error_values = tp_results[f'{metric}_sem'] if f'{metric}_sem' in tp_results.columns else None
            
            ax.errorbar(x_positions, y_values, yerr=error_values,
                       marker='o', linewidth=3, markersize=8,
                       color=colors[i % len(colors)], capsize=5, capthick=2)
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.set_xlabel('Developmental Stage', fontweight='bold')
            ax.set_ylabel(f'{metric.capitalize()} Similarity', fontweight='bold')
            ax.set_title(f'{metric.capitalize()}', fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)
    
    plt.suptitle('GRN Similarity Across Development (All Metrics)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Plot celltype analysis for each metric
    for i, metric in enumerate(metrics):
        if f'{metric}_mean' in ct_results.columns:
            ax = axes2[i]
            ax.set_facecolor('white')
            
            # Sort celltypes by similarity
            ct_sorted = ct_results.copy().sort_values(f'{metric}_mean', ascending=False)
            
            x_positions = range(len(ct_sorted))
            y_values = ct_sorted[f'{metric}_mean']
            error_values = ct_sorted[f'{metric}_sem'] if f'{metric}_sem' in ct_sorted.columns else None
            celltype_labels = ct_sorted['celltypes']
            
            ax.errorbar(x_positions, y_values, yerr=error_values,
                       marker='o', linewidth=0, markersize=5,
                       color=colors[i % len(colors)], capsize=2, capthick=1,
                       linestyle='none')
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(celltype_labels, rotation=45, ha='right')
            ax.set_xlabel('Cell Types (Sorted by Similarity)', fontweight='bold')
            ax.set_ylabel(f'{metric.capitalize()} Similarity', fontweight='bold')
            ax.set_title(f'{metric.capitalize()}', fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)
    
    plt.suptitle('GRN Temporal Dynamics by Cell Type (All Metrics)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_pdf:
        fig1.savefig('grn_timepoint_comparison_all_metrics.pdf', 
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        fig2.savefig('grn_celltype_comparison_all_metrics.pdf', 
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        print("✅ Saved: grn_timepoint_comparison_all_metrics.pdf")
        print("✅ Saved: grn_celltype_comparison_all_metrics.pdf")
    
    plt.show()
    
    return fig1, fig2

# Usage example:
"""
# Create individual cosine similarity plots
fig_tp, fig_ct = plot_grn_scatter_plots(tp_results, ct_results, 
                                       metric='cosine', save_pdf=True)

# Create comparison across all metrics
fig_tp_all, fig_ct_all = plot_all_metrics_comparison(tp_results, ct_results, 
                                                   metrics=['pearson', 'cosine', 'spearman'], 
                                                   save_pdf=True)
"""

# %%
figpath

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from difflib import get_close_matches
from matplotlib.patches import Patch

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set your figure save path
# figpath = "/your/path/here/"  # Change this to your desired path

# Your data (assumes tp_results and ct_results are already loaded)
metric = 'cosine'  # Change to 'pearson', 'spearman', or 'jaccard' as needed

# Your color scheme and lineage mapping
lineage_colors = {
    'CNS': '#DAA520',                    # Golden/orange
    'Endoderm': '#6A5ACD',              # Blue/purple  
    'Epiderm': '#DC143C',               # Red
    'Germline': '#DA70D6',              # Magenta/orchid
    'Lateral Mesoderm': '#228B22',      # Forest green
    'Neural Crest': '#20B2AA',          # Light sea green/teal
    'Paraxial Mesoderm': '#4169E1'      # Royal blue
}

celltype_to_lineage = {
    "CNS": ["neural", "neural_optic", "neural_posterior", "spinal_cord", "neurons", "hindbrain", "optic_cup", "differentiating_neurons", "floor_plate", "neural_floor_plate", "enteric_neurons"],
    "Neural Crest": ["neural_crest"],
    "Paraxial Mesoderm": ["somites", "fast_muscle", "muscle", "PSM", "NMPs", "tail_bud", "notochord"],
    "Lateral Mesoderm": ["lateral_plate_mesoderm", "heart_myocardium", "hematopoietic_vasculature", "pharyngeal_arches", "pronephros", "hemangioblasts", "hatching_gland"],
    "Endoderm": ["endoderm", "endocrine_pancreas"],
    "Epiderm": ["epidermis"],
    "Germline": ["primordial_germ_cells"]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_celltype_lineage(celltype_name, celltype_to_lineage):
    """Find the best matching lineage for a celltype using fuzzy matching"""
    
    # First try exact match
    for lineage, celltypes in celltype_to_lineage.items():
        if celltype_name in celltypes:
            return lineage
    
    # If no exact match, try fuzzy matching
    all_reference_celltypes = []
    lineage_lookup = {}
    
    for lineage, celltypes in celltype_to_lineage.items():
        for ct in celltypes:
            all_reference_celltypes.append(ct)
            lineage_lookup[ct] = lineage
    
    # Find closest matches (top 3, minimum 60% similarity)
    closest_matches = get_close_matches(celltype_name, all_reference_celltypes, 
                                      n=3, cutoff=0.6)
    
    if closest_matches:
        best_match = closest_matches[0]
        matched_lineage = lineage_lookup[best_match]
        print(f"  Fuzzy match: '{celltype_name}' → '{best_match}' ({matched_lineage})")
        return matched_lineage
    
    # If still no match, try partial string matching
    for lineage, celltypes in celltype_to_lineage.items():
        for ref_ct in celltypes:
            if ref_ct in celltype_name or celltype_name in ref_ct:
                print(f"  Partial match: '{celltype_name}' → '{ref_ct}' ({lineage})")
                return lineage
    
    print(f"  ⚠️ No match found for: '{celltype_name}' → using 'Other'")
    return 'Other'

# Set clean matplotlib style
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Timepoint mapping for better labels
hpf_labels = {
    'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
    'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'
}

print("🎨 Mapping celltypes to lineages...")

# ============================================================================
# PLOT 1: TIMEPOINT ANALYSIS
# ============================================================================

print("\n📊 Creating timepoint analysis plot...")

# Calculate figure width based on number of timepoints
n_timepoints = len(tp_results)
tp_width = max(6, n_timepoints * 1.2)

# Create figure
fig1, ax1 = plt.subplots(1, 1, figsize=(tp_width, 5), facecolor='white')
ax1.set_facecolor('white')

# Prepare data
x_labels = [hpf_labels.get(tp, tp) for tp in tp_results['timepoints']]
x_positions = range(len(x_labels))
y_values = tp_results[f'{metric}_mean']
error_values = tp_results[f'{metric}_sem'] if f'{metric}_sem' in tp_results.columns else None

# Create scatter plot with error bars
ax1.errorbar(x_positions, y_values, yerr=error_values,
            marker='o', linewidth=3, markersize=8,
            color='#2E86C1', capsize=5, capthick=2,)
            #markerfacecolor='#2E86C1', markeredgecolor='white', markeredgewidth=1)

# Formatting
ax1.set_xticks(x_positions)
ax1.set_xticklabels(x_labels, rotation=45, ha='right')
ax1.set_xlabel('Developmental Stage', fontweight='bold')
ax1.set_ylabel(f'{metric.capitalize()} Similarity', fontweight='bold')
ax1.set_title(f'{metric.capitalize()} Similarity Across Development\n(Celltype Similarity Within Timepoints)', 
             fontweight='bold', pad=20)

# Clean styling
ax1.spines['bottom'].set_linewidth(1.5)
ax1.spines['left'].set_linewidth(1.5)
ax1.grid(False)

plt.tight_layout()

# Save timepoint plot
plt.savefig(f'{figpath}/grn_{metric}_timepoint_scatter.pdf', 
           bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✅ Saved: {figpath}/grn_{metric}_timepoint_scatter.pdf")
plt.show()

# ============================================================================
# PLOT 2: CELLTYPE ANALYSIS WITH LINEAGE COLORS
# ============================================================================

print("\n📊 Creating celltype analysis plot with lineage colors...")

# Calculate figure width based on number of celltypes
n_celltypes = len(ct_results)
ct_width = max(12, n_celltypes * 0.4)

# Create figure
fig2, ax2 = plt.subplots(1, 1, figsize=(ct_width, 6), facecolor='white')
ax2.set_facecolor('white')

# Sort celltypes by similarity (high to low)
ct_sorted = ct_results.copy().sort_values(f'{metric}_mean', ascending=False)

# Prepare data
x_positions = range(len(ct_sorted))
y_values = ct_sorted[f'{metric}_mean']
error_values = ct_sorted[f'{metric}_sem'] if f'{metric}_sem' in ct_sorted.columns else None
celltype_labels = ct_sorted['celltypes']

# Map celltypes to lineages and get colors
point_colors = []
lineage_assignments = {}

print("Celltype → Lineage mapping:")
for ct in celltype_labels:
    lineage = find_celltype_lineage(ct, celltype_to_lineage)
    color = lineage_colors.get(lineage, '#808080')
    point_colors.append(color)
    lineage_assignments[ct] = lineage

# Create scatter plot with lineage-based colors
for i, (x, y, color, ct) in enumerate(zip(x_positions, y_values, point_colors, celltype_labels)):
    # Plot error bar in same color as point
    if error_values is not None:
        ax2.errorbar(x, y, yerr=error_values.iloc[i],
                    marker='o', markersize=6, color=color,
                    markerfacecolor=color, #markeredgecolor='white', markeredgewidth=1,
                    capsize=4, capthick=1.5, linestyle='none')
    else:
        ax2.scatter(x, y, color=color, s=40, 
                   edgecolors='white', linewidth=1)

# Formatting
ax2.set_xticks(x_positions)
ax2.set_xticklabels(celltype_labels, rotation=45, ha='right')
ax2.set_xlabel('Cell Types (Sorted: High→Low Temporal Similarity)', fontweight='bold')
ax2.set_ylabel(f'{metric.capitalize()} Similarity', fontweight='bold')
ax2.set_title(f'{metric.capitalize()} Temporal Dynamics by Cell Type\n(High = Stable Over Time, Low = Dynamic Over Time)', 
             fontweight='bold', pad=20)

# Clean styling
ax2.spines['bottom'].set_linewidth(1.5)
ax2.spines['left'].set_linewidth(1.5)
ax2.grid(False)

# Add legend for lineage colors (only for lineages that appear in the data)
unique_lineages = set(lineage_assignments.values())
legend_elements = [Patch(facecolor=lineage_colors.get(lineage, '#808080'), label=lineage) 
                  for lineage in sorted(unique_lineages) if lineage != 'Other']

if legend_elements:
    legend = ax2.legend(handles=legend_elements, loc='upper right', frameon=False, title='Lineages')
    # Set title fontweight (fix for the matplotlib error)
    legend.get_title().set_fontweight('bold')

plt.tight_layout()

# Save celltype plot
plt.savefig(f'{figpath}/grn_{metric}_celltype_scatter.pdf', 
           bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✅ Saved: {figpath}/grn_{metric}_celltype_scatter.pdf")
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

# print(f"\n📊 {metric.upper()} SIMILARITY SUMMARY:")
# print("="*50)

# print(f"Timepoint Analysis:")
# print(f"  Range: {tp_results[f'{metric}_mean'].min():.3f} - {tp_results[f'{metric}_mean'].max():.3f}")
# print(f"  14hpf (TDR128): {tp_results[tp_results['timepoints']=='TDR128'][f'{metric}_mean'].iloc[0]:.3f}")
# trend = 'Decreasing' if tp_results[f'{metric}_mean'].iloc[-1] < tp_results[f'{metric}_mean'].iloc[0] else 'Increasing'
# print(f"  Overall trend: {trend}")

# print(f"\nCelltype Analysis:")
# print(f"  Most stable: {ct_sorted['celltypes'].iloc[0]} ({ct_sorted[f'{metric}_mean'].iloc[0]:.3f})")
# print(f"  Most dynamic: {ct_sorted['celltypes'].iloc[-1]} ({ct_sorted[f'{metric}_mean'].iloc[-1]:.3f})")
# print(f"  Range: {ct_results[f'{metric}_mean'].min():.3f} - {ct_results[f'{metric}_mean'].max():.3f}")

# # Lineage assignment summary
# print(f"\n🎨 LINEAGE ASSIGNMENT SUMMARY:")
# print("="*50)
# lineage_counts = {}
# for lineage in lineage_assignments.values():
#     lineage_counts[lineage] = lineage_counts.get(lineage, 0) + 1

# for lineage, count in sorted(lineage_counts.items()):
#     color = lineage_colors.get(lineage, '#808080')
#     print(f"  {lineage}: {count} celltypes (color: {color})")

# # Stability by lineage
# print(f"\n📊 TEMPORAL STABILITY BY LINEAGE:")
# print("="*50)
# for lineage in sorted(unique_lineages):
#     if lineage != 'Other':
#         lineage_cts = [ct for ct, lin in lineage_assignments.items() if lin == lineage]
#         if lineage_cts:
#             lineage_similarities = ct_sorted[ct_sorted['celltypes'].isin(lineage_cts)][f'{metric}_mean']
#             if not lineage_similarities.empty:
#                 mean_sim = lineage_similarities.mean()
#                 std_sim = lineage_similarities.std()
#                 print(f"  {lineage}: {mean_sim:.3f} ± {std_sim:.3f}")
                
#                 # Most/least stable in this lineage
#                 lineage_data = ct_sorted[ct_sorted['celltypes'].isin(lineage_cts)]
#                 most_stable = lineage_data.iloc[0]['celltypes']
#                 most_dynamic = lineage_data.iloc[-1]['celltypes']
#                 print(f"    Most stable: {most_stable} ({lineage_data.iloc[0][f'{metric}_mean']:.3f})")
#                 print(f"    Most dynamic: {most_dynamic} ({lineage_data.iloc[-1][f'{metric}_mean']:.3f})")

# print(f"\n✅ Analysis complete! Plots saved to: {figpath}")


# %%
