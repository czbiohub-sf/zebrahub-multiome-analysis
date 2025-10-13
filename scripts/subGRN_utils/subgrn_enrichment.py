"""
TF enrichment analysis for similarity blocks

This module provides functions for analyzing transcription factor enrichment
in similarity blocks, identifying block-defining TFs, and distinguishing
shared vs block-specific regulatory programs.

Author: Extracted from EDA_extract_subGRN_reg_programs_Take2.py
Date: 2025-01-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hypergeom, chi2_contingency
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)


def analyze_tf_enrichment_in_blocks(cluster_tf_matrix: pd.DataFrame,
                                    blocks_data: Dict[str, List[str]],
                                    min_frequency: float = 0.3,
                                    min_enrichment_ratio: float = 1.5,
                                    max_tfs_per_block: int = 15,
                                    statistical_test: str = 'hypergeometric') -> Dict:
    """
    Analyze TF enrichment in similarity blocks using hypergeometric test

    Parameters
    ----------
    cluster_tf_matrix : pd.DataFrame
        Binary matrix (clusters × TFs)
    blocks_data : Dict[str, List[str]]
        Dictionary with block_name -> list of cluster IDs
    min_frequency : float, default=0.3
        Minimum frequency of TF within block to consider
    min_enrichment_ratio : float, default=1.5
        Minimum enrichment ratio vs background
    max_tfs_per_block : int, default=15
        Maximum TFs to report per block
    statistical_test : str, default='hypergeometric'
        Statistical test: 'hypergeometric', 'chi2', or 'none'

    Returns
    -------
    enrichment_results : Dict
        Dictionary with block-specific enrichment data:
        {block_name: {'top_tfs': DataFrame, 'all_significant': DataFrame,
                      'block_size': int, 'valid_clusters': List}}

    Examples
    --------
    >>> matrix = pd.DataFrame(np.random.randint(0, 2, (100, 50)))
    >>> blocks = {'Block1': ['0_0', '0_1', '0_2'], 'Block2': ['1_0', '1_1']}
    >>> results = analyze_tf_enrichment_in_blocks(matrix, blocks)
    >>> print(f"Found enriched TFs in {len(results)} blocks")
    """
    logger.info("=== TF ENRICHMENT ANALYSIS FOR SIMILARITY BLOCKS ===")

    # Verify cluster availability
    available_clusters = set(cluster_tf_matrix.index)
    all_analysis_clusters = set()
    for clusters in blocks_data.values():
        all_analysis_clusters.update(clusters)

    missing_clusters = all_analysis_clusters - available_clusters
    if missing_clusters:
        logger.warning(f"{len(missing_clusters)} clusters not found in TF matrix")
        logger.warning(f"Missing: {list(missing_clusters)[:10]}{'...' if len(missing_clusters) > 10 else ''}")

    # Calculate background TF frequencies
    background_tf_freq = cluster_tf_matrix.mean(axis=0)
    total_clusters = len(cluster_tf_matrix)

    logger.info(f"Background: {total_clusters} total clusters, {len(background_tf_freq)} TFs")

    # Analyze each block
    enrichment_results = {}

    for block_name, cluster_list in blocks_data.items():
        logger.info(f"\n--- Analyzing {block_name} ({len(cluster_list)} clusters) ---")

        # Get clusters that exist in TF matrix
        valid_clusters = [c for c in cluster_list if c in available_clusters]
        logger.info(f"Valid clusters in TF matrix: {len(valid_clusters)}/{len(cluster_list)}")

        if len(valid_clusters) < 2:
            logger.info(f"Skipping {block_name}: insufficient valid clusters")
            continue

        # Extract TF matrix for this block
        block_tf_matrix = cluster_tf_matrix.loc[valid_clusters]
        block_size = len(valid_clusters)

        # Calculate TF frequencies within block
        block_tf_freq = block_tf_matrix.mean(axis=0)
        block_tf_counts = block_tf_matrix.sum(axis=0)

        # Calculate enrichment metrics
        enrichment_data = []

        for tf in cluster_tf_matrix.columns:
            # Frequencies
            freq_in_block = block_tf_freq[tf]
            freq_background = background_tf_freq[tf]

            # Counts for statistical testing
            count_in_block = block_tf_counts[tf]
            count_in_background = cluster_tf_matrix[tf].sum()

            # Skip if too infrequent in block
            if freq_in_block < min_frequency:
                continue

            # Calculate enrichment ratio
            enrichment_ratio = freq_in_block / freq_background if freq_background > 0 else float('inf')

            # Statistical testing
            p_value = 1.0
            if statistical_test == 'hypergeometric':
                # Hypergeometric test: is this TF overrepresented?
                p_value = hypergeom.sf(
                    count_in_block - 1,  # observed - 1 (for survival function)
                    total_clusters,      # population size
                    count_in_background, # successes in population
                    block_size          # sample size
                )
            elif statistical_test == 'chi2':
                # Chi-square test for independence
                contingency_table = [
                    [count_in_block, block_size - count_in_block],
                    [count_in_background - count_in_block,
                     total_clusters - block_size - (count_in_background - count_in_block)]
                ]
                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                except:
                    p_value = 1.0

            enrichment_data.append({
                'tf': tf,
                'freq_in_block': freq_in_block,
                'freq_background': freq_background,
                'enrichment_ratio': enrichment_ratio,
                'count_in_block': count_in_block,
                'count_background': count_in_background,
                'p_value': p_value,
                'neg_log_p': -np.log10(p_value) if p_value > 0 else 10
            })

        # Sort by enrichment ratio and significance
        enrichment_df = pd.DataFrame(enrichment_data)

        if len(enrichment_df) > 0:
            # Filter by enrichment ratio
            significant_tfs = enrichment_df[
                (enrichment_df['enrichment_ratio'] >= min_enrichment_ratio) &
                (enrichment_df['freq_in_block'] >= min_frequency)
            ].copy()

            # Sort by combined score (enrichment × significance)
            significant_tfs['combined_score'] = significant_tfs['enrichment_ratio'] * significant_tfs['neg_log_p']
            significant_tfs = significant_tfs.sort_values('combined_score', ascending=False)

            # Keep top TFs
            top_tfs = significant_tfs.head(max_tfs_per_block)

            enrichment_results[block_name] = {
                'top_tfs': top_tfs,
                'all_significant': significant_tfs,
                'block_size': block_size,
                'valid_clusters': valid_clusters
            }

            logger.info(f"Top {len(top_tfs)} enriched TFs:")
            for _, row in top_tfs.iterrows():
                logger.info(f"  {row['tf']}: {row['freq_in_block']:.2%} (vs {row['freq_background']:.2%} bg), "
                          f"ratio={row['enrichment_ratio']:.2f}, p={row['p_value']:.2e}")
        else:
            logger.info("No TFs passed enrichment criteria")
            enrichment_results[block_name] = None

    return enrichment_results


def visualize_tf_enrichment(enrichment_results: Dict, top_n: int = 10) -> None:
    """
    Create comprehensive visualizations of TF enrichment

    Parameters
    ----------
    enrichment_results : Dict
        Output from analyze_tf_enrichment_in_blocks()
    top_n : int, default=10
        Number of top TFs to show per block

    Examples
    --------
    >>> results = analyze_tf_enrichment_in_blocks(matrix, blocks)
    >>> visualize_tf_enrichment(results, top_n=8)
    """
    logger.info("=== CREATING TF ENRICHMENT VISUALIZATIONS ===")

    # Collect data for visualization
    plot_data = []
    all_top_tfs = set()

    for block_name, result in enrichment_results.items():
        if result is not None:
            top_tfs = result['top_tfs'].head(top_n)
            for _, row in top_tfs.iterrows():
                plot_data.append({
                    'Block': block_name,
                    'TF': row['tf'],
                    'Frequency': row['freq_in_block'],
                    'Background': row['freq_background'],
                    'Enrichment_Ratio': row['enrichment_ratio'],
                    'Neg_Log_P': row['neg_log_p'],
                    'Combined_Score': row['combined_score']
                })
                all_top_tfs.add(row['tf'])

    if not plot_data:
        logger.info("No enrichment data available for visualization")
        return

    df_plot = pd.DataFrame(plot_data)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))

    # 1. Heatmap of TF frequencies across blocks
    ax1 = plt.subplot(3, 3, (1, 2))

    # Create matrix for heatmap
    tf_block_matrix = df_plot.pivot_table(
        index='TF', columns='Block', values='Frequency', fill_value=0
    )

    # Sort TFs by maximum enrichment across blocks
    tf_max_enrichment = df_plot.groupby('TF')['Enrichment_Ratio'].max()
    tf_order = tf_max_enrichment.sort_values(ascending=False).index
    tf_block_matrix = tf_block_matrix.reindex(tf_order)

    sns.heatmap(tf_block_matrix, annot=True, fmt='.2f', cmap='Reds',
                ax=ax1, cbar_kws={'label': 'TF Frequency'})
    ax1.set_title('TF Frequency Across Blocks', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Similarity Block')
    ax1.set_ylabel('Transcription Factor')

    # 2. Enrichment ratio heatmap
    ax2 = plt.subplot(3, 3, (3, 3))

    enrichment_matrix = df_plot.pivot_table(
        index='TF', columns='Block', values='Enrichment_Ratio', fill_value=1
    )
    enrichment_matrix = enrichment_matrix.reindex(tf_order)

    sns.heatmap(enrichment_matrix, annot=True, fmt='.1f', cmap='RdYlBu_r',
                center=1, ax=ax2, cbar_kws={'label': 'Enrichment Ratio'})
    ax2.set_title('TF Enrichment Ratios', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Similarity Block')
    ax2.set_ylabel('')

    # 3. Top TFs by block (bar plots)
    for i, (block_name, result) in enumerate(enrichment_results.items()):
        if result is None:
            continue

        ax = plt.subplot(3, 3, 4 + i)

        top_tfs = result['top_tfs'].head(8)  # Top 8 for space

        if len(top_tfs) > 0:
            bars = ax.barh(range(len(top_tfs)), top_tfs['enrichment_ratio'],
                          color='steelblue', alpha=0.7)

            # Add frequency annotations
            for j, (_, row) in enumerate(top_tfs.iterrows()):
                ax.text(row['enrichment_ratio'] + 0.1, j,
                       f"{row['freq_in_block']:.1%}",
                       va='center', fontsize=8)

            ax.set_yticks(range(len(top_tfs)))
            ax.set_yticklabels(top_tfs['tf'], fontsize=9)
            ax.set_xlabel('Enrichment Ratio')
            ax.set_title(f'{block_name} Top TFs', fontweight='bold', fontsize=12)
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No enriched TFs', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'{block_name} Top TFs', fontweight='bold', fontsize=12)

        # Limit subplots to available blocks
        if i >= 4:  # Only show first 5 blocks
            break

    plt.suptitle('TF Enrichment Analysis Across Similarity Blocks',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig("tf_enrichment_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


def create_block_tf_summary(enrichment_results: Dict,
                            blocks_data: Dict[str, List[str]]) -> Dict:
    """
    Create comprehensive summary of block-defining TFs

    Parameters
    ----------
    enrichment_results : Dict
        Output from analyze_tf_enrichment_in_blocks()
    blocks_data : Dict[str, List[str]]
        Original block cluster lists

    Returns
    -------
    summary : Dict
        Summary information per block

    Examples
    --------
    >>> results = analyze_tf_enrichment_in_blocks(matrix, blocks)
    >>> summary = create_block_tf_summary(results, blocks)
    >>> print(summary['Block1']['top_tfs'])
    """
    logger.info("=== BLOCK-DEFINING TF SUMMARY ===")

    summary = {}

    for block_name, result in enrichment_results.items():
        if result is None:
            continue

        top_tfs = result['top_tfs']
        block_size = result['block_size']

        if len(top_tfs) > 0:
            # Get top 5 TFs for summary
            top_5 = top_tfs.head(5)

            tf_summary = []
            for _, row in top_5.iterrows():
                tf_info = f"{row['tf']} ({row['freq_in_block']:.1%}, {row['enrichment_ratio']:.1f}×)"
                tf_summary.append(tf_info)

            # Identify highly specific TFs (high enrichment, low background)
            highly_specific = top_tfs[
                (top_tfs['enrichment_ratio'] > 3.0) &
                (top_tfs['freq_background'] < 0.1)
            ]

            summary[block_name] = {
                'size': block_size,
                'top_tfs': tf_summary,
                'n_enriched': len(top_tfs),
                'highly_specific': list(highly_specific['tf'].head(3)),
                'best_tf': top_tfs.iloc[0]['tf'] if len(top_tfs) > 0 else None,
                'best_enrichment': top_tfs.iloc[0]['enrichment_ratio'] if len(top_tfs) > 0 else 0
            }

            logger.info(f"\n{block_name} ({block_size} clusters):")
            logger.info(f"  Block-defining TFs: {', '.join(tf_summary[:3])}")
            if len(highly_specific) > 0:
                logger.info(f"  Highly specific TFs: {', '.join(list(highly_specific['tf'].head(3)))}")
            logger.info(f"  Total enriched TFs: {len(top_tfs)}")
        else:
            summary[block_name] = {'size': block_size, 'no_enrichment': True}
            logger.info(f"\n{block_name}: No significantly enriched TFs")

    return summary


def find_shared_vs_specific_tfs(enrichment_results: Dict) -> Tuple[Dict, Dict]:
    """
    Identify TFs that are shared across blocks vs block-specific

    Parameters
    ----------
    enrichment_results : Dict
        Output from analyze_tf_enrichment_in_blocks()

    Returns
    -------
    shared_tfs : Dict[str, List[str]]
        TFs enriched in multiple blocks: {tf: [block1, block2, ...]}
    specific_tfs : Dict[str, List[str]]
        Block-specific TFs: {block: [tf1, tf2, ...]}

    Examples
    --------
    >>> results = analyze_tf_enrichment_in_blocks(matrix, blocks)
    >>> shared, specific = find_shared_vs_specific_tfs(results)
    >>> print(f"Found {len(shared)} shared TFs")
    """
    logger.info("=== SHARED vs BLOCK-SPECIFIC TF ANALYSIS ===")

    # Collect all enriched TFs by block
    block_tfs = {}
    all_enriched_tfs = set()

    for block_name, result in enrichment_results.items():
        if result is not None and len(result['top_tfs']) > 0:
            block_enriched = set(result['top_tfs']['tf'])
            block_tfs[block_name] = block_enriched
            all_enriched_tfs.update(block_enriched)

    # Categorize TFs
    tf_block_membership = defaultdict(list)
    for tf in all_enriched_tfs:
        for block_name, tf_set in block_tfs.items():
            if tf in tf_set:
                tf_block_membership[tf].append(block_name)

    # Categorize by sharing pattern
    shared_tfs = {}  # TF -> list of blocks
    specific_tfs = {}  # Block -> list of specific TFs

    for tf, blocks in tf_block_membership.items():
        if len(blocks) > 1:
            shared_tfs[tf] = blocks
        else:
            block_name = blocks[0]
            if block_name not in specific_tfs:
                specific_tfs[block_name] = []
            specific_tfs[block_name].append(tf)

    logger.info(f"Shared TFs (enriched in multiple blocks): {len(shared_tfs)}")
    for tf, blocks in list(shared_tfs.items())[:10]:  # Show top 10
        logger.info(f"  {tf}: {', '.join(blocks)}")

    logger.info("\nBlock-specific TFs:")
    for block_name, tf_list in specific_tfs.items():
        logger.info(f"  {block_name}: {len(tf_list)} specific TFs")
        logger.info(f"    {', '.join(tf_list[:5])}{'...' if len(tf_list) > 5 else ''}")

    return shared_tfs, specific_tfs


def create_enrichment_ranking_table(enrichment_results: Dict,
                                    output_file: str = "tf_enrichment_ranking.csv") -> Optional[pd.DataFrame]:
    """
    Create comprehensive ranking table of all enriched TFs

    Parameters
    ----------
    enrichment_results : Dict
        Output from analyze_tf_enrichment_in_blocks()
    output_file : str, default="tf_enrichment_ranking.csv"
        Output CSV filename

    Returns
    -------
    df_ranked : pd.DataFrame or None
        Ranked enrichment table sorted by combined score

    Examples
    --------
    >>> results = analyze_tf_enrichment_in_blocks(matrix, blocks)
    >>> ranked = create_enrichment_ranking_table(results, "my_ranking.csv")
    >>> print(ranked.head())
    """
    logger.info("=== CREATING COMPREHENSIVE TF RANKING ===")

    all_enrichment_data = []

    for block_name, result in enrichment_results.items():
        if result is not None and len(result['top_tfs']) > 0:
            for _, row in result['top_tfs'].iterrows():
                all_enrichment_data.append({
                    'Block': block_name,
                    'TF': row['tf'],
                    'Frequency_in_Block': row['freq_in_block'],
                    'Background_Frequency': row['freq_background'],
                    'Enrichment_Ratio': row['enrichment_ratio'],
                    'Count_in_Block': row['count_in_block'],
                    'P_Value': row['p_value'],
                    'Neg_Log_P': row['neg_log_p'],
                    'Combined_Score': row['combined_score']
                })

    df_all_enrichment = pd.DataFrame(all_enrichment_data)

    if len(df_all_enrichment) > 0:
        # Sort by combined score
        df_ranked = df_all_enrichment.sort_values('Combined_Score', ascending=False)

        # Save to CSV
        df_ranked.to_csv(output_file, index=False)
        logger.info(f"Comprehensive ranking saved to: {output_file}")

        # Show top overall TFs
        logger.info("\nTop 10 TFs across all blocks (by combined score):")
        for i, (_, row) in enumerate(df_ranked.head(10).iterrows()):
            logger.info(f"  {i+1:2d}. {row['TF']} ({row['Block']}): "
                      f"{row['Frequency_in_Block']:.1%} freq, {row['Enrichment_Ratio']:.1f}× enrichment")

        return df_ranked
    else:
        logger.info("No enrichment data to rank")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Module: subgrn_enrichment.py")
    print("Contains functions for TF enrichment analysis in similarity blocks")
