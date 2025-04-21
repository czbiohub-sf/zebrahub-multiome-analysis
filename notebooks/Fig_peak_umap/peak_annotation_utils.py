# This script is a collection of functions used to annotate the peaks using the genomic annotation
import pandas as pd
import numpy as np
import pyranges as pr
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# Annotate peaks as promoter, exonic, intronic, or intergenic
def annotate_peak_types(peaks, gtf_file, upstream_promoter=2000, downstream_promoter=100):
    """
    Annotate peaks as promoter, exonic, intronic, or intergenic
    
    Parameters:
    peaks: DataFrame with columns 'Chromosome', 'Start', 'End'
    gtf_file: Path to GTF annotation file
    upstream_promoter: bases upstream of TSS to consider as promoter (default 2000)
    downstream_promoter: bases downstream of TSS to consider as promoter (default 100)
    """
    # Create PyRanges object for peaks
    peaks_gr = pr.PyRanges(peaks)
    
    # Read GTF file
    gtf = pr.read_gtf(gtf_file)
    
    # Get genes
    genes = gtf[gtf.Feature == 'gene']
    
    # Create promoter regions based on strand
    plus_promoters = genes[genes.Strand == '+'].copy()
    minus_promoters = genes[genes.Strand == '-'].copy()
    
    # Adjust coordinates for plus strand
    plus_promoters.Start = plus_promoters.Start - upstream_promoter
    plus_promoters.End = plus_promoters.Start + downstream_promoter + upstream_promoter
    
    # Adjust coordinates for minus strand
    minus_promoters.End = minus_promoters.End + upstream_promoter
    minus_promoters.Start = minus_promoters.End - (downstream_promoter + upstream_promoter)
    
    # Combine promoters
    promoters = pr.concat([plus_promoters, minus_promoters])
    
    # Get exons
    exons = gtf[gtf.Feature == 'exon']
    
    # Initialize peak types
    peaks_df = peaks.copy()
    peaks_df['peak_type'] = 'intergenic'
    
    # Find overlaps
    promoter_peaks = peaks_gr.overlap(promoters).as_df()
    exon_peaks = peaks_gr.overlap(exons).as_df()
    gene_peaks = peaks_gr.overlap(genes).as_df()
    
    # Create sets of overlapping peaks for efficient lookup
    promoter_set = set(zip(promoter_peaks.Chromosome, promoter_peaks.Start, promoter_peaks.End))
    exon_set = set(zip(exon_peaks.Chromosome, exon_peaks.Start, exon_peaks.End))
    gene_set = set(zip(gene_peaks.Chromosome, gene_peaks.Start, gene_peaks.End))
    
    # Annotate peaks
    for idx, row in peaks_df.iterrows():
        peak_tuple = (row.Chromosome, row.Start, row.End)
        if peak_tuple in promoter_set:
            peaks_df.at[idx, 'peak_type'] = 'promoter'
        elif peak_tuple in exon_set:
            peaks_df.at[idx, 'peak_type'] = 'exonic'
        elif peak_tuple in gene_set:
            peaks_df.at[idx, 'peak_type'] = 'intronic'
    
    return peaks_df

def add_peak_annotations_to_adata(adata, annotated_peaks):
    """
    Add peak annotations to AnnData object's observation fields.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData object containing peak data
    annotated_peaks : pandas.DataFrame
        DataFrame with peak annotations including Chromosome, Start, End, and peak_type
    """
    # Add each column as observation field
    adata.obs['chromosome'] = annotated_peaks['Chromosome'].astype(str)
    adata.obs['peak_start'] = annotated_peaks['Start']
    adata.obs['peak_end'] = annotated_peaks['End']
    adata.obs['peak_type'] = annotated_peaks['peak_type']
    
def plot_chromosome_umaps(adata, output_path=None, figsize=(20, 20)):
    """
    Create a multi-panel UMAP plot colored by chromosomes.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData object with UMAP coordinates and chromosome annotations
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height) in inches
    """
    # Get unique chromosomes
    chromosomes = sorted(adata.obs['chromosome'].unique())
    n_chroms = len(chromosomes)
    
    # Calculate grid dimensions (5x5 or adjust based on number of chromosomes)
    n_rows = int(np.ceil(np.sqrt(n_chroms)))
    n_cols = n_rows
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each chromosome
    for idx, chrom in enumerate(chromosomes):
        ax = axes[idx]
        mask = adata.obs['chromosome'] == chrom
        
        # Plot UMAP
        sc.pl.umap(
            adata[mask], 
            show=False, 
            ax=ax, 
            size=1,
            title=f'Chromosome {chrom}'
        )
        
    # Remove empty subplots
    for idx in range(len(chromosomes), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()