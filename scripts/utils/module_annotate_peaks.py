# A module for annotating peaks using genomic annotation and Signac

# import libraries
import pandas as pd
import numpy as np
import pyranges as pr


# 1) Annotate peaks as promoter, exonic, intronic, or intergenic (Argelaguet et al. 2022)
# NOTE. Look into the ArchR on how to annotate the peaks using genomic annotation
def annotate_peaks(peaks, gtf_file, upstream_promoter=2000, downstream_promoter=100):
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