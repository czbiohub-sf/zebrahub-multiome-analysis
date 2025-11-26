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
    
# associate peaks to genes based on genomic proximity
def associate_peaks_to_genes(adata_peaks, gtf_file, max_distance=50000, chunk_size=1000):
    """
    Associate peaks with genes using chunked processing to reduce memory usage
    
    Parameters:
    -----------
    adata_peaks : AnnData
        AnnData object containing peak information
    gtf_file : str
        Path to GTF file
    max_distance : int
        Maximum distance to consider for nearest gene association
    chunk_size : int
        Number of peaks to process in each chunk
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with gene associations
    """
    import pyranges as pr
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    
    # Create DataFrame from obs_names (which contain the coordinates)
    print("Preparing peak coordinates...")
    coords = [x.split('-') for x in adata_peaks.obs.index]
    peaks_df = pd.DataFrame(coords, columns=['Chromosome', 'Start', 'End'])
    
    # Convert Start and End to integers
    peaks_df['Start'] = peaks_df['Start'].astype(int)
    peaks_df['End'] = peaks_df['End'].astype(int)
    
    # Add peak_type from obs if it exists
    if 'peak_type' in adata_peaks.obs:
        peaks_df['peak_type'] = adata_peaks.obs['peak_type']
    
    # Create PyRanges object for peaks
    print("Initializing PyRanges...")
    peaks_gr = pr.PyRanges(peaks_df)
    
    # Read GTF file and get genes
    print("Reading GTF file...")
    gtf = pr.read_gtf(gtf_file)
    genes = gtf[gtf.Feature == 'gene']
    
    print("Processing TSS coordinates...")
    # Create TSS coordinates for + and - strands separately
    plus_tss = genes[genes.Strand == '+'].copy()
    plus_tss.End = plus_tss.Start + 1
    
    minus_tss = genes[genes.Strand == '-'].copy()
    minus_tss.Start = minus_tss.End - 1
    
    # Combine TSS coordinates
    tss = pr.concat([plus_tss, minus_tss])
    
    # Initialize results DataFrame
    result_df = peaks_df.copy()
    result_df['gene_body_overlaps'] = ''
    result_df['nearest_gene'] = ''
    result_df['distance_to_tss'] = np.nan
    
    # Process in chunks
    total_chunks = len(peaks_df) // chunk_size + (1 if len(peaks_df) % chunk_size else 0)
    
    for chunk_idx in tqdm(range(total_chunks), desc="Processing peaks in chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(peaks_df))
        
        # Get chunk of peaks
        chunk_peaks = peaks_df.iloc[start_idx:end_idx]
        chunk_gr = pr.PyRanges(chunk_peaks)
        
        # Find overlaps for this chunk
        gene_overlaps = chunk_gr.join(genes, suffix='_gene', apply_strand_suffix=False)
        nearest_tss = chunk_gr.nearest(tss, suffix='_tss', apply_strand_suffix=False)
        
        # Process gene body overlaps
        if not gene_overlaps.empty:
            overlaps_df = gene_overlaps.as_df()
            overlaps_df['gene_name'] = overlaps_df['gene_name'].astype(str)
            
            # Group by peak coordinates
            for _, peak_group in overlaps_df.groupby(['Chromosome', 'Start', 'End']):
                peak_genes = ','.join(set(peak_group['gene_name']))
                idx = result_df[
                    (result_df['Chromosome'] == peak_group['Chromosome'].iloc[0]) &
                    (result_df['Start'] == peak_group['Start'].iloc[0]) &
                    (result_df['End'] == peak_group['End'].iloc[0])
                ].index
                result_df.loc[idx, 'gene_body_overlaps'] = peak_genes
        
        # Process nearest TSS
        if not nearest_tss.empty:
            nearest_df = nearest_tss.as_df()
            nearest_df['gene_name'] = nearest_df['gene_name'].astype(str)
            
            for _, row in nearest_df.iterrows():
                peak_center = (row['Start'] + row['End']) // 2
                # Use Strand (not Strand_tss) since apply_strand_suffix=False
                tss_pos = row['Start_tss'] if row['Strand'] == '+' else row['End_tss']
                distance = abs(peak_center - tss_pos)
                
                if distance <= max_distance:
                    idx = result_df[
                        (result_df['Chromosome'] == row['Chromosome']) &
                        (result_df['Start'] == row['Start']) &
                        (result_df['End'] == row['End'])
                    ].index
                    # Use gene_name (not gene_name_tss) since apply_strand_suffix=False
                    result_df.loc[idx, 'nearest_gene'] = row['gene_name']
                    result_df.loc[idx, 'distance_to_tss'] = distance
    
    # Reset the index to match the original AnnData
    result_df.index = adata_peaks.obs_names
    
    return result_df