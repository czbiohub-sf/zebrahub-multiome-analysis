# A module to annotate the peaks with associated genes
import os
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches


# define a function to associate the peaks with "genes"
# we will prioritize the genes on the following order:
# (1) linked genes (high correlation with RNA), (2) overlapping with gene body, (3) the nearest gene (within 50kb), otherwise None
def create_gene_associations(adata):
    """
    Create associated_gene and association_type columns based on priority:
    1. linked_gene (if available)
    2. gene_body_overlaps (if linked_gene is empty, if multiple, pick the nearest gene)
    [DEPRECATED] 3. nearest_gene (if both above are empty)
    4. None (if all are empty)
    """
    
    # Create copies of the relevant columns for easier manipulation
    linked_gene = adata.obs['linked_gene'].copy()
    gene_body_overlaps = adata.obs['gene_body_overlaps'].copy()
    nearest_gene = adata.obs['nearest_gene'].copy()
    
    # Initialize the new columns
    associated_gene = pd.Series(index=adata.obs.index, dtype='object')
    association_type = pd.Series(index=adata.obs.index, dtype='object')
    
    # Helper function to check if a value is empty/null
    def is_empty(value):
        if pd.isna(value):
            return True
        if isinstance(value, str) and (value.strip() == '' or value.lower() in ['nan', 'none', 'null']):
            return True
        return False
    
    # Apply priority logic
    for idx in adata.obs.index:
        # Priority 1: linked_gene
        if not is_empty(linked_gene.loc[idx]):
            associated_gene.loc[idx] = linked_gene.loc[idx]
            association_type.loc[idx] = 'linked'
        
        # Priority 2: gene_body_overlaps
        elif not is_empty(gene_body_overlaps.loc[idx]):
            overlap_genes = gene_body_overlaps.loc[idx]
            # Check if there are multiple genes (comma-separated)
            if ',' in str(overlap_genes):
                # Multiple genes in overlap - fall back to nearest_gene
                if not is_empty(nearest_gene.loc[idx]):
                    associated_gene.loc[idx] = nearest_gene.loc[idx]
                    association_type.loc[idx] = 'overlap'  # From overlap fallback
                else:
                    # If no nearest gene, take the first overlap gene
                    first_gene = str(overlap_genes).split(',')[0].strip()
                    associated_gene.loc[idx] = first_gene
                    association_type.loc[idx] = 'overlap'
            else:
                # Single gene in overlap
                associated_gene.loc[idx] = overlap_genes
                association_type.loc[idx] = 'overlap'
        
        # # Priority 3: nearest_gene
        # elif not is_empty(nearest_gene.loc[idx]):
        #     associated_gene.loc[idx] = nearest_gene.loc[idx]
        #     association_type.loc[idx] = 'nearest'
        
        # Priority 4: None
        else:
            associated_gene.loc[idx] = None
            association_type.loc[idx] = 'none'
    
    # Add the new columns to adata_sub.obs
    adata.obs['associated_gene'] = associated_gene
    adata.obs['association_type'] = association_type.astype('category')
    
    return adata

# analyze the gene associations
def analyze_gene_associations(adata, subclust_key):
    """
    Analyze different gene association strategies for all subclusters
    """
    results = []
    
    for subclust in sorted(adata.obs[subclust_key].unique()):
        peaks_sub = adata[adata.obs[subclust_key] == subclust]
        n_peaks = len(peaks_sub)
        
        # Strategy 1: Linked genes only
        linked_genes = peaks_sub.obs['linked_gene'].dropna().unique()
        linked_clean = [g for g in linked_genes if isinstance(g, str) and g.strip() != '' and '/' not in g]
        
        # Strategy 2: Linked + Overlap genes
        linked_overlap_mask = peaks_sub.obs['association_type'].isin(['linked', 'overlap'])
        linked_overlap_peaks = peaks_sub[linked_overlap_mask]
        linked_overlap_genes = linked_overlap_peaks.obs['associated_gene'].dropna().unique()
        linked_overlap_clean = [g for g in linked_overlap_genes if isinstance(g, str) and g.strip() != '' and '/' not in g]
        
        # Strategy 3: All associations (linked + overlap + nearest)
        all_genes = peaks_sub.obs['associated_gene'].dropna().unique()
        all_clean = [g for g in all_genes if isinstance(g, str) and g.strip() != '' and '/' not in g]
        
        # Count peaks with each association type
        assoc_counts = peaks_sub.obs['association_type'].value_counts()
        
        results.append({
            'subcluster': subclust,
            'n_peaks': n_peaks,
            'linked_genes': len(linked_clean),
            'linked_overlap_genes': len(linked_overlap_clean),
            'all_genes': len(all_clean),
            'linked_peaks': assoc_counts.get('linked', 0),
            'overlap_peaks': assoc_counts.get('overlap', 0),
            'nearest_peaks': assoc_counts.get('nearest', 0),
            'none_peaks': assoc_counts.get('none', 0),
            'linked_gene_list': linked_clean,
            'linked_overlap_gene_list': linked_overlap_clean,
            'all_gene_list': all_clean
        })
    
    return pd.DataFrame(results)