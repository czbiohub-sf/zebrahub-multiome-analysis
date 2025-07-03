import pandas as pd
import numpy as np

# Part 1. Create motif-to-factors dictionary
def create_motif_to_factors_dict(motif_factors_df):
    """
    Convert motif-factor pairs to dictionary format.
    
    Parameters:
    -----------
    motif_factors_df : pd.DataFrame
        Raw dataframe from motif2factors.txt
        
    Returns:
    --------
    motif_to_factors : dict
        {motif_id: [list_of_factors]}
    """
    
    print("Converting to motif → factors dictionary...")
    
    # Group by motif and aggregate factors into lists
    motif_to_factors = motif_factors_df.groupby('Motif')['Factor'].apply(list).to_dict()
    
    print(f"Created mapping for {len(motif_to_factors)} motifs")
    
    # Show some examples
    print(f"\nExample mappings:")
    for i, (motif, factors) in enumerate(list(motif_to_factors.items())[:3]):
        print(f"  {motif}: {factors[:5]}{'...' if len(factors) > 5 else ''} ({len(factors)} total)")
    
    return motif_to_factors


def create_motif_factors_dataframe(motif_factors_df):
    """
    Create a clean dataframe with motifs as index and factors as a column.
    
    Parameters:
    -----------
    motif_factors_df : pd.DataFrame
        Raw dataframe from motif2factors.txt
        
    Returns:
    --------
    motifs_df : pd.DataFrame
        Clean dataframe with motif as index, 'factors' column containing lists
    """
    
    print("Creating clean motifs dataframe...")
    
    # Group by motif and create aggregated dataframe
    motif_groups = motif_factors_df.groupby('Motif').agg({
        'Factor': list,  # List of all factors
        'Evidence': lambda x: list(set(x)),  # Unique evidence types
        'Curated': lambda x: list(set(x))   # Unique curated values
    }).reset_index()
    
    # Rename columns for clarity
    motif_groups.columns = ['motif', 'factors', 'evidence_types', 'curated_status']
    
    # Set motif as index
    motifs_df = motif_groups.set_index('motif')
    
    # Add summary statistics
    motifs_df['n_factors'] = motifs_df['factors'].apply(len)
    
    print(f"Created dataframe: {motifs_df.shape}")
    print(f"Factors per motif - Mean: {motifs_df['n_factors'].mean():.1f}, "
          f"Median: {motifs_df['n_factors'].median():.0f}, "
          f"Range: {motifs_df['n_factors'].min()}-{motifs_df['n_factors'].max()}")
    
    print(f"\nDataframe structure:")
    print(motifs_df.head())
    
    return motifs_df