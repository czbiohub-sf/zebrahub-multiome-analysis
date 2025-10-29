# module for exporting peaks for GimmeMotifs

import os
import pandas as pd

# export peaks for GimmeMotifs maelstrom
def export_peaks_for_gimmemotifs(adata_peaks, cluster="leiden", out_dir=None, out_name=None):
    """
    Export peaks in a format suitable for GimmeMotifs.

    Parameters:
    -----------
    adata_peaks : AnnData
        AnnData object containing peak information.
    cluster : str, default="leiden"
        Column in `adata_peaks.obs` specifying cluster labels.
    out_dir : str or None, default=None
        Directory where the output file will be saved. If None, the file will not be saved.
    out_name : str or None, default=None
        Name of the output file. If None, the file will not be saved.

    Returns:
    --------
    export_df : pd.DataFrame
        DataFrame containing the formatted peak locations and cluster labels.
    """
    # Format peak locations (assuming 'chr' prefix is needed)
    export_df = pd.DataFrame({
        'loc': 'chr' + adata_peaks.obs_names.str.replace('-', ':', 1).str.replace('-', '-', 1),
        'cluster': adata_peaks.obs[cluster]  # Use the specified cluster column
    })

    # Remove potential duplicates
    export_df = export_df.drop_duplicates()

    # Save file only if out_dir is specified
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        output_file = os.path.join(out_dir, f'peaks_{out_name}.txt')
        export_df.to_csv(output_file, sep='\t', index=False)
        print(f"Exported {len(export_df)} peaks to {output_file}")
        print("Clusters present:", export_df['cluster'].unique())
    
    return export_df
