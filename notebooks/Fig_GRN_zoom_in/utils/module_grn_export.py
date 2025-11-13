import os
import pandas as pd

# Recommended directory structure:
# grn_exports/
# ├── filtered/
# │   ├── timepoint_00_somites/
# │   │   ├── NMPs.csv
# │   │   ├── PSM.csv
# │   │   └── differentiating_neurons.csv
# │   ├── timepoint_05_somites/
# │   └── ...
# ├── unfiltered/
# │   ├── timepoint_00_somites/
# │   └── ...
# ├── metadata/
# │   ├── timepoint_mapping.csv
# │   ├── celltype_counts.csv
# │   └── export_log.txt
# └── combined/
#     ├── all_filtered_grns.csv
#     └── all_unfiltered_grns.csv

def setup_export_directories(base_dir="grn_exports"):
    """Create the directory structure for GRN exports"""
    dirs = [
        f"{base_dir}/filtered",
        f"{base_dir}/unfiltered", 
        f"{base_dir}/metadata",
        f"{base_dir}/combined"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return base_dir

def create_timepoint_mapping(dict_links, base_dir):
    """Create a mapping file for timepoint codes to somite stages"""
    # Based on your description: 0, 5, 10, 15, 20, 30 somites
    timepoint_codes = list(dict_links.keys())
    somite_stages = [0, 5, 10, 15, 20, 30]  # Adjust based on your actual stages
    
    mapping_df = pd.DataFrame({
        'timepoint_code': timepoint_codes,
        'somite_stage': somite_stages,
        'timepoint_name': [f"timepoint_{stage:02d}_somites" for stage in somite_stages]
    })
    
    mapping_df.to_csv(f"{base_dir}/metadata/timepoint_mapping.csv", index=False)
    return mapping_df

def export_grn_data(dict_links, base_dir="grn_exports"):
    """Export all GRN data with organized file structure"""
    
    # Setup directories
    setup_export_directories(base_dir)
    
    # Create timepoint mapping
    mapping_df = create_timepoint_mapping(dict_links, base_dir)
    
    # Track export statistics
    export_stats = []
    all_filtered = []
    all_unfiltered = []
    
    for timepoint_code, links_obj in dict_links.items():
        # Get the corresponding somite stage name
        timepoint_name = mapping_df[mapping_df['timepoint_code'] == timepoint_code]['timepoint_name'].iloc[0]
        somite_stage = mapping_df[mapping_df['timepoint_code'] == timepoint_code]['somite_stage'].iloc[0]
        
        # Create timepoint-specific directories
        filtered_dir = f"{base_dir}/filtered/{timepoint_name}"
        unfiltered_dir = f"{base_dir}/unfiltered/{timepoint_name}"
        os.makedirs(filtered_dir, exist_ok=True)
        os.makedirs(unfiltered_dir, exist_ok=True)
        
        # Export filtered GRNs
        if hasattr(links_obj, 'filtered_links') and links_obj.filtered_links:
            for celltype, grn_df in links_obj.filtered_links.items():
                if grn_df is not None and len(grn_df) > 0:
                    # Add metadata columns
                    grn_df_export = grn_df.copy()
                    grn_df_export['timepoint_code'] = timepoint_code
                    grn_df_export['somite_stage'] = somite_stage
                    grn_df_export['celltype'] = celltype
                    grn_df_export['grn_type'] = 'filtered'
                    
                    # Export individual file
                    filename = f"{filtered_dir}/{celltype.replace(' ', '_').replace('/', '_')}.csv"
                    grn_df_export.to_csv(filename, index=False)
                    
                    # Add to combined dataset
                    all_filtered.append(grn_df_export)
                    
                    # Track stats
                    export_stats.append({
                        'timepoint_code': timepoint_code,
                        'somite_stage': somite_stage,
                        'celltype': celltype,
                        'grn_type': 'filtered',
                        'n_edges': len(grn_df),
                        'n_tfs': grn_df['source'].nunique(),
                        'n_targets': grn_df['target'].nunique(),
                        'filename': filename
                    })
        
        # Export unfiltered GRNs
        if hasattr(links_obj, 'links_dict') and links_obj.links_dict:
            for celltype, grn_df in links_obj.links_dict.items():
                if grn_df is not None and len(grn_df) > 0:
                    # Add metadata columns
                    grn_df_export = grn_df.copy()
                    grn_df_export['timepoint_code'] = timepoint_code
                    grn_df_export['somite_stage'] = somite_stage
                    grn_df_export['celltype'] = celltype
                    grn_df_export['grn_type'] = 'unfiltered'
                    
                    # Export individual file
                    filename = f"{unfiltered_dir}/{celltype.replace(' ', '_').replace('/', '_')}.csv"
                    grn_df_export.to_csv(filename, index=False)
                    
                    # Add to combined dataset
                    all_unfiltered.append(grn_df_export)
                    
                    # Track stats
                    export_stats.append({
                        'timepoint_code': timepoint_code,
                        'somite_stage': somite_stage,
                        'celltype': celltype,
                        'grn_type': 'unfiltered',
                        'n_edges': len(grn_df),
                        'n_tfs': grn_df['source'].nunique(),
                        'n_targets': grn_df['target'].nunique(),
                        'filename': filename
                    })
    
    # Export combined datasets
    if all_filtered:
        combined_filtered = pd.concat(all_filtered, ignore_index=True)
        combined_filtered.to_csv(f"{base_dir}/combined/all_filtered_grns.csv", index=False)
    
    if all_unfiltered:
        combined_unfiltered = pd.concat(all_unfiltered, ignore_index=True)
        combined_unfiltered.to_csv(f"{base_dir}/combined/all_unfiltered_grns.csv", index=False)
    
    # Export statistics
    stats_df = pd.DataFrame(export_stats)
    stats_df.to_csv(f"{base_dir}/metadata/export_statistics.csv", index=False)
    
    # Create summary statistics
    summary_stats = stats_df.groupby(['somite_stage', 'celltype', 'grn_type']).agg({
        'n_edges': 'sum',
        'n_tfs': 'first',
        'n_targets': 'first'
    }).reset_index()
    summary_stats.to_csv(f"{base_dir}/metadata/summary_statistics.csv", index=False)
    
    print(f"Export completed! Files saved to: {base_dir}")
    print(f"Total datasets exported: {len(export_stats)}")
    print(f"Combined filtered GRNs: {len(all_filtered) if all_filtered else 0} datasets")
    print(f"Combined unfiltered GRNs: {len(all_unfiltered) if all_unfiltered else 0} datasets")
    
    return stats_df

# Usage example:
# export_stats = export_grn_data(dict_links)

# Helper functions for loading data later:

def load_grn_by_timepoint_celltype(base_dir, somite_stage, celltype, grn_type="filtered"):
    """Load a specific GRN by timepoint and celltype"""
    timepoint_name = f"timepoint_{somite_stage:02d}_somites"
    celltype_clean = celltype.replace(' ', '_').replace('/', '_')
    filepath = f"{base_dir}/{grn_type}/{timepoint_name}/{celltype_clean}.csv"
    
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print(f"File not found: {filepath}")
        return None

def load_all_grns_for_celltype(base_dir, celltype, grn_type="filtered"):
    """Load all timepoints for a specific celltype"""
    all_timepoints = []
    celltype_clean = celltype.replace(' ', '_').replace('/', '_')
    
    grn_dir = f"{base_dir}/{grn_type}"
    for timepoint_dir in os.listdir(grn_dir):
        filepath = f"{grn_dir}/{timepoint_dir}/{celltype_clean}.csv"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            all_timepoints.append(df)
    
    if all_timepoints:
        return pd.concat(all_timepoints, ignore_index=True)
    else:
        return None

def get_export_summary(base_dir):
    """Get a summary of what was exported"""
    stats_path = f"{base_dir}/metadata/export_statistics.csv"
    if os.path.exists(stats_path):
        return pd.read_csv(stats_path)
    else:
        return None