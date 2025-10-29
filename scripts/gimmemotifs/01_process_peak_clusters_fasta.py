# A script for computing de novo motifs per cluster using gimmemotifs
import os
import scanpy as sc
from gimmemotifs.denovo import gimme_motifs
import celloracle as co
from celloracle import motif_analysis as ma
import sys
from atac_seq_motif_analysis import ATACSeqMotifAnalysis

def setup_gimmemotifs_config():
    """Setup GimmeMotifs configuration"""
    from gimmemotifs.config import MotifConfig
    config = MotifConfig()
    config.set_motif_dir("/hpc/mydata/yang-joon.kim/genomes/danRer11/")
    config.set_bg_dir("/hpc/mydata/yang-joon.kim/genomes/danRer11/")
    config.write(open(config.user_config, "w"))
    return config

def prepare_peaks_by_cluster(adata, cluster_key='leiden'):
    """Generate dictionary of peaks for each cluster"""
    cluster_peaks = {}
    for cluster in adata.obs[cluster_key].unique():
        # Get peaks for this cluster
        cluster_mask = adata.obs[cluster_key] == cluster
        cluster_peaks[cluster] = adata[cluster_mask].obs_names.tolist()
    return cluster_peaks

def save_cluster_fasta(peaks, cluster_id, output_dir, atac_analysis):
    """Convert peaks to FASTA and save for a specific cluster"""
    # Convert peaks to DataFrame
    peaks_df = atac_analysis.list_peakstr_to_df(peaks)
    
    # Validate peaks
    valid_peaks = atac_analysis.check_peak_format(peaks_df)
    
    # Convert to FASTA
    fasta = atac_analysis.peak_to_fasta(
        valid_peaks["chr"] + "-" + 
        valid_peaks["start"].astype(str) + "-" + 
        valid_peaks["end"].astype(str)
    )
    
    # Remove zero-length sequences
    filtered_fasta = atac_analysis.remove_zero_seq(fasta)
    
    # Create cluster-specific directory
    cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Save FASTA file
    fasta_path = os.path.join(cluster_dir, f"peaks_cluster_{cluster_id}.fasta")
    with open(fasta_path, "w") as f:
        for name, seq in zip(filtered_fasta.ids, filtered_fasta.seqs):
            f.write(f">{name}\n{seq}\n")
    
    return fasta_path

def main():
    # Setup paths
    input_h5ad = "/path/to/your/peaks_by_pseudobulk.h5ad"  # Update this path
    output_base_dir = "/path/to/output/directory"  # Update this path
    genomes_dir = "/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/fasta/"
    
    # Initialize analysis
    ref_genome = "danRer11"
    atac_analysis = ATACSeqMotifAnalysis(ref_genome, genomes_dir)
    
    # Read data
    print("Reading input data...")
    adata = sc.read_h5ad(input_h5ad)
    
    # Setup GimmeMotifs
    setup_gimmemotifs_config()
    
    # Get peaks by cluster
    print("Processing peaks by cluster...")
    cluster_peaks = prepare_peaks_by_cluster(adata, cluster_key='leiden')
    
    # Process each cluster
    cluster_info = []
    for cluster_id, peaks in cluster_peaks.items():
        print(f"Processing cluster {cluster_id}...")
        
        # Save FASTA file for this cluster
        fasta_path = save_cluster_fasta(peaks, cluster_id, output_base_dir, atac_analysis)
        
        # Store information for SLURM submission
        cluster_info.append({
            'cluster_id': cluster_id,
            'fasta_path': fasta_path,
            'output_dir': os.path.join(output_base_dir, f"cluster_{cluster_id}")
        })
    
    # Save cluster information for SLURM submission
    import json
    with open(os.path.join(output_base_dir, 'cluster_info.json'), 'w') as f:
        json.dump(cluster_info, f)

if __name__ == "__main__":
    main()