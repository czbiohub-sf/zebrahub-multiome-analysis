import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad

print('Creating synthetic dataset for chromatin velocity testing...')
np.random.seed(42)

# Parameters
n_peaks = 500
n_pseudobulks = 50 
n_clusters = 5
peaks_per_cluster = n_peaks // n_clusters

# 1. Create peak names (mimic chromosome-start-end format)
peak_names = []
for i in range(n_peaks):
    chrom = np.random.choice(['chr1', 'chr2', 'chr3', 'chr4', 'chr5'])
    start = np.random.randint(1000000, 50000000)
    end = start + np.random.randint(200, 2000)
    peak_names.append(f'{chrom}-{start}-{end}')

print(f'Generated {len(peak_names)} peak names')

# 2. Create pseudobulk names
celltypes = ['neural', 'muscle', 'endoderm', 'mesoderm', 'ectoderm']
timepoints = ['0hr', '4hr', '8hr', '12hr', '16hr', '20hr', '24hr', '30hr', '36hr', '48hr']
pseudobulk_names = []

for i in range(n_pseudobulks):
    celltype = np.random.choice(celltypes)
    timepoint = np.random.choice(timepoints)
    pseudobulk_names.append(f'{celltype}_{timepoint}_{i}')

print(f'Generated {len(pseudobulk_names)} pseudobulk names')

# 3. Create clustered accessibility matrix
accessibility_matrix = np.random.lognormal(0, 0.5, (n_pseudobulks, n_peaks))

# Add cluster structure
for cluster_id in range(n_clusters):
    peak_start = cluster_id * peaks_per_cluster
    peak_end = peak_start + peaks_per_cluster
    
    # Select pseudobulks that have high accessibility for this cluster
    high_access_pseudobulks = np.random.choice(n_pseudobulks, size=n_pseudobulks//3, replace=False)
    
    # Increase accessibility for this cluster in selected pseudobulks
    accessibility_matrix[high_access_pseudobulks, peak_start:peak_end] *= 3.0

print(f'Created accessibility matrix shape: {accessibility_matrix.shape}')

# 4. Create AnnData object
adata = ad.AnnData(X=accessibility_matrix)
adata.obs_names = pseudobulk_names
adata.var_names = peak_names

# Add metadata
adata.obs['celltype'] = [name.split('_')[0] for name in pseudobulk_names]
adata.obs['timepoint'] = [name.split('_')[1] for name in pseudobulk_names]

print(f'Created synthetic AnnData object: {adata.shape}')

# 5. Create co-accessibility matrix
coaccess_pairs = []

# Within-cluster high co-accessibility
for cluster_id in range(n_clusters):
    peak_start = cluster_id * peaks_per_cluster
    peak_end = peak_start + peaks_per_cluster
    cluster_peaks = peak_names[peak_start:peak_end]
    
    print(f'Creating cluster {cluster_id}: peaks {peak_start}-{peak_end-1}')
    
    for i, peak1 in enumerate(cluster_peaks):
        for j, peak2 in enumerate(cluster_peaks[i+1:], i+1):
            if np.random.random() < 0.3:  # Sample 30% of within-cluster pairs
                score = np.random.uniform(0.3, 0.9)
                coaccess_pairs.append({'Peak1': peak1, 'Peak2': peak2, 'coaccess': score})

print(f'Created {len(coaccess_pairs)} within-cluster pairs')

# Add inter-cluster connections (lower co-accessibility)
n_inter_cluster = 1000
inter_added = 0

while inter_added < n_inter_cluster:
    peak1_idx = np.random.randint(0, n_peaks)
    peak2_idx = np.random.randint(0, n_peaks)
    
    if peak1_idx != peak2_idx:
        cluster1 = peak1_idx // peaks_per_cluster
        cluster2 = peak2_idx // peaks_per_cluster
        
        if cluster1 != cluster2:  # Different clusters
            score = np.random.uniform(-0.1, 0.2)  # Lower co-accessibility
            coaccess_pairs.append({
                'Peak1': peak_names[peak1_idx], 
                'Peak2': peak_names[peak2_idx], 
                'coaccess': score
            })
            inter_added += 1

# Create DataFrame
coaccess_df = pd.DataFrame(coaccess_pairs)

print(f'Final co-accessibility matrix: {len(coaccess_df)} pairs')
print(f'Score range: {coaccess_df["coaccess"].min():.3f} - {coaccess_df["coaccess"].max():.3f}')

# Check coverage
unique_peaks_coaccess = set(coaccess_df['Peak1'].tolist() + coaccess_df['Peak2'].tolist())
print(f'Peak coverage: {len(unique_peaks_coaccess)}/{len(peak_names)} ({len(unique_peaks_coaccess)/len(peak_names)*100:.1f}%)')

# 6. Save synthetic datasets
print('Saving synthetic datasets...')
adata.write_h5ad('./synthetic_peaks_by_pseudobulks.h5ad')
coaccess_df.to_csv('./synthetic_coaccessibility.csv', index=False)

print('✓ Synthetic datasets created and saved!')
print('Files:')
print('  - synthetic_peaks_by_pseudobulks.h5ad (accessibility data)')
print('  - synthetic_coaccessibility.csv (co-accessibility data)')