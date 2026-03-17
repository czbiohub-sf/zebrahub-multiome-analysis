# Cross-Species Peak UMAP via Motif-Space Embedding
## Execution Plan

---

## Overview

Build a cross-species peak UMAP that embeds regulatory elements from zebrafish, mouse, and human into a shared space based on transcription factor binding motif composition. Peaks with similar regulatory grammar will co-embed regardless of sequence conservation.

---

## Inputs

```
# h5ad objects: peaks-by-pseudobulk matrices
# Each h5ad has:
#   - .X: peaks (rows) × pseudobulk_samples (columns) accessibility matrix
#   - .obs: peak metadata (chr, start, end, peak_id, etc.)
#   - .var: pseudobulk sample metadata (celltype, timepoint, etc.)

ZEBRAFISH_H5AD = "<path_to_zebrafish_peaks_by_pseudobulk.h5ad>"
MOUSE_H5AD     = "<path_to_mouse_peaks_by_pseudobulk.h5ad>"
HUMAN_H5AD     = "<path_to_human_peaks_by_pseudobulk.h5ad>"

# Reference genomes (for peak sequence extraction)
ZEBRAFISH_GENOME = "danRer11"   # or path to fasta
MOUSE_GENOME     = "mm10"       # or path to fasta
HUMAN_GENOME     = "hg38"       # or path to fasta

# Motif database
JASPAR_MOTIF_DB = "JASPAR2024_CORE_vertebrates_non-redundant"
# Will be downloaded and converted to HOMER format
```

---

## Phase 1: Preparation

### Step 1.1: Download and prepare JASPAR core vertebrate motif database

```bash
# Download JASPAR 2024 CORE vertebrate non-redundant PWMs
# https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt

# Convert JASPAR PFM → HOMER motif format:
python convert_jaspar_to_homer.py \
    --input JASPAR2024_CORE_vertebrates_non-redundant_pfms.txt \
    --output jaspar_core_vertebrate.motif
```

**Key decision:** Use JASPAR 2024 CORE vertebrates non-redundant collection (~900 motifs).

- ~900 motifs may be too many due to TF family redundancy.
- **Recommended:** Cluster motifs by similarity (HOMER `compareMotifs.pl` or MEME `TOMTOM`) and retain representative motifs per cluster. Target: 300–500 non-redundant motifs.
- Alternative: use all JASPAR motifs and let PCA handle redundancy (correlated motifs load on the same PCs).

### Step 1.2: Extract peak sequences from each species

```python
import anndata as ad

def extract_peak_sequences(h5ad_path, output_bed):
    """
    Extract peak coordinates from h5ad .obs and write BED file.
    Expected .obs columns: 'chr', 'start', 'end'
    (or 'peak_id' parseable as chr:start-end)
    """
    adata = ad.read_h5ad(h5ad_path)
    peaks_bed = adata.obs[['chr', 'start', 'end']].copy()
    peaks_bed['name'] = adata.obs_names
    peaks_bed['score'] = 0
    peaks_bed['strand'] = '.'
    peaks_bed.to_csv(output_bed, sep='\t', header=False, index=False)
    return output_bed
```

```bash
# For each species, extract sequences with bedtools:
bedtools getfasta -fi ${ZEBRAFISH_GENOME}.fa -bed zebrafish_peaks.bed \
    -fo zebrafish_peaks.fa -name
bedtools getfasta -fi ${MOUSE_GENOME}.fa -bed mouse_peaks.bed \
    -fo mouse_peaks.fa -name
bedtools getfasta -fi ${HUMAN_GENOME}.fa -bed human_peaks.bed \
    -fo human_peaks.fa -name
```

**Note on peak width:** Resize all peaks to a fixed width (e.g., 500 bp centered on summit) before sequence extraction for fair motif scoring comparison. Variable-width peaks require downstream length normalization.

### Step 1.3: Verify peak counts and basic stats

```python
# Log number of peaks per species
# Check peak width distributions
# Verify genome builds match the h5ad coordinates
```

---

## Phase 2: Motif Scanning with HOMER

### Step 2.1: Run HOMER `annotatePeaks.pl` for motif scanning

```bash
# Scan each species' peaks against the shared JASPAR motif database.
# -mscore outputs the best log-odds score per motif per peak.

# Zebrafish
annotatePeaks.pl zebrafish_peaks.bed ${ZEBRAFISH_GENOME} \
    -m jaspar_core_vertebrate.motif \
    -mscore -nogene -noann \
    > zebrafish_motif_scores.txt

# Mouse
annotatePeaks.pl mouse_peaks.bed ${MOUSE_GENOME} \
    -m jaspar_core_vertebrate.motif \
    -mscore -nogene -noann \
    > mouse_motif_scores.txt

# Human
annotatePeaks.pl human_peaks.bed ${HUMAN_GENOME} \
    -m jaspar_core_vertebrate.motif \
    -mscore -nogene -noann \
    > human_motif_scores.txt
```

**Alternative if `-mscore` format is unsuitable:**

```bash
# Use findMotifsGenome.pl in "find" mode:
findMotifsGenome.pl zebrafish_peaks.bed ${ZEBRAFISH_GENOME} output_dir/ \
    -find jaspar_core_vertebrate.motif \
    > zebrafish_motif_hits.txt
```

### Step 2.2: Parse HOMER output into peaks × motifs matrix

```python
import pandas as pd
import numpy as np

def parse_homer_motif_scores(homer_output_path, peak_ids):
    """
    Parse HOMER annotatePeaks.pl -mscore output into a
    peaks × motifs DataFrame.

    Returns:
        pd.DataFrame: rows = peaks, columns = motif names, values = log-odds scores
    """
    df = pd.read_csv(homer_output_path, sep='\t', index_col=0)

    # HOMER prepends annotation columns before motif score columns.
    # Adjust this filter based on actual column names in the output.
    motif_cols = [c for c in df.columns if 'Score' in c or c.startswith('MA')]
    motif_matrix = df[motif_cols].fillna(0)
    motif_matrix.index = peak_ids

    return motif_matrix

zf_motifs = parse_homer_motif_scores("zebrafish_motif_scores.txt", zf_peak_ids)
mm_motifs = parse_homer_motif_scores("mouse_motif_scores.txt", mm_peak_ids)
hs_motifs = parse_homer_motif_scores("human_motif_scores.txt", hs_peak_ids)
```

**Critical check:** Verify that motif column names are identical across all three species outputs — this shared vocabulary is what makes cross-species embedding possible.

---

## Phase 3: Build Cross-Species Motif Matrix

### Step 3.1: Concatenate peaks × motifs matrices

```python
import anndata as ad

def build_cross_species_motif_adata(motif_matrices, h5ad_paths):
    """
    motif_matrices: {'zebrafish': df, 'mouse': df, 'human': df}
    h5ad_paths:     {'zebrafish': path, 'mouse': path, 'human': path}
    """
    all_obs, all_peaks = [], []

    for species, motif_df in motif_matrices.items():
        adata_orig = ad.read_h5ad(h5ad_paths[species])

        obs = adata_orig.obs.copy()
        obs['species'] = species
        obs['peak_id_global'] = [f"{species}_{pid}" for pid in obs.index]

        # Cell type with highest accessibility per peak
        X = adata_orig.X.toarray() if hasattr(adata_orig.X, 'toarray') else adata_orig.X
        obs['celltype_max_accessibility'] = adata_orig.var.index[np.argmax(X, axis=1)]

        all_obs.append(obs)
        all_peaks.append(motif_df)

    combined_motif = pd.concat(all_peaks, axis=0)
    combined_obs   = pd.concat(all_obs, axis=0)
    combined_obs.index = combined_obs['peak_id_global']
    combined_motif.index = combined_obs.index

    adata_motif = ad.AnnData(
        X=combined_motif.values,
        obs=combined_obs,
        var=pd.DataFrame(index=combined_motif.columns)
    )

    print(f"Combined motif matrix: {adata_motif.shape}")
    for sp in ['zebrafish', 'mouse', 'human']:
        n = (adata_motif.obs['species'] == sp).sum()
        print(f"  {sp}: {n} peaks")

    return adata_motif
```

### Step 3.2: Normalize the motif score matrix

```python
def normalize_motif_matrix(adata_motif):
    """
    Within-species z-score normalization per motif.
    Removes GC-content and genome-composition differences between species.
    """
    adata_motif.layers['raw_motif_scores'] = adata_motif.X.copy()

    for species in ['zebrafish', 'mouse', 'human']:
        mask = adata_motif.obs['species'] == species
        X = adata_motif.X[mask, :]
        mean = np.mean(X, axis=0)
        std  = np.std(X, axis=0)
        std[std == 0] = 1
        adata_motif.X[mask, :] = (X - mean) / std

    return adata_motif
```

**Design decision:** Within-species z-scoring (above) is recommended as the default — analogous to batch correction where species is the batch variable. Alternatives:
- **Quantile normalization:** forces identical per-motif distributions across species (more aggressive)
- **Log-transform then z-score:** useful if raw scores are highly skewed

---

## Phase 4: Dimensionality Reduction and UMAP

### Step 4.1: PCA on the combined motif matrix

```python
import scanpy as sc

sc.tl.pca(adata_motif, n_comps=50, svd_solver='arpack')
# Inspect: sc.pl.pca_variance_ratio(adata_motif)
# Check PC loadings: top motifs per PC should reflect known regulatory programs
```

### Step 4.2: Optional — Harmony correction for species batch effect

```python
import harmonypy as hm

harmony_out = hm.run_harmony(
    adata_motif.obsm['X_pca'][:, :50],
    adata_motif.obs,
    'species',
    max_iter_harmony=20
)
adata_motif.obsm['X_pca_harmony'] = harmony_out.Z_corr.T
```

**Trade-off:**
- Without Harmony: species may separate even for conserved programs (GC content, motif DB biases)
- With Harmony: risk of over-correcting and merging truly divergent programs

**Recommendation:** Run both and compare; save separate h5ad files for each.

### Step 4.3: Compute neighbors, UMAP, and Leiden clustering

```python
def compute_umap(adata_motif, use_harmony=False, n_neighbors=15, min_dist=0.1):
    use_rep = 'X_pca_harmony' if use_harmony else 'X_pca'
    sc.pp.neighbors(adata_motif, n_neighbors=n_neighbors, use_rep=use_rep)
    sc.tl.umap(adata_motif, min_dist=min_dist)
    for res in [0.5, 1.0, 2.0]:
        sc.tl.leiden(adata_motif, resolution=res, key_added=f'leiden_res{res}')
    return adata_motif
```

---

## Phase 5: Visualization and Validation

### Step 5.1: Diagnostic UMAP plots

```python
# 1. Color by species — check for mixing vs. separation
sc.pl.umap(adata_motif, color='species')

# 2. Color by cell type of max accessibility
sc.pl.umap(adata_motif, color='celltype_max_accessibility')

# 3. Per-species panels within the shared UMAP
for species in ['zebrafish', 'mouse', 'human']:
    mask = adata_motif.obs['species'] == species
    sc.pl.umap(adata_motif[mask], color='celltype_max_accessibility',
               title=f'{species} peaks in cross-species UMAP')

# 4. Leiden clusters
sc.pl.umap(adata_motif, color='leiden_res1.0')

# 5. Spot-check key TF motif scores
for motif in ['GATA4', 'SOX2', 'PAX6', 'MYOD1', 'TAL1', 'FOXP1']:
    if motif in adata_motif.var_names:
        sc.pl.umap(adata_motif, color=motif)
```

### Step 5.2: Validate cross-species cluster coherence

```python
def validate_cluster_coherence(adata_motif):
    # Species mixing per cluster
    cluster_species = pd.crosstab(
        adata_motif.obs['leiden_res1.0'],
        adata_motif.obs['species'],
        normalize='index'
    )
    print(cluster_species)

    # Cell type enrichment per cluster, per species
    for species in ['zebrafish', 'mouse', 'human']:
        mask = adata_motif.obs['species'] == species
        ct_enrich = pd.crosstab(
            adata_motif.obs.loc[mask, 'leiden_res1.0'],
            adata_motif.obs.loc[mask, 'celltype_max_accessibility'],
            normalize='index'
        )
        print(f"\n{species}:\n{ct_enrich}")

    # Top motifs per cross-species cluster
    for cluster in adata_motif.obs['leiden_res1.0'].unique():
        mask = adata_motif.obs['leiden_res1.0'] == cluster
        mean_scores = pd.Series(
            np.mean(adata_motif.X[mask, :], axis=0),
            index=adata_motif.var_names
        )
        print(f"\nCluster {cluster} top motifs: {mean_scores.nlargest(10).index.tolist()}")
```

### Step 5.3: Cross-reference with species-specific UMAPs

Build a confusion matrix mapping species-specific leiden clusters (from individual-species UMAPs) to cross-species leiden clusters. Homologous cell-type clusters across species should map to the same cross-species cluster.

---

## Phase 6: Biological Analysis

### Step 6.1: Identify conserved vs. species-specific regulatory programs

```python
def classify_conservation(adata_motif):
    cluster_species = pd.crosstab(
        adata_motif.obs['leiden_res1.0'],
        adata_motif.obs['species'],
        normalize='index'
    )
    for cluster in cluster_species.index:
        fracs = cluster_species.loc[cluster]
        if all(fracs > 0.15):
            label = "deeply_conserved"
        elif fracs['zebrafish'] < 0.1 and fracs['mouse'] > 0.15 and fracs['human'] > 0.15:
            label = "mammal_specific"
        elif max(fracs) > 0.8:
            label = f"species_specific_{fracs.idxmax()}"
        else:
            label = "partially_conserved"
        print(f"Cluster {cluster}: {label} | {dict(fracs.round(2))}")
```

### Step 6.2: Temporal dynamics of conservation (zebrafish-specific)

For zebrafish peaks in each cross-species cluster, map their timepoint of maximum accessibility.

**Hypothesis:** deeply conserved regulatory programs are enriched in earlier developmental stages (phylotypic period), while species-specific programs emerge later.

---

## Output Files

```
notebooks/EDA_peak_umap_cross_species/
├── outputs/
│   ├── motif_database/
│   │   ├── jaspar_core_vertebrate.motif
│   │   └── motif_metadata.csv
│   ├── peak_sequences/
│   │   ├── zebrafish_peaks.bed / zebrafish_peaks.fa
│   │   ├── mouse_peaks.bed    / mouse_peaks.fa
│   │   └── human_peaks.bed    / human_peaks.fa
│   ├── motif_scores/
│   │   ├── zebrafish_motif_scores.txt
│   │   ├── mouse_motif_scores.txt
│   │   └── human_motif_scores.txt
│   └── cross_species_embedding/
│       ├── cross_species_motif_adata.h5ad
│       ├── cross_species_umap_no_harmony.h5ad
│       └── cross_species_umap_with_harmony.h5ad
└── figures/
    ├── umap_by_species.pdf
    ├── umap_by_celltype.pdf
    ├── umap_by_leiden.pdf
    ├── umap_per_species_panels.pdf
    ├── cluster_species_composition.pdf
    ├── cluster_motif_enrichment.pdf
    └── conservation_classification.pdf
```

---

## Dependencies

```
# Python
scanpy >= 1.9
anndata >= 0.8
pandas, numpy
harmonypy
pybedtools
matplotlib, seaborn

# Command-line tools
homer  (annotatePeaks.pl, findMotifsGenome.pl)
bedtools
```

---

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Peak width for motif scanning | 500 bp centered on summit | Wider = more motifs but more noise |
| Motif score type | Continuous (log-odds) | Binary is alternative but less informative |
| Within-species normalization | Z-score per motif | Critical for removing GC/genome bias |
| Number of PCs | 50 | Check variance explained; may need 30–100 |
| Harmony correction | Run both with/without | Compare to assess over-correction |
| UMAP n_neighbors | 15 | Standard; may adjust for dataset size |
| UMAP min_dist | 0.1 | Lower = tighter clusters |
| Leiden resolution | 0.5, 1.0, 2.0 | Multiple resolutions for hierarchy |

---

## Execution Order

```
Phase 1: Preparation
  1.1  Download + convert JASPAR motifs to HOMER format
  1.2  Extract peak sequences from each species h5ad → BED → FASTA
  1.3  Verify peak counts and genome build consistency

Phase 2: Motif Scanning
  2.1  Run HOMER annotatePeaks.pl for each species (parallelizable)
  2.2  Parse HOMER output into peaks × motifs DataFrames

Phase 3: Build Combined Matrix
  3.1  Concatenate across species into single AnnData
  3.2  Within-species z-score normalization

Phase 4: Embedding
  4.1  PCA (n=50)
  4.2  Harmony correction (optional, run in parallel with 4.3)
  4.3  Neighbors → UMAP → Leiden (with and without Harmony)

Phase 5: Visualization + Validation
  5.1  Diagnostic UMAP plots (species, celltype, motifs, leiden)
  5.2  Cluster coherence analysis
  5.3  Cross-reference with species-specific UMAPs

Phase 6: Biological Analysis
  6.1  Conservation classification (conserved / mammal-specific / species-specific)
  6.2  Temporal dynamics analysis (zebrafish timepoint enrichment per cluster)
```
