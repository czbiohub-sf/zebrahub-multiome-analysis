# Cross-Species Peak UMAP via Motif-Space Embedding
## Execution Plan

---

## Overview

Build a cross-species peak UMAP embedding zebrafish (~640K), mouse (~192K), and human (~1M) regulatory elements into a shared space based on TF binding motif composition (JASPAR2024 CORE vertebrates). Peaks with similar regulatory grammar co-embed regardless of sequence conservation.

---

## Inputs

| Species | h5ad path | Peaks | Genome |
|---------|-----------|-------|--------|
| Zebrafish | `/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/objects_v2/peaks_by_ct_tp_pseudobulked_all_peaks_pca_concord.h5ad` | 640,834 | danRer11 |
| Mouse | `/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated_v2.h5ad` | 192,251 | mm10 |
| Human | `/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/public_data/human_domcke_2020/peaks_by_pb_celltype_stage_annotated.h5ad` | 1,041,455 | **hg19** ⚠️ |

**Total peaks: ~1.87M across three species.**

> ⚠️ **Genome version note:** The human h5ad has `obs['genome'] == 'hg19'`. The GTF provided (`GRCH38.gencode.v47`) is hg38. Need to confirm whether to use an hg19 genome FASTA for sequence extraction, or liftover the peaks to hg38 first. Use hg19 FASTA for motif scanning to match the stored coordinates.

GTF files (for gene annotation reference only — not used in motif scanning):
- Mouse: `/hpc/reference/sequencing_alignment/alignment_references/mouse_gencode_M31_GRCm39_cellranger/genes/genes.gtf.gz`
- Human: `/hpc/reference/sequencing_alignment/alignment_references/GRCH38.gencode.v47.primary_assembly_Cellranger_20250321/genes/genes.gtf.gz`

### h5ad Data Structure

**Zebrafish** (`peaks × 190 pseudobulk groups`):
- `.obs` index format: `1-32-526` (chromosome-start-end, **no "chr" prefix**)
- `.obs` key columns: `celltype`, `timepoint` (dominant annotation per peak)
- Coordinate parsing: split on `-` → `chr={parts[0]}`, `start={parts[1]}`, `end={parts[2]}`; prepend `chr` prefix when writing BED
- Peak widths: variable (~500–700bp typical, some wider)
- `.layers`: `log_norm`, `normalized`, `sum`

**Mouse** (`peaks × 145 pseudobulk groups`):
- `.obs` index format: `chr1-3035602-3036202`
- `.obs` key columns: `chr`, `start`, `end` (Categorical dtype → cast to int), `top_celltype`, `top_timepoint`
- Peak widths: ~600bp (uniform)
- `.layers`: `normalized`, `sum`

**Human** (`peaks × 249 pseudobulk groups`):
- `.obs` index format: `chr1-752336-752980`
- `.obs` key columns: `chr`, `start`, `end` (Categorical dtype → cast to int), `top_celltype`, `top_timepoint`, `genome`
- Peak widths: variable (266–1813bp)
- `.layers`: `normalized`, `sum`

### Motif Database
- JASPAR 2024 CORE vertebrates non-redundant (already downloaded — **fill in path**)
- HOMER binary: `/hpc/mydata/yang-joon.kim/homer/bin/annotatePeaks.pl`
- **Preferred scanner:** GimmeMotifs (`gimmemotifs`) — reads JASPAR PFM natively, parallelized, Python-native

### Output Directory
```
/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap/
├── peak_sequences/
├── motif_database/
├── motif_scores/
├── cross_species_motif_adata.h5ad
├── cross_species_embedded.h5ad
├── cluster_conservation_classification.csv
└── figures/
```

---

## Script Structure

All scripts live in `notebooks/EDA_peak_umap_cross_species/scripts/`:

| Script | Purpose |
|--------|---------|
| `00_config.py` | All paths and parameters (import everywhere) |
| `01_prepare_peaks.py` | Extract coordinates → BED → FASTA per species |
| `02_prepare_motifs.py` | Load/convert JASPAR motifs, build metadata CSV |
| `03_scan_motifs.py` | Parallelized motif scanning → `{species}_motif_scores.npz` |
| `04_build_matrix.py` | Concatenate + normalize → `cross_species_motif_adata.h5ad` |
| `05_embed_umap.py` | PCA → Harmony → UMAP → Leiden |
| `06_visualize.py` | Diagnostic plots |
| `07_validate.py` | Cluster coherence + conservation classification |
| `run_all.py` | Master execution script |
| `slurm_scripts/` | SLURM submission scripts for compute-heavy steps |

---

## Phase 1: Preparation

### Step 1.1: Per-species coordinate extraction → BED → FASTA

Each species requires species-specific parsing of `.obs` to produce a 6-column BED file,
then `bedtools getfasta` to extract sequences.

**Species-specific parsing logic:**

```python
def parse_peak_coordinates(adata, species):
    """Returns DataFrame with columns: chr, start, end"""
    if species == 'zebrafish':
        # Index format: '1-32-526' — no chr prefix
        parts = adata.obs_names.str.split('-', expand=True)
        df = pd.DataFrame({
            'chr':   'chr' + parts[0],   # prepend 'chr'
            'start': parts[1].astype(int),
            'end':   parts[2].astype(int),
        }, index=adata.obs_names)
    else:
        # Mouse/Human: separate chr, start, end columns (Categorical → int)
        df = pd.DataFrame({
            'chr':   adata.obs['chr'].astype(str),
            'start': adata.obs['start'].astype(int),
            'end':   adata.obs['end'].astype(int),
        }, index=adata.obs_names)
    return df
```

**Resize all peaks to 500 bp centered:**

```python
center = (df['start'] + df['end']) // 2
df['start'] = (center - 250).clip(lower=0)
df['end']   = center + 250
```

**Sequence extraction:**

```bash
bedtools getfasta -fi ${GENOME}.fa -bed {species}_peaks.bed -fo {species}_peaks.fa -name
```

**Genome FASTAs needed:**
- `danRer11.fa` — zebrafish (danRer11 / GRCz11)
- `mm10.fa` — mouse (GRCm38)
- `hg19.fa` — human (**hg19**, matching h5ad coordinates)

**Metadata to save per species:**

From `.obs`: species label, original peak ID, `celltype_max` / `top_celltype`, `timepoint` / `top_timepoint`, `lineage` (if present), `leiden_coarse`.

Column name mapping:

| Field | Zebrafish col | Mouse col | Human col |
|-------|---------------|-----------|-----------|
| Dominant cell type | `celltype` | `top_celltype` | `top_celltype` |
| Dominant timepoint | `timepoint` | `top_timepoint` | `top_timepoint` |
| Lineage | `peak_lineage` (check) | `peak_lineage` | `peak_lineage` |
| Leiden coarse | *(none yet)* | `leiden_coarse` | `leiden_coarse` |

### Step 1.2: Verify peak counts and widths

```python
# Sanity checks per species
assert len(peaks_df) == adata.n_obs
assert (peaks_df['end'] - peaks_df['start']).eq(500).all(), "Peak widths not 500bp"
assert not peaks_df['chr'].str.contains('nan').any(), "NaN chromosomes"
```

---

## Phase 2: Motif Database Preparation

### Step 2.1: Load JASPAR motifs

GimmeMotifs reads JASPAR PFM format directly:

```python
from gimmemotifs.motif import read_motifs

JASPAR_PFM = "<path_to_downloaded_JASPAR2024_CORE_vertebrates_non-redundant.pfm>"
motifs = read_motifs(JASPAR_PFM, fmt='jaspar')
print(f"Loaded {len(motifs)} motifs")  # expect ~800-900
```

### Step 2.2: Build motif metadata CSV

```python
meta = pd.DataFrame([{
    'motif_id': m.id,
    'tf_name':  m.id.split('.')[-1] if '.' in m.id else m.id,
    'consensus': m.consensus,
} for m in motifs])
meta.to_csv(f"{OUTPUT_DIR}/motif_database/motif_metadata.csv", index=False)
```

---

## Phase 3: Motif Scanning (Main Compute Step)

**Scale:** ~1.87M peaks × ~800 motifs. This is the bottleneck.
- GimmeMotifs with 32 CPUs: ~30–60 min per species
- Human (~1M peaks) will be the slowest; run in parallel with zebrafish/mouse

### Step 3.1: GimmeMotifs Scanner (recommended)

```python
from gimmemotifs.motif import read_motifs
from gimmemotifs.scanner import Scanner
from scipy import sparse
import numpy as np

def scan_species(fasta_path, motif_path, n_cpus, output_npz):
    motifs = read_motifs(motif_path)
    motif_names = [m.id for m in motifs]

    s = Scanner(ncpus=n_cpus)
    s.set_motifs(motif_path)
    s.set_threshold(threshold=0.0)  # keep all scores

    scores_list, seq_names = [], []
    for name, scores in s.best_score(fasta_path):
        scores_list.append(scores)
        seq_names.append(name)

    score_matrix = np.array(scores_list)   # (n_peaks, n_motifs)
    sparse.save_npz(output_npz, sparse.csr_matrix(score_matrix))
    np.savez(output_npz.replace('.npz', '_meta.npz'),
             peak_names=np.array(seq_names),
             motif_names=np.array(motif_names))
    print(f"Shape: {score_matrix.shape}")
```

### Step 3.2: HOMER fallback (split-parallel-merge)

If GimmeMotifs is unavailable, split peaks into chunks of ~10K, run
`annotatePeaks.pl` in parallel with `multiprocessing.Pool`, then merge.
HOMER binary: `/hpc/mydata/yang-joon.kim/homer/bin/annotatePeaks.pl`

### SLURM resources for scanning

| Species | Peaks | Recommended resources |
|---------|-------|-----------------------|
| Zebrafish | 640K | 32 CPUs, 64G, 4h |
| Mouse | 192K | 32 CPUs, 32G, 2h |
| Human | 1.04M | 32 CPUs, 128G, 6h |

Run all three as independent SLURM jobs in parallel.

---

## Phase 4: Build Cross-Species Motif Matrix

### Step 4.1: Concatenate and normalize

```python
# Within-species z-score per motif (removes GC/genome-composition bias)
def zscore_within_species(X):
    X = X.toarray() if sparse.issparse(X) else X.copy()
    mean = np.mean(X, axis=0, keepdims=True)
    std  = np.std(X, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std
```

Critical check: motif names must be identical (same order) across all three
`_meta.npz` files before concatenating.

### Step 4.2: Create AnnData and save

```python
adata = ad.AnnData(
    X=np.vstack([X_zf, X_mm, X_hs]),           # (1.87M, n_motifs)
    obs=pd.concat([obs_zf, obs_mm, obs_hs]),    # species, peak metadata
    var=pd.DataFrame(index=motif_names),         # motif IDs
)
adata.write(f"{OUTPUT_DIR}/cross_species_motif_adata.h5ad")
```

---

## Phase 5: Embedding

```python
# PCA
sc.tl.pca(adata, n_comps=50, svd_solver='arpack')

# Harmony (optional — run both with and without, save separately)
import harmonypy as hm
ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, 'species', max_iter_harmony=20)
adata.obsm['X_pca_harmony'] = ho.Z_corr.T

# Neighbors + UMAP + Leiden at multiple resolutions
for suffix, use_rep in [('_raw', 'X_pca'), ('_harmony', 'X_pca_harmony')]:
    sc.pp.neighbors(adata, n_neighbors=15, use_rep=use_rep, key_added=f'neighbors{suffix}')
    sc.tl.umap(adata, neighbors_key=f'neighbors{suffix}', min_dist=0.1)
    adata.obsm[f'X_umap{suffix}'] = adata.obsm['X_umap'].copy()
    for res in [0.5, 1.0, 2.0]:
        sc.tl.leiden(adata, resolution=res, neighbors_key=f'neighbors{suffix}',
                     key_added=f'leiden{suffix}_res{res}')

adata.write(f"{OUTPUT_DIR}/cross_species_embedded.h5ad")
```

---

## Phase 6: Visualization

Key plots (both `_raw` and `_harmony` embeddings):

1. **Color by species** — assess mixing vs. separation
2. **Color by `celltype_max` / `top_celltype`** — do homologous cell types co-locate?
3. **Per-species panels** — highlight each species on the shared UMAP background
4. **Leiden clusters** (`res=0.5, 1.0, 2.0`)
5. **Key TF motif scores** — GATA1/4, SOX2/10, PAX6, MYOD1, TAL1, HAND2, NKX2-5, TBX5

---

## Phase 7: Validation and Conservation Analysis

### Cluster species composition

```python
species_comp = pd.crosstab(
    adata.obs['leiden_harmony_res1.0'], adata.obs['species'], normalize='index'
)
```

### Conservation classification

| Category | Criteria |
|----------|----------|
| `deeply_conserved` | All 3 species > 10% |
| `mammal_specific` | zebrafish < 5%, mouse & human > 10% |
| `species_specific:{name}` | One species > 85% |
| `partially_conserved` | Everything else |

### Cell type enrichment per cluster per species

Cross-reference cross-species Leiden clusters with species-specific `celltype_max` / `top_celltype`.
Homologous cell types should co-cluster in conserved clusters.

### Temporal dynamics (zebrafish)

For zebrafish peaks in each cross-species cluster, map `timepoint` to assess whether
deeply conserved regulatory programs are enriched at earlier developmental stages.

---

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Peak width | 500 bp | Centered on midpoint |
| Scanner | GimmeMotifs | Fallback: HOMER with parallel chunks |
| Normalization | Z-score within species | Removes GC/genome bias |
| n_PCs | 50 | Check variance explained |
| Harmony | Run both with/without | Compare to detect over-correction |
| n_neighbors | 15 | Standard |
| min_dist | 0.1 | |
| Leiden resolutions | 0.5, 1.0, 2.0 | |

---

## Open Questions / TODOs Before Starting

- [ ] **Confirm JASPAR file path** — user has JASPAR motifs downloaded; fill in path in `00_config.py`
- [ ] **Human genome version** — h5ad has `hg19` coordinates; need hg19 FASTA for `bedtools getfasta`, or liftover to hg38 first
- [ ] **Mouse genome FASTA** — verify mm10 FASTA path (GTF is GRCm39 but peaks likely mm10/GRCm38)
- [ ] **Zebrafish FASTA** — confirm danRer11 FASTA path
- [ ] **GimmeMotifs availability** — check if `gimmemotifs` is installed in `sc_rapids` or another env
- [ ] **`peak_lineage` in zebrafish** — verify column name for lineage metadata in zebrafish h5ad

---

## Execution Order

```
Phase 1:  01_prepare_peaks.py        (fast, CPU only)
Phase 2:  02_prepare_motifs.py       (fast, CPU only)
Phase 3:  03_scan_motifs.py          (SLURM, 3 parallel jobs, GPU not needed)
Phase 4:  04_build_matrix.py         (CPU, ~30 min for 1.87M peaks)
Phase 5:  05_embed_umap.py           (CPU/GPU, ~1–2h)
Phase 6:  06_visualize.py            (CPU, ~15 min)
Phase 7:  07_validate.py             (CPU, ~15 min)
```
