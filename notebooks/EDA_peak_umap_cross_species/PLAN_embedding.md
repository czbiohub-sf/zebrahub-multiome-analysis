# Plan: Cross-Species Peak UMAP Embedding + Visualization

## Context

We completed Steps 1–4b of the cross-species motif scanning pipeline:
- 879 JASPAR2024 motifs scanned across 1.87M peaks (zebrafish/mouse/human)
- Output: `cross_species_motif_scores_FPR_0.010_binarized.h5ad` (1,874,537 × 879, binary 0/1, 0.28 GB)
- No embeddings computed yet (empty `.obsm`)

**Goal:** Compute PCA → UMAP embedding on the binarized motif matrix and generate visualizations colored by species, specific TF motifs, TF family groupings, and Leiden cluster × motif enrichment.

## Files to Create

| File | Purpose |
|------|---------|
| `notebooks/EDA_peak_umap_cross_species/05_embed_motif_umap.py` | Embedding: TF-IDF → PCA → neighbors → UMAP → Leiden |
| `notebooks/EDA_peak_umap_cross_species/06_visualize_motif_umap.py` | All visualizations |
| `scripts/jaspar-motif-scan/slurm/run_embed_umap.sh` | SLURM GPU job for embedding |

## Step 1: Embedding Script (`05_embed_motif_umap.py`)

**Env:** `sc_rapids` (GPU) at `/hpc/user_apps/data.science/conda_envs/sc_rapids`

**Input:** `/hpc/scratch/.../cross_species_motif_scores_FPR_0.010_binarized.h5ad`

### Pipeline

1. **Load** binarized h5ad, save raw binary to `layers["binary"]`
2. **TF-IDF weighting** via `sklearn.feature_extraction.text.TfidfTransformer(norm='l2', smooth_idf=True)` — upweights rare/specific motifs, downweights ubiquitous ones. Standard for binary occurrence matrices. Store IDF weights in `var["idf_weight"]`
3. **PCA** (100 comps, GPU via `rsc.pp.pca`). Save scree plot. Use ~50 PCs for neighbors (guided by variance explained)
4. **Neighbors** (`n_neighbors=30, n_pcs=50, metric="cosine"`) — cosine is natural for L2-normalized TF-IDF vectors
5. **UMAP** (`min_dist=0.1, spread=1.0`) — tight clusters for discrete motif space
6. **Leiden** at resolutions 0.5, 1.0, 2.0, 5.0
7. **Save** embedded h5ad → `cross_species_motif_embedded.h5ad` + lightweight CSV with coordinates

### Verification
- Shape preserved (1,874,537 × 879)
- UMAP coordinates finite
- Species counts unchanged (zebrafish=640,831, mouse=192,251, human=1,041,455)
- All Leiden columns non-null

## Step 2: SLURM Script (`slurm/run_embed_umap.sh`)

```
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
```

Expected runtime: ~20-30 min (TF-IDF ~1 min, PCA ~3 min, neighbors ~10 min, UMAP ~10 min).

## Step 3: Visualization Script (`06_visualize_motif_umap.py`)

### 3a. Species Overlay
- Combined UMAP colored by species (3 colors)
- Per-species highlight panels (one species colored, others gray)
- Use `size=0.5, rasterized=True` for 1.87M points

### 3b. Specific TF Overlays
Binary presence/absence on UMAP for key developmental TFs:
- **Hematopoietic:** GATA1, GATA2, TAL1, RUNX1, SPI1
- **Neural:** SOX2, SOX10, PAX6, NEUROD1, ASCL1
- **Cardiac:** HAND2, NKX2-5, TBX5, GATA4, MEF2A
- **Muscle:** MYOD1
- **General:** CTCF, FOXA2, POU5F1

Layout: 4×4 grid, gray background (absent) → colored (present)

### 3c. TF Family Grouping
Classify 879 motifs into ~20 structural families using regex dictionary on `tf_name`:
- bHLH, Forkhead, GATA, HMG/Sox, Homeodomain/HOX, ETS, C2H2 zinc finger/KLF, Nuclear receptor, bZIP/AP-1, T-box, POU, TEAD, CTCF, etc.
- Heterodimers (e.g., `GATA1::TAL1`): classify by first partner

**Dominant family per peak** = family with most motif hits → color UMAP by this

### 3d. Leiden × Motif Enrichment Heatmap
- For each Leiden cluster: compute log2 fold enrichment vs global motif frequency
- Filter to top 50 most variable motifs
- `seaborn.clustermap`, RdBu_r, centered at 0

### 3e. Additional Diagnostics
- PCA scree plot (variance explained)
- Leiden clusters colored on UMAP
- Species composition per Leiden cluster (stacked bar)
- Celltype and timepoint overlays

## Output Locations

```
/hpc/scratch/.../multiome-cross-species-peak-umap/
├── cross_species_motif_embedded.h5ad       ← main output
└── cross_species_umap_coords.csv.gz        ← lightweight coords

figures/cross_species_motif_umap/
├── pca_variance_explained.pdf
├── umap_species.pdf
├── umap_species_per_panel.pdf
├── umap_motif_{TF_NAME}.pdf  (×16 TFs)
├── umap_dominant_tf_family.pdf
├── leiden_motif_enrichment_heatmap.pdf
├── umap_leiden_res1.0.pdf
└── leiden_species_composition.pdf
```

## Execution Order

0. Save this plan → `notebooks/EDA_peak_umap_cross_species/PLAN_embedding.md` ✅
1. Write `05_embed_motif_umap.py` ✅
2. Write `slurm/run_embed_umap.sh` ✅
3. Submit SLURM GPU job, monitor
4. After embedding completes: write `06_visualize_motif_umap.py`, run it
5. Inspect plots, iterate on parameters if needed
