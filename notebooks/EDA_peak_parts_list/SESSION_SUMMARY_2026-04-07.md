# Peak Parts List Session Summary — 2026-04-07

## Goal

Build publication-ready figures and analyses demonstrating that the 640K ATAC-seq peaks × 32 celltypes × 6 timepoints pseudobulk matrix constitutes a queryable "parts list" of cis-regulatory elements with defined celltype and temporal specificity. Target: Nature Methods Resource manuscript.

---

## 1. Publication Figure Settings (RESOLVED)

### Problem
PDFs had character-by-character text in Illustrator (Type 3 fonts), not editable sentences.

### Root cause
- `pdf.fonttype=42` alone is insufficient — SLURM jobs resolved the `sans-serif` font family to DejaVu Sans (Type 3) instead of Arial (TrueType)
- `sns.set()` resets `savefig.dpi` back to ~72

### Solution — 5-step rcParams block
```python
import matplotlib as _mpl
_mpl.rcParams.update(_mpl.rcParamsDefault)   # 1. reset
_mpl.rcParams['font.family'] = 'Arial'      # 2. CRITICAL — explicit Arial
_mpl.rcParams["pdf.fonttype"] = 42          # 3. TrueType embedding
_mpl.rcParams["ps.fonttype"]  = 42
import seaborn as _sns
_sns.set(style="whitegrid", context="paper") # 4. seaborn AFTER fonttype
_mpl.rcParams["savefig.dpi"]  = 300         # 5. DPI AFTER sns.set()
```

### Verification
`pdffonts file.pdf` → must show `CID TrueType ArialMT`, not `Type 3 DejaVuSans`

### Environment
Use `single-cell-base` for visualization scripts (confirmed working). `gReLu` only for scripts needing pysam/pymemesuite.

---

## 2. V3 Celltype-Level Specificity (existing, refined)

### Approach
- Average log-norm accessibility across reliable timepoints per celltype → 640K × 31 matrix
- Leave-one-out z-score across celltypes: `z = (x - mean_other) / std_other`
- Separates celltype specificity from temporal dynamics

### Key result
Top-50 peaks per celltype recover known biology: cardiac peaks near gata4/tnnc1a, muscle peaks near myod1/myog, etc.

---

## 3. Tau Specificity Index (NEW)

### Rationale
Need a single number per peak capturing "how celltype-specific is this peak?" for the parts list narrative. Better than showing z-score distributions.

### Approach
Tau index (Yanai et al. 2005): `tau = sum(1 - x_hat_i) / (N-1)` where `x_hat_i = x_i / max(x)`. Ranges 0 (uniform/housekeeping) to 1 (perfectly specific).

### Outputs
- `V3_peak_specificity_metrics.csv` — tau + Gini for all 640K peaks
- Three UMAP encodings: color (magma), dot size, transparency
- Tau distribution histogram

### Key finding
Visual: celltype-specific peaks (high tau) cluster tightly on UMAP; broadly accessible peaks (low tau) are diffuse. This directly shows the parts list structure.

---

## 4. RNA-ATAC Concordance (NEW)

### Problem
Initial attempt used pre-computed RNA z-scores from `genes_by_ct_tp_top_5068genes.h5ad` — these were z-scores of z-scores (wrong scale, y-axis compressed to -0.3–12 vs ATAC 0–150).

### Solution
Built RNA pseudobulk from scratch using `integrated_RNA_ATAC_counts_RNA_master_filtered.h5ad` (94K cells × 32K genes):
1. Sum raw counts per (celltype × timepoint) group
2. Median-scale normalization (same as ATAC pipeline)
3. log1p
4. Celltype-level z-score with `std_floor = 0.5` (robust for zero-inflated RNA)

### Why std_floor = 0.5?
RNA has many genes with zero expression in most celltypes. The ATAC `1e-10` floor produces z-scores up to 65,000 for RNA. A floor of 0.5 (≈2-fold variability on log1p scale) caps max z at ~25, biologically interpretable.

### Output
- `rna_by_ct_tp_pseudobulked.h5ad` — genes × conditions
- `rna_specificity_matrix_celltype_level.h5ad` — genes × celltypes z-scores
- Scatter plots: ATAC z-score vs RNA z-score per celltype

---

## 5. Motif Enrichment Heatmap (NEW — key figure)

### Problem with original approach
- Only 7 focal celltypes had FIMO results
- Background for Fisher's exact test was inconsistent (3 new celltypes used pool of 2 others; original 7 used pool of 6)
- Z-score denominator was only N=6 or N=7

### Revised approach
1. Run FIMO on top-50 peaks for ALL 31 celltypes (~18 min)
2. Fisher's exact: each celltype vs all other 30 pooled (~1500 peaks background)
3. Enrichment z-score across all 31 celltypes (robust N=31 denominator)
4. Focused heatmap: 6 interesting celltypes × union of top-5 TFs each (30 TFs)

### Key results (6 interesting celltypes)
| Celltype | Top TFs | Biology |
|---|---|---|
| epidermis | P53, P63, P73, IRX3 | p53/p63 family — known epidermal regulators |
| neural_crest | TFEB, ZNF74, AP2C | AP2 family — classic neural crest |
| hemangioblasts | TRPS1, GATA1-3, GATA6 | GATA family — canonical hematopoietic |
| hindbrain | EGR2, EGR1, WT1 | EGR2 (Krox20) — rhombomere segmentation |
| optic_cup | NOTO*, HME2, GSX2, LHX9, VAX1 | Homeobox TFs — eye/brain patterning |
| hatching_gland | HMBX1, FOXC1, HSF2 | FOX family |

*NOTO PWM in optic_cup = shared homeobox TAAT-core motif, not the actual NOTO TF. JASPAR limitation: homeobox TFs have nearly indistinguishable PWMs.

### Outputs
- `V3_interesting6_motif_zscore_hitrate.pdf` — Panel A (z-score) + Panel B (raw hit rate)
- `V3_interesting6_motif_log2fold.pdf` — log2 fold enrichment
- `V3_interesting6_motif_heatmap_colorbar.pdf` — manuscript version with celltype colors
- `V3_all31_motif_enrichment.csv` — full results for all 31 celltypes

---

## 6. Marker Gene Profiles — Reverse Lookup (NEW)

### Rationale
Instead of "top peaks → what gene?", do "known gene → which peak?" This is more intuitive for readers and demonstrates the query interface.

### Key examples: broad → specific regulatory elements

**Muscle lineage:**
| Gene | Accessibility pattern | Story |
|---|---|---|
| myf5 | muscle(5.7), PSM(2.3), fast_muscle(2.0) | Shared across muscle lineage |
| myod1 | fast_muscle(7.2), muscle(2.6) | Committed myogenic |
| myog | fast_muscle(26.2) only | Terminal differentiation |

**Cardiac lineage:**
| Gene | Accessibility pattern | Story |
|---|---|---|
| nkx2.5 | heart(4.2), hematopoietic_vasculature(3.6) | Shared cardiac/vascular |
| hand2 | heart(8.8), pronephros(2.1) | Lateral plate derivatives |
| gata4 | heart(14.6) only | Committed cardiac |

### Output
Bar plots at `V3/peak_profiles/marker_genes/{gene}_{celltype}.{pdf,png}`

---

## 7. Gene Locus View Recommendations

Best loci for the "parts list" narrative (multiple peaks, each belonging to different celltypes):

| Gene | Peaks | Celltypes with z>4 | Story |
|---|---|---|---|
| **gata4** | 34 | 8 (heart, endoderm, neurons, vasculature...) | One gene, 8 celltype-specific enhancers |
| **pax6a** | 50 | 8 (telencephalon, optic_cup, PSM...) | Unexpected regulatory complexity |
| **sox9a** | 52 | 13 | Diversity champion — 13 celltypes |

**Recommendation:** Show gata4 (most recognizable, spans germ layers).

---

## 8. Proposed SI Figure Panel Layout

| Panel | Content | Source |
|---|---|---|
| A | Tau distribution + UMAP (specificity landscape) | `specificity_overview/` |
| B | Peak UMAP — 2 example peaks highlighted (tnnc1a, slc4a1a or myod1) | `peak_umap/` |
| C | Celltype + temporal bar plots for those peaks | `peak_profiles/` |
| D | 6-panel peak UMAP (interesting 6) | `peak_umap/interesting6_panel` |
| E | Motif enrichment heatmap (z-score + hit rate) | `motif_enrichment/interesting6` |
| F | RNA-ATAC concordance scatter | `rna_atac_concordance/` |
| G | Temporal heatmap (1-2 celltypes, row-normalized) | `temporal/heatmaps/` |
| H | Gene locus view (gata4) | `gene_locus_views/` |

A-C: "the resource exists and is queryable"
D-E: "it reveals known biology"
F-H: "it's validated and multi-dimensional"

---

## 9. Commits & PR

4 commits on `peak-parts-list` branch → PR #16:
1. `1a70016` — V3 pipeline + publication figure settings (09, 09b, 09c)
2. `345c363` — UMAP visualizations (09d family)
3. `f662774` — RNA pseudobulk + concordance (09e)
4. `cac72fc` — Motif heatmap + Tau UMAP + marker gene profiles (09f, 09g, 09c-marker)

---

## 10. Open Questions / Next Steps

- **Cross-species conservation**: do the interesting-6 celltypes' top peaks overlap with conserved anchors from the cross-species UMAP project?
- **GO enrichment**: for each celltype's top-50 peaks, what biological processes are the linked genes involved in? (dot plot: celltype × GO term)
- **Gene locus view generation**: create the gata4 locus figure with multi-celltype peak annotations
- **Broader marker gene set**: extend reverse-lookup profiles to the 6 interesting celltypes (foxd3, sox10, egr2b, pax6a, vsx2, ctslb)
