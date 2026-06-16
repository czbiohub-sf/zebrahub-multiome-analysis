# Peak Parts List — Rationale, Background, and Context

*A catalog of celltype-specific cis-regulatory DNA elements from the zebrafish embryo multiome atlas.*

---

## The question

**What are the discrete regulatory elements that define each cell type during early vertebrate development, and can we assemble them into a queryable "parts list" for synthetic biology?**

Cell type identity is encoded by which genes are expressed, which in turn is controlled by which DNA regulatory elements (enhancers, promoters) are accessible and bound by transcription factors. Our multiome dataset — 94,562 cells profiled simultaneously for chromatin accessibility (ATAC) and gene expression (RNA) across 32 cell types and 6 developmental timepoints (0–30 somites) — gives us an unprecedented view of which ~640,000 genomic loci are "open for business" in each cell type at each stage.

This analysis transforms that raw map into a **parts list**: for any given cell type, the top regulatory elements that define its identity, annotated with their genomic coordinates, associated genes, TF binding motifs, and temporal dynamics.

---

## Why a "parts list"?

Three applications motivate this framing:

1. **Biological discovery** — Which TFs drive which cell types? A ranked catalog of specific elements per cell type lets us read out the regulatory logic without hypothesis-free guessing. The fact that known master regulators (GATA for hemangioblasts, AP2 for neural crest, EGR2/Krox20 for hindbrain) emerge directly from this approach validates it.

2. **Functional hypothesis generation** — If you want to know "what regulates gene X in cell type Y?", our parts list answers it: here are the peaks near gene X, ranked by how specific each is to cell type Y, along with which TFs likely bind there.

3. **Synthetic regulatory DNA design** — Deep learning models like [CREsted (Kempynck et al.)](https://www.biorxiv.org/content/10.1101/2025.04.02.646812v1) learn cell type-specific enhancer grammar from exactly this type of data. Our parts list provides both:
   - **Training exemplars**: high-confidence celltype-specific elements to learn from
   - **Design templates**: validated natural sequences that can be used as starting points for generating synthetic enhancers with cell type-specific activity

---

## Why this dataset is ideal

Most published ATAC/RNA atlases have one of these limitations:
- Single timepoint (can't separate celltype vs developmental stage)
- Coarse cell type annotations (lineage-level, not cell type-level)
- Small numbers of cell types (can't robustly compare across many backgrounds)

Our zebrafish multiome dataset covers:
- **31 reliably annotated cell types** (ML_coarse labels, >=20 cells per condition)
- **6 timepoints** spanning gastrulation → early organogenesis
- **640,830 consensus peaks** called across the full atlas
- **Matched RNA** for every cell, enabling cross-modal validation

This breadth is what makes the enrichment tests robust — when we ask "is GATA1 enriched in hemangioblast peaks?", the background is ~6,000 peaks from 30 other cell types, giving high statistical power.

---

## The methodology: why celltype-level z-score (V3)

We evolved through three versions before settling on V3:

**V1 — per-condition z-score** (celltype × timepoint)
- Problem: a peak accessible across all timepoints in one cell type was *penalized* as "uniformly accessible" even though it's actually highly celltype-specific. The method conflated celltype specificity with temporal specificity.

**V2 — shrinkage-regularized z-score**
- Improved statistical handling of low-n conditions, but still computed per-condition, not per-celltype.

**V3 — celltype-level leave-one-out z-score** *(current approach)*
- Step 1: For each cell type, average log-normalized accessibility across its reliable timepoints → 640,830 × 31 celltype-mean matrix
- Step 2: For each peak in each cell type: `z = (x - mean_others) / std_others`
- **This cleanly separates celltype specificity from temporal dynamics.** A peak constitutively accessible in heart_myocardium across all timepoints gets a high celltype-level z-score. Its temporal profile is captured separately.

The top-200 peaks per cell type (ranked by z-score) form the parts list.

---

## Validation: the approach recovers known biology

**1. Canonical TF motifs match known cell type biology**

Running FIMO on top-200 peaks and testing enrichment via Fisher's exact test against all 30 other cell types pooled (~6,000 peaks background), we recover:

| Cell type | Top enriched TFs (FDR < 0.05) | Known biology |
|---|---|---|
| Epidermis | P63, P73, P53, TEAD4, AP2 family | p63 is THE epidermal master TF; TEAD/Hippo known in skin |
| Neural crest | AP2C, AP2B, AP2A, SOX8 | AP2α is a defining NC specifier; SOX10 lineage marker |
| Hemangioblasts | GATA1, GATA6, GATA2, KLF4, GATA3 | GATA/KLF axis is canonical hematopoietic specification |
| Hindbrain | EGR2, EGR1, EGR3, WT1 | EGR2 (Krox20) defines rhombomere 3/5 identity |
| Optic cup | NOTO*, HXD3, HME2, LHX1, EMX2 | Homeobox TFs for eye field specification |
| Hatching gland | FOXI1, FOXA2, FOXJ3, FOXA1 | FOX family expressed in hatching gland |

*NOTO PWM likely reflects shared homeobox TAAT-core binding; the specific TF is ambiguous within the family.

**2. Marker gene promoters are captured unsupervised**

Examining hemangioblast top-20 peaks reveals that **promoters of well-known hematopoietic genes** are ranked highly purely from chromatin accessibility patterns, with no gene-level supervision:
- Rank 1: **slc4a1a** promoter (z=144.8) — Band 3, erythroid
- Rank 17: **hbbe2** promoter (z=78.4) — hemoglobin beta embryonic
- Rank 18: **gata1a** promoter (z=77.8) — master hematopoietic TF's own promoter
- Rank 19: **hemgn** promoter (z=75.6) — hemogen

That the approach independently recovers marker gene promoters from pure chromatin data corroborates the "parts list" claim.

**3. ATAC specificity correlates with RNA specificity**

For peaks linked to genes, the ATAC-based V3 z-score (peak specificity) correlates with the RNA-based z-score for the corresponding gene — confirming that the accessibility signal reflects gene regulatory activity, not chromatin noise.

---

## A key insight: motif presence as a causal filter

Not all celltype-specific peaks are **TF-driven enhancers**. A peak can be accessible in one cell type because:
- (A) TFs bind there directly and open the chromatin → **functional enhancer** ✓
- (B) Nearby chromatin remodeling passively exposes the region → **passenger**
- (C) It's a structural element (CTCF boundary, SINE repeat) → **architectural**
- (D) TFs bind via protein-protein tethering without recognizing DNA → **cofactor-driven**

For synthetic regulatory DNA design, we specifically want class (A). Our FIMO-based filter flags this: **peaks with >=3 enriched TF motif hits are candidate functional enhancers** (`has_motif_support = True`). Peaks without motif support may still be celltype-specific biologically, but they're poor templates for synthetic design because we don't have a mechanistic model for their activity.

This two-stage filter — (1) celltype specificity via V3 z-score, (2) motif support via FIMO enrichment — produces a distilled set of candidate building blocks.

---

## The "building blocks" story

The motif position maps reveal that many top peaks contain **clusters of multiple enriched TF motifs** within a ~500bp window. For example, hemangioblast rank 3 (near CABZ01069040.1, 472bp) contains:
- A GATA cluster (GATA1/2/3/4/5/6 all within ~50bp at position ~330)
- A TAL1/bHLH cluster at positions 130–190 and 300–340
- Flanked by SALL4 and KLF motifs

These elements look like **natural regulatory circuits** — co-localized TF binding sites that cooperatively specify cell type identity. For synthetic design, they can be used:
- As-is (direct reuse of natural sequences with validated activity)
- As templates (shuffle motif order, spacing, add/remove sites to tune activity)
- As training data for generative models that learn the grammar of celltype-specific enhancers

---

## Output: what's in the parts list

For each of 31 cell types, we provide **top 200 peaks** with:

- Genomic coordinates (chr:start-end, danRer11) + length
- Peak type (promoter, exonic, intronic, intergenic)
- Linked gene (Cicero co-accessibility) and nearest gene (TSS distance)
- V3 specificity z-score (how celltype-specific)
- Tau index (global specificity metric)
- Temporal accessibility profile (6 timepoints, with reliability flag)
- TF motif hits (which enriched TFs bind where within the sequence)
- `has_motif_support` flag (>=3 enriched TFs → candidate functional enhancer)

Users can query by cell type, filter by motif support, select candidates, extract FASTA sequences, and feed them into downstream tools (CREsted, enhancer activity prediction models, synthetic biology design pipelines).

---

## Data access

All files are on the HPC at paths documented in `PORTAL_HANDOFF.md`. Key entry points:

- **Primary parts list** (portal-ready, with motif annotations):
  `/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/V3_all_celltypes_top200_peaks_with_motifs.csv`
- **Motif positions** (exact TF binding site coordinates within each peak):
  `/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/V3_top200_motif_positions.csv`
- **Temporal profiles** (6-timepoint accessibility per peak):
  `/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_peak_parts_list/outputs/V3/V3_all_celltypes_top200_temporal_profiles.csv`
- **Reference genome** (for sequence extraction):
  `/hpc/reference/sequencing_alignment/fasta_references/danRer11.primary.fa`

For implementation details, see `PORTAL_HANDOFF.md`. For computational methodology, see `METHODS_DRAFT.md`. For the full session narrative, see `SESSION_SUMMARY_2026-04-07.md`.
