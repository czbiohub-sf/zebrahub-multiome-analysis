# Peak Parts List — Biological Validation Plan

## Core question
Do the genes near the most celltype-specific ATAC peaks actually make biological sense for that celltype?

---

## Tier 1 — RNA-ATAC concordance (PRIORITY) ✅ IN PROGRESS

**Concept**: Compare ATAC peak specificity (V3 z-score) against RNA expression specificity for the same gene in the same celltype.

**Scatter plot**:
- x-axis: V3 ATAC z-score (celltype-level specificity for that peak)
- y-axis: RNA expression z-score (same celltype, same gene)
- One dot = one peak-gene pair (top 50 peaks per celltype, filtered to those with linked/associated genes)
- Color by celltype
- Show Pearson/Spearman correlation per celltype

**Peak → gene assignment**:
- Use `linked_gene` first, fall back to `nearest_gene`
- Only include peaks where at least one gene annotation exists (no unannotated peaks)
- If multiple peaks map to the same gene, keep all (each dot = peak-gene pair)

**RNA z-score**:
- Compute per-celltype mean expression across reliable timepoints (same logic as ATAC V3)
- Apply leave-one-out z-score across celltypes → directly comparable to ATAC z-score

**Output figures**:
- Per-celltype scatter (grid): ATAC z vs RNA z, with Pearson r and p-value
- Combined scatter: all celltypes overlaid, color-coded, with regression line
- Supplementary table: all peak-gene pairs with both z-scores

**Script**: `notebooks/EDA_peak_parts_list/09e_rna_atac_concordance.py`

---

## Tier 2 — GO/pathway enrichment

**Concept**: Run GO biological process enrichment on the gene list per celltype.

- Tool: `gprofiler-official` Python package (REST API, no R needed)
- Background: all annotated genes in the 640K peak set
- Expected: fast_muscle → "muscle contraction", heart → "cardiac development", neural_crest → "neural crest migration"
- Output: dotplot or heatmap — celltypes × top GO terms

**Script**: `notebooks/EDA_peak_parts_list/09f_go_enrichment.py`

---

## Tier 3 — Known marker gene overlap

**Concept**: Compute overlap between top-50 peak genes and curated celltype marker lists.

- Sources: ZFIN expression DB, CellMarker 2.0, published Zebrahub RNA markers
- Metric: fraction of top-50 peak genes in known marker list; Fisher's exact enrichment
- Output: supplementary table + bar plot

**Script**: `notebooks/EDA_peak_parts_list/09g_marker_overlap.py`

---

## Tier 4 — LiteMind biological interpretation

**Concept**: Feed gene lists into `scripts/litemind_peak_analysis/` for automated PubMed/ZFIN/GO queries.

- Produces narrative paragraphs per celltype
- Already integrated in repo
- Use as supplementary text or to guide figure annotation

---

## Priority order for manuscript

1. **Tier 1** (RNA-ATAC concordance) — most novel, directly validates parts list concept
2. **Tier 2** (GO enrichment) — standard validation, expected by referees
3. **Tier 3** (marker overlap) — quick sanity check, good for supplementary

---

## Key design decisions

- **Gene definition**: `linked_gene` > `nearest_gene`; skip unannotated peaks
- **Top N**: top 50 per celltype, filtered to gene-annotated peaks only
- **Celltype scope**: all 31 celltypes (or focus on ★★★★★ first)
- **RNA z-score**: leave-one-out across celltypes, same as ATAC V3 formula
