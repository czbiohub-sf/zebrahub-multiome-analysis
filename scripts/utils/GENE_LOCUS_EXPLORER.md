# Gene Locus Explorer

**Vision:** *"Pick a gene → see its regulatory peak landscape across the
chromosome, color-coded by which cell types use each peak, scaled by
absolute openness — then drill into any peak to see motifs, temporal
dynamics, and TF biology."*

This page is the entry point for that workflow. All utilities are
gene-agnostic — pass any gene name in the danRer11 / GRCz11 annotation
and the pipeline runs end-to-end.

---

## Status (2026-05-05)

- ✅ Static figures and CSVs work for any gene
- 🟡 The "click each peak" interactivity is not yet built — current
  outputs are PDF/PNG. The CSV side has all the per-peak metadata
  ready for an interactive frontend.
- 🔭 Future: Streamlit / D3.js interactive locus track that loads the
  ranking CSV + motif hits and reveals per-peak panels on click.

---

## One-command end-to-end

```bash
PYTHON=/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python
ROOT=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis

$PYTHON $ROOT/scripts/utils/gene_locus_explore.py \
    --gene sox10 \
    --output-dir results/sox10/ \
    --target-celltype neural_crest \
    --permissive-celltypes differentiating_neurons,neural_optic,enteric_neurons \
    --prefer-distal --compact \
    --top-n-3panel 10
```

This produces (in `results/sox10/`):

```
sox10_peaks.csv                     # all peaks linked/nearest to sox10
sox10_jaspar_fimo_*.csv / .npz      # FIMO TF motif hits (JASPAR2024)
sox10_enhancer_ranking.csv          # master ranked DataFrame
sox10_summary.txt                   # top-15 quick view
sox10_locus_view.{pdf,png}          # CHROMOSOME-LEVEL view (the
                                      "comparable across peaks" plot)
sox10_motif_maps/rank{NN}_*.{pdf,png} # per-peak motif tracks
sox10_3panel/peak_summary_rank{NN}_*.{pdf,png}  # top-N drill-down
```

Sanity-tested on `pax2a` (33 peaks, MHB target) and `sox10` (26 peaks,
neural_crest target). Total runtime ≈ 1–3 min per gene.

---

## The locus-view figure (the headline plot)

**Single image. All peaks for the gene. Comparable at a glance.**

| Visual encoding | Meaning |
|---|---|
| X-position | Genomic coordinate on the host chromosome |
| Block width | Peak length (with min visible width for tiny peaks) |
| **Block height** | **Absolute accessibility (log_norm)** in the peak's top1 celltype — the predictor of "will this region drive activity" |
| Block color | Top1 celltype (canonical Zebrahub palette) |
| Gray bar (row 2) | Chromosome ruler with kb/Mb tick marks |
| Gene track (row 3) | Gene body parsed from GTF — thin line = introns, blue boxes = exons, red arrow = TSS |
| Right legend | Celltypes present in the peak set |

**Key choice — accessibility over z-score:** Earlier versions used the
V3 specificity z-score for height. We switched to absolute log_norm
accessibility because the two metrics are nearly uncorrelated for
real peaks (r = -0.12 for pax2a). For predicting *activity* of a
synthesized construct, **how open is the chromatin?** beats **how
specific?**.

---

## Workflow chain (each script is reusable on its own)

```
                 [GTF] gene_name → TSS coordinate
                          ↓
gene_locus_explore.py  ──► one-command driver
   │
   │  (1) marker_gene_peaks.py
   │      gene → peaks_for_genes() → peaks CSV
   │      (peak coords + V3 z-scores per celltype + tau)
   │
   │  (2) run_fimo_on_peaks.py
   │      peaks CSV + danRer11 FASTA + JASPAR2024 → FIMO hits CSV
   │      (one row per peak × motif × position; 1067 hits for 33 pax2a peaks)
   │
   │  (3) rank_synthetic_enhancers.py
   │      peaks + FIMO hits → master ranking CSV
   │      composite_score (specificity × activity × tf_density × motif_strength
   │                       × peak_type_factor)
   │      use_case_score = composite + target_bonus + permissive_bonus
   │                                  + distal_bonus, × synthesis_length_factor
   │      + 200 bp core window per peak
   │      + (optional) compact-segment hubs for long peaks
   │
   │  (4) plot_peaks_locus_view.py
   │      peaks + GTF → locus-track figure
   │      (the headline "all peaks at once" view)
   │
   │  (5) make_peak_3panel_figures.py
   │      top-N peaks → per-peak deep dive PDF/PNG
   │      Panel 1A celltype z-score bar plot (31 celltypes)
   │      Panel 1B timepoint z-score within top1 celltype
   │      Panel 2  TF motif track (family-colored, 200 bp core, compact hubs)
   │      Panel 3  text summary classifying TFs as EXPLICIT / IMPLICIT MHB-relevant
   │              (the EXPLICIT/IMPLICIT lists are MHB-curated — see "Generalizing
   │              to other tissues" below)
```

---

## Generalizing to other tissues / genes

The pipeline is gene-agnostic. Three things still need tissue context:

1. **TF biology curation** (the EXPLICIT / IMPLICIT classification
   shown in each peak's text panel). Default: hardcoded MHB list in
   `make_peak_3panel_figures.py`. **For non-MHB tissues, use the
   agent-curation workflow below — no code edits required.**

2. **`--target-celltype` and `--permissive-celltypes`.** Already
   user-supplied CLI args — pass any combination relevant to your
   gene. No code change needed.

3. **GTF.** Default is danRer11 zebrafish
   (`/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/genes/genes.gtf.gz`).
   For other species pass `--gtf` to the relevant utilities.

### Agent-driven TF curation workflow (replaces hardcoded MHB list)

Instead of hand-curating TF lists per tissue, use this two-step flow:

```bash
# 1. After running gene_locus_explore.py, generate the research brief:
$PYTHON scripts/utils/tf_biology_lookup.py \
    --fimo-hits results/<gene>/<gene>_jaspar_fimo_hits.csv \
    --label <gene> \
    --target-tissue <celltype> \
    --target-gene <gene> \
    --output-dir results/<gene>/ \
    --top-n 25
```

This produces:
- `<gene>_tf_research_brief.md` — markdown with prepared **ZFIN, NCBI
  Gene, Alliance Genome, PubMed** query URLs for each of the top-N
  TFs. URLs include the target tissue and gene as search terms so the
  TF + tissue association is what's being looked up.
- `<gene>_tf_biology_table.csv` — stub table with columns `tf,
  n_peaks_bound, fraction_peaks, best_pvalue, jaspar_accession,
  category, citations, notes`. The `category, citations, notes`
  columns are blank, ready for an agent to fill.

```bash
# 2. An agent (Claude with WebSearch / WebFetch, or a researcher) opens
#    the brief, follows the prepared URLs, and fills in for each TF:
#       category   ∈ {EXPLICIT, IMPLICIT, IRRELEVANT}
#       citations  PMID / DOI / ZFIN entries (semicolon-delimited)
#       notes      1-line mechanism / role
```

```bash
# 3. Re-run the 3-panel figures pointing at the filled CSV:
$PYTHON scripts/utils/make_peak_3panel_figures.py \
    --ranking-csv results/<gene>/<gene>_enhancer_ranking.csv \
    --fimo-hits   results/<gene>/<gene>_jaspar_fimo_hits.csv \
    --output-dir  results/<gene>/<gene>_3panel/ \
    --top-n       10 \
    --tf-biology-csv results/<gene>/<gene>_tf_biology_table.csv
```

The text panels now classify TFs based on the agent's tissue-specific
curation rather than the hardcoded MHB list.

**Why this design.** The set of "MHB explicit" TFs (PAX, EN, OTX, …)
is hand-curation that breaks for every new tissue. By moving the
classification into a per-query CSV that an agent fills via
literature search, the pipeline scales to any gene/tissue without
code edits — and the agent's reasoning + citations are preserved as
part of the analysis trail.

---

## Vision: interactive locus explorer (not yet implemented)

Current outputs are static figures. The data is already structured
for interactivity — the master ranking CSV has every per-peak field
needed:

```
peak_id, chrom, start, end, length, peak_type, distance_to_target_tss,
top1_celltype, top1_z, top2_celltype, top2_z, max_accessibility,
n_unique_tfs, top_tfs, core_200bp_start/end, compact_segments, …
```

A future interactive version would:

- Render the locus track in the browser (D3.js / Plotly / Streamlit)
- Show tooltip on hover (gene, accessibility, top TFs, distance to TSS)
- Click → expand the per-peak 3-panel view inline
- Filter by celltype / peak_type / score / distance
- Export selected peaks → seed sequences for AgenticCRE

The static PDFs in `{label}_3panel/` show what the per-peak panel
should contain. The CSVs already drive every visual encoding.

---

## Utilities (all gene-agnostic; in `scripts/utils/`)

| File | Purpose |
|---|---|
| `gene_locus_explore.py` | **One-command driver** — runs steps 1–5 for any gene |
| `marker_gene_peaks.py` | `peaks_for_genes()` — gene → peaks CSV |
| `build_peak_metadata_cache.py` | Build the 77 MB parquet cache from the master h5ad (one-time) |
| `run_fimo_on_peaks.py` | FIMO TF motif scan on any peaks CSV (JASPAR2024 / H12CORE / CIS-BP) |
| `rank_synthetic_enhancers.py` | Composite-score ranking + 200 bp core + compact hubs |
| `plot_peaks_locus_view.py` | Chromosome-track view (the headline figure) |
| `make_peak_3panel_figures.py` | Per-peak 3-panel deep dive (accepts `--tf-biology-csv` for agent-curated tissue classification) |
| `tf_biology_lookup.py` | Aggregates top-N TFs from FIMO output → research-brief markdown + stub CSV for agent curation |
| `gtf_helpers.py` | `get_gene_tss()`, `get_gene_struct()` — shared GTF utilities |
| `SYNTHETIC_ENHANCER_PIPELINE.md` | AgenticCRE-focused pipeline doc (DE-genes → synthesis seeds) |
| `GENE_LOCUS_EXPLORER.md` | This file |

All utilities accept `--help`. Each is independently runnable; the
orchestrator (`gene_locus_explore.py`) just chains them with the
right argument plumbing.

---

## Quickstart for a new gene (no MHB)

```bash
# Neural crest — sox10 example
$PYTHON $ROOT/scripts/utils/gene_locus_explore.py \
    --gene sox10 \
    --output-dir results/sox10/ \
    --target-celltype neural_crest

# Heart — gata4 example
$PYTHON $ROOT/scripts/utils/gene_locus_explore.py \
    --gene gata4 \
    --output-dir results/gata4/ \
    --target-celltype heart_myocardium \
    --permissive-celltypes endoderm,lateral_plate_mesoderm

# Skeletal muscle — myod1 example
$PYTHON $ROOT/scripts/utils/gene_locus_explore.py \
    --gene myod1 \
    --output-dir results/myod1/ \
    --target-celltype fast_muscle \
    --permissive-celltypes muscle,somites
```

The pipeline figures out the rest — TSS lookup, peaks, motif scanning,
ranking, and figures.
