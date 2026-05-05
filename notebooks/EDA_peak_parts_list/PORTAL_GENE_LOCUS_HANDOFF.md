# Gene Locus Explorer — Web Portal Handoff

Companion to `PORTAL_HANDOFF.md` (celltype-driven view). Adds a
**gene-driven** entry point: user types a gene → sees all its
associated regulatory peaks at the genomic locus → can drill into any
peak.

---

## Overview

> *"Pick a gene → see its regulatory peak landscape across the chromosome,
> color-coded by which cell types use each peak, scaled by absolute
> openness — then click a peak to see motifs, temporal dynamics,
> and TF biology."*

Two complementary views are now available on the portal:

| Existing page | New page |
|---|---|
| Celltype → top peaks | Gene → all associated peaks |
| "Show me MHB-specific peaks" | "Where are pax2a's regulatory elements?" |
| Driven by V3 z-score ranking | Driven by `linked_gene` / `nearest_gene` annotation |

The two link to each other: clicking a gene name in the existing page
opens the new gene-locus view; clicking a peak in either view opens
the same per-peak detail panel.

---

## User Flow

```
1. Gene search box
   └─ Autocomplete from danRer11 GTF gene_name attribute (~32K genes)

2. Gene selected (e.g., pax2a)
   └─ Locus view (the headline plot, see "Visual encoding" below)
   └─ Side panel: peak table — sortable by composite_score / max_z /
                                distance_to_target_tss / peak_type
   └─ Filter chips: peak_type, top1_celltype, "in pax2a-expressed
      tissues vs off-tissue", min_z slider

3. Click a peak block (or table row)
   └─ Deep-dive panel (3 sub-panels, see "Per-peak detail" below)

4. Multi-select peaks → download
   └─ BED, FASTA (full peak), FASTA (200 bp core), CSV with metadata
```

---

## Visual encoding for the locus view (the headline plot)

A single horizontal track. All peaks for the gene visible at once.

| Visual element | Encoding |
|---|---|
| X-position | Genomic coordinate on the host chromosome |
| Block width | Peak length (with min visible width for tiny peaks) |
| **Block height** | **Absolute log_norm accessibility in top1 celltype** — the predictor of "will this region drive activity" |
| Block color | Top1 celltype (canonical Zebrahub palette) |
| Hover | Tooltip: peak_id, top1_celltype + z, length, peak_type, top 3 TFs, distance_to_target_tss |
| Below: gray bar | Chromosome ruler (kb / Mb tick marks) |
| Below: gene track | Thin line = introns, blue boxes = exons, red arrow = TSS |
| Right legend | Celltypes present in this gene's peak set |

**Important: height = accessibility, NOT z-score.** They are nearly
uncorrelated (r = −0.12 for pax2a). Z-score = specificity (how
cell-type-restricted); accessibility = activity potential. For
predicting whether a synthesized construct will drive expression,
accessibility is the right metric.

Static example: `figures/peak_parts_list/V3/marker_gene_queries/pax2a_locus_view.{pdf,png}`

---

## Per-peak detail panel (3 sub-panels)

Clicking a peak block opens a modal / inline expansion with three
sub-panels:

**Sub-panel A — Cell-type specificity** (bar chart)
- Bars: V3 z-score for this peak in all 31 celltypes, sorted desc
- Top1 celltype highlighted with red edge
- Y-axis: z-score; bars colored by celltype canonical color

**Sub-panel B — Temporal specificity** (bar chart)
- Within the top1 celltype: leave-one-out z-score across the 6 somite
  stages (0, 5, 10, 15, 20, 30 somites)
- Annotates raw log_norm value above each bar
- Bar color matches top1 celltype

**Sub-panel C — TF motif track**
- Horizontal track of the peak (gray)
- FIMO hits as boxes, colored by **TF family** (PAX, SOX, FOX, HOX,
  GATA, TBX, OTX, EN, ETV, KLF, NK, HD, bHLH, NR, IRF, …)
- Top 8 hits labeled with TF name above
- 200 bp core window highlighted with dashed red
- Compact-segment "hubs" (for long peaks) shaded light blue

**Sub-panel D — TF biology summary** (text)
- Counts: "EXPLICIT regulators: N | IMPLICIT (neural/dev): N |
  background ZF: N"
- Lists the explicit TFs with PMID citations (when agent-curated table
  is available; falls back to MHB hardcoded list otherwise)

Static example: `figures/peak_parts_list/V3/marker_gene_queries/pax2a_mhb_3panel/peak_summary_rank01_*.{pdf,png}`

---

## Backend / data files

The static pipeline produces all the per-gene data the portal needs.
Run for any gene via:

```bash
PYTHON=/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python
ROOT=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis

$PYTHON $ROOT/scripts/utils/gene_locus_explore.py \
    --gene <gene_name> \
    --output-dir <portal-data-dir>/gene_locus/<gene_name>/ \
    --target-celltype <primary_tissue>          # optional
    --permissive-celltypes <a,b,c>              # optional
    --prefer-distal --compact                   # optional
```

**Per-gene output directory** (drop into `portal-data/gene_locus/<gene>/`):

| File | Purpose |
|---|---|
| `<gene>_peaks.csv` | All peaks linked/nearest the gene + V3 z-scores |
| `<gene>_jaspar_fimo_hits.csv` | All FIMO motif hits (peak × motif × position) |
| `<gene>_jaspar_fimo_binary.npz` | Boolean (n_peaks × 1443) hit matrix |
| `<gene>_enhancer_ranking.csv` | **Master per-peak table** — drives the locus view + detail panel |
| `<gene>_summary.txt` | Top-15 quick view (human-readable) |
| `<gene>_locus_view.{pdf,png}` | Static fallback |
| `<gene>_motif_maps/rank{NN}_*.{pdf,png}` | Static motif track per peak |
| `<gene>_3panel/peak_summary_rank{NN}_*.{pdf,png}` | Static deep-dive panels (top-N) |
| `<gene>_tf_research_brief.md` | Markdown w/ ZFIN/NCBI/PubMed query URLs (agent input) |
| `<gene>_tf_biology_table.csv` | TF → category mapping (agent fills) |

For an interactive portal you'll want to **read the CSVs** and
re-render the figures dynamically (D3.js / Plotly / vega-lite). The
PNG/PDFs serve as static fallback or for download.

---

## Master ranking CSV schema (the portal's primary data source)

`<gene>_enhancer_ranking.csv` — one row per peak. Sorted by
`use_case_score` (or `composite_score` if no bonuses applied).

**Identity / coordinates**
- `peak_id`, `chrom`, `start`, `end`, `length`
- `peak_type` ∈ {promoter, exonic, intronic, intergenic}
- `distance_to_tss` (nearest gene's TSS)
- `distance_to_target_tss` (centroid → target gene's TSS, in bp; this gene's TSS)
- `linked_gene`, `nearest_gene`, `query_gene`, `via_linked`, `via_nearest`

**Cell-type specificity**
- `top1_celltype`, `top1_z` (V3 leave-one-out z-score)
- `top2_celltype`, `top2_z`, `top3_celltype`, `top3_z`
- `max_z`, `max_accessibility`, `tau`, `gini`

**TF content**
- `n_motif_hits`, `n_unique_tfs`, `n_unique_motifs`, `median_neg_log_p`
- `top_tfs` (comma-separated top 8 TFs by p-value)

**Synthesis-friendly geometry**
- `core_200bp_start`, `core_200bp_end`, `core_200bp_n_tfs` (densest 200 bp window)
- `compact_segments` (semi-colon-delimited absolute coords of motif hubs)
- `compact_length`, `compact_n_segments` (sum of hub widths; n hubs)
- `synthesis_length_factor` ∈ [0.4, 1.0] (IDT-cost proxy)

**Scoring components**
- `rank_specificity`, `rank_activity`, `rank_tf_density`, `rank_motif_strength`
- `base_score`, `peak_type_factor`, `composite_score`
- `target_match`, `top1_in_permissive` (booleans for the bonuses)
- `use_case_score` (final sort)
- `rank` (1-indexed, sorted by `use_case_score`)

---

## Agent-curated TF biology table

`<gene>_tf_biology_table.csv` — produced as a stub by
`tf_biology_lookup.py`, filled by an agent reading the `_brief.md`.

| Column | Meaning |
|---|---|
| `tf` | TF name from JASPAR2024 motif label |
| `n_peaks_bound`, `fraction_peaks` | Coverage stats |
| `best_pvalue`, `median_pvalue` | Hit strength |
| `jaspar_accession` | Up to 3 JASPAR IDs |
| **`category`** | Agent fills: `EXPLICIT` / `IMPLICIT` / `IRRELEVANT` |
| **`citations`** | Agent fills: PMID/DOI/ZFIN (semicolon-delimited) |
| **`notes`** | Agent fills: 1-line mechanism / role |

The portal can **show citations on hover** in sub-panel D, with each
one as a clickable link to PubMed/ZFIN. This makes the TF biology
classification verifiable rather than opaque.

---

## Implementation suggestions

**Locus view rendering** — a horizontal D3.js / Plotly track works
well. Anchor the X-axis to chromosome coordinates. Use the canonical
celltype colors from `scripts/utils/module_dict_colors.py` (this is
the source of truth for the palette across all Zebrahub products).

**Tooltip** — show peak_id, top1_celltype + z, length, peak_type,
top_tfs, distance_to_target_tss. The full per-peak detail panel
expands on click.

**Per-gene precomputation** — the pipeline takes 1–3 min per gene. For
a portal with hundreds of marker genes, run `gene_locus_explore.py`
in batch (SLURM array) once and ship the resulting per-gene
directories. The bulk of compute is FIMO; everything else is fast.

**Gene autocomplete** — pull names from the GTF:
`/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/genes/genes.gtf.gz`,
`gene_name` attribute on `transcript` lines.

**Cross-linking** — the existing celltype-page peak table has
`linked_gene` and `nearest_gene` columns. Make those clickable → opens
the new gene-locus page for that gene.

**Reuse** — the master peak metadata cache parquet
(`outputs/V3/peak_metadata_cache.parquet`, 77 MB) is already keyed by
peak_id and joined with V3 z-scores + tau metrics. The portal can
read this once at startup and serve any per-peak query in O(1) without
loading the 4.4 GB master h5ad.

---

## Source code (repo: `zebrahub-multiome-analysis`, branch `peak-parts-list`)

All in `scripts/utils/`:
- `gene_locus_explore.py` — one-command driver
- `marker_gene_peaks.py` — gene → peaks
- `run_fimo_on_peaks.py` — TF motif scan
- `rank_synthetic_enhancers.py` — composite-score ranking
- `plot_peaks_locus_view.py` — locus-track figure
- `make_peak_3panel_figures.py` — per-peak deep-dive figure
- `tf_biology_lookup.py` — TF research brief generator
- `gtf_helpers.py` — shared TSS / exon lookup
- `GENE_LOCUS_EXPLORER.md` — vision + extension guide
- `SYNTHETIC_ENHANCER_PIPELINE.md` — AgenticCRE-focused pipeline doc

Sanity-tested on `pax2a` (33 peaks, MHB target) and `sox10` (26 peaks,
neural crest target).
