# Handoff — Run `gene_locus_explore.py` on multiple genes

*For: a fresh Claude Code session in this repo.*
*Repo:* `zebrahub-multiome-analysis`
*Branch:* `gene-locus-explore-batch` (already created — `git checkout`
this branch in your session).
*Prerequisite work:* PR #16 (the peak-parts-list pipeline). All
utilities and the peak metadata cache already exist on this branch.

---

## Mission

Run the `gene_locus_explore.py` end-to-end pipeline on **a curated set
of marker genes spanning different cell types**, producing per-gene
output directories with synthesis-ready ≤500 bp DNA elements and a
cross-gene summary table.

The pipeline is gene-agnostic and already sanity-tested on pax2a (MHB)
and sox10 (neural crest). The goal of this session is to (a) extend the
catalog, and (b) exercise the principles document on real new
analyses to validate it.

**Time budget:** ~1–3 min per gene of FIMO compute + manual curation
~5–10 min per gene = 2–4 hours total for 6 genes.

---

## TL;DR — the one-command driver

```bash
PYTHON=/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python
ROOT=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis

$PYTHON $ROOT/scripts/utils/gene_locus_explore.py \
    --gene <gene_name> \
    --output-dir $ROOT/notebooks/EDA_peak_parts_list/outputs/V3/marker_gene_queries/<gene_name>/ \
    --target-celltype <primary_celltype> \
    --permissive-celltypes <comma,separated,list> \
    --prefer-distal --compact \
    --top-n-3panel 10
```

This chains 5 steps: peaks → FIMO → ranking → locus view → top-10
3-panel deep dives. It also auto-generates a TF research brief
(`*_tf_research_brief.md`) and stub biology table for agent curation.

Each gene takes ~1–3 min compute. No SLURM needed.

---

## Candidate genes — recommended set

Six genes across five tissue types. Picked for biological clarity (each
has a canonical top-1 celltype expectation) and diversity (covers
neural crest, heart, skeletal muscle, notochord, hindbrain, endoderm).

| Gene | Expected top1 | Permissive (other tissues) | Why this gene |
|---|---|---|---|
| **sox10** | neural_crest | differentiating_neurons, neural_optic, enteric_neurons | Master regulator of neural crest; well-studied enhancers (NC1, NC4) |
| **foxd3** | neural_crest | neural, fast_muscle, somites | Compare with sox10 — earlier NC regulator; multi-tissue |
| **gata4** | heart_myocardium | endoderm, lateral_plate_mesoderm | Heart driver; endoderm precursor signal |
| **myod1** | fast_muscle | muscle, somites | Muscle bHLH master regulator |
| **tbxta** (brachyury) | notochord | NMPs, tail_bud | Strongest single-tissue marker — good contrast with multi-tissue pax2a |
| **egr2b** (krox20) | hindbrain | neural_posterior, differentiating_neurons | Hindbrain rhombomere 3/5 marker |

You can also extend with **gata1a** / **tal1** (hemangioblasts) or **otx2a** (forebrain/midbrain) if time permits.

---

## Workflow per gene

For each gene in the candidate set:

### Step 1 — Run the driver

```bash
$PYTHON $ROOT/scripts/utils/gene_locus_explore.py \
    --gene sox10 \
    --output-dir $ROOT/notebooks/EDA_peak_parts_list/outputs/V3/marker_gene_queries/sox10/ \
    --target-celltype neural_crest \
    --permissive-celltypes differentiating_neurons,neural_optic,enteric_neurons \
    --prefer-distal --compact --top-n-3panel 10
```

Wait for the "DONE" message. Inspect:
- `{gene}_peaks.csv` — how many peaks (should be ~10–60)
- `{gene}_summary.txt` — top-15 ranked peaks
- `{gene}_locus_view.{pdf,png}` — chromosome track view

### Step 2 — Apply principle ② (verify annotations)

For the top 5 peaks, check `peak_type`:

- If `peak_type == 'exonic'` for an upstream/downstream peak, **verify it's
  the target gene's exon**, not a neighbor's:
  ```bash
  GTF=/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/genes/genes.gtf.gz
  zcat $GTF | awk '$3=="exon" && $1=="<chr>" && $4<=<peak_end> && $5>=<peak_start>' \
    | grep -o 'gene_name "[^"]*"' | sort -u
  ```
- If the exon overlaps a different gene, **drop that peak** from
  synthesis candidates (it was the −10.7 kb pax2a case → CT025909.4).

### Step 3 — Pick 2–4 synthesis candidates manually

Open `{gene}_locus_view.pdf` (chromosome track) + the top-N 3-panel
PDFs. Apply principles ①, ⑥:

- Want at least 1 peak with `top1_celltype == target` AND high
  accessibility (taller bar in locus view)
- If the gene is multi-tissue (like pax2a), pick 1 peak per permissive
  tissue too — these are **alternative regulatory elements**, not
  off-target noise
- Promoter peak (close to TSS): worth including for proximal context
- Avoid "exonic-in-neighbor-gene" peaks (per Step 2)

Build a `selected_peaks.csv` with a `selection_label` and
`selection_note` column for each. Example:
```csv
selection_label,peak_id,selection_note
promoter,13-29768232-29769128,canonical promoter ~2 kb upstream
distal_NC_enhancer,3-1456789-1457400,strong NC peak ~10 kb 5' of TSS
alternative_otic,...,alternative tissue regulator
```

### Step 4 — Generate ≤500 bp designs

```bash
$PYTHON $ROOT/scripts/utils/design_short_element.py \
    --ranking-csv {gene}/<selected_peaks_ranking>.csv \
    --fimo-hits   {gene}/{gene}_jaspar_fimo_hits.csv \
    --output-dir  {gene}/selected_for_synthesis/designs/ \
    --max-len 500
```

This produces Scheme A (single 500 bp window) and Scheme B (stitched
hubs ≤500 bp). For each selected peak, the PDF shows the comparison.

### Step 5 — Extract DNA sequences

```python
# inside a python script or notebook
import pysam, pandas as pd
fa = pysam.FastaFile("/hpc/reference/sequencing_alignment/fasta_references/danRer11.primary.fa")
# Read design_summary.csv, fetch each schemeA + schemeB sequence, write sequences.fasta
```

Use the same FASTA-extraction pattern as in
`pax2a/selected_for_synthesis/sequences.fasta` (see that file for the
exact format).

### Step 6 — Write per-gene INTERPRETATION.md

A short markdown (1 page) summarizing:
- What we picked and why
- The top biological TFs (cite from `tf_research_brief.md`)
- Caveats specific to this gene (principle ⑨, ⑭)

Template:

```markdown
# {gene} synthetic enhancer candidates

**Gene:** {gene}, expressed in {tissues}
**Pipeline:** gene_locus_explore.py + design_short_element.py
**Peaks examined:** {N total} → {N selected}

## Selected peaks
- **{label1}** ({coords}, {peak_type}, top1={top1}, z={z})
  Picked because: ...
- ...

## Top TFs (from FIMO + biology brief)
- EXPLICIT: ...
- IMPLICIT: ...

## Caveats / open questions
- ...
```

### Step 7 — Apply principle ⑫ (cleanup)

Verify `{gene}/` directory structure matches the canonical layout:
```
{gene}/
├── {gene}_peaks.csv
├── fimo/{gene}_jaspar_fimo_*.csv/.npz
├── ranking/{gene}_enhancer_ranking.csv, {gene}_summary.txt
├── tf_biology/{gene}_tf_research_brief.md, {gene}_tf_biology_table.csv
├── locus_view/{gene}_locus_view.{pdf,png}
├── 3panel_top10/peak_summary_rank{NN}_*.{pdf,png}
├── motif_maps_all/rank{NN}_*.{pdf,png}
└── selected_for_synthesis/
    ├── selected_peaks.csv
    ├── 3panel/
    ├── designs/
    ├── design_summary.csv
    ├── sequences.fasta
    └── INTERPRETATION.md
```

The driver creates the first six folders automatically. You create the
`selected_for_synthesis/` subtree manually.

---

## Cross-gene summary (do this at session end)

After all genes are processed, write a top-level summary:

```
outputs/V3/marker_gene_queries/CROSS_GENE_SUMMARY.md
```

Contents:
- Table: gene | n_peaks | n_selected | top1_celltypes | seq count
- Comparisons:
  - Which gene had the most peaks? Fewest?
  - Which had the cleanest top1 = target (single-tissue) vs spread
    (multi-tissue)?
  - Notable activator/repressor patterns across genes
- Aggregate `all_genes_sequences.fasta` combining all picks

Apply principle ⑪ — what insights generalize across the catalog?

---

## Apply the 14 reasoning principles throughout

This session is also a **stress-test of the principles doc**
(`scripts/utils/AGENT_REASONING_LESSONS.md`). At each step:

- **① "What metric?"** — locus-view block HEIGHT = absolute accessibility,
  not z-score. Don't mix them up when comparing.
- **② "Verify annotations"** — Step 2 above. Run the GTF check.
- **③ "Don't hard-code biology"** — use `--target-celltype` and
  `--permissive-celltypes` CLI args, not script edits.
- **⑥ "Bonuses, not penalties for off-target"** — for multi-tissue
  genes, permissive list captures alternative regulators.
- **⑨ "Caveats next to output"** — INTERPRETATION.md per gene.
- **⑭ "Communicate decisions"** — `selection_note` column.

Full principles list:
`scripts/utils/AGENT_REASONING_LESSONS.md` (long-form)
`scripts/utils/AGENT_REASONING_QUICKREF.md` (1-page)

If you find a new failure mode this session that's not in the
principles doc, **flag it for §15** — add to the doc with the concrete
example.

---

## Definition of done — per gene

- [ ] Driver ran successfully (`{gene}_summary.txt` exists)
- [ ] Annotation verification done for top-5 peaks (no neighbor-exon
      issues)
- [ ] `selected_peaks.csv` written with 2–4 peaks + `selection_note`
- [ ] `design_short_element.py` ran on the selection
- [ ] `sequences.fasta` extracted
- [ ] `INTERPRETATION.md` written

## Definition of done — for the session

- [ ] 4–6 genes processed (suggested: sox10, foxd3, gata4, myod1, tbxta, egr2b)
- [ ] Each gene has a complete `{gene}/` directory
- [ ] `CROSS_GENE_SUMMARY.md` written at the top level
- [ ] All-genes `all_genes_sequences.fasta` aggregated (NEW)
- [ ] Commits pushed to `gene-locus-explore-batch` branch
- [ ] PR opened against `peak-parts-list` (or `main` if PR #16 has merged)

---

## Background context — for the new session

The pax2a worked example lives at:
```
outputs/V3/marker_gene_queries/pax2a/
```

Read it as a reference. Specifically:
- `pax2a/selected_for_synthesis/selected_peaks.csv` — example of curation
- `pax2a/tf_function_analysis/INTERPRETATION.md` — example of writeup
- `pax2a/selected_for_synthesis/sequences.fasta` — example of FASTA format

Source-of-truth docs:
- Pipeline overview: `scripts/utils/GENE_LOCUS_EXPLORER.md`
- Per-step details: `scripts/utils/SYNTHETIC_ENHANCER_PIPELINE.md`
- Reasoning principles: `scripts/utils/AGENT_REASONING_LESSONS.md`
- Repo navigator: `outputs/V3/marker_gene_queries/README.md`

---

## Self-check before declaring this session done

- ⑪ Did you generalize anything that should be a utility?
   (E.g., the FASTA-extraction pattern could become
   `scripts/utils/extract_design_sequences.py` if reused.)
- ⑫ Did you write `CROSS_GENE_SUMMARY.md` and aggregate FASTA?
- ⑬ Were any destructive operations user-approved?
- ⑭ Does each per-gene `selected_peaks.csv` have a `selection_note`
   column explaining "why this peak"?

If any principle was hard to apply, note that for the registry — the
principles doc is a living document and should be improved when new
edge cases surface.

---

## Quick reference — files this session will create

| Path | Created by |
|---|---|
| `outputs/V3/marker_gene_queries/{gene}/...` | `gene_locus_explore.py` (auto) |
| `outputs/V3/marker_gene_queries/{gene}/selected_for_synthesis/selected_peaks.csv` | You, manually |
| `outputs/V3/marker_gene_queries/{gene}/selected_for_synthesis/designs/` | `design_short_element.py` |
| `outputs/V3/marker_gene_queries/{gene}/selected_for_synthesis/sequences.fasta` | Inline pysam script |
| `outputs/V3/marker_gene_queries/{gene}/selected_for_synthesis/INTERPRETATION.md` | You, per gene |
| `outputs/V3/marker_gene_queries/CROSS_GENE_SUMMARY.md` | You, at session end |
| `outputs/V3/marker_gene_queries/all_genes_sequences.fasta` | You, at session end |

All of the above are gitignored (`outputs/V3/` is gitignored). Only
this handoff doc + any new scripts you add go in git.
