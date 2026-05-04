# Synthetic Enhancer Selection Pipeline

End-to-end workflow for going from a **list of marker / DE genes** to a
ranked set of **synthetic-enhancer candidate peaks** with TF motif maps
and 200 bp seed sequences.

Designed for handoff to AgenticCRE (motif-centered ISM seeding) or any
downstream synthetic-biology / regulatory-design project.

---

## Pipeline overview

```
              marker / DE gene list
                       │
                       ▼
       ┌────────────────────────────────────┐
       │ 1. peaks_for_genes()               │
       │    → peaks linked/nearest the gene │
       │    + V3 specificity z-scores       │
       │    (scripts/utils/marker_gene_peaks.py)
       └────────────────────────────────────┘
                       │
                       ▼  (CSV with chrom/start/end)
       ┌────────────────────────────────────┐
       │ 2. run_fimo_on_peaks()             │
       │    → JASPAR2024 FIMO scan          │
       │    (scripts/utils/run_fimo_on_peaks.py)
       └────────────────────────────────────┘
                       │
                       ▼  (peaks_fimo_hits.csv)
       ┌────────────────────────────────────┐
       │ 3. rank_synthetic_enhancers()      │
       │    → composite score per peak      │
       │    + 200 bp core window            │
       │    + per-peak motif map plots      │
       │    (scripts/utils/rank_synthetic_enhancers.py)
       └────────────────────────────────────┘
                       │
                       ▼
              top-ranked enhancer table
              + 200 bp seed coordinates
              + motif track PDFs
```

---

## Step 1 — peaks_for_genes

Returns one row per peak associated with a gene (via Cicero-linked or
nearest-TSS within 50 kb). Pulls from the precomputed peak metadata
cache (`outputs/V3/peak_metadata_cache.parquet`).

**Python:**
```python
from marker_gene_peaks import peaks_for_genes
peaks = peaks_for_genes(["pax2a"], min_z=0.0)
peaks.to_csv("pax2a_peaks.csv", index=False)
```

**CLI:**
```bash
PYTHON=/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python
$PYTHON scripts/utils/marker_gene_peaks.py --genes pax2a -o pax2a_peaks.csv
```

Key columns on output: `peak_id, chrom, start, end, peak_type,
distance_to_tss, linked_gene, nearest_gene, top1/2/3_celltype,
top1/2/3_z, max_z, max_accessibility, tau, gini`.

---

## Step 2 — run_fimo_on_peaks

Scans each peak against a TF motif database using FIMO (pymemesuite).

**Default DB:** `jaspar2024` (182 vertebrate-curated motifs, 172 unique
TFs). Cleaner than `h12core` which is mammalian-derived and dominated
by zinc-finger background.

**CLI:**
```bash
$PYTHON scripts/utils/run_fimo_on_peaks.py \
    --peaks-csv pax2a_peaks.csv \
    --label pax2a \
    --output-dir results/ \
    --motif-db jaspar2024     # or h12core / cisbpv2_danrer
```

**Outputs** (in `--output-dir`):

| File | Content |
|---|---|
| `{label}_fimo_hits.csv` | One row per (peak, motif, position) hit — for downstream analyses |
| `{label}_fimo_binary.npz` | Boolean matrix (n_peaks × n_motifs) for fast set ops |
| `{label}_fimo_peaks.csv` | Peak IDs in scan order |
| `{label}_fimo_tf_summary.csv` | Per-TF: number of peaks with at least one hit |

**Available motif databases:**

| DB key | Source | Motifs / TFs | When to use |
|---|---|---|---|
| `jaspar2024` (default) | JASPAR 2024 vertebrate consensus | 182 / 172 | Most cases — clean signal |
| `h12core` | HOCOMOCO v12 core (mammalian) | 1443 / 949 | Comprehensive scan, accept ZF background |
| `cisbpv2_danrer` | CIS-BP v2 zebrafish-specific | varies | Zebrafish paralog-specific calls |

---

## Step 3 — rank_synthetic_enhancers

Joins peak metadata + FIMO hits → produces a master ranked table, finds
a 200 bp core window per peak, optionally generates per-peak motif map
plots.

**CLI:**
```bash
$PYTHON scripts/utils/rank_synthetic_enhancers.py \
    --peaks-csv pax2a_peaks.csv \
    --fimo-hits results/pax2a_fimo_hits.csv \
    --label pax2a_mhb \
    --output-dir results/ \
    --target-celltype midbrain_hindbrain_boundary \  # optional bonus
    --prefer-distal \                                # optional bonus
    --plot                                           # optional motif maps
```

---

## Composite score — exactly how it's computed

Every numeric peak feature is converted to a **percentile rank ∈ [0, 1]**
within the input peak set:

| Feature | Source | What it captures |
|---|---|---|
| `rank_specificity` | rank of `top1_z` (V3 max z-score) | How cell-type-restricted the peak is |
| `rank_activity` | rank of `max_accessibility` | How "open" the peak is in its top celltype |
| `rank_tf_density` | rank of `n_unique_tfs` (FIMO hits) | Regulatory complexity / TF cluster strength |
| `rank_motif_strength` | rank of median(−log10 p-value) of best hit per TF | Statistical strength of motif matches |

These are averaged into a `base_score`:

```
base_score = mean(rank_specificity, rank_activity,
                   rank_tf_density, rank_motif_strength)     # ∈ [0, 1]
```

A `peak_type_factor` then biases toward distal-enhancer-friendly types:

| peak_type | Factor |
|---|---|
| intronic, intergenic | 1.0 (preferred — distal) |
| promoter | 0.7 |
| exonic | 0.5 |

```
composite_score = base_score × peak_type_factor               # ∈ [0, 1]
```

---

## use_case_score — context-specific boost (optional)

Two flags add bonus terms on top of `composite_score`:

| Flag | Bonus |
|---|---|
| `--target-celltype <name>` | +0.15 if `top1_celltype == <name>` |
| `--prefer-distal` | +0.10 for `intergenic`, +0.05 for `intronic` |

```
use_case_score = composite_score
               + (0.15 if top1 matches target_celltype else 0)
               + (0.10 if intergenic, 0.05 if intronic, 0 otherwise)
```

When either flag is set, the output is sorted by `use_case_score`
(descending). When neither is set, sort falls back to `composite_score`.

The base components (`rank_*`, `composite_score`) are always preserved
in the output CSV — bonuses are additive and transparent so you can
re-rank by any column you want.

---

## Master output: `{label}_enhancer_ranking.csv`

One row per peak. Key columns:

| Column | Meaning |
|---|---|
| `rank` | 1 = best by chosen sort score |
| `use_case_score` | Final ranking score (if bonuses applied) |
| `composite_score` | Pure data-driven composite (no context bonuses) |
| `peak_id`, `chrom`, `start`, `end`, `length` | Peak coordinates |
| `peak_type` | promoter / exonic / intronic / intergenic |
| `distance_to_tss` | bp to nearest TSS |
| `linked_gene` / `nearest_gene` | Gene annotations |
| `top1_celltype` / `top1_z` | Most-specific celltype + V3 z-score |
| `top2_celltype` / `top2_z` | 2nd-ranked celltype |
| `top3_celltype` / `top3_z` | 3rd-ranked |
| `max_z`, `max_accessibility`, `tau`, `gini` | Specificity metrics |
| `n_motif_hits`, `n_unique_tfs`, `n_unique_motifs` | FIMO hit counts |
| `median_neg_log_p` | Median −log10(p) of best hit per TF |
| `top_tfs` | Comma-separated top 8 TFs by p-value |
| `core_200bp_start`, `core_200bp_end` | Coords of densest 200 bp window |
| `core_200bp_n_tfs` | Unique TFs whose midpoint falls in the core window |
| `rank_specificity`, `rank_activity`, `rank_tf_density`, `rank_motif_strength` | Feature percentile ranks |
| `base_score`, `peak_type_factor` | Components of `composite_score` |
| `mhb_target_match` | Boolean: did this peak match `--target-celltype`? |

---

## Per-peak motif maps (`--plot` flag)

For each peak, generates a horizontal track PDF (and PNG):

- Gray peak background
- Color-coded boxes for each FIMO motif hit, colored by **TF family**
  (PAX, SOX, FOX, HOX, GATA, TBX, OTX, EN, ETV, KLF, NK, MYC, ZF, HD,
  bHLH, HMG, RFX, NR, IRF, NFKB, CREB, other)
- Top 8 strongest hits labeled with TF name
- Dashed red rectangle = chosen 200 bp core window
- Title with peak metadata (coords, peak_type, top1 celltype + z,
  composite score, n unique TFs)

Output dir: `{output-dir}/{label}_motif_maps/rank{NN}_{peak_id}.{pdf,png}`

---

## Worked example — pax2a → MHB synthetic enhancer

```bash
PYTHON=/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python
ROOT=/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis
OUT=$ROOT/notebooks/EDA_peak_parts_list/outputs/V3/marker_gene_queries

# 1. Extract pax2a peaks (33 peaks)
$PYTHON $ROOT/scripts/utils/marker_gene_peaks.py \
    --genes pax2a -o $OUT/pax2a_peaks.csv

# 2. FIMO scan with JASPAR2024 (~12s)
$PYTHON $ROOT/scripts/utils/run_fimo_on_peaks.py \
    --peaks-csv $OUT/pax2a_peaks.csv \
    --label pax2a_jaspar \
    --output-dir $OUT

# 3. Rank with MHB target + distal preference
$PYTHON $ROOT/scripts/utils/rank_synthetic_enhancers.py \
    --peaks-csv $OUT/pax2a_peaks.csv \
    --fimo-hits $OUT/pax2a_jaspar_fimo_hits.csv \
    --label pax2a_mhb \
    --output-dir $OUT \
    --target-celltype midbrain_hindbrain_boundary \
    --prefer-distal \
    --plot
```

**Result:** rank 1 = `13-29781733-29782356` (intronic, MHB top1, z=9.1,
25 unique JASPAR TFs, composite=0.799, use_case=0.999). Core 200 bp:
`chr13:29,781,749–29,781,949`. This is the canonical pax2a MHB intronic
enhancer.

---

## Extracting seed sequences

Use the `core_*bp_start` / `core_*bp_end` columns directly with pysam
on the danRer11 FASTA:

```python
import pysam, pandas as pd
df = pd.read_csv("pax2a_mhb_enhancer_ranking.csv")
fa = pysam.FastaFile(
    "/hpc/reference/sequencing_alignment/fasta_references/danRer11.primary.fa")

top10 = df.head(10)
for _, r in top10.iterrows():
    seq = fa.fetch(f"chr{r['chrom']}", int(r['core_200bp_start']), int(r['core_200bp_end']))
    print(f">{r['peak_id']}_rank{r['rank']}_{r['top1_celltype']}_z{r['top1_z']:.1f}")
    print(seq)
fa.close()
```

These 200 bp sequences are the **AgenticCRE seed candidates** for
ISM-based enhancer optimization or direct synthesis testing.

---

## Caveats

1. **JASPAR2024 motif names are sometimes dimer/cluster IDs** (e.g.,
   `PAX_PHOX2`, `PRDM_ZNF`) — these represent groups of TFs that share
   a motif. The `top_tfs` column reports the database's motif label as
   given.

2. **Composite is rank-based** — it's relative within the input peak
   set, not absolute. A peak with `composite=0.8` is in the top 20% of
   peaks queried, not "objectively good" on a global scale.

3. **`peak_type_factor` favors enhancer-like types.** If you want to
   include promoters as candidates (e.g., for proximal-element design),
   either ignore `composite_score` and re-sort by individual ranks, or
   edit `PEAK_TYPE_FACTOR` in `rank_synthetic_enhancers.py`.

4. **`min_z` filtering happens upstream in `peaks_for_genes()`.** If
   you want only meaningfully-specific peaks before running FIMO (saves
   compute on noisy peaks), set `--min-z 2` or higher.

5. **Distance cap.** `distance_to_tss` is bounded at 50 kb in the
   upstream pipeline. For long-range enhancers, `linked_gene`
   (Cicero co-accessibility) extends beyond this window.

---

## Files

```
scripts/utils/
  marker_gene_peaks.py             # Step 1
  build_peak_metadata_cache.py     # one-time cache builder
  run_fimo_on_peaks.py             # Step 2
  rank_synthetic_enhancers.py      # Step 3
  SYNTHETIC_ENHANCER_PIPELINE.md   # this document

notebooks/EDA_peak_parts_list/outputs/V3/
  peak_metadata_cache.parquet      # 77 MB — built once
  marker_gene_queries/             # query outputs (gitignored)
    pax2a_peaks.csv
    pax2a_jaspar_fimo_hits.csv
    pax2a_jaspar_fimo_binary.npz
    pax2a_jaspar_fimo_tf_summary.csv
    pax2a_mhb_enhancer_ranking.csv
    pax2a_mhb_summary.txt
    pax2a_mhb_motif_maps/
      rank01_*.pdf  ...  rank33_*.pdf
```
