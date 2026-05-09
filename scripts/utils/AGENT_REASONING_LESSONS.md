# Agent Reasoning Lessons — distilled from the pax2a synthetic-enhancer session

Reflection on what worked (and what didn't) when an Agent-driven pipeline
was used to go from a single gene name (`pax2a`) to a curated set of
synthesis-ready DNA elements with biology checks at every step. These
are *transferable* principles — not specific to peaks, ATAC, or zebrafish.

Each lesson has three parts:
- **Principle** — the one-line takeaway.
- **What happened** — the concrete moment in this session.
- **What an Agent should do** — the operational rule.

---

## 1. Always re-ask "what metric answers what question?"

**Principle.** Two metrics with similar names can answer fundamentally
different questions. Pick the one that matches user intent, not the one
that happens to be computed.

**What happened.** The locus-view figure first used **V3 z-score
(specificity)** as block height. After review we switched to **absolute
log_norm accessibility (activity)**. The two are nearly uncorrelated for
real peaks (r = −0.12 for pax2a). A peak with z = 5.95 and accessibility
54 is "very specific to MHB" but unlikely to drive strong reporter
expression; conversely a peak with z = 4.3 and accessibility 833 is more
predictive of activity.

**What an Agent should do.** When asked "is this peak / element good?",
clarify the optimization target before scoring. *Specificity*,
*activity potential*, *synthesizability*, *biological match* are
different metrics. State which one drives each ranking, and surface the
others as separate columns so the user can re-rank.

---

## 2. Verify annotations against the question, not the database default

**Principle.** Categorical labels (peak_type, "exonic", "promoter",
"5'UTR") are computed relative to the *nearest gene*, not necessarily
the *gene the user is asking about*.

**What happened.** A peak 10.7 kb upstream of pax2a TSS was annotated
`peak_type=exonic`. We initially picked it as one of three synthesis
candidates. A direct GTF lookup revealed the "exonic" annotation was
relative to a *different* gene, `CT025909.4`. Using a regulatory
element overlapping another gene's exon for synthesis is risky (it
could disrupt that gene). We dropped it.

**What an Agent should do.** When a label seems suspicious given the
spatial relationship to the gene of interest, **verify against
ground-truth source (the GTF)** rather than trust the cached label.
For peaks linked to gene X, "exonic" should mean exonic *to X* — if
not, flag it.

---

## 3. Don't hard-code tissue / context biology

**Principle.** TF function, "off-target", "validated marker" all depend
on cellular context. Hard-coded lists don't scale beyond the single
context they were curated for.

**What happened.** The 3-panel figure originally classified TFs as
"EXPLICIT MHB" / "IMPLICIT" / "background" using a hand-curated MHB-
specific prefix list. This works for pax2a / MHB analyses but breaks
for any other gene/tissue. We refactored to:
1. Generate a research-brief markdown with prepared
   ZFIN/NCBI/PubMed/Alliance query URLs for the top-N TFs.
2. Output a stub CSV (category / citations / notes columns blank).
3. An agent reads the brief, runs WebSearch on the URLs, fills the CSV.
4. The figure-rendering script accepts the filled CSV via
   `--tf-biology-csv` to override the hardcoded list.

**What an Agent should do.** When you find yourself writing "if
celltype == 'MHB' then …", stop. Externalize tissue knowledge into
either (a) a CLI argument the user provides, or (b) an agent-driven
literature lookup that fills a structured table. Keep the
classification logic generic; the data is per-query.

---

## 4. Layer evidence; don't trust a single signal

**Principle.** A high-confidence finding is supported by multiple,
independent signals. A single signal is a starting hypothesis.

**What happened.** MHB marker genes were first computed from chromatin
(peak parts list) — 1864 candidates. Many were Ensembl-style lncRNAs
without functional annotation. The same celltype was then queried in a
separate scRNA atlas (Zebrahub v01), producing 89 RNA Wilcoxon markers.
The **intersection (54 markers)** is the high-confidence MHB marker set
— and it recapitulates canonical MHB biology (pax5, pax2a, pax7a/b,
en2a/b, cnpy1, fgf18b, lmx1bb, otx2a, gbx2, …). The *ATAC-only* and
*RNA-only* sets are weaker hypotheses.

**What an Agent should do.** Default to the **intersection** when
multiple data modalities are available. Treat single-source results as
"candidates", multi-source confirmation as "validated". Always show
both numbers explicitly.

---

## 5. Real-world constraints shape the algorithm, not just filter the output

**Principle.** When the user has a downstream constraint (cost, time,
size limit), bake it into the optimization, not as a post-hoc filter.

**What happened.** IDT charges substantially more for synthesis above
500 bp. Initial ranking ignored peak length entirely, so #2 (1706 bp)
beat smaller peaks despite being expensive. We added:
- A `synthesis_length_factor ∈ [0.4, 1.0]` multiplier in `use_case_score`
- A "compact-segments" algorithm that finds motif hubs and stitches
  ≤500 bp from a long peak (Scheme B)
- Side-by-side Scheme-A-vs-B comparison so the user sees the tradeoff

**What an Agent should do.** Solicit downstream constraints early
("cost cap", "time cap", "size limit") and incorporate them into
scoring with transparent multipliers. Don't just filter "drop peaks
> 500 bp" — degrade the score continuously so a great long peak can
still beat a mediocre short one.

---

## 6. Bonuses for context-match; not penalties for "off-target"

**Principle.** "Off-target" can be on-target for an alternative
biological context. The user's intent is "what I want to optimize for",
not "what's wrong".

**What happened.** pax2a is expressed in MHB *and* optic_cup *and*
pronephros *and* hindbrain. An early version of the ranker treated any
non-MHB top1 celltype as a penalty. We refactored to:
- `--target-celltype` (primary): +0.15 bonus
- `--permissive-celltypes` (alternative tissues where the gene is also
  normally expressed): +0.05 bonus, *NOT* penalized as off-target
- Other celltypes: no bonus, no penalty (still ranked by base composite)

**What an Agent should do.** Frame multi-tissue genes as *opportunities
for dissection*, not *off-target noise*. Provide additive bonuses
rather than multiplicative penalties so off-target peaks remain
visible and the user can re-rank.

---

## 7. Make scoring transparent; preserve every component

**Principle.** A black-box score is useless when the user wants to
re-rank by a different criterion. Always emit the components alongside
the final score.

**What happened.** `composite_score = mean(rank_specificity,
rank_activity, rank_tf_density, rank_motif_strength) × peak_type_factor`.
`use_case_score` adds optional bonuses. Both are in the output CSV
**and** all `rank_*`, `peak_type_factor`, `synthesis_length_factor`,
`target_match`, `top1_in_permissive` columns are preserved. The user
can sort by any of them.

**What an Agent should do.** Treat the final score as one *view*, not
the only view. Always emit every component. Document the formula. Use
percentile-rank components (not raw values) so they're comparable
across different feature scales.

---

## 8. Pick defaults for the 80% case, but make them switchable

**Principle.** A good default is the choice that's right for most
users. A great default is one that's right for most users *and* easy to
change.

**What happened.** The motif scanner defaults to JASPAR2024 (182
vertebrate-curated motifs, ~12× faster, less zinc-finger flooding) but
accepts `--motif-db {h12core, jaspar2024, cisbpv2_danrer}`. Each option
documents when to use it. Why JASPAR by default: cleaner signal for
typical cases. Why H12CORE optional: comprehensive scan when you
*want* the noise.

**What an Agent should do.** Document why the default is the default,
not just what it is. Always provide an opt-out for power users.
Anti-pattern: a default that's "the first thing the developer
implemented" with no rationale.

---

## 9. State caveats next to the output

**Principle.** Every classifier, scorer, and prediction has
assumptions. The user reading the output should not have to re-derive
them.

**What happened.** The activator/repressor TF classifier produces nice
stacked bars, but TF function is profoundly context-dependent (LEF1/TCF
activates with β-catenin, represses with TLE; nuclear receptors are
ligand-dependent; "background" zinc-fingers are partly noise). We wrote
caveats directly into:
- The script docstring
- The `INTERPRETATION.md` next to every output figure
- The main `GENE_LOCUS_EXPLORER.md` doc

**What an Agent should do.** When you produce a classification or
prediction, immediately produce the caveat document. State what's
*not* claimed. Anti-pattern: "the model says peak X is an enhancer" —
say instead "FIMO finds matches for these TFs; whether they're bound
in vivo requires expression + ChIP".

---

## 10. Iterate on edge cases; don't paper over them

**Principle.** When an algorithm produces a clearly nonsensical result
on a specific input, the algorithm has a bug — fix the assumption, not
the symptom.

**What happened.** The "stitched motif hubs" algorithm produced 36 bp
designs (instead of ≤500 bp) when motifs were evenly distributed across
a long peak (max gap = 89, gap threshold = 80). Fix: iterate the gap
threshold (80 → 40 → 20 → 10) until the result uses ≥50% of the budget.
Falls back to "densest 500 bp window inside the largest hub" if every
hub individually exceeds the budget.

**What an Agent should do.** When a result looks anomalous on a
specific input, *trace the algorithm against that input*. Identify the
implicit assumption that broke (here: "motifs cluster into hubs
separated by big gaps"). Generalize the assumption (iterate the
parameter). Anti-pattern: adding a special-case `if` for the failing
input.

---

## 11. Build for the next gene, not just this one

**Principle.** If you ran the analysis for gene X, an agent or human
should be able to run it for gene Y in one command. Generalization
discipline pays back in 2x.

**What happened.** After the pax2a analysis, we built
`gene_locus_explore.py` as a single-command driver: `--gene <name>` +
optional `--target-celltype` / `--permissive-celltypes`. The pipeline
chains five sub-utilities (peaks, FIMO, ranking, locus view, 3-panel),
auto-looks up the gene's TSS from the GTF, and produces a per-gene
output directory with the same layout regardless of which gene was
queried. Sanity-tested on `sox10` (neural crest, 26 peaks) immediately
after pax2a.

**What an Agent should do.** Whenever a multi-step analysis is
completed for one input, write a thin orchestrator that takes the
input as a parameter. Test with one different input before declaring
the pipeline done. Anti-pattern: a Jupyter notebook with hand-edited
cells for "the pax2a run".

---

## 12. Cleanup as part of the workflow

**Principle.** A directory full of `_v1`, `_v2`, `_mhb`, `_3selected`
duplicates is a usability bug. Plan the cleanup pass and write a
README.

**What happened.** After ~10 iterations of re-running the pipeline
with different parameters, the output directory had:
- Three versions of the ranking CSV (v1 / mhb / v2)
- Two FIMO scans (h12core + jaspar)
- Four sub-directories with overlapping content (3panel, motif_maps,
  3selected, promoter)
- 32 MB of clutter
We proposed a clean 2-folder structure (`mhb_markers/`, `pax2a/`),
moved files (preserving the canonical version, removing the rest),
extracted a single `sequences.fasta` for synthesis, and wrote a
top-level README. Disk: 32 MB → 20 MB. Top-level entries: 32 → 3.

**What an Agent should do.** After every "feature complete" milestone,
inventory the output dir, identify duplicates / obsolete versions, and
either consolidate or delete. Always end with a README named for what
the directory contains. Anti-pattern: leaving `_v3_final`, `_v4_real`,
`_v5_actually_final` files in place "just in case".

---

## 13. Ask the user when reversibility is at stake

**Principle.** Auto-mode is not a license to delete. Confirm before
destructive operations even when the user has said "go fast".

**What happened.** Cleanup involved deleting ~half of the contents of a
gitignored directory. Even though everything was reproducible from the
scripts, we proposed the plan in writing first, listed every file as
DELETE / KEEP / CONSOLIDATE, and waited for the user's approval before
executing. Took ~2 minutes of agent time, prevented the irreversible
case where we'd have to re-run the whole 1.5-hour pipeline.

**What an Agent should do.** Default to action only on
*reversible* operations. For destructive operations (delete, force-push,
overwrite, etc.) — propose the plan, wait for explicit approval, then
execute. Auto-mode means "minimize interruptions on safe work", not
"never check on dangerous work".

---

## 14. Communicate decisions, not just results

**Principle.** Numbers without rationale are noise. Each ranking, each
recommendation, each filter should come with a one-sentence "why".

**What happened.** Every output figure / table is accompanied by a
plain-English explanation: "block height = absolute accessibility,
NOT z-score, because the two are nearly uncorrelated and accessibility
is the better activity predictor"; "we picked the −2.2 kb promoter
peak instead of the −10.7 kb 5' peak because the latter overlaps
CT025909.4's exon"; "Scheme B is recommended for Peak 3 because Scheme
A drops all EXPLICIT MHB TFs". The user knows *what* and *why*.

**What an Agent should do.** Pair every metric / decision / filter
with its rationale at output time, not just on request. Anti-pattern:
a results table with no headline interpretation. The user reading the
output six months from now should still understand the decisions.

---

## How to use this document

For a new agent picking up an analysis like this:

1. **Before starting**: skim §1, §3, §6 to align metric/scope with user intent.
2. **During the analysis**: §4, §5, §7, §8 guide the algorithm and scoring.
3. **At each output step**: §9, §14 — caveats and rationale.
4. **At edge cases**: §2, §10 — verify against the source of truth, fix the assumption.
5. **At completion**: §11, §12, §13 — generalize, clean up, confirm before destructive cleanup.

Each principle has a corresponding artifact in this repo:
- §1 / §7 — `rank_synthetic_enhancers.py` scoring components
- §2 — `gtf_helpers.py`
- §3 — `tf_biology_lookup.py` (agent-curation workflow)
- §4 — `09f_mhb_rna_vs_atac_markers.py` (ATAC × RNA intersection)
- §5 / §10 — `design_short_element.py` (synthesis length factor + iterative compacting)
- §6 — `compute_composite_score()` (target_match + permissive bonuses)
- §8 — `run_fimo_on_peaks.py` (motif DB defaults)
- §9 — `INTERPRETATION.md` files
- §11 — `gene_locus_explore.py` (one-command driver)
- §12 — `outputs/V3/marker_gene_queries/README.md`
