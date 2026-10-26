# Peak specialization and temporal accessibility: prototype v3

Updated 2026-10-16. Every point is a **fixed genomic peak**. This prototype draws
supported paths through a peak accessibility atlas, then maps measured changes
in celltype specificity onto those paths. Opening/closing rates remain a separate
quantity. The simulations establish computational behavior; they do not establish
a developmental ancestry or transport process among genomic elements.

## Start with the two explorers

- [Tau at 24 hpf: paths toward late-stage restriction](outputs/specialization_flow_v3/tau/specialization_arrow_explorer.html).
- [Duration-averaged entropy: paths toward concentrated accessibility mass](outputs/specialization_flow_v3/entropy/specialization_arrow_explorer.html).

Both viewers use the same fixed 2D and 3D coordinates. Switch between structural
arrows and time-guided arrows, choose an interval/celltype, and click a peak to
inspect measured and true trajectories. Gold marks the constitutive reference;
gray includes broad-opening and weak-signal controls. A gold reference peak has
zero true temporal rate and need not emit temporal arrows.

Full filepath for the primary, late-restriction explorer:

```text
/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_chromatin_velocity/outputs/specialization_flow_v3/tau/specialization_arrow_explorer.html
```

Useful figures, available in each method's output directory:

| File | What it shows |
| --- | --- |
| `01_structural_arrows_2d.png` | 2D paths, colored by family and by the chosen specificity score |
| `02_structural_arrows_3d.png` | Static 3D view of structural paths |
| `03_temporal_guided_arrows_2d.png` | All five developmental intervals, with specificity-rate colors |
| `04_parameter_calibration.png` | Estimator, neighbor, and rate-threshold comparisons |
| `05_ground_truth_controls.png` | Representative stationary, opening, reversing, and weak-signal controls |
| `06_arrow_iteration_comparison.png` | Initial versus selected versus display-filtered arrows |
| `07_time_order_control.png` | Correct stage order compared with time-label permutations |

PNG figures also have PDF versions. The explorer is a self-contained HTML file.
The parent output directory also contains [a side-by-side entropy/tau comparison](outputs/specialization_flow_v3/08_entropy_vs_tau_arrows_2d.png) and [the edge-uncertainty calibration](outputs/specialization_flow_v3/09_edge_uncertainty_calibration.png).
It contains [the specificity-definition audit](outputs/specialization_flow_v3/specificity_definition_audit.png)
and JSON/CSV diagnostics documenting why the two orientations differ.

## What was simulated

The matrix has **3,400 peaks × 120 conditions**: 20 celltypes at
10, 12, 14, 16, 19, and 24 hpf. Entries are continuous accessibility propensities,
with six independently sampled beta-noise pseudobulk replicates. Condition-specific
cell counts and sequencing-depth factors affect measurement precision. Individual
cells and biological batches are not simulated.

| Population | Peaks | Temporal behavior |
| --- | ---: | --- |
| Constitutive core | 200 | Broad accessibility at every stage; exactly zero true rate |
| Family bridges | 800 | Continuous mixtures between broad and family-associated profiles |
| Celltype branches | 1,600 | Continuous mixtures between family and celltype-associated profiles |
| Stable celltype-specific | 160 | Restricted accessibility, constant through time |
| Uniform opening | 160 | Accessibility rises in every celltype, without increasing true specificity |
| Lineage opening | 160 | A selected celltype opens from a shared low background |
| Transient lineage | 160 | A selected celltype opens, then returns to its initial background |
| Low signal | 160 | Constant accessibility below the analysis signal floor |

The original 2,600-peak hierarchy is preserved. Its planted profile coordinate
indexes different peaks; it is independent of developmental time. Additional
controls have no value for that coordinate. Some controls form separate clouds;
no graph edges or UMAP coordinates are fabricated to connect them.

## Two specificity definitions answer different questions

For accessibility `A[c]` across the observed celltypes at one stage:

```text
p[c] = A[c] / sum(A)
entropy_specificity = 1 - H(p) / log(C)
tau = sum_c(1 - A[c] / max(A)) / (C - 1)
```

Both are zero for uniform accessibility and one when only one celltype is
accessible. Entropy summarizes the concentration of **total accessibility mass**.
Tau measures restriction **relative to the most accessible celltype**. Tau was
introduced for tissue-specific gene expression; applying it to peak accessibility
is an adaptation. [Yanai et al., 2005](https://academic.oup.com/bioinformatics/article/21/5/650/220059).

A published comparison documents substantial differences among expression
specificity metrics. That benchmark motivates comparing definitions here; its
results do not validate this ATAC simulation or establish a universally preferred
metric. [Kryuchkova-Mostacci and Robinson-Rechavi, 2017](https://academic.oup.com/bib/article/18/2/205/2562739).

The two saved structural orientations are:

- **Entropy / duration:** the mean entropy specificity over developmental time,
  using trapezoidal weights `[1, 2, 2, 2.5, 4, 2.5] / 14` for the six stages.
- **Tau / terminal:** tau at 24 hpf, with weights `[0, 0, 0, 0, 0, 1]`. This asks
  which neighboring peak profiles are more restricted at the late stage.

Equal-stage and late-emphasized alternatives are also evaluated. Late weights
are proportional to `(1 + (t - 10) / 14)^2`; they do not mean the same thing as
terminal-stage restriction. An invalid terminal observation is invalid, with no
fallback to an earlier stage.

### The definition audit changed our interpretation

The existing simulation's family-to-celltype coordinate was **not** monotonic in
entropy specificity. On seed 42, median Spearman correlations across the 20
celltype branches were:

| Score | Equal stages | Duration | Late weighting | 24 hpf only |
| --- | ---: | ---: | ---: | ---: |
| Entropy specificity | −0.957 | −0.966 | −0.944 | −0.885 |
| Tau | −0.914 | −0.871 | −0.239 | +0.995 |

All four family bridges had positive correlations above 0.999 for both metrics
and all four weightings. Two effects explain the celltype-branch disagreement:

1. **Background mass:** with one celltype at 0.85 and nineteen at 0.10, off-target
   values still comprise 69% of the total accessibility mass. For a five-celltype
   program with the same maximum and background, off-target mass is only 26%.
   Entropy can therefore fall when a program narrows from five target celltypes
   to one. Tau retains the relative-breadth interpretation in this example.
2. **Timing:** family-associated changes occur earlier than celltype-associated
   changes in this simulation. Averaging across time can rank an earlier family
   program above a later subtype program even when using tau.

We retained the generator and both orientations. Terminal tau expresses the
requested late-program restriction most directly; entropy-duration remains a
useful alternative question. The plotted coordinate is not a proven specificity
truth, and arrows should not be flipped merely to match its visual ordering.
The [audit JSON](outputs/specialization_flow_v3/specificity_definition_audit.json)
and [group-level CSV](outputs/specialization_flow_v3/specificity_definition_audit.csv)
contain exact values, example trajectories, and the background experiment.

## How the arrows are constructed

### Structural orientation

A sparse neighbor graph is built from 30 PCA components of measured accessibility.
For an existing edge from peak `i` to peak `j`, let `S` be the chosen structural
specificity score, `G` the neighbor weight, `epsilon` the minimum score difference,
and `q` the gradient exponent. A leave-one-out SE for the aggregate score adds
an empirical uncertainty margin:

```text
threshold[i,j] = epsilon + z_edge * sqrt(SE[i]^2 + SE[j]^2)
up_weight[i,j] = G[i,j] * max(S[j] - S[i] - threshold[i,j], 0)^q
P_up[i,j] = up_weight[i,j] / sum_j(up_weight[i,j])
u_up[i] = sum_j P_up[i,j] * (UMAP[j] - UMAP[i]) / ||UMAP[j] - UMAP[i]||
```

The edge margin was added after development-seed inspection found arrows driven
by small measurement gradients inside uniform-accessibility controls. It does
not model between-peak covariance or graph/embedding uncertainty, so it is an
empirical edge filter rather than a formal confidence interval. Its multiplier
is calibrated using seeds 42/43.

Rows without supported outgoing edges have zero arrows. A second field uses
`-S` to obtain `u_down` from actual lower-specificity neighbors. This preserves
supported reversals at tips, where an uphill field may have no direction.

Neighbor selection and score orientation use measured profiles. Embeddings only
supply the displayed unit directions. The synthetic co-accessibility surrogate
is stored separately and **is not used in these fields**; this stage tests the
specificity-based interpretation before adding cis smoothing.

The always-accessible core is a simulated reference population, identified for
coloring. Arrows are not imposed by distance from that core, and no genealogy is
inferred from its location. Truth labels and noiseless scores are used for
calibration, evaluation, and illustrative colors—not to construct or filter the
held-out field.

### Temporal guidance and opening/closing

Specificity rates use matched observed celltypes at adjacent stages:

```text
r[i,t] = (S_matched[i,t+1] - S_matched[i,t]) / (time[t+1] - time[t])
guided[i,t] = abs(r[i,t]) * unit(u_up[i])    when r > 0
guided[i,t] = abs(r[i,t]) * unit(u_down[i])  when r < 0
```

Unsupported directions and rates failing the display filters remain hidden.
This is a chosen mapping of a scalar onto an atlas direction. A scalar rate does
not identify a unique 2D/3D vector, and the arrows do not establish peak transport.
A decreasing-specificity arrow follows supported downhill neighbors rather than
simply negating an uphill arrow into empty space.

Opening/closing in an individual celltype is separately calculated as
`(A[c,t+1] - A[c,t]) / delta_hpf`. Increasing specificity can result from opening
in a target lineage, closing in other lineages, or both. All five intervals use
their actual widths: `[2, 2, 2, 3, 5]` hpf.

Arrows are length-normalized for display and spatially subsampled to reduce
occlusion. Their drawn length is not a physical speed. The saved arrays retain
rates, graph probabilities, coherence, and underlying vectors separately.

## Calibration and executed results

Seeds **42 and 43** select estimator and arrow settings. Seed **44** was inspected
during development, including an additional check of spurious structural arrows
inside uniformly accessible controls. Seed **45** provides the final computational
evaluation after that refinement, with parameters fixed before its generation.
These are draws from the same simulator, rather than independent biological
validation. Specificity
and stage-weighting definitions represent different biological questions; the
script compares alternatives without selecting their meaning from an outward
appearance.

Estimator candidates include pseudocounts `0, 0.001, 0.01`, probability shrinkage
`0, 0.05`, and optional jackknife bias correction. A pseudocount is added in
accessibility units; shrinkage mixes the resulting celltype proportions toward
uniform before the chosen score is calculated. Mean program-level rate RMSE
prevents the larger hierarchy from overwhelming the controls.

Structural candidates use `k = 15, 30, 60`, score tolerances
`0, 0.0001, 0.0005, 0.001`, and gradient exponents `0.5, 1, 2`. The selection score
rewards edges toward the true chosen score and penalizes missing hierarchy
coverage. Agreement with the planted profile coordinate is recorded separately.
An additional edge-margin sweep compares multipliers `0, 0.5, 1, 1.5, 2` using
aggregate-score SEs calculated from the leave-one-out trajectories.

Final seed-45 settings and results from each `run_summary.json`:

| Setting/result | Tau / terminal | Entropy / duration |
| --- | ---: | ---: |
| Pseudocount | 0.01 | 0.001 |
| Uniform shrinkage | 0.05 | 0 |
| Jackknife bias correction | No | Yes |
| State-graph neighbors | 60 | 60 |
| Edge tolerance | 0.001 | 0.0005 |
| Gradient exponent | 2 | 2 |
| Edge-SE margin multiplier | 2 | 1 |
| Rate-SE multiplier / absolute floor | 2.57 / 0 | 2.57 / 0 |
| Probability-weighted edges toward true chosen specificity | 98.91% | 99.94% |
| Probability-weighted hierarchy edges along planted coordinate | 98.73% | 35.03% |
| Dynamic hierarchy peaks with structural support | 96.00% | 98.625% |
| Active-rate recall | 64.21% | 94.83% |
| Sign accuracy among selected active rates | 99.92% | 100% |
| Pooled stationary-interval false positives | 2.38% | 1.65% |
| Worst stationary-program false positives | 4.75% | 5.00% |
| Eligible structural arrows, 2D | 2,484 | 2,752 |

Active rates have true magnitude at least `0.0005 / hpf`; stationary rates have
true magnitude below `1e-10 / hpf`. The stationary-program selection target was
5% on calibration seeds; the final runs meet it empirically, with stable-specific
peaks the hardest stationary control. Development seed 44 had exceeded the target
for this class, illustrating why the filter is not a guaranteed 95% confidence
rule. These percentages describe scalar-rate selection before additional arrow
projection/display filters. Sign accuracy is conditional on accepted active
rates: tau accepts about 64% of active intervals, not all of them.

Tau has stage-specificity macro RMSE 0.08058 and rate macro RMSE
`0.02868 / hpf`; entropy has 0.00478 and `0.002456 / hpf`, respectively. Tau
remains sensitive to noise and bias near uniform profiles. The definitions have
different numerical scales and active sets, so raw RMSE and recall do not
constitute a like-for-like contest between methods. Accurate signs among accepted
changes do not establish accurate magnitudes or a physical velocity.

The fixed UMAP uses 30 PCA components, 30 neighbors, `min_dist=0.4`, PCA-derived
initialization, `gamma=0.5`, 1,000 iterations, and random seed 42. Its graph is
separate from the selected 60-neighbor field graph. Both specificity variants
reuse the same seed-45 atlas; neither redraws it to favor a field.

The edge-margin iteration is recorded in `edge_uncertainty_sweep.csv`. For
example, on calibration seed 42, increasing the tau edge multiplier from 0 to 2
raised weighted agreement with true score increases from 89.53% to 97.25%, while
reducing dynamic-hierarchy coverage from 99.13% to 94.92%. This exposes the trade-off
between coverage and suppressing weak, noise-driven gradients. Seed-44 outputs
are retained under `development_seed44/` as development evidence.

Exact results: [tau summary](outputs/specialization_flow_v3/tau/run_summary.json),
[entropy summary](outputs/specialization_flow_v3/entropy/run_summary.json), and
`held_out_program_metrics.csv` in each directory. The `entropy_sweep.csv` filename
is retained for estimator comparisons in both methods, including tau.

## Projection and viewer validation

The [projection diagnostics](outputs/specialization_flow_v3/final_projection_metrics.csv)
compare estimates with a truth-score reference using the same graph and fixed
embedding. Among tau structural directions with reference support, 99.48% in 2D
and 99.64% in 3D have positive cosine agreement. This evaluates recovery of the
chosen score orientation, not developmental ancestry. No constitutive-core or
uniform-opening peaks retain guided temporal arrows after the final filters in
either dimension. A small fraction of uniform-opening peaks (1.25%) retain tau
structural directions; structural and temporal null behavior remain separate.

[Artifact checks](outputs/specialization_flow_v3/artifact_validation.json) verify
peak ordering, exact stored/display coordinate agreement, vector scaling, and
weak-signal masks for both methods. Actual Chromium runs exercised 2D/3D fields,
peak selection, rotation, and PNG download. Both final viewers loaded without
JavaScript errors or external requests. Reports and screenshots are saved in
`outputs/specialization_flow_v3/browser_validation/`. Rate colors use a fixed
symmetric range across all five intervals; measured and true specificity rates
share that range. Opening-rate colors also share a range across celltypes.
Values beyond the displayed range saturate. The full scientific test suite passed
65 tests (24 existing, 32 field-method, and 9 synthetic-data tests).

## Uncertainty and missing-data limits

Six independent synthetic measurement replicates provide leave-one-out
jackknife rate SEs. These quantify the specified measurement model, not biological
replicate variation. Tau contains a non-smooth maximum, particularly relevant to
flat profiles with tied celltypes; jackknife behavior there requires empirical
null controls. Bias correction is optional and was not selected for tau.

A structural arrow also requires at least two supported neighbors, projected
coherence at least 0.25, and mean leave-one-out direction agreement at least 0.8.
The graph and embedding stay fixed for this check. Leave-one-out fits share most
of their data; their agreement is a conditional stability diagnostic, not an
independent reproducibility probability. Display-filtered arrows are not nominal
confidence statements.

The [independent stress report](outputs/specialization_flow_v3/control_stress_checks.json)
uses seeds 42/43 and checks the following:

- Explicit missing masks and NaNs produce identical paired-support rates.
  Physically dropping unmatched celltypes agrees to numerical precision.
- A stable-specific peak can show a spurious rate as large as `0.0942 / hpf` when
  an absent target celltype changes the stage-wise denominator. Recomputing both
  endpoints on their common celltypes gives exactly zero in that control.
- Bias correction is applied to paired endpoint scores before differencing;
  separately correcting stage scores is insufficient when support changes.
- Weak peaks are excluded using mean accessibility below 0.02, before adding any
  pseudocount. Celltypes are categories of equal weight, rather than weighted by
  their simulated number of cells.
- Exact rate reversal requires the reflected time grid
  `[10,15,18,20,22,24]`, as well as reversed states. Reversing values while keeping
  the original unequal intervals does not yield equal-magnitude negative rates.

The time-permutation figures evaluate scalar rate alignment while keeping atlas
orientation fixed. They support the role of temporal order in estimating changes;
they do not independently validate a spatial transport interpretation.

## Reproduce and inspect

Run from the repository root:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export NUMBA_NUM_THREADS=2
export NUMBA_CACHE_DIR=/tmp/chromatin_velocity_numba
export MPLCONFIGDIR=/tmp/chromatin_velocity_mpl
CHROMATIN_PYTHON=/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python

"$CHROMATIN_PYTHON" notebooks/EDA_chromatin_velocity/specialization_benchmark.py \
  --method tau --weighting terminal --test-seed 45 \
  --output-dir notebooks/EDA_chromatin_velocity/outputs/specialization_flow_v3/tau

"$CHROMATIN_PYTHON" notebooks/EDA_chromatin_velocity/specialization_benchmark.py \
  --method entropy --weighting duration --test-seed 45 \
  --output-dir notebooks/EDA_chromatin_velocity/outputs/specialization_flow_v3/entropy

"$CHROMATIN_PYTHON" notebooks/EDA_chromatin_velocity/finish_specialization_report.py

"$CHROMATIN_PYTHON" -m unittest discover \
  -s notebooks/EDA_chromatin_velocity -p 'test_specialization*.py'
```

To refresh only HTML viewers and temporal PNGs from saved H5AD results, run
`refresh_specialization_views.py`, followed by `finish_specialization_report.py`;
this does not refit the models.

Main sources:

- [Synthetic generator](../../scripts/chrom_velo/specialization_synthetic.py): continuous hierarchy and temporal controls.
- [Specificity and fields](../../scripts/chrom_velo/specialization_flow.py): matched-support scores, graph orientation, and separate uphill/downhill guidance.
- [Benchmark driver](specialization_benchmark.py): calibration, fixed embedding, diagnostics, and export.
- [Standalone viewer](specialization_viewer.py): 2D/3D arrows and selected-peak curves.

Each method saves `specialization_flow.h5ad`, measured and true scores/rates,
uphill/downhill sparse graphs, masks and projected vectors, peak/condition tables,
and the viewer payload. The shared `fixed_atlas_seed45.h5ad` preserves the exact
coordinates used for comparisons. The definition audit and independent stress
JSON/CSV/PNG artifacts record separate seed-42/43 diagnostic experiments; they
are not regenerated by the main benchmark command.

Before applying this to the real atlas, keep its existing coordinates and confirm
peak alignment, accessibility normalization, missing celltype-stage support, and
replicate structure. Choose the specificity question explicitly, then compare
lineage-specific opening/closing against the resulting guided field.
