# Exploratory chromatin accessibility dynamics

This work studies accessibility changes of genomic peaks across zebrafish
celltypes and developmental stages. A fixed-atlas point is **one genomic peak**;
a shared-state point is the same peak at a particular stage. Rates use the actual
times **10, 12, 14, 16, 19, and 24 hpf** and remain separate from UMAP arrow directions.

**Current conclusion:** six-stage trajectories and variable-peak selection provide
useful descriptions of opening, closing, and regulatory programs. A global
root-to-branch stream and a consistent predictive advantage from Cicero have not
been established. This conclusion applies to the tested models and connected
peak subset; it does not establish an absence of biological cis coupling.

## Latest benchmark: variable peaks and shared Cicero weights (2026-10-26)

- **Selection spans celltype differences and temporal changes.** Total variability
  is variance of log1p(CPM) across reliable past celltype × stage pseudobulks,
  adjusted using training-defined mean-accessibility bins. Gini is a separate
  measure of concentration in linear CPM; high Gini can also reflect sparsity.
- **Group and peak support are distinct.** Groups with fewer than 20 pooled cells
  are excluded; absent groups remain missing. The balanced cell split gives at
  least 10 cells per half for retained groups. Selection uses 89, 120, and 151 past
  groups at origins 14, 16, and 19 hpf. A separate peak threshold requires at least
  20 source-half raw counts over those groups. Cutoffs use source-half observations
  and outer-training chromosomes only.
- **The shared model has six signed graph coefficients**, shared across peaks and
  celltypes, plus celltype-specific own-history terms. It evaluates 4,525
  Cicero-connected targets and 21 complete response celltypes. The selector uses
  all reliable past groups across 31 represented celltypes, so its scope is broader
  than the response panel. Regulatory partners remain available when targets are
  selected.
- **Predictive gain is not consistent.** Pooled squared-rate error exceeds
  persistence by 6.10% for all eligible targets, 19.13% for high-total-variance
  targets, and 27.55% for high-Gini targets. At 14→16 hpf, high-variance targets
  improve on persistence by 2.43%, but the additional gain over own history is
  only 0.29%, with chromosome-resampling uncertainty spanning zero.
- **Selected-cohort refits are a separate exploratory sensitivity.** The primary
  analysis fits all eligible targets and stratifies evaluation. The refits change
  both training and tuning populations while retaining identical held-out target
  masks; they also fail to establish a consistent gain. Late zero forecasts are
  validation-selected abstention, not evidence of biological stability.

These percentages describe the completed 2026-10-26 benchmark, not every possible
model or future rerun. Cicero was estimated using all cells and stages, so the
forecast tests condition on a fixed graph. Cell halves assess sampling
reproducibility, not biological replication; chromosome resampling does not supply
biological-replicate confidence intervals. CPM measures relative accessibility.
The existing atlas arrows are retrospective profile correspondence and are not
outputs of the new forecast model.

## Where to start

See [REPRODUCING.md](REPRODUCING.md) for prerequisites, executable commands,
validation scope, browser checks, and the generated-output boundary. The
[tested-version snapshot](requirements-tested.txt) records selected package
versions; it is not a complete environment lock.

| Analysis | Main source entry points | Local generated outputs |
|---|---|---|
| Synthetic trajectories and continuous atlases | `toy_chromatin_velocity.py`, `synthetic_celltype_benchmark.py`, `continuous_atlas_benchmark.py` | `outputs/toy_v2/` and corresponding synthetic runs |
| Six-stage trajectory modes | `real_trajectory_velocity.py`, `verify_trajectory_velocity.py` | `outputs/trajectory_velocity_v1/` |
| Shared peak-state PCA/UMAP | `state_atlas_prepare.py`, `state_atlas_umap.py`, `state_atlas_validate.py` | `outputs/state_trajectory_atlas_v1/` |
| Celltype atlas and independent-cell controls | `ct_peak_atlas.py`, `ct_flow_noise.py`, `ct_cis_validation.py` | `outputs/ct_peak_atlas_v1/`, `outputs/ct_flow_noise_v1/`, `outputs/ct_cis_validation_v1/` |
| Projection and display sensitivity | `ct_velocity_flow_sweep.py`, `ct_velocity_experiments_report.py` | `outputs/ct_velocity_experiments_v1/` |
| Variable-peak sampling pilot | `ct_variable_peak_pilot.py` | `outputs/cis_variable_peak_design_v1/filter_pilot/` |
| Past-only selection and shared forecasts | `ct_variable_peak_forecast_selection.py`, `ct_shared_weight_forecast.py` | `outputs/ct_shared_weight_forecast_v1/selection_audit/` and `model/` |
| Selected training/tuning sensitivity | `ct_shared_weight_selected_refit.py` | `outputs/ct_shared_weight_forecast_v1/model_selected_refit/` |
| Independent forecast audits | `ct_shared_weight_forecast_audit.py`, `ct_shared_weight_selected_refit_audit.py` | `outputs/ct_shared_weight_forecast_v1/independent_audit/` |

All `outputs/` paths are generated local artifacts and are absent from a fresh
clone. Once reproduced or restored, the latest run is summarized by
`outputs/ct_shared_weight_forecast_v1/HIGH_LEVEL_SUMMARY.txt` and
`DETAILED_SUMMARY.txt`, with source tables and audit records in the same run directory.
Source scripts, tests, and viewer templates are versioned; generated data, figures,
rendered reports, model fits, and delivery records are not.

## Historical development notes

**The dated sections below are preserved records of earlier experiments.** Their
claims, test counts, environment workarounds, and proposed next steps refer to
those runs. Read them together with the current validation conclusions above;
they do not establish predictive velocity or supersede the latest benchmark.

## Parameter sensitivity and Cicero forecasts (2026-10-21)

Start with the [combined experiment report](outputs/ct_velocity_experiments_v1/report.html)
and [offline parameter explorer](outputs/ct_velocity_experiments_v1/parameter_explorer.html).

- **Projection sweep:** 20,000 fixed random query peaks, full-atlas candidate targets;
  k=10/30/60/100 and neighbor-rate blends 0/0.25/0.5, all five intervals.
  Local directions are stable among jointly displayable peaks. Mean raw graph-hop
  drift is rootward in every tested setting, but the underlying graph shares this
  inward tendency. Late excess weakens on a fixed common-projectable cohort.
  The broad reference is a distributed set, not one UMAP center.
- **Display smoothing:** nine settings retain the same underlying per-peak field;
  directions agree closely where both settings support curves.
- **Cicero forecasting:** current weighted neighbor state and recent neighbor
  changes do not demonstrate a predictive gain in rolling temporal tests with
  held-out chromosomes and reciprocal cell-half outcomes. Own-history prediction
  also loses to persistence overall. The graph remains fixed from all cells/stages,
  and cutoff 0.25 connects only 4,525 peaks; this is a conditional selected-subset test.
- **Checks:** independent reconstruction of projections, all summary rows, shortest
  graph distances, forecast coefficients/predictions, and planted/null controls.
  Rates, source fits and source atlas remain unchanged. No parameter was selected
  to maximize an apparent root-to-branch stream.

Arrays and tables live in `outputs/ct_velocity_flow_sweep_v1/`,
`outputs/ct_cis_forecast_v1/`, and `outputs/ct_cis_forecast_lagged_v1/`.
The combined output folder contains PNG/PDF comparisons, display grids,
interactive HTML, audit results, and email delivery records.

## Reprojected velocities on the CT atlas (2026-10-21)

The existing six-stage fitted CPM/hour rates are now reused on the new fixed
celltype × stage atlas. Start with the
[reprojection report](outputs/ct_velocity_reprojection_v1/report.html),
[2D arrows](outputs/ct_velocity_reprojection_v1/01_ct_umap_velocity_arrows.png),
[local stream display](outputs/ct_velocity_reprojection_v1/03_ct_umap_velocity_streams.png),
and [offline 2D/3D viewer](outputs/ct_velocity_reprojection_v1/ct_velocity_explorer.html).

The primary projection rebuilds 30-neighbor candidates in the full 640,830-peak,
50-PC log-CPM space, then aligns stage-specific linear CPM profile differences
with the unchanged 21-celltype rate vectors. A separate coordinate-only control
reuses the original directed transition weights. Five interval fields remain
separate; neither root direction nor additional entropy/cis coupling is imposed.
Opening/closing colors remain distinct from the direction of arrows on the map.

`updated_projection.h5` and `reused_projection.h5` preserve all arrays and sparse
transition weights. `chromatin_velocity_ct_umap.h5ad` contains the original
biological layers and updated coordinates/projected fields. Figures subsample
points/arrows only for display; every peak is evaluated. These are descriptive
projections of conditional fitted rates, and the independent-cell/count-noise
limitations in the preceding validation still apply.

## Celltype atlas and independent-cell validation (2026-10-21)

Start with the [combined analysis report](outputs/ct_validation_delivery_v1/combined_report.html),
[full 2D atlas](outputs/ct_peak_atlas_v1/04_full_peak_atlas_2d.png), and
[offline 2D/3D explorer](outputs/ct_peak_atlas_v1/ct_peak_atlas_explorer.html).
These results provide the atlas and earlier validation underlying the experiments above.

- **Geometry:** one point is one peak, with 190 celltype × stage features.
  The CT and Leiden objects had different matrices but identical saved coordinates.
  Eight controlled fits compare aggregation, linear/log CPM, low-cell columns,
  PC count, neighborhood settings, seed, and dimensions. All **640,830 peaks**
  are projected through the default frozen models trained on 20,000 peaks;
  [full coordinates and annotations](outputs/ct_peak_atlas_v1/full_ct_peak_atlas.h5)
  preserve the original ordered peak IDs. The explorer displays 8,038 peaks.
- **Flow and noise:** [independent-cell tests](outputs/ct_flow_noise_v1/report.html)
  compare five interval-specific rates in 21 adequately sampled celltypes,
  held-out peak identities, count-noise controls, and planted dynamics.
  Log geometry with linear CPM/hour rates retains modest reproducible structure,
  concentrated in larger, often abundant/broad changes. A global stream is not established.
- **Cis smoothing:** [separate conditional test](outputs/ct_cis_validation_v1/report.html).
  At cutoff 0.25 only 4,525 peaks are connected. The small gain beyond ordinary
  shrinkage is exploratory; the fixed graph was learned from all cells.
- **Validation:** source/count reconciliation, sampled and full embedding audits,
  flow metric reconstruction, cis control review, and offline browser checks passed.
  Source objects remain unchanged. Cell halves test sampling stability, not biological replication.

Reports, figures, tables, code, and email delivery receipts are indexed under
`outputs/ct_validation_delivery_v1/`. Each analysis report describes its limits
and reproduction commands. These results supersede interpreting the earlier
smooth arrow or stream displays as independent evidence for global temporal flow.

## Real-data atlas application (2026-10-20)

Start with the [depth-matched 2D/3D explorer](outputs/real_specialization_depth_matched_v1/real_specialization_explorer.html)
and [original versus matched comparison](outputs/real_specialization_depth_matched_v1/07_original_vs_depth_matched_entropy.png).
The model uses all **640,830 peaks × 190 pseudobulks**, retaining the published
coordinates and graph. Every point is a peak.

Original-depth and matched-depth outputs are paired sensitivity views. A
constant-accessibility simulation can reproduce a late specificity increase at
the observed depths; this does not identify how much of the real trend is
technical or exclude biological change. The matched estimate equalizes coverage
across stages within each of 21 celltypes, averages five count thinnings, and
filters temporal arrows for conditional sign stability. Measured CPM and
opening/closing curves stay unchanged. Downsampling can obscure weak real
signals; neither view establishes biological velocity or supplies biological
confidence intervals.

The [parameter audit](REAL_SPECIALIZATION.md#what-was-actually-tuned-or-swept)
distinguishes synthetic ground-truth calibration, nearby real-data sensitivity
checks, and display choices. The real arrows were not globally optimized.
[Streamplots and smoothing checks](REAL_SPECIALIZATION.md#interpreting-streamplots)
add local mean-direction views of the saved fields. Curves are display summaries,
not time-evolving trajectories of genomic peaks.

See [REAL_SPECIALIZATION.md](REAL_SPECIALIZATION.md) for methods, support losses,
validation and the preserved original-depth comparison. Primary H5AD, offline
explorer and PNG/PDF outputs are in `outputs/real_specialization_depth_matched_v1/`;
the original comparison and depth-null diagnostic remain in
`outputs/real_specialization_v1/`.

## New: specificity-guided arrows in 2D and 3D (2026-10-16)

See [SPECIALIZATION_FLOW.md](SPECIALIZATION_FLOW.md) for the 3,400-peak ×
120-condition benchmark, calibrated arrows, temporal reversal controls, and
interactive explorers. The primary view uses late-stage tau specificity; an
entropy comparison shows why metric and time weighting change the interpretation.
Every point is a peak. Structural paths and measured temporal changes are shown
separately, with uncertainty filters and ground-truth diagnostics.

## New: continuous atlas with a constitutive core (2026-09-17)

See [CONTINUOUS_20CT.md](CONTINUOUS_20CT.md) for the 2,600-peak × 120-condition
simulation with an always-accessible core, smooth family/celltype branches,
a 3D explorer, and root-addition / bridge-removal controls. Run
`continuous_atlas_benchmark.py`. The core is a profile-space reference with zero
true temporal rate, not a developmental ancestor.

## 20-celltype synthetic benchmark (2026-09-17)

See [SYNTHETIC_20CT.md](SYNTHETIC_20CT.md) for the 2,400-peak × 120-condition
simulation, matched clean/noisy/stress datasets, and interactive 2D/3D explorer.
Its main entry point is `synthetic_celltype_benchmark.py`. The earlier prototype
and conceptual findings are preserved below.

Updated 2026-09-15. Start with `toy_chromatin_velocity.py` (Jupytext percent
format). Results below come from executed experiments, not expected outcomes.

## What was recovered

- The original July/September 2025 implementation is in
  `scripts/chromatin_velocity*.py` and `scripts/chrom_velo/archive/`. It used
  propagated accessibility as an unspliced surrogate; a symmetric co-accessibility
  graph cannot by itself identify the direction of time.
- Commit `1df2f65` (2025-10-17) describes a deprecated v1. The newer peak-based
  implementation is `e838ab1` (2025-10-27): `scripts/chrom_velo/chrom_velo_core.py`,
  `01_chrom_velo_notebook.py`, and the associated README. These were included in
  [PR #5](https://github.com/czbiohub-sf/zebrahub-multiome-analysis/pull/5).
  The PR has no recoverable discussion comments. A November 25 notebook refactor
  (`a21f4f1`) preserved the earlier analysis wrapper; it is not a newer algorithm.
- The October hybrid uses within-celltype spline derivatives, cis regularization,
  and a local UMAP projection. Two reasons to replace its numerical core: it uses
  **somite counts as the time coordinate**, and its projection regresses neighboring
  **velocities**, rather than state differences, onto UMAP displacements. Its
  fallback to mean neighbor displacement can also create an apparent direction
  unsupported by a temporal signal.
- The local `multivelo` branch ends at `4393f3f` (2026-05-14). Its
  `scripts/multivelo_whole_embryo/` pipeline estimates joint RNA/ATAC velocity in
  **cells**, with 3D visualization. It is a separate analysis from peak-profile
  dynamics. Local branch history survives, but `git ls-remote --heads origin`
  currently returns only `main`; the origin/* branch entries in this checkout
  are stale. No dedicated chromatin-velocity PR was found in accessible PR history.
- The previous conversation itself was not recovered. This note combines the
  recovered code/history with the supplied implementation handoff.

## The central distinction

Let A[i,c,t] be normalized accessibility of peak i in celltype c at stage t.
The defensible temporal quantity is

    rate[i,c,t] = (A[i,c,next(t)] - A[i,c,t]) / (next(t) - t).

Positive means increasing accessibility; negative means decreasing accessibility.
These are retrospective changes in a population summary, not a kinetic prediction
for an individual cell. With compositional normalization, they describe relative
accessibility changes, not an absolute probability that a DNA region is open.

Use the actual hours `[10,12,14,16,19,24]`, not somite numbers or equally spaced
stage indices. Compute within celltype, and inspect each lineage before averaging.
Celltype labels do not establish ancestor/descendant links; changing subpopulation
composition can generate changes within a coarse annotation.

A peak UMAP embeds a peak's **whole trajectory across conditions**. Each peak is a
fixed genomic element, not a sampled state along the trajectory of another peak.
An arrow to another peak therefore needs an explicit interpretation.

### Why the handoff's early-to-late expectation can reverse

Take peaks with the same pulse shape but different peak times:

    x_tau(t) = f(t - tau)
    x_tau(t + h) = x_(tau - h)(t).

Advancing the profile makes it resemble an **earlier** program. Thus a positive
temporal derivative projected against other peaks' whole profiles can point
late-to-early even when it is computed correctly. The executed pulse experiment
has 120 nonzero interior arrows; **100% point toward smaller tau**. The diagnostic
phase axis is explicitly labeled as a known coordinate, not UMAP; a separate
figure shows actual 2D UMAP arrows and supported streamlines for the same family.

Do not flip the sign solely to make an early-to-late validation plot look right.
Negating the derivative corresponds to delaying a profile, which is a different
quantity. Likewise, the sign of a 2D arrow component does not encode opening.

### Recommended visualization and next model

1. **Primary result: time-resolved signed rates on the fixed peak atlas.** Blue =
   closing, red = opening, gray = unsupported. Use the same color range and
   coordinates across all five transitions, with celltype selection/facets.
   A time slider works identically in 2D and in rotatable 3D. Show RMS/absolute rate
   separately: a transient opening-then-closing peak can have almost zero mean.
2. **Secondary result: explicitly labeled profile-shift arrows.** The prototype
   implements the handoff projection with scVelo in both dimensions. Normalize
   arrows for display and retain rates in feature space. Streamlines are local
   interpolation, masked outside supported neighborhoods; they are not observed
   trajectories. Disconnected peak islands may not support useful streamlines.
3. **If the desired arrows mean early-to-late regulatory succession**, fit/test a
   distinct lagged-profile graph within a lineage: similar profiles, positive
   fitted lag, and high-dimensional neighbor support. Use a pulse/onset timing
   parameter appropriate to the program; argmax alone is poor for monotonic
   opening/closing and flat profiles. Direction from stage ordering is an imposed
   constraint, not independent validation. Benchmark lag and successor recovery
   against the continuous toy before adding such arrows to the real atlas.
4. An accessibility-weighted centroid can summarize where activity shifts on the
   map, but its movement is a population summary. A transport model would need
   explicit opening/closing source terms: accessibility is not conserved mass.
   Derivatives alone cannot uniquely identify peak-to-peak transport.

## Implemented in this prototype

`../../scripts/chrom_velo/temporal_velocity.py` provides:

- Forward differences and a `form='slope'` OLS summary option, with explicit metadata
  and unordered-column support. `divide_by_dt=True` is the default.
- No interpolation or bridging of missing stages. Terminal/missing/low-count
  intervals have a separate validity mask. Stored zeros are placeholders, never
  counted as observed zero rates.
- Sparse positive cis smoothing with self edges, then `alpha=0.6` mixing. The cis
  graph replaces the expression-neighbor graph **only for regularization**. The
  embedding neighbor graph remains separate. Isolated peaks retain their own rate.
- Mask-aware signed, opening, closing and RMS summaries. These are equal-condition
  averages, not a duration-weighted net change; time-resolved panels also report
  how many celltypes contribute.
- scVelo projection only: no fake spliced/unspliced matrices and no kinetic fitting.
  Explicit state layer, `approx=False`, no square-root transform, one-step graph
  neighborhood, positive transitions, existing coordinates unchanged.
- Conservative projection masks: features must be valid across all retained peaks.
  Per-peak missingness may remove extra features; scVelo cannot directly handle an
  edge-specific feature mask. Unsupported fields return zero arrows explicitly.

scVelo centers vectors across features before cosine comparison, so shared opening
across all selected conditions may have no projected direction. Preserve signed
rates independently. Its embedding also subtracts a uniform-neighbor baseline;
raw arrows are not biological speed. See the official
[velocity graph source](https://github.com/theislab/scvelo/blob/v0.3.4/scvelo/tools/velocity_graph.py),
[embedding source](https://github.com/theislab/scvelo/blob/v0.3.4/scvelo/tools/velocity_embedding.py),
and [projection API](https://scvelo.readthedocs.io/en/stable/scvelo.tl.velocity_graph.html).

## Executed toy results

Two complementary datasets:

- **540 peaks, 3 celltypes, 6 irregular stages (16 observed conditions):** opening,
  closing, pulse, delayed opening, opposing lineages, constant controls. Gaussian
  measurement noise; planted sparse cis modules; two absent conditions, including
  an internal gap. Compare estimates with exact noiseless interval-average rates,
  rather than with an instantaneous derivative at a different time.
- **180 continuous pulse programs:** known phase/onset continuum, separate 2D/3D
  UMAPs computed once from states, no derivative information used for the embedding.
  The same high-dimensional graph supports all projections.

Five-seed benchmark, threshold 0.25 (the default 0.35 retains the same planted edges):

| Graph / intrinsic weight | Rate RMSE | Dynamic-entry sign accuracy |
|---|---:|---:|
| Temporal only, alpha=1 | 0.04556 | 84.98% |
| Planted cis, alpha=0.6 | 0.03119 | 90.61% |
| Planted cis, alpha=0 | 0.01872 | 96.76% |
| Rewired cis, alpha=0.6 | 0.03769 | 84.74% |
| Rewired cis, alpha=0 | 0.05630 | 69.03% |
| Time permutation, temporal only | 0.16715 | 51.36% |

Sign accuracy excludes true rates with absolute value <=0.015, so nearly static
entries do not dominate the metric. Constant-control RMS and the full 0.25/0.5/0.8
threshold sweep are in the CSV. The default mixture reduces RMSE by 31.5%.
Some generic averaging improves RMSE even on the rewired graph; stronger recovery
with the planted graph is the relevant comparison. Pure coupling wins here because
modules were simulated to be coherent; this does **not** establish the best alpha
for biological data. No replicate/cell-count model or bootstrap confidence is
implemented yet.

Eight regression tests cover units, irregular time, missing stages, finite/low-cell
masks, order invariance, cis restriction, isolates, masked smoothing, pulses, and
zero/uniform projected fields. The end-to-end experiment also exercises the actual
scVelo graph and both embedding dimensions.

### Outputs (generated, gitignored)

Under `outputs/toy_v2/`:

- `01_ground_truth_profiles.{png,pdf}`
- `02_peak_umap_rates_and_arrows.{png,pdf}`
- `03_time_resolved_opening_closing.{png,pdf}`
- `04_validation.{png,pdf}`
- `05_interactive_3d.html` — signed rates and arrows, rotatable, self-contained
- `06_profile_shift_counterexample.{png,pdf}`
- `07_continuous_program_umap.{png,pdf}`
- `08_continuous_program_3d.html` — the known continuous program family
- `toy_chromatin_velocity.h5ad`, `phase_counterexample.h5ad`
- `peak_summary.csv`, `cluster_summary.csv` (Leiden means),
  `validation_metrics.csv`, `run_summary.json`

## Real data audit: findings and what they change

`audit_real_inputs.py` reads the state in bounded blocks and streams the edge CSV.
It does not alter inputs or run a real-data velocity inference.

State:
`../data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad`
(relative to repository root).

Cicero:
`../data/processed_data/13_peak_umap_analysis/cicero_integrated_ATAC_v2/02_integrated_ATAC_v2_cicero_connections_peaks_integrated_peaks.csv`.

1. **640,830 unique peak rows × 190 condition columns**, 32 celltypes × 6 stages.
   Absent columns are hemangioblasts/10 hpf and notochord/24 hpf. Present X has no
   nonfinite values; its 27,893,354 zeros are observed matrix entries, not placeholders
   for those two absent columns. There are 158 supported forward-difference columns,
   or 144 with an n_cells>=20 endpoint requirement. Fourteen conditions have n<20.
2. The state, 2D/3D coordinates and saved neighbor matrices share the same AnnData
   observation axis. Map external Cicero IDs to this exact ordered peak index.
   Edge lists lack a dense ordered index; they must never be aligned by row number.
3. **X equals `normalized`, not `log_norm`.** On 512 spaced sample peaks,
   log_norm equals log1p(normalized). Notebook comments claiming X=log_norm are
   incorrect for this saved object. Sample reconstruction of stored PCA from
   current layers and saved scaling has relative RMSE 0.750 for normalized and
   1.029 for log_norm: neither reproduces stored scores. Embedding preprocessing
   provenance remains unresolved. Do not assume an exact PCA/state relationship.
4. The recovered Cicero CSV has 5,405,656 rows, all cis, including 45,428 nonfinite
   scores. 5,405,108 rows have both endpoint IDs in the atlas. Explicitly report and
   drop invalid scores/unmatched edges before constructing a graph; deduplicate
   undirected pairs rather than summing repeated scores.
5. Only **31,611 peaks** appear as any Cicero endpoint. Usable coverage is much
   lower after thresholding:

   | Score threshold | Matched edge rows | Peaks touched |
   |---|---:|---:|
   | >0.25 | 17,550 | 4,525 (0.71% of atlas) |
   | >0.5 | 1,910 | 807 |
   | >0.8 | 24 | 20 |

   Row counts are not unique undirected edge counts. Do not infer regulatory
   isolation from missing graph coverage. Retain the atlas with self-only
   smoothing where no valid edge is available, and report that support separately.
   A literal intersection with graph endpoints would discard most peaks.

The complete audit and condition list are saved under `outputs/real_input_audit/`.
These findings support a **temporal-only, within-celltype pilot first**, followed
by a cis-coupling comparison on well-covered loci (or a more comprehensive verified
Cicero graph). Real-data velocity inference has intentionally not been run yet.

Before that pilot:

- Establish which state/feature transformation generated each chosen atlas. Compute
  biological deltas in a declared normalization, then transform state and tangent
  consistently. For a fixed feature scaler z=(x-mu)/sigma, use v_z=v_x/sigma;
  differencing separately standardized stage columns can manufacture changes.
- Keep the existing UMAP and its original high-dimensional neighbor graph. For a
  subset, audit neighbor retention/isolation; do not quietly replace it with 2D kNN.
- Estimate uncertainty by resampling embryos/replicates where available, otherwise
  within-condition cells (the latter measures sampling, not biological replication).
  Control coarse-celltype composition and unequal coverage; report intervals where
  a celltype emerges as unsupported instead of calling them de novo opening.
- Evaluate rates at slc4a1a, cdx1a and late neuronal loci in the relevant lineages.
  Treat those as hypotheses to check against measured profiles, not mandatory signs.
  Test alpha, threshold, graph rewiring, time permutations, and embedding stability.
- Only then decide whether the paper needs profile-shift arrows, a separately
  validated succession graph, or primarily animated signed-rate maps.

## Reproduce on this HPC

The existing `single-cell-base` has scVelo 0.3.2 with a newer SciPy: `.A` sparse
access fails. This run used scVelo 0.3.4 in a temporary import directory, leaving
shared packages unchanged. Recreate that directory if /tmp has been cleared:

```bash
CV_PY=/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python
"$CV_PY" -m pip install --no-deps --target /tmp/chromatin_velocity_scvelo034 scvelo==0.3.4
export PYTHONPATH=/tmp/chromatin_velocity_scvelo034${PYTHONPATH:+:$PYTHONPATH}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export NUMBA_CACHE_DIR=/tmp/chromatin_velocity_numba
export MPLCONFIGDIR=/tmp/chromatin_velocity_mpl
"$CV_PY" notebooks/EDA_chromatin_velocity/test_temporal_velocity.py
"$CV_PY" notebooks/EDA_chromatin_velocity/toy_chromatin_velocity.py
"$CV_PY" notebooks/EDA_chromatin_velocity/audit_real_inputs.py
```

The toy needs numpy, scipy, pandas, anndata, scanpy, scvelo, matplotlib, plotly,
and Leiden support. Version metadata is recorded in `run_summary.json`.
