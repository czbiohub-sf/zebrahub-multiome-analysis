# Synthetic peak accessibility across 20 celltypes and 6 stages

For the newer connected 3D atlas with an always-accessible core, see
[CONTINUOUS_20CT.md](CONTINUOUS_20CT.md). This document describes the preceding
discrete-program benchmark.

2026-09-17. This is the next benchmark after the simple single-lineage pulse toy.
The goal is to test recovery of known accessibility changes within celltypes.

## Start here

Open `outputs/synthetic_20ct_v1/synthetic_celltypes_explorer.html` in a browser.
It is self-contained. Each dot is a **peak** with 120 measurements, not a cell.
Select a celltype, a stage or interval, and true accessibility / measured
accessibility / true rate / estimated rate. Click a peak to inspect its continuous
truth curves, measured means, and estimated rates. Switch between fixed 2D and 3D
UMAPs. At the last stage, the rate view is gray because no forward interval exists.

Choose the **opposite celltypes** example and compare **CT16** with **CT01** in the
seed-42 run: the same peak opens in CT16 and closes in CT01. Averaging those rates
would erase the change. The faint curves show the other celltypes.

For inspecting the matrix before considering UMAP:

- `01_example_peak_trajectories.png`: one example peak per program; gray lines are
  other celltypes, colored lines are two highlighted celltypes, points are noisy means.
- `02_peaks_by_120_conditions.png`: sampled peaks across every column, in 20 blocks
  of six ordered timepoints. True and observed accessibility use the same scale.
- `03_celltype_time_rate_maps.png`: true opening/closing rates on one fixed atlas.
- `04_rate_recovery_benchmark.png`: recovery and false-change behavior over three seeds.

All outputs live under `notebooks/EDA_chromatin_velocity/outputs/synthetic_20ct_v1/`.
They are gitignored and reproducible from the source files.

## Dataset design

- **2,400 peaks**: 80 regulatory modules, each containing 30 related but nonidentical
  peaks. Five synthetic chromosomes carry multiple distinct programs.
- **20 synthetic celltypes**, CT01–CT20, organized into four similarity families
  of five. These groups are not an asserted developmental lineage tree.
- **6 stages:** 10, 12, 14, 16, 19, 24 hpf.
- **120 columns:** CT01_10hpf, CT01_12hpf, …, CT20_24hpf.
- **Continuous scores between 0 and 1.** These are latent accessibility propensities
  and noisy measurements of them; they are not fragment counts, TF-IDF values,
  or simulated single cells.

Eight equally represented program types:

| Program | What a peak does |
|---|---|
| Opening | Smooth sigmoid increase in selected celltypes |
| Closing | Smooth sigmoid decrease |
| Transient opening | Opens and then closes |
| Transient closing | Closes and then reopens |
| Late opening | Increase concentrated near the end of the sampled interval |
| Opposite celltypes | Opens in one celltype, closes in another, stable elsewhere |
| Stable specific | Celltype-specific accessibility, constant over time |
| Stable broad | Broad accessibility, constant over time |

Programs can target one celltype, a related family, or many celltypes. Baseline,
amplitude, center time, and curve width vary; related celltypes can have different
amplitudes and timing offsets. Peaks within a module have continuous variation.
The opposite-celltype case deliberately has matching rate magnitudes to test exact
cancellation. The active rates are still evaluated separately, never as one
across-celltype scalar.

For a peak i, celltype c and time t:

    A[i,c,t] = baseline[i,c] + amplitude[i] * loading[i,c] * f[i,c](t)

The generator saves both the analytic derivative dA/dt and the interval-average
rate `(A(t_next)-A(t))/dt`. Forward differences are evaluated against the latter:
these are not interchangeable for a nonlinear curve across a five-hour gap.

### Continuous sampling noise

There are three independent **sampling replicates** per condition. These are not
biological embryo replicates. For true mean mu and precision phi:

    Y_rep ~ Beta(mu * phi, (1-mu) * phi)
    phi = simulated_n_cells * simulated_depth_factor * noise_precision
    X = mean(Y_rep)

This preserves E[X]=mu while making sparse or shallow conditions noisier. No count
matrix is generated. Random streams for latent programs, graph construction, and
sampling are separate: changing noise strength does not change the ground truth.
The default noise_precision=1.5 is a chosen benchmark setting, not fitted to real
Zebrahub data. Tests check the sampling mean and variance against this model.

### An imperfect regulatory graph

Same-module peaks supply candidate cis relationships. The observed graph retains
75% of candidate edges, prescribes 10% isolated peaks, and targets a 15% share of
incorrect edges joining different modules on the same chromosome. True and wrong
edge-score distributions overlap. Wrong edges can join different temporal programs.

The seed-42 observed graph contains 7,889 undirected edges, including 1,183 wrong
edges; 238 peaks were prescribed to be isolated. These settings are deliberately
much better covered than the recovered real-data graph. Sparse-coverage experiments
can increase `graph_isolated_fraction`; this simulation does not assume the real
graph has the same coverage. The oracle candidate graph is saved separately and
used only as a labeled benchmark reference.

## Three matched scenarios

| File | Shape | Purpose |
|---|---|---|
| `synthetic_clean.h5ad` | 2,400 × 120 | Exact ground truth; temporal baseline should have zero error |
| `synthetic_noisy.h5ad` | 2,400 × 120 | Heterogeneous sampling precision; main explorer dataset |
| `synthetic_stress.h5ad` | 2,400 × 116 | Stronger noise, missing stages, and low-cell-count conditions |

All three share the same latent programs and graph for a given seed. The stress
scenario removes CT20/10 hpf, CT20/12 hpf, CT08/14 hpf, and CT12/24 hpf. CT04/24 hpf
and CT13/12 hpf each contain nine simulated cells across the sampling replicates.
The estimator masks intervals touching a missing or n_cells<20 condition. It does
not bridge a missing internal stage or encode absence as zero accessibility.

The noisy dataset has **100 valid forward-rate columns** (five per celltype).
The stress dataset has **92** after missingness and low-count masking. Final-stage
zeros are storage placeholders with validity=False.

### AnnData schema

- `.X`: measured means; peaks × observed celltype/stage columns.
- `.obs`: synthetic peak ID, chromosome/interval, true module, program/scope, target
  celltypes, curve center/width/amplitude. `true_*` fields are evaluation metadata.
- `.var`: explicit celltype, family, time_hpf, counts/depth per replicate, total
  n_cells, interval length and rate-supported flag.
- `.layers['truth_state']`: noise-free mean accessibility.
- `.layers['truth_instantaneous_rate']`: analytic derivative at each stage.
- `.layers['truth_velocity']`, `truth_velocity_valid`: exact observed-interval truth.
- `.layers['sampling_rep1'/'sampling_rep2'/'sampling_rep3']`: continuous noisy samples.
- `.layers['velocity_temporal']`: estimated forward differences, with n_cells mask.
- `.layers['velocity']`: alpha=0.6 mixture with observed cis smoothing, threshold>0.35.
- `.layers['velocity_valid']`: validity mask, separate from stored numeric zeros.
- `.obsm['true_celltype_loadings'/'true_celltype_lags_hpf'/'true_celltype_baseline']`:
  ground-truth curve parameters, ordered CT01–CT20 even in the missing-column scenario.
- `.obsp['coaccessibility']`: imperfect observed graph, aligned to obs order.
- `.obsp['true_cis_candidates']`: oracle candidate graph, for evaluation only.
- Noisy dataset: `.obsm['X_umap']`, `X_umap3d`, PCA, and neighbors. Both UMAPs use
  one neighbor graph built from measured accessibility. Ground-truth labels and
  derivatives never enter the embedding computation.

The explorer displays rates as colors. A known sign of accessibility change is not
a ground-truth direction of peak-to-peak motion on UMAP. Profile-shift projection
remains a separate experiment, documented in the earlier README.

## Executed benchmark: seeds 42, 43, 44

The numerical estimator is the existing `temporal_velocity.py`. Compare temporal
only, mixed imperfect cis, mixed oracle cis, mixed rewired cis, and pure imperfect
cis. Metrics are saved globally, per celltype, and per true program.

Noisy scenario, averages across three seeds:

| Method | Dynamic-entry RMSE | Opening/closing sign accuracy | Static entries exceeding 0.01/hpf |
|---|---:|---:|---:|
| Temporal | 0.01533 | 95.55% | 35.62% |
| Mixed imperfect cis | 0.01233 | 97.44% | 23.46% |
| Mixed oracle cis | 0.01145 | 97.94% | 20.72% |
| Mixed rewired cis | 0.01889 | 95.13% | 25.49% |
| Pure imperfect cis | 0.01506 | 98.18% | 10.86% |

Dynamic-entry metrics use true absolute interval rate >0.01/hpf. Static entries
have true absolute rate <1e-7/hpf. The 0.01 threshold is a chosen effect-size cutoff,
**not a statistical significance test**. Intermediate small changes are excluded
from both subsets but included in global RMSE. A high sign accuracy does not imply
well-calibrated magnitudes or reliable detection of changes in stable peaks.

On clean data, temporal error is zero and graph smoothing introduces a small bias
because related peaks have different true rates. On noisy data, partial coupling
improves dynamic RMSE by about 20%, while pure coupling oversmooths more. This is a
useful tradeoff to test, not evidence that a particular alpha is generally optimal.
Static false changes remain frequent: uncertainty calibration is a necessary next
algorithmic step.

## Reproduce and modify

From repository root:

```bash
CV_PY=/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export NUMBA_CACHE_DIR=/tmp/chromatin_velocity_numba
export MPLCONFIGDIR=/tmp/chromatin_velocity_mpl
"$CV_PY" notebooks/EDA_chromatin_velocity/test_synthetic_accessibility.py
"$CV_PY" notebooks/EDA_chromatin_velocity/synthetic_celltype_benchmark.py
```

This benchmark does not need the temporary scVelo upgrade: rate estimation uses
NumPy/SciPy and atlas construction uses Scanpy. Nine generator tests pass. Saved
H5ADs are reopened and checked for index/layer alignment. The JavaScript continuous
curve formula agrees with all 288,000 saved latent entries to within 1e-6.

For a different size or number of seeds:

```bash
"$CV_PY" notebooks/EDA_chromatin_velocity/synthetic_celltype_benchmark.py \
  --n-modules 80 --peaks-per-module 50 --benchmark-seeds 5 \
  --output-dir notebooks/EDA_chromatin_velocity/outputs/synthetic_20ct_larger
```

The generator is `scripts/chrom_velo/synthetic_accessibility.py`. Its
`SyntheticConfig` controls the noise precision, replicate count, family sharing,
graph retention, wrong-edge fraction, and isolated-peak fraction. Keep latent seed
fixed when changing one factor so comparisons have the same underlying truth.

## What to refine before calling it realistic

This is a controlled first benchmark, not an empirically calibrated zebrafish
simulation. Each peak currently has one main temporal program; future variants
can mix programs, introduce true lineage bifurcations, and add module-correlated
or embryo-level variation. Biological replication is not modeled yet. A grid of
noise/coverage/missingness settings will be more informative than one attractive
UMAP. Calibrating accessibility distributions, mean–variance behavior, and graph
coverage to real data should come after reviewing these explicit trajectories.
