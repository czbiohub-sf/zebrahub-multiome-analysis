# Reproducing the chromatin dynamics exploration

These are project-specific analysis scripts, not an installed software package.
Run commands from the repository root. The small scientific tests use synthetic
arrays; real-data analyses require the original local inputs and earlier
intermediate artifacts. A fresh clone contains the code and templates, but no
`outputs/` directory or downloaded experimental data.

## Environment

The clean-checkout tests used Python **3.10.13** in the existing Linux
`single-cell-base` environment. [requirements-tested.txt](requirements-tested.txt)
records selected observed package versions. It is not a complete dependency lock
or a guarantee that a new environment solver will recreate the same installation.
The repository's `environments/` files provide broader context and retain the
portability limitations described in the top-level README.

The latest selection/forecast code uses numpy, scipy, pandas, h5py,
scikit-learn, and matplotlib. Publication-figure checks also need **pypdf** and
Poppler's **`pdftotext` and `pdffonts` executables** on `PATH`. Earlier embedding
and interactive work additionally uses anndata, scanpy, scVelo, UMAP, plotly,
and, for some benchmarks, Leiden/igraph support. Browser checks require Node.js,
Playwright, and an installed Chromium browser.

```bash
# After activating the desired Python environment:
CV_PY="${CHROMATIN_PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export NUMBA_NUM_THREADS=2
export NUMBA_CACHE_DIR=/tmp/chromatin_velocity_numba
export MPLCONFIGDIR=/tmp/chromatin_velocity_mpl
```

The historical toy run used a temporary scVelo 0.3.4 import override because of a
sparse-matrix compatibility issue in that run. Its exact command is preserved in
the historical README section. The current tested installation contains scVelo
0.3.2, and the latest shared-weight forecast does not call scVelo. Do not infer a
single environment specification from every historical run.

## Small scientific tests

```bash
"$CV_PY" -m unittest discover -s notebooks/EDA_chromatin_velocity -p 'test_*.py'
"$CV_PY" notebooks/EDA_chromatin_velocity/test_activation_geometry.py
```

The isolated clean-checkout check passed **157 unittest tests**, plus **five
numerical geometry controls** run by the second command. Those five function-based
checks are not discovered by unittest. No pytest installation is required.
These tests cover numerical and scientific invariants on small arrays; they do
not rerun the full atlas or establish biological validity.

The shared-weight model also has data-free checks against an independent dense
ridge solve and a planted partner-specific signal:

```bash
"$CV_PY" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'notebooks/EDA_chromatin_velocity')
import ct_shared_weight_forecast as model
import ct_shared_weight_forecast_audit as audit
out = Path('notebooks/EDA_chromatin_velocity/outputs/shared_weight_synthetic_checks')
out.mkdir(parents=True, exist_ok=True)
print(audit.direct_ridge_solve_check(model))
print(audit.partner_specific_production_check(model, out))
PY
```

In the reviewed run, nine dense-solve comparisons had a maximum prediction
difference of **7.77 × 10⁻¹⁶**; the strong planted partner-signal control had
**99.18%** error reduction versus the constant-graph comparator. The
partner-feature-removed control also passed. Zero-feature
baseline equivalence and shuffled controls passed. These checks verify the
implementation for the planted mechanism; they do not establish power for subtle
biological effects or demonstrate a real-data forecasting gain.

## Input and artifact layout

Source-path constants are near the top of each analysis script. The real-data
scripts generally expect a sibling `../data/` directory relative to the repository,
including:

- `annotated_data/objects_v2/peaks_by_ct_tp_master.h5ad` and the annotated peak-atlas
  objects used by the earlier trajectory analyses.
- `processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered.h5ad`.
- The Cicero edge CSV under
  `processed_data/13_peak_umap_analysis/cicero_integrated_ATAC_v2/`.

Check the exact constants and saved input provenance before running elsewhere.
Original experimental inputs are not distributed in this PR. Input objects are
opened read-only; generated results are written under this notebook directory's
`outputs/` tree. Rerunning a stage can replace its generated artifacts, so preserve
an earlier run directory if it is needed for comparison.

The latest benchmark resumes from these earlier artifacts:

| Prerequisite | Producer / purpose |
|---|---|
| `outputs/trajectory_velocity_v1/trajectory_fits.h5` | `real_trajectory_velocity.py`: six-stage fits and peak metadata |
| `outputs/state_trajectory_atlas_v1/sample_data.h5` | `state_atlas_prepare.py --stage prepare`: fixed whole-peak sample |
| `outputs/ct_flow_noise_v1/cell_split_input.h5` | `ct_cell_split_input.py`: split cells, counts, coverage, and group cell numbers |
| `outputs/ct_cis_validation_v1/cis_split_input.h5` | `ct_cis_validation.py --stage prepare`: connected targets and required partners |
| `outputs/ct_cis_validation_v1/cis_directed_matched_control_edges.csv.gz` | Same preparation stage: true and distance-matched control partners |

The Cicero preparation stage also needs the edge audit produced by
`ct_cis_validation.py --stage audit`. Earlier historical README sections describe
the toy, atlas, and trajectory stages. The following commands are a **resume
sequence with these prerequisites available**, not an automatic fresh-clone data
bootstrap.

## Latest selection and forecast benchmark

```bash
# Expand selection inputs to all 32 celltypes, retaining reliable past groups.
"$CV_PY" notebooks/EDA_chromatin_velocity/ct_variable_peak_forecast_selection.py

# Primary all-eligible fit, with selected cohorts as evaluation strata.
"$CV_PY" notebooks/EDA_chromatin_velocity/ct_shared_weight_forecast.py
"$CV_PY" notebooks/EDA_chromatin_velocity/ct_shared_weight_forecast_audit.py

# Separate exploratory sensitivity: select both fitting and tuning populations.
"$CV_PY" notebooks/EDA_chromatin_velocity/ct_shared_weight_selected_refit.py
"$CV_PY" notebooks/EDA_chromatin_velocity/ct_shared_weight_selected_refit_audit.py

# Local coverage and selected-fit comparison figures.
"$CV_PY" notebooks/EDA_chromatin_velocity/ct_shared_weight_report.py coverage
"$CV_PY" notebooks/EDA_chromatin_velocity/ct_shared_weight_refit_figure.py
```

Outputs are under `outputs/ct_shared_weight_forecast_v1/`:

- `selection_audit/`: aligned 32-celltype target counts, source-half and
  origin-specific masks, training cutoffs, diagnostics, and input audits.
- `model/`: primary predictions, coefficients/scalers, nested tuning records,
  chromosome folds, losses, comparisons, and plots.
- `model_selected_refit/`: separate high-Gini and high-total-variance fitting/tuning
  sensitivities. Their outer evaluation masks match the primary analysis.
- `independent_audit/`: source/graph reconstruction, independent numerical solves,
  feature/prediction checks, planted/null controls, and future-data mutation tests.
- `figures/`: coverage and comparison figures, with source tables.

The full primary audit includes future-value perturbations. Its optional
`--skip-future` flag records a **partial** audit and should not be presented as
completion of the full leakage check. The selected-refit audit is also a separate
required check for those exploratory refits.

The completed historical benchmark passed source/count reconciliation, independent
feature and prediction reconstruction, disjoint chromosome checks, and future-data
invariance. Those checks do not make the Cicero graph prospective: it was learned
from all stages and cells. The fixed 21-celltype response panel also uses
whole-course coverage metadata. Cell halves and chromosome resampling quantify
sampling behavior rather than biological replication.

## Atlas displays and local report packages

The cohort-atlas display additionally requires
`outputs/ct_peak_atlas_v1/full_ct_peak_atlas.h5` and
`outputs/ct_velocity_reprojection_v1/updated_projection.h5`:

```bash
"$CV_PY" notebooks/EDA_chromatin_velocity/ct_shared_weight_report.py atlas
```

It reuses the existing retrospective six-stage arrows on fixed coordinates;
it does not display new shared-model forecasts. Count support and arrow
projectability can reduce the displayed subset further.

The milestone scripts assemble local HTML/text reports and ZIP packages from
completed results:

```bash
"$CV_PY" notebooks/EDA_chromatin_velocity/ct_shared_weight_milestone1.py
"$CV_PY" notebooks/EDA_chromatin_velocity/ct_shared_weight_milestone2.py
```

The first requires the selector and atlas figures; the second also requires the
primary and selected-refit audit records to report `passed`, plus their figures.
These packagers contain reviewed, run-specific narrative and numerical summaries.
If a rerun changes the inputs, model, or results, update that narrative before
reusing the package. `CHROMATIN_REPORT_RECIPIENT` optionally populates local
manifest metadata and defaults to an empty string. **The packagers create local
files only; they do not send email.**

## Browser validation

After generating the corresponding offline HTML and source arrays, run the
checks from the repository root:

```bash
node notebooks/EDA_chromatin_velocity/check_state_viewer.cjs
node notebooks/EDA_chromatin_velocity/check_ct_peak_atlas_viewer.cjs
node notebooks/EDA_chromatin_velocity/check_ct_velocity_reproject_viewer.cjs
node notebooks/EDA_chromatin_velocity/check_ct_velocity_sweep_viewer.cjs
```

By default these load the `playwright` package through Node's module resolution.
Set `CHROMATIN_PLAYWRIGHT_MODULE` to an installed module path if it lives elsewhere.
The sweep check also uses `CHROMATIN_PYTHON` (default `python`) for its H5 payload
comparison; that interpreter needs numpy and h5py. Its optional
`CT_PARAMETER_VIEWER_OUT` changes the parameter-viewer output directory, while the
source sweep H5 path still uses the repository-relative analysis location.

Browser checks verify rendered coordinates, masks, controls, and offline loading.
They require previously generated artifacts and an installed browser; they are
separate from the small unittest suite.

## Versioned source versus generated results

Versioned files include analysis Python, reusable numerical modules under
`scripts/chrom_velo/`, tests, viewer HTML/JavaScript templates, and human-written
documentation. The notebook-local `.gitignore` excludes `outputs/` and
`__pycache__/`; root ignore rules also exclude large data.

Generated matrices, fitted models, figures, CSV results, rendered HTML reports,
source audit records, logs, ZIP/email packages, delivery receipts, and external
model-review transcripts remain local. Do not force-add generated output folders.
Links to `outputs/` in historical documentation refer to those local run artifacts
and will not resolve in a fresh clone. The code and dated status describe the
methods and conclusions; regenerating the full numerical record requires the
original data and recorded preprocessing artifacts.
