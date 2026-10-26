# A continuous peak atlas with an always-accessible core

## Open the prototype

Main interactive viewer (self-contained HTML; open locally in a browser):

```text
/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis/notebooks/EDA_chromatin_velocity/outputs/synthetic_20ct_continuous_v2/continuous_peak_explorer.html
```

The default is a rotatable 3D UMAP. **Every dot is a peak.** Gold marks peaks
that remain accessible in all 20 celltypes at all six stages. The four colored
families each contain five celltypes. Select “Profile specialization” to see the
continuum, or select an opening/closing rate and celltype to inspect temporal
changes on the same coordinates. Click a peak for its time courses. “Example peak”
steps through the core, a family branch, and a celltype branch.

The same directory contains `continuity_comparison_3d.html` for rotating the three
control datasets, and `01_continuity_comparison.png` / `.pdf` for a static overview.

## What changed in the simulation

The previous simulation used mostly separate temporal program groups. It already
included stable broad accessibility, but that did not guarantee intermediate
profiles between groups. Adding another always-accessible group alone retained
those gaps.

This version samples a **connected family of accessibility profiles**:

```text
200 constitutive peaks
  └─ 800 intermediate peaks: broad accessibility → four family-specific profiles
       └─ 1,600 intermediate peaks: family-specific → 20 celltype-specific profiles
```

Total: **2,600 peaks × 120 columns** (20 celltypes × 10, 12, 14, 16, 19, 24 hpf).
The diagram describes similarities among different peaks. A peak is a fixed
regulatory element; it does not become another peak.

For each peak and celltype, true accessibility is

```text
A(c,t) = baseline + amplitude × [w_family × family_contrast(c) × g(t)
                             + w_celltype × celltype_contrast(c) × h(t)]
```

- Baselines lie around 0.65, with small peak-to-peak variation.
- The family contrast is +0.18 inside a family and −0.55 outside.
- The celltype contrast is +0.23 in the target celltype and −0.55 elsewhere.
- `g(t)` and `h(t)` are smooth sigmoid gates, normalized to zero at 10 hpf and
  one at 24 hpf. Their centers are approximately 15.5 and 18.5 hpf.
- Core weights are `(0,0)`; family bridges use `(s,0)` for `s` from 0 to 1;
  celltype branches use `(1−u,u)` for `u` from 0 to 1. Junctions have matching
  profiles, with timing variation tapering away near the endpoints.
- Three noisy beta-distributed measurements are averaged. Sampling precision
  depends on simulated cell counts and depth. No individual cells are generated.

This gives target-specific opening, off-target closing, and some transient
opening in related celltypes. All peaks have shared accessibility across celltypes
at the first stage, then specialize to different extents. That is a deliberate,
simplified model; it has not been fitted to the real data and emphasizes closing
outside target celltypes. It is not yet a general model of developmental ATAC.

**An always-accessible core can be a reference in profile space, but it is not a
temporal ancestor.** Its exact temporal derivative is zero. Measured estimates
can be nonzero because of sampling noise and smoothing. The specialization
coordinate (0 = core, 1 = family-specific, 2 = celltype-specific) indexes different
peaks and is independent of developmental time.

## Does the connectivity come from the accessibility data?

We compared three datasets under identical embedding settings. The table counts
connected components in a 15-nearest-neighbor graph in measured-profile PCA space,
**before considering UMAP coordinates**:

| Dataset | Peaks | Components at k = 15 |
|---|---:|---:|
| Original discrete programs | 2,400 | 67 |
| Original + always-accessible peaks only | 2,600 | 67 |
| Constitutive core + continuous intermediate profiles | 2,600 | 1 |
| New model with its 800 family bridges removed | 1,800 | 5 |

The removal control keeps the full model's PCA basis and rebuilds its neighbor
graph after removing the bridges. Full results at k = 5, 15, and 30 are in
`connectivity_metrics.csv`. At k = 5 the new model still has two components
(the larger contains about 98.2% of peaks); connectedness depends on neighborhood
scale. The original model is not a matched resampling control for the new model;
the bridge-removal comparison more directly tests the designed connections.

UMAP uses measured accessibility only: PCA with 30 components, 30 neighbors,
`min_dist=0.4`, seed 42. The final display uses PCA initialization, repulsion
strength 0.5, and 1,000 iterations in both 2D and 3D. The same choices apply to
all three panels. No peak coordinates or neighbor edges are manually prescribed.
The cis graph is not used to fit UMAP.

A connected input graph can still look torn in an embedding. Spectral
initialization separated one family visually in this run; increasing the neighbor
count alone did not reliably fix that. PCA initialization and lower repulsion
showed the known connections more clearly. Those are display choices, not extra
biological evidence. See `03_neighbor_sensitivity.png` and
`04_initialization_sensitivity.png`; `inspect_continuous_layouts.py` reproduces
these comparisons. Parameter meanings are documented in the
[UMAP parameter guide](https://umap-learn.readthedocs.io/en/latest/parameters.html)
and [Scanpy UMAP API](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.umap.html).

## Rates, graph assumptions, and files

The H5AD `synthetic_continuous.h5ad` stores noisy means in `.X`, exact state and
instantaneous derivatives, interval rates, temporal estimates, cis-coupled
estimates, masks, condition metadata, and fixed 2D/3D UMAP coordinates. Interval
rates divide by the actual 2, 2, 2, 3, and 5 hpf gaps. The final stage has no
following interval and is masked. All 120 conditions are present in this version;
use the preceding benchmark's stress dataset for missing-condition tests.

The cis graph is a favorable synthetic surrogate: nearest neighbors of noiseless
profiles restricted to the same synthetic chromosome, with missing edges and
isolated peaks. It is separate from the measured-state graph used for UMAP.
It is not inferred Cicero or a causal network. Coupling uses threshold 0.35 and
60% intrinsic / 40% neighborhood rates. This version tests continuity and provides
a known-truth rate explorer; it does not establish a preferred velocity estimator
or an interpretation for arrows between peaks.

Additional files:

- `02_root_to_specific_profiles.png` / `.pdf`: actual temporal curves along one
  family of profiles.
- `peak_metadata.csv`, `conditions.csv`, `run_summary.json`: simulation labels,
  condition counts, embedding settings, and numerical checks.
- `discrete_plus_constitutive.h5ad`: the root-addition control.
- `explorer_data.json`: the numerical data embedded in the viewer.

## Reproduce

From the repository root, using the existing scientific environment:

```bash
CV_PY=/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
  NUMBA_CACHE_DIR=/tmp/chromatin_velocity_numba MPLCONFIGDIR=/tmp/chromatin_velocity_mpl \
  "$CV_PY" notebooks/EDA_chromatin_velocity/continuous_atlas_benchmark.py
```

The benchmark loads the prior `synthetic_20ct_v1/synthetic_noisy.h5ad` if present,
otherwise regenerates the same original model. Source:
`../../scripts/chrom_velo/continuous_accessibility.py` and
`continuous_atlas_benchmark.py`. Seven continuity-specific tests check the
constant core, shared initial accessibility, exact analytic derivatives, matching
junctions, signed target/off-target rates, cis restriction, and terminal masks.
The complete existing velocity/synthetic suite can be run with:

```bash
PYTHONPATH=/tmp/chromatin_velocity_scvelo034 "$CV_PY" -m unittest discover \
  -s notebooks/EDA_chromatin_velocity -p 'test_*.py'
```

The temporary scVelo 0.3.4 path is the compatibility overlay used by the earlier
velocity tests; see the main README for its installation command. The new atlas
generator does not import scVelo.

## Executed validation

All 24 scientific tests passed (seven for this model plus the previous seventeen).
The exported H5AD was reopened and checked against the viewer for ordered peak
IDs, layers, masks, and embedding coordinates. Viewer truth curves matched all
312,000 saved peak-by-condition values within 5.3e-7; the preceding viewer model
also passed its numerical check. Controls were exercised with a mocked DOM and
Plotly; this is not a browser rendering test. The static 3D comparison was
visually inspected. Sampled 3D trustworthiness was 0.9963 at 15 neighbors
(1,000 sampled peaks); this measures local neighbor preservation, not biological
validity.
