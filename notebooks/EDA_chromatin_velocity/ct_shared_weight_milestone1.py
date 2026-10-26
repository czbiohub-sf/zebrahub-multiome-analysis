import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ct_shared_weight_report as r
s=r.OUT/'selection_audit'
summary='''Milestone 1 is complete: peak selection now uses all reliable past celltype × stage pseudobulks, preserving both celltype differences and temporal changes. Of 190 observed groups, 176 have at least 20 cells. At forecast origins 14, 16 and 19 hpf, the selectors use 89, 120 and 151 past groups across 31 represented celltypes. Future accessibility values do not affect selection.

Gini and mean-adjusted total variance select different peaks. At 16 hpf, the Gini group has median accessibility 0.25 CPM and detection in 56% of past groups; the total-variance group has 1.28 CPM and 82.5% detection. The total-variance group retains substantial differences among celltype means, as you intended. I would keep Gini as a specificity-oriented comparison and use adjusted total variance as the primary variable-peak cohort.

The attached fixed-UMAP comparison shows these cohorts with our existing retrospective 16→19 hpf arrows. No UMAP or arrow directions were refitted for this display. The new six-weight Cicero forecast model is being tested next against persistence, own history, an equally flexible own-only model, matched neighbors and generic smoothing. This selection milestone does not yet establish improved forecasting or a global root-to-branch flow.'''
detailed='''MILESTONE 1 — RELIABLE PSEUDOBULKS AND VARIABLE-PEAK SELECTION
26 October 2026

COMPLETED
1. Audited cell numbers for all 32 celltypes × 6 developmental stages.
2. Re-extracted both independent cell-half counts for all 4,525 connected Cicero targets across all 32 celltypes.
3. Matched every existing 21-celltype count exactly and reconstructed all 190 pooled pseudobulk count columns exactly.
4. Built origin-specific, chromosome-held-out Gini and total-variance selections and numerical checks.
5. Plotted cohort diagnostics and reused the frozen peak atlas and retrospective arrows for a display-only comparison.

GROUP AND PEAK ELIGIBILITY
There are 190 observed groups, 14 observed groups below 20 cells, and two absent combinations. A group is retained if its pooled A+B cell count is at least 20. The balanced split guarantees at least 10 cells per half under this rule; among the past groups actually used at 14/16/19 hpf, the observed minima are 18/18/16 cells per half. Missing and excluded groups do not become zero accessibility.

A separate peak-level rule requires at least 20 source-half raw counts summed over eligible past groups. This count-support rule is distinct from the 20-cell pseudobulk rule. All eligible past groups have equal weight after full-library CPM normalization; pseudobulks with more cells do not receive greater biological weight.

SELECTION SCORES
Gini: inequality of the nonnegative linear-CPM values across eligible past pseudobulks. It can capture celltype specificity and temporal concentration but is not the same statistic as variance.
Total variance: population variance of log1p(CPM) across the same groups. This retains both differences among celltypes and changes across stages. To control the strong relationship between mean and variance, rank each peak against training peaks in its mean-CPM decile. The 80th percentile of eligible training scores defines each top-20% cohort.

All calibration uses the source half and other chromosomes only. Five chromosome folds are assembled into out-of-fold target masks. Selection is repeated at 14, 16 and 19 hpf, separately for A→B and B→A forecasts. These are top-20% cohorts within the eligible connected Cicero population, not within the complete 640,830-peak atlas. Neighbor partners are retained regardless of target selection.

HALF-A COHORTS
Origin    Eligible    High Gini    Adjusted total variance
14 hpf       4,482          893                         884
16 hpf       4,522          891                         913
19 hpf       4,522          910                         929

At 16 hpf, all eligible / Gini / total-variance median CPM is 1.24 / 0.25 / 1.28. Median past detection is 89.2% / 55.8% / 82.5%. Median share of total log-accessibility variance attributable to differences among celltype means is 55.1% / 40.7% / 73.6%. The weighted variance decomposition is exact over observed eligible groups. Unequal stage availability means the between-celltype component also includes differences in stage composition; it does not isolate a causal cell-identity effect.

FIXED-ATLAS DISPLAY
The map uses source-half A selection through 16 hpf, with thresholds learned on other chromosomes. Of 4,522 eligible / 891 Gini / 913 total-variance targets, 1,509 / 119 / 353 have a previously supported and projectable retrospective 16→19 arrow. At most 180 arrows per panel are shown using one deterministic priority over source identities, without choosing favorable directions. Arrow lengths are normalized; orange/blue indicate signed net opening/closing, respectively.

These vectors were previously fitted using the full six-stage course. Coordinates and neighbors also use the full course. They are retrospective profile-correspondence arrows, not new out-of-time forecasts, physical peak movement, or ancestral root directions. This figure isolates the display effect of choosing different source cohorts.

NUMERICAL VALIDATION
Thirty source-half × origin × chromosome-fold combinations passed future-value mutation and training-preprocessing invariance checks. Replacing all future accessibilities/counts with missing values and future cell counts with zero leaves every selector score and mask unchanged. Multiplying held-out peak values cannot change the training mean bins, training cutoffs or training scores. Source files are unchanged. Peak identities match the frozen atlas and stored projection exactly.

INTERPRETATION
The adjusted-total-variance cohort follows your requested biological scope and has much better count support than the Gini cohort in this connected subset. Gini can still identify biologically interesting restricted programs, but low detection must be separated from biological specificity. Much of the selected variability consists of differences across celltype contexts, so high variability alone cannot guarantee predictable future change. That is what the next forecast milestone tests.

SCOPE AND LIMITS
Cicero ≥0.25 connects only 4,525 of 640,830 atlas peaks (0.71%). Selection covers 31 represented celltypes in reliable past groups; forecast responses remain the existing 21 complete celltypes. Some selected variation therefore comes from contexts outside the response panel. The graph was estimated from all cells and stages; the forthcoming forecast test conditions on that graph. Independent cell halves assess sampling repeatability, not biological replication. CPM represents relative accessibility.

NEXT MILESTONE
Six global signed graph coefficients, shared across peaks and response celltypes, modify neighborhood contributions using partner state, recipient state, distance and score. Nested past-only validation selects shrinkage and ridge strength. Success requires additional predictive information beyond persistence, own history, own-only flexibility, matched neighbors and generic smoothing. Forecast results will be reported separately when computation and independent checks finish.
'''
figs=[(r.FIG/'00_data_coverage.png','Cell counts, inclusion rule, and sparse Cicero graph coverage.'),(s/'01_selector_diagnostics_170mm.png',(s/'selector_diagnostic_caption.txt').read_text()),(r.FIG/'01_variable_peak_fixed_atlas.png','Frozen peak UMAP and existing retrospective arrows, subsetted by source-half A selection through 16 hpf. These arrows are not the new forecasts.')]
extra=list(s.glob('*.csv'))+list(s.glob('*.csv.gz'))+list(s.glob('*.json'))+[s/'selection_methods.txt',s/'origin_fold_selection.h5']+list(r.FIG.glob('*.csv'))+list(r.FIG.glob('*.json'))+[r.HERE/'ct_variable_peak_forecast_selection.py',r.HERE/'ct_shared_weight_report.py']
r.package_milestone('milestone_1_selection','Chromatin dynamics milestone 1: reliable groups and variable peaks',summary,detailed,figs,extra)
