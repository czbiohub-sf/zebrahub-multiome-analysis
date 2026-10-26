#!/usr/bin/env python3
"""Package final reviewed shared-weight forecast results; no sending side effects."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pandas as pd
import ct_shared_weight_report as r
m=r.OUT/'model';s=r.OUT/'model_selected_refit';a=r.OUT/'independent_audit'
assert json.loads((a/'independent_validation.json').read_text())['status']=='passed'
assert json.loads((a/'selected_refit/selected_refit_independent_validation.json').read_text())['status']=='passed'
summary='''Milestone 2 is complete: the shared-weight Cicero model, all benchmark models, and selected-peak refits have been run and independently audited. The model learns six signed coefficients shared across peaks and celltypes, allowing partner accessibility, recipient accessibility, genomic distance and Cicero score to change each partner's contribution. Peak selection excludes pseudobulks with fewer than 20 cells and uses only measurements available at each forecast origin.

The main conclusion is that variable-peak selection remains useful for describing regulatory programs, but this Cicero model has not established additional predictive information. At 14→16 hpf, variance-selected peaks show 2.43% lower change-prediction error than persistence; the extra reduction versus own history is only 0.29%, with a chromosome-resampling range spanning zero. Across all three forecast intervals, shared Cicero increases error versus persistence by 6.10% on all eligible connected peaks, 19.13% on variance-selected peaks, and 27.55% on Gini-selected peaks.

We also fitted and tuned the models specifically within each selected cohort. This did not rescue the result: shared Cicero error remained 43.91% above persistence for high-variance targets and 4.14% above persistence for high-Gini targets. Gini-specific fitting reduced the original overprediction, but did not beat own history or generic smoothing. These refits are an explicitly exploratory sensitivity, with the same held-out targets and no new tuning options.

My interpretation: earlier relationships transfer poorly across 16→19 hpf, while late zero forecasts reflect model abstention after past validation—not biological stability. We can keep the variable-peak atlas and six-stage opening/closing descriptions as useful results. I would not present these arrows as validated future velocities or a global root-to-branch flow. The next scientific step should address transfer and measurement uncertainty, ideally with independent developmental samples; changing arrow appearance will not resolve the forecasting limitation.'''
primary=pd.read_csv(m/'pooled_metrics.csv')
q=primary[(primary.interval=='pooled')&(primary.error=='rate_sse')&(primary.cohort.isin(['all_eligible','gini_top20','total_variance_top20']))].copy()
q['error_increase_vs_persistence_percent']=-100*q.vs_persistence
q['error_increase_vs_own_history_percent']=-100*q.vs_own_history
q['error_increase_vs_generic_smoothing_percent']=-100*q.vs_generic_smoothing
q[['cohort','model','error_increase_vs_persistence_percent','error_increase_vs_own_history_percent','error_increase_vs_generic_smoothing_percent']].to_csv(r.OUT/'final_primary_comparison.csv',index=False)
table=q.pivot(index='model',columns='cohort',values='error_increase_vs_persistence_percent').reindex(['persistence','own_history','own_nonlinear','constant_cis','flexible_cis','distance_flexible','generic_smoothing'])
detailed='''MILESTONE 2 — SHARED CICERO WEIGHTS AND VARIABLE-PEAK FORECASTS
26 October 2026

WHAT WAS BUILT
A small transparent forecast model whose candidate partners are constrained by positive Cicero links. There are six globally shared graph coefficients, rather than a parameter for every peak or edge. The model predicts a peak's next accessibility change across 21 response celltypes. It makes no assumption that every peak should move away from a broadly accessible root.

DATA AND SCOPE
Cicero score >=0.25 provides 4,525 connected target peaks out of a 640,830-peak atlas (0.71%). All true partners and the matched-distance control partners are retained, giving 11,334 available source nodes. The graph has 17,550 directed entries, excludes self edges, stays within chromosome, and is row normalized.

Peak selection uses every reliable observed past celltype x stage pseudobulk, with >=20 pooled cells; absent groups remain absent. There are 89/120/151 eligible past groups across 31 represented celltypes at origins 14/16/19 hpf. Responses and model feature vectors retain the existing 21 complete celltypes. Selection therefore includes some contextual variability outside the response panel. A separate peak support floor requires >=20 source-half raw counts over the eligible past groups.

Gini is computed on linear CPM. Total variability is variance of log1p(CPM) across past pseudobulks, ranked within training-defined mean-CPM deciles. Both top-20% cutoffs are calibrated on other chromosomes using the source half only. They are percentiles within the connected eligible population, not the whole atlas. Neighbor partners are retained even when targets are filtered.

MODEL
Let a_i be current CPM and r_i the preceding observed CPM/hour change. The own-history part has a separate intercept and two slopes per celltype for log1p(a_i) and r_i. Define z as current log1p(CPM), standardized using training data. The six graph terms are:
  1. (W a)_i - a_i: current neighbor residual.
  2. (W r)_i: previous neighbor rate.
  3. z_i (W r)_i: recipient-state gating.
  4. W(z r)_i: partner-specific state gating.
  5. Sum_j W_ij d_ij r_j: distance-weighted partner rates.
  6. Sum_j W_ij s_ij r_j: original-score-weighted partner rates.
Here d is log1p(distance)/log(500001), clipped to [0,1], and s is the original Cicero score. Thus the effective signed lag weight is proportional to:
  W_ij [theta_0 + theta_recipient z_i + theta_partner z_j + theta_distance d_ij + theta_score s_ij].
An additional coefficient multiplies the current neighbor residual. Partner state/distance/score can change relative contributions among a receiver's partners. Negative fitted coefficients are predictive associations, not evidence of repression or causal direction.

The six graph slopes are shared across all peaks AND all response celltypes. Own slopes and intercepts are celltype-specific. Feature centering, scaling and state-gate scaling are learned separately inside every fitting split. The penalized objective is mean squared residual across examples/celltypes plus lambda times the mean squared own slopes and the sum of squared shared slopes. A Schur-complement solve implements this efficiently and matches an independent dense solve.

BENCHMARKS
Persistence predicts zero accessibility change. Six fitted models are compared:
- Own history: current log accessibility and preceding own rate.
- Own nonlinear: own history plus six shared nonlinear own-only features, including raw current CPM, to control for the raw-state component in the graph residual.
- Constant Cicero: own history plus current neighbor residual and previous neighbor rate.
- Shared Cicero: own history plus all six graph features.
- Distance control: identical six-feature graph model with the existing same-chromosome, approximately distance-matched control partners and original edge weights. This control does not preserve every abundance or degree property.
- Generic smoothing: own history plus current-state k30 neighbor residual and previous neighbor rate. Neighbors use only current source-half log-CPM profiles, exclude self, and have exponential distance weights. This is transductive across available peak identities but does not use future accessibility.

VALIDATION DESIGN
Forecasts are evaluated for 14->16, 16->19 and 19->24 hpf. Rates divide differences by the actual 2, 3 and 5 hour gaps. Source-half A measurements predict (B_next - B_current)/dt; the reciprocal B->A direction is also evaluated. The current source value is never subtracted from an outcome-half future value to define the change target.

Five chromosome folds separate coefficient-fitting targets from evaluation targets. All training outcomes end at or before the forecast origin. The latest completed interval supplies inner validation, with earlier completed intervals for training and three inner chromosome folds. At the first origin only one history-available interval exists, so inner validation uses chromosome separation within that past interval; it cannot supply a temporal inner holdout.

Nested validation chooses lambda from {0.1,1,10} and whole-forecast shrinkage from {0,0.25,0.5,1}. Zero is an allowed forecast. Ties favor smaller forecasts and then stronger regularization. No outer future outcome chooses hyperparameters, feature subsets or selection thresholds.

The primary fits all eligible targets and evaluates all connected, all past-count-eligible, Gini-selected and total-variance-selected strata. A bounded secondary experiment then restricts both training and inner validation to each selected cohort. Those selectors are recomputed at the inner origin and inner training chromosomes. Outer evaluation masks, neighbors, feature families, settings and folds remain identical. This secondary comparison changes BOTH fitting and tuning populations and is not independent confirmation.

RESULTS: PRIMARY POOLED CHANGE ERROR
Positive entries below mean a percentage INCREASE in squared CPM/hour error versus persistence; zero is persistence itself.

'''+table.to_string(float_format=lambda x:f'{x:.2f}')+'''

The shared-weight model also fails to outperform the own nonlinear, constant-Cicero, distance-control and generic-smoothing comparisons in pooled point estimates. For all eligible targets, the shared-Cicero improvement versus persistence is -6.10%, with a chromosome-bootstrap range of -9.95% to +0.11%. For variance-selected targets it is -19.13%, range -31.82% to +1.88%. These are conditional block-resampling ranges, not uncertainty across biological replicates or newly estimated graphs. They do not establish a universal negative claim about regulatory coupling.

EARLY INTERVAL SIGNAL AND FAILURE TO TRANSFER
For high-total-variance targets at 14->16 hpf, shared Cicero improves over persistence by 2.43% (chromosome-bootstrap range 0.63% to 4.83%). Its additional gain over own history is 0.29% (range -0.25% to +0.84%), and over equally flexible own-only features is 0.16% (range -0.60% to +1.22%). This supports a modest early forecasting signal, without establishing an added Cicero contribution. These are exploratory, unadjusted interval comparisons.

At 16->19 hpf, primary shared-Cicero error rises 25.57% above persistence for all eligible targets, 37.83% for high-total-variance targets and 44.73% for Gini targets. The other fitted models also transfer poorly. This demonstrates that the fitted earlier relationships generalize poorly across this interval in this dataset. It does not identify a biological transition, batch effect or normalization effect as the cause.

At 19->24 hpf, past validation selects an exactly zero forecast in all ten half/fold fits for most models, and nine of ten for shared Cicero. The nearly flat result therefore means abstention after shrinkage, not a claim that late chromatin biology is static.

SELECTED FITTING AND TUNING
The high-total-variance refit has shared-Cicero error 43.91% above persistence, compared with 19.13% under the primary all-eligible fit. On that same held-out cohort, the selected fit has 20.80% more error than the primary shared-Cicero model. It still does not beat own history or generic smoothing.

The Gini-specific refit reduces shared-Cicero error to 4.14% above persistence, compared with 27.55% under primary all-eligible fitting. This reduces overprediction, but shared Cicero remains 1.87% worse than its own-history benchmark and 1.89% worse than generic smoothing. The selected-vs-all improvement has a chromosome-bootstrap range spanning zero. It is not evidence that Cicero supplied predictive information.

INTERVAL CHANGE VERSUS FUTURE LEVEL
Primary error measures rates in CPM/hour. A secondary table weights each rate squared error by dt^2 to evaluate the interval accessibility change. This equals the error of B_current + dt*predicted_rate against B_next, using B_current only as a common evaluation anchor. It is not an operational source-only absolute future-level forecast. Future-level correlations were not used to claim success.

INDEPENDENT IMPLEMENTATION CHECKS
- Exact source peak alignment and raw-count/CPM reconstruction.
- Independent sparse edge reconstruction, no self edges, cis-only and row-normalization checks.
- Dense ridge versus Schur-complement coefficient agreement within 7.8e-16.
- A planted, deliberately strong partner-specific synthetic signal: the implemented partner gate reduces error by 99.18% versus constant weighting or removal of that partner term. Shuffled labels and zero-feature controls pass. This checks the implementation, not realistic power on these biological data.
- Entire-fold future perturbations leave selection, tuning losses/choices, fitted parameters and predictions bitwise unchanged.
- Reconstruction of all 540 saved fitted models and predictions across the three training strategies, plus the metric, chromosome-loss and pooled comparison tables.
- Exact selected training and 1,080 inner training/validation membership checks; identical primary/secondary outer targets and masks.
- Primary maximum relative SSE reconstruction discrepancy 9.7e-10.
Original data and frozen primary results remain unchanged.

INTERPRETATION AND NEXT SCIENTIFIC PRIORITY
Mean-adjusted total variance is useful for selecting heterogeneous regulatory programs. High Gini supplies a different, sparser specificity-oriented cohort. Selection and retrospective arrows can improve how we describe the measured developmental course. Neither primary forecasts nor selected refits demonstrate that Cicero neighbors add reliable information about the next change beyond own state/history and generic pooling.

The supported result is a descriptive chromatin-dynamics atlas, with limited early own-history predictability on variable peaks. A validated global root-to-branch velocity field is not established. Broadly accessible peaks remain a useful reference, not proven ancestors of specialized peaks.

A more persuasive future test would use independent, lineage-matched developmental samples and denser coverage around 14-19 hpf, learning the graph and coefficients in training samples only. With the current data, count-aware or program-level forecasts are possible new hypotheses, but they would need to pass the same baseline comparisons. They have not been run in this milestone. I would preserve the current benchmark and avoid treating a more coherent-looking stream as predictive validation.

LIMITS
Only 4,525 strongly connected peaks enter this test. The graph was learned using all stages and cells, so this is conditional evaluation of that graph, not a fully prospective graph-learning test. The fixed 21-celltype response panel uses whole-course cell-count coverage metadata. Cell halves assess sampling, not biological replication. There are only three forecast intervals, which share stages. Raw-CPM squared error emphasizes larger accessibility changes and high-amplitude peaks. CPM cannot distinguish global absolute chromatin changes from relative composition. Top-20% selections and model families were fixed; comparisons after examining prior project stages remain exploratory.

FILES
All results are under notebooks/EDA_chromatin_velocity/outputs/ct_shared_weight_forecast_v1/.
model/: primary predictions.h5, shared coefficients, nested tuning, per-celltype/interval/chromosome losses, pooled comparisons and plots.
model_selected_refit/total_variance_top20/ and model_selected_refit/gini_top20/: same artifacts for the two bounded selected-fit sensitivities.
selection_audit/: full 32-celltype target counts, past-only masks, validation and diagnostic figures.
independent_audit/: both independent numerical validation manifests and details.
figures/: cell-count coverage, retrospective cohort atlas, selected-fitting comparison, plot data and exact 170 mm canvas checks.

The three full prediction H5 files remain in the workspace because each is approximately 20-24 MB. The email archive includes the detailed text, plotted PDFs, metric/coefficient/tuning tables, audit/design records and all scripts; the first milestone archive already included the selection masks and diagnostics. Reproduction requires the recorded original local inputs. No new constrained-model arrows were presented as validated forecasts.
'''
(r.OUT/'HIGH_LEVEL_SUMMARY.txt').write_text(summary+'\n');(r.OUT/'DETAILED_SUMMARY.txt').write_text(detailed+'\n')
figs=[(m/'01_shared_weight_forecast.png','Primary interval forecasts on identical evaluation cohorts. Bands are conditional chromosome-bootstrap ranges; late zero forecasts reflect model shrinkage.'),(m/'02_shared_weight_comparators.png','Shared-Cicero error reductions versus each comparator; rate and interval-change scores. Positive values would favor shared Cicero.'),(r.FIG/'02_selected_fit_comparison_170mm.png','Same held-out targets, with fitting and tuning performed on all eligible versus selected targets. Positive error increases are worse than persistence. This 170 mm figure summarizes the explicitly exploratory selected-fit sensitivity.'),(m/'03_selected_shrinkage_and_penalty.png','Nested past-only choices across 30 outer fits per model. A shrinkage of zero produces an exactly zero change forecast.')]
extra=[r.OUT/'HIGH_LEVEL_SUMMARY.txt',r.OUT/'DETAILED_SUMMARY.txt',r.OUT/'final_primary_comparison.csv',r.OUT/'run_scope.txt',r.FIG/'selected_fit_comparison_plot_data.csv',r.FIG/'figure_canvas_validation.json']
for folder in [m,s,a]:
 extra+=list(folder.rglob('*.json'))+list(folder.rglob('*.csv'))+list(folder.rglob('*.csv.gz'))+list(folder.rglob('*.txt'))
extra += [r.HERE/name for name in ['ct_shared_weight_forecast.py','ct_shared_weight_selected_refit.py','ct_shared_weight_forecast_audit.py','ct_shared_weight_selected_refit_audit.py','ct_variable_peak_forecast_selection.py','ct_shared_weight_report.py','ct_shared_weight_refit_figure.py','ct_shared_weight_milestone1.py','ct_shared_weight_milestone2.py']]
r.package_milestone('milestone_2_forecasts','Chromatin dynamics milestone 2: shared Cicero forecasts and selected-peak refits',summary,detailed,figs,extra)
# Compact workspace entrypoint links the portable reports and the exact arrays.
index='''<!doctype html><html><head><meta charset="utf-8"><title>Variable peaks and shared Cicero forecasts</title><style>body{max-width:950px;margin:40px auto;padding:0 20px;font:16px/1.6 Arial;color:#20303b}a{color:#176d8a}li{margin:12px 0}</style></head><body><h1>Variable peaks and shared Cicero forecasts</h1><p>26 October 2026</p><h2>Conclusion</h2><p>Variable-peak selection captures heterogeneous regulatory programs. This small shared-weight Cicero model does not establish predictive improvement over persistence, own history and generic smoothing. A modest 14→16 hpf signal in variable peaks is largely explained by own history.</p><ul><li><a href="email_package/milestone_1_selection/milestone_1_selection_report.html">Milestone 1: selection audit and retrospective atlas plots</a></li><li><a href="email_package/milestone_2_forecasts/milestone_2_forecasts_report.html">Milestone 2: forecast results, refits and detailed interpretation</a></li><li><a href="HIGH_LEVEL_SUMMARY.txt">High-level summary</a> · <a href="DETAILED_SUMMARY.txt">Detailed methods and results</a></li><li><a href="model/predictions.h5">Primary predictions and fitted coefficients</a></li><li><a href="model_selected_refit/total_variance_top20/predictions.h5">Variance-selected fitting/tuning predictions</a></li><li><a href="model_selected_refit/gini_top20/predictions.h5">Gini-selected fitting/tuning predictions</a></li><li><a href="independent_audit/independent_validation.json">Primary independent audit</a> · <a href="independent_audit/selected_refit/selected_refit_independent_validation.json">Selected-refit audit</a></li></ul><p>Both portable milestone reports include embedded plots. The full prediction arrays remain available locally. Cohort atlas arrows reuse the previous six-stage fits and are retrospective.</p></body></html>'''
(r.OUT/'index.html').write_text(index)
