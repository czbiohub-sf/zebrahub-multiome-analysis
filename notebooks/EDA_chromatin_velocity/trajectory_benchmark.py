"""Bounded known-truth benchmark for six-stage accessibility trajectories.

Every observation is a fixed peak x celltype accessibility trajectory in CPM.
Library exposure varies by celltype and stage; simulated counts are independent
Poisson samples conditional on truth. These are computational controls, not
biological replicates. No trajectory is min/max normalized. The narrow-pulse and
technical-spike families deliberately have identical observations but different
truth: six measurements cannot distinguish those explanations on their own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/chrom_velo"))
TIMES = np.array([10., 12., 14., 16., 19., 24.])
FAMILIES = (
    "stable_broad_high", "stable_broad_low", "stable_specific",
    "uniform_opening", "uniform_closing", "lineage_opening",
    "lineage_closing", "broad_pulse", "broad_trough", "lineage_pulse",
    "lineage_trough", "celltype_switch", "narrow_true_pulse",
    "isolated_technical_spike",
)
MODE_NAMES = ("stable", "opening", "closing", "pulse", "trough", "complex")
DEFAULT_OUTPUT = Path(__file__).parent / "outputs/trajectory_velocity_v1/synthetic"
RATE_FLOOR = .05  # A declared practical effect in synthetic CPM/hpf, not a P value.


def _tent(times, center=16., left=10., right=24.):
    """A bounded broad pulse; sampled interval derivatives are exact differences."""
    times = np.asarray(times)
    return np.maximum(0., np.minimum((times-left)/(center-left), (right-times)/(right-center)))


def make_benchmark(seed=42, n_per_family=120, n_celltypes=20):
    """Return an independent <=2,000-peak test atlas and complete latent truth.

    Exposures refer to a larger unmodelled atlas, so the counts of this subset do
    not sum to the exposure. The simulation has no cell-level sampling model or
    biological overdispersion. Low-signal controls are retained to expose lack
    of information rather than silently replacing noisy zeros by missing values.
    """
    if n_per_family < 1 or n_celltypes < 2:
        raise ValueError("Need positive family size and at least two celltypes")
    if n_per_family * len(FAMILIES) > 2000:
        raise ValueError("This benchmark is intentionally bounded to 2,000 peaks")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 300]))
    depth_rng = np.random.default_rng(np.random.SeedSequence([seed, 301]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 302]))
    c, t = n_celltypes, len(TIMES)
    stage_depth = np.array([2.2, 3.3, 2.4, 2.8, 2.0, .5])
    coverage = 3e6 * depth_rng.lognormal(0., .55, (c, 1)) * stage_depth
    coverage *= depth_rng.lognormal(-.5*.18**2, .18, (c, t))
    coverage = np.clip(coverage, .5e6, 30e6)
    n = n_per_family * len(FAMILIES)
    truth = np.empty((n, c, t), dtype=float)
    measurement = np.empty_like(truth)
    target_mask = np.zeros((n, c), dtype=bool)
    metadata = []
    linear = (TIMES-TIMES[0])/(TIMES[-1]-TIMES[0])
    pair_template = []
    for fi, family in enumerate(FAMILIES):
        for j in range(n_per_family):
            i = fi*n_per_family+j
            target = j % c
            partner = (target+max(1,c//2)) % c
            baseline = np.exp(rng.uniform(np.log(.05), np.log(2.)))
            amplitude = np.exp(rng.uniform(np.log(.2), np.log(20.)))
            center = 14. if j % 2 else 16.
            pulse = _tent(TIMES, center)
            current = np.full((c,t), baseline)
            target_mask[i,target] = True
            if family == "stable_broad_high":
                current[:] = np.exp(rng.uniform(np.log(2.), np.log(20.)))
                target_mask[i] = True
            elif family == "stable_broad_low":
                current[:] = np.exp(rng.uniform(np.log(.03), np.log(.2)))
                target_mask[i] = True
            elif family == "stable_specific":
                current[target] += amplitude
            elif family == "uniform_opening":
                current += amplitude*linear
                target_mask[i] = True
            elif family == "uniform_closing":
                current += amplitude*(1-linear)
                target_mask[i] = True
            elif family == "lineage_opening":
                current[target] += amplitude*linear
            elif family == "lineage_closing":
                current[target] += amplitude*(1-linear)
            elif family in ("broad_pulse", "lineage_pulse"):
                if family == "broad_pulse":
                    current += amplitude*pulse
                    target_mask[i] = True
                else:
                    current[target] += amplitude*pulse
            elif family in ("broad_trough", "lineage_trough"):
                if family == "broad_trough":
                    current += amplitude*(1-pulse)
                    target_mask[i] = True
                else:
                    current[target] += amplitude*(1-pulse)
            elif family == "celltype_switch":
                current[target] += amplitude*(1-linear)
                current[partner] += amplitude*linear
                target_mask[i,partner] = True
            elif family == "narrow_true_pulse":
                # The true pulse is restricted to (15.2,16.8) hpf, so only one
                # sampled stage sees it. The technical twin receives exactly the
                # same observed counts below, with a constant biological truth.
                current[target,3] += amplitude
                pair_template.append((baseline, amplitude, target, current.copy()))
            elif family == "isolated_technical_spike":
                baseline, amplitude, target, twin = pair_template[j]
                current[:] = baseline
                target_mask[i] = False
                target_mask[i,target] = True
            truth[i] = current
            measurement[i] = twin if family == "isolated_technical_spike" else current
            metadata.append(dict(peak=f"sim_peak_{i:05d}", family=family,
                                 target_celltype=target, partner_celltype=partner,
                                 baseline_cpm=float(current.min()), amplitude_cpm=amplitude,
                                 temporal_amplitude_cpm=float(np.ptp(current,axis=-1).max()),
                                 turn_hpf=(16. if family == "narrow_true_pulse" else center) if "pulse" in family or "trough" in family else np.nan,
                                 identifiability_pair=j if family in FAMILIES[-2:] else -1))
    counts = noise_rng.poisson(measurement*coverage[None,:,:]/1e6)
    pulse_start = FAMILIES.index("narrow_true_pulse")*n_per_family
    spike_start = FAMILIES.index("isolated_technical_spike")*n_per_family
    counts[spike_start:spike_start+n_per_family] = counts[pulse_start:pulse_start+n_per_family]
    values = counts/coverage[None,:,:]*1e6
    observed = np.ones_like(values,dtype=bool)
    if not np.isfinite(truth).all() or (truth < 0).any():
        raise AssertionError("Synthetic truth must be finite and nonnegative")
    return dict(seed=seed, times=TIMES.copy(), truth=truth, values=values, counts=counts,
                exposure=coverage, observed=observed, target_mask=target_mask,
                measurement_variance=(counts+.5)*(1e6/coverage[None,:,:])**2,
                count_supported=counts.sum(axis=-1)>=20,
                metadata=pd.DataFrame(metadata).set_index("peak"),
                true_rates=np.diff(truth,axis=-1)/np.diff(TIMES),
                raw_rates=np.diff(values,axis=-1)/np.diff(TIMES))


def rate_mode(rates, floor=RATE_FLOOR):
    """Classify the supported interval signs; missing intervals are not zeros."""
    rates = np.asarray(rates)
    flat = rates.reshape(-1,rates.shape[-1])
    result = np.full(len(flat), -1,dtype=np.int8)
    for i,row in enumerate(flat):
        if not np.isfinite(row).all():
            continue
        signs = np.sign(row[np.abs(row) >= floor])
        if not len(signs):
            result[i] = 0
        elif (signs > 0).all():
            result[i] = 1
        elif (signs < 0).all():
            result[i] = 2
        else:
            changes = np.flatnonzero(np.diff(signs))
            result[i] = (3 if signs[0] > 0 else 4) if len(changes) == 1 else 5
    return result.reshape(rates.shape[:-1])


def _safe_mean(values):
    values = np.asarray(values)
    return float(np.mean(values)) if values.size else np.nan


def score_rates(data, rates, estimator, floor=RATE_FLOOR, fit=None, selected_mask=None, core_mode=None):
    """Family-stratified errors on comparable observed interval support.

    Accuracy is evaluated against known interval-average rates. The sign recall
    denominator includes all true active supported intervals; conditional sign
    accuracy only includes intervals selected by the estimate. Stationary
    fractions distinguish per-interval errors from any false event per trajectory.
    """
    truth = data["true_rates"]
    observed = data["observed"]
    support = observed[...,:-1] & observed[...,1:] & np.isfinite(rates)
    true_mode = rate_mode(truth,floor)
    selected_mask = (np.abs(rates) >= floor) if selected_mask is None else np.asarray(selected_mask,dtype=bool)
    estimated_mode = rate_mode(np.where(selected_mask,rates,0.),floor)
    if core_mode is not None:
        # Core codes: insufficient, stable, increasing, decreasing, peak,
        # trough, uncertain/complex. Keep uncertainty out of the stable class.
        estimated_mode = np.asarray(core_mode,dtype=int)-1
    family_values = data["metadata"].family.to_numpy()
    records=[]
    for family in FAMILIES:
        peaks = family_values == family
        for scope in ["all_celltypes", "program_celltypes", "strong_program_celltypes"]:
            row = np.broadcast_to(peaks[:,None],truth.shape[:-1]).copy()
            if scope != "all_celltypes": row &= data["target_mask"]
            if scope == "strong_program_celltypes":
                row &= data["metadata"].amplitude_cpm.to_numpy()[:,None] >= 5.
            valid = support & row[...,None]
            true_active = valid & (np.abs(truth) >= floor)
            selected = valid & selected_mask
            both = true_active & selected
            stationary = row & (np.abs(truth)<1e-12).all(axis=-1)
            stationary_valid = valid & stationary[...,None]
            active_rows = row & np.isin(true_mode,[3,4]) & np.isfinite(rates).all(axis=-1)
            active_mode = true_mode[active_rows]
            row_valid = row & np.isfinite(rates).all(axis=-1)
            errors = (rates-truth)[valid]
            same = np.sign(rates) == np.sign(truth)
            out=dict(seed=data["seed"],estimator=estimator,family=family,scope=scope,
                     n_trajectories=int(row.sum()),supported_intervals=int(valid.sum()),
                     rate_rmse=float(np.sqrt(np.mean(errors**2))) if errors.size else np.nan,
                     rate_mae=_safe_mean(np.abs(errors)),
                     true_active_intervals=int(true_active.sum()),
                     sign_recall=_safe_mean((selected & same)[true_active]),
                     selected_active_sign_accuracy=_safe_mean(same[both]),
                     stationary_interval_fp=_safe_mean(selected[stationary_valid]),
                     stationary_trajectory_fp=_safe_mean((selected.any(axis=-1))[stationary & row_valid]),
                     stationary_peak_fp=_safe_mean(selected.any(axis=(1,2))[peaks & (np.abs(truth)<1e-12).all(axis=(1,2)) & row_valid.any(axis=1)]),
                     broad_reversal_recall=_safe_mean(estimated_mode[active_rows] == active_mode),
                     mode_accuracy=_safe_mean(estimated_mode[row_valid] == true_mode[row_valid]))
            if fit is not None:
                state_valid = observed & row[...,None] & np.isfinite(fit)
                out["state_rmse"] = float(np.sqrt(np.mean((fit-data["truth"])[state_valid]**2)))
            records.append(out)
    return pd.DataFrame(records),true_mode,estimated_mode


def _clean_json(value):
    if isinstance(value,dict): return {str(k):_clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_clean_json(v) for v in value]
    if isinstance(value,np.ndarray): return _clean_json(value.tolist())
    if isinstance(value,np.generic): return _clean_json(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def _raw_selection(data,rate_z=3.):
    variance = data["measurement_variance"]
    se = np.sqrt(variance[...,:-1]+variance[...,1:])/np.diff(TIMES)
    return (np.abs(data["raw_rates"]) >= np.maximum(RATE_FLOOR,rate_z*se)) & data["count_supported"][...,None]


def _fit(data, penalty, rate_z=3.):
    from trajectory_modes import fit_trajectories
    result=fit_trajectories(data["values"],data["times"],observed=data["observed"],
                            weights=np.broadcast_to(data["exposure"],data["values"].shape),
                            measurement_variance=data["measurement_variance"],
                            complexity_penalty=penalty,rate_floor=RATE_FLOOR,rate_z=rate_z)
    # Fit observed low-count states so closed CT coordinates remain represented.
    # Count support limits mode/change claims, not availability of fitted states.
    result["rate_supported"] &= data["count_supported"][...,None]
    result["mode"][~data["count_supported"]] = 0
    return result


def peak_detection_metrics(data,rates,se,selected,lack_of_fit=None):
    """Apply the real pipeline's per-peak/interval source activation rule.

    Low-count fitted coordinates remain in the vector and its sampling-noise
    denominator. A source needs any supported CT change AND whole-vector SNR>=2.
    Conditional SEs omit model-selection and biological uncertainty.
    """
    observed=data["observed"]
    feature=observed[...,:-1]&observed[...,1:]&np.isfinite(rates)&np.isfinite(se)
    if lack_of_fit is not None:
        feature &= (np.asarray(lack_of_fit)<=4.)[...,None]
    vector=np.where(feature,rates,0.)
    noise=np.sqrt(np.sum(np.where(feature,se,0.)**2,axis=1))
    magnitude=np.linalg.norm(vector,axis=1)
    snr=np.divide(magnitude,noise,out=np.zeros_like(magnitude),where=noise>1e-12)
    active=(selected&feature).any(axis=1)&(snr>=2.)
    true_active=(np.abs(data["true_rates"])>=RATE_FLOOR).any(axis=1)
    stationary=(np.abs(data["true_rates"])<1e-12).all(axis=(1,2))
    truth=data["true_rates"]
    cosine=np.divide(np.sum(vector*truth,axis=1),magnitude*np.linalg.norm(truth,axis=1),out=np.full_like(magnitude,np.nan),where=magnitude*np.linalg.norm(truth,axis=1)>1e-12)
    rows=[]
    for family in FAMILIES:
        use=data["metadata"].family.to_numpy()==family
        strong=use&(data["metadata"].amplitude_cpm.to_numpy()>=5.)
        true_intervals=true_active&use[:,None]
        strong_intervals=true_active&strong[:,None]
        selected_true=active&true_intervals
        rows.append(dict(family=family,n_peaks=int(use.sum()),
                         stationary_peak_fp=_safe_mean(active.any(axis=1)[use&stationary]),
                         stationary_peak_interval_fp=_safe_mean(active[use&stationary]),
                         true_active_intervals=int(true_intervals.sum()),
                         source_active_recall=_safe_mean(active[true_intervals]),
                         strong_source_active_recall=_safe_mean(active[strong_intervals]),
                         selected_true_vector_cosine=_safe_mean(cosine[selected_true]),
                         selected_true_wrong_direction=_safe_mean(cosine[selected_true]<0),
                         source_active_fraction=_safe_mean(active[use])))
    return pd.DataFrame(rows),dict(source_active=active,source_snr=snr)


def _result_arrays(result):
    # The shared core contract uses fitted/rates; explicit errors are preferable
    # to silently reshaping or reconstructing a different scientific estimator.
    return np.asarray(result["fitted"]),np.asarray(result["rates"])


def _calibration_summary(metrics,peak_metrics=None):
    table=metrics[metrics.scope == "program_celltypes"]
    primary=table[~table.family.isin(FAMILIES[-2:])]
    stationary=table[table.family.isin(FAMILIES[:3])]
    strong=metrics[metrics.scope == "strong_program_celltypes"]
    sustained=strong[strong.family.isin(["uniform_opening","uniform_closing","lineage_opening","lineage_closing","celltype_switch"])]
    transient=strong[strong.family.isin(["broad_pulse","broad_trough","lineage_pulse","lineage_trough"])]
    result=dict(macro_rate_rmse=float(primary.rate_rmse.mean()),
                worst_stationary_interval_fp=float(stationary.stationary_interval_fp.max()),
                worst_stationary_trajectory_fp=float(stationary.stationary_trajectory_fp.max()),
                worst_stationary_peak_fp=float(metrics[(metrics.scope == "all_celltypes") & metrics.family.isin(FAMILIES[:3])].stationary_peak_fp.max()),
                strong_sustained_sign_recall=float(sustained.sign_recall.mean()),
                strong_broad_reversal_recall=float(transient.broad_reversal_recall.mean()))
    if peak_metrics is not None:
        null=peak_metrics[peak_metrics.family.isin(FAMILIES[:3])]
        dynamic=peak_metrics[peak_metrics.family.isin(["uniform_opening","uniform_closing","lineage_opening","lineage_closing","broad_pulse","broad_trough","lineage_pulse","lineage_trough","celltype_switch"])]
        result["worst_stationary_source_peak_fp"]=float(null.stationary_peak_fp.max())
        result["strong_source_active_recall"]=float(dynamic.strong_source_active_recall.mean())
    return result


def _plot_outputs(out,data,results,metrics,calibration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    fit,rates=_result_arrays(results)
    fig,axes=plt.subplots(4,4,figsize=(15,12),sharex=True,layout="constrained")
    for ax,family in zip(axes.flat,FAMILIES):
        rows=np.flatnonzero(data["metadata"].family.to_numpy()==family)
        # Fixed 75th amplitude percentile is representative, not best-fit selection.
        order=np.argsort(data["metadata"].iloc[rows].amplitude_cpm.to_numpy())
        i=rows[order[int(.75*(len(rows)-1))]];ct=int(data["metadata"].iloc[i].target_celltype)
        ax.plot(TIMES,data["truth"][i,ct],color="black",lw=2,label="Truth")
        ax.scatter(TIMES,data["values"][i,ct],color="#878787",s=23,label="Measured")
        ax.plot(TIMES,fit[i,ct],color="#007b98",lw=1.8,label="Fitted")
        ax.set_title(family.replace("_"," "));ax.set_xlabel("hpf");ax.set_ylabel("CPM")
        ax.set_xticks(TIMES)
    for ax in axes.flat[len(FAMILIES):]:ax.set_visible(False)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Held-out synthetic trajectories: original CPM units; one fixed peak × celltype per panel",fontsize=13)
    for suffix in ["png","pdf"]:fig.savefig(out/f"01_synthetic_trajectories.{suffix}",dpi=180)
    plt.close(fig)
    table=metrics[metrics.scope == "program_celltypes"]
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout="constrained")
    x=np.arange(len(FAMILIES))
    for estimator,offset,color in [("raw",-.18,"#9c9c9c"),("fitted",.18,"#007b98")]:
        d=table[table.estimator==estimator].set_index("family").reindex(FAMILIES)
        axes[0].bar(x+offset,d.rate_rmse,.35,label=estimator,color=color)
        axes[1].bar(x+offset,d.sign_recall,.35,label=estimator,color=color)
        axes[2].bar(x+offset,d.mode_accuracy,.35,label=estimator,color=color)
    for ax,title in zip(axes,["Rate RMSE (CPM/hpf)","True-active sign recall","Trajectory-mode accuracy"]):
        ax.set_xticks(x,[f.replace("_"," ") for f in FAMILIES],rotation=75,ha="right")
        ax.set_title(title)
    axes[0].legend(frameon=False);axes[1].set_ylim(0,1);axes[2].set_ylim(0,1)
    fig.suptitle("Seed43 evaluation: narrow true pulse and technical spike are observationally identical",fontsize=12)
    for suffix in ["png","pdf"]:fig.savefig(out/f"02_synthetic_rate_metrics.{suffix}",dpi=180)
    plt.close(fig)
    true_mode=rate_mode(data["true_rates"]);estimated_mode=np.asarray(results["mode"],dtype=int)-1
    rows=data["target_mask"] & np.isfinite(rates).all(axis=-1)
    confusion=np.zeros((len(MODE_NAMES),len(MODE_NAMES)+1),int)
    np.add.at(confusion,(true_mode[rows],estimated_mode[rows]+1),1)
    normalized=np.divide(confusion,confusion.sum(axis=1,keepdims=True),out=np.zeros_like(confusion,dtype=float),where=confusion.sum(axis=1,keepdims=True)>0)
    fig,ax=plt.subplots(figsize=(7,6),layout="constrained")
    im=ax.imshow(normalized,cmap="Blues",norm=Normalize(0,1))
    for y in range(len(MODE_NAMES)):
        for x in range(len(MODE_NAMES)+1):
            ax.text(x,y,f"{confusion[y,x]}\n{normalized[y,x]:.0%}",ha="center",va="center",color="white" if normalized[y,x]>.6 else "black",fontsize=8)
    prediction_labels=("insufficient","stable","opening","closing","pulse","trough","uncertain/complex")
    ax.set_xticks(range(len(prediction_labels)),prediction_labels,rotation=45,ha="right")
    ax.set_yticks(range(len(MODE_NAMES)),MODE_NAMES);ax.set_xlabel("Fitted interval-sign mode");ax.set_ylabel("True interval-sign mode")
    ax.set_title("Held-out mode confusion, program celltypes\nIncludes deliberately ambiguous pulse/spike controls")
    fig.colorbar(im,ax=ax,label="Fraction within true mode")
    for suffix in ["png","pdf"]:fig.savefig(out/f"03_synthetic_mode_confusion.{suffix}",dpi=180)
    plt.close(fig)
    pd.DataFrame(confusion,index=MODE_NAMES,columns=prediction_labels).to_csv(out/"mode_confusion_counts.csv")

    peak_metrics=pd.read_csv(out/"heldout_seed43_peak_detection.csv")
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout="constrained")
    for z,color in [(3.,"#d08729"),(3.5,"#007b98"),(4.,"#7b5a9c")]:
        table=calibration[calibration.rate_z==z]
        if table.empty:continue
        axes[0,0].plot(table.complexity_penalty,table.macro_rate_rmse,"o-",color=color,label=f"z={z:g}")
        axes[0,1].plot(table.worst_stationary_source_peak_fp,table.strong_source_active_recall,"o-",color=color,label=f"z={z:g}")
    selected=calibration[(calibration.rate_z==3.5)&(calibration.complexity_penalty==.2)].iloc[0]
    axes[0,1].scatter(selected.worst_stationary_source_peak_fp,selected.strong_source_active_recall,marker="*",s=200,color="black",zorder=5,label="Chosen: penalty .2, z3.5")
    axes[0,0].set(xlabel="Complexity penalty",ylabel="Family-macro rate RMSE (CPM/hpf)",title="Seed42: continuous rate error")
    axes[0,0].text(.03,.96,"Frozen setting rechecked; no retuning" if len(calibration)==1 else "Curves coincide: z changes support, not fitted rates",transform=axes[0,0].transAxes,va="top",fontsize=8)
    axes[0,1].set(xlabel="Worst stationary family: any false source / peak",ylabel="Strong true source recall",title="Seed42: frozen support/power recheck" if len(calibration)==1 else "Seed42: support versus power",ylim=(0,1))
    axes[0,1].axvline(.05,color="#888888",ls="--",lw=1);axes[0,1].legend(frameon=False,fontsize=8)
    dynamic=list(FAMILIES[3:13]);null=list(FAMILIES[:3])+[FAMILIES[-1]]
    for estimator,offset,color in [("raw",-.18,"#9c9c9c"),("fitted",.18,"#007b98")]:
        table=peak_metrics[peak_metrics.estimator==estimator].set_index("family")
        axes[1,0].bar(np.arange(len(dynamic))+offset,table.loc[dynamic,"strong_source_active_recall"],.35,label=estimator,color=color)
        axes[1,1].bar(np.arange(len(null))+offset,table.loc[null,"stationary_peak_fp"],.35,label=estimator,color=color)
    axes[1,0].set_xticks(np.arange(len(dynamic)),[f.replace("_"," ") for f in dynamic],rotation=65,ha="right")
    axes[1,1].set_xticks(np.arange(len(null)),[f.replace("_"," ") for f in null],rotation=30,ha="right")
    axes[1,0].set(title="Seed43: strong true changes detected",ylabel="Source-active interval recall",ylim=(0,1))
    axes[1,1].set(title="Seed43: stationary peaks with any false source",ylabel="Fraction of peaks",ylim=(0,.5))
    axes[1,1].axhline(.05,color="#888888",ls="--",lw=1);axes[1,1].legend(frameon=False)
    fig.suptitle("Conditional sampling filters: improved broad-trend detection costs narrow-pulse sensitivity",fontsize=13)
    for suffix in ["png","pdf"]:fig.savefig(out/f"04_synthetic_source_detection.{suffix}",dpi=180)
    plt.close(fig)


def run_benchmark(out=DEFAULT_OUTPUT,n_per_family=120,penalties=(0.,.02,.05,.1,.2),rate_z_candidates=(3.,3.5,4.),frozen_only=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source_hashes={name:hashlib.sha256((ROOT/"scripts/chrom_velo"/name).read_bytes()).hexdigest() for name in ["trajectory_modes.py","trajectory_classification.py"]}
    if frozen_only:
        penalties,rate_z_candidates=(.2,),(3.5,)
    training=make_benchmark(42,n_per_family)
    raw_training,_,_=score_rates(training,training["raw_rates"],"raw",selected_mask=_raw_selection(training))
    raw_reference=_calibration_summary(raw_training)
    calibration=[]
    for rate_z in rate_z_candidates:
        for penalty in penalties:
            print(f"Calibrating seed42: penalty={penalty:g}, conditional z={rate_z:g}",flush=True)
            result=_fit(training,float(penalty),float(rate_z));fitted,rates=_result_arrays(result)
            metrics,_,_=score_rates(training,rates,"fitted",fit=fitted,selected_mask=result["rate_supported"],core_mode=result["mode"])
            peak_metrics,_=peak_detection_metrics(training,rates,result["rate_se"],result["rate_supported"],result["lack_of_fit"])
            summary=_calibration_summary(metrics,peak_metrics)
            summary.update(complexity_penalty=float(penalty),rate_z=float(rate_z),seed=42)
            summary["passes_predeclared_gates"]=(summary["worst_stationary_source_peak_fp"]<=.05 and summary["strong_sustained_sign_recall"]>=.90 and summary["strong_broad_reversal_recall"]>=.80 and summary["macro_rate_rmse"]<=raw_reference["macro_rate_rmse"])
            calibration.append(summary)
    calibration=pd.DataFrame(calibration)
    # The seed42 sweep showed that no candidate met all original recall and
    # false-positive gates. The declared conservative operating point prioritizes
    # peak-level false events over the last few percent of strong-signal recall.
    # This setting is fixed from seed42 only; seed43 never chooses parameters.
    penalty,rate_z=.2,3.5
    match=calibration[(calibration.complexity_penalty==penalty)&(calibration.rate_z==rate_z)]
    if len(match)!=1:raise ValueError("The declared operating point .2 / 3.5 must be in the calibration grid")
    best=match.iloc[0]
    calibration_file="frozen_recheck_seed42.csv" if frozen_only else "calibration_seed42.csv"
    calibration.to_csv(out/calibration_file,index=False)
    operating_point=dict(source_sha256=source_hashes,complexity_penalty=penalty,rate_z=rate_z,rate_floor=RATE_FLOOR,
                         min_total_counts_for_mode=20,source_snr_threshold=2.,
                         source_active="any supported celltype interval AND whole-vector L2 signal/sqrt(sum rate_SE_squared)>=2",
                         fit_low_count_states=True,calibration_seed=42,evaluation_seed=43,
                         recheck_only=bool(frozen_only),current_seed42_metrics=calibration_file,
                         original_calibration_table="calibration_seed42_pre_clipping_se_fix.csv" if frozen_only else calibration_file,
                         conditional_se="max(unclipped linear variance, active-set clipped-prediction variance); model-selection uncertainty excluded",
                         selection="Conservative operating point fixed from seed42: peak-level false-positive priority; original recall targets are reported even when missed.",
                         calibration_metrics=best.to_dict())
    (out/"operating_point.json").write_text(json.dumps(_clean_json(operating_point),indent=2,allow_nan=False)+"\n")
    print("Frozen operating point saved to "+str(out/"operating_point.json"),flush=True)
    test=make_benchmark(43,n_per_family)
    result=_fit(test,penalty,rate_z);fitted,rates=_result_arrays(result)
    raw_metrics,_,_=score_rates(test,test["raw_rates"],"raw",fit=test["values"],selected_mask=_raw_selection(test,rate_z))
    fitted_metrics,true_mode,estimated_mode=score_rates(test,rates,"fitted",fit=fitted,selected_mask=result["rate_supported"],core_mode=result["mode"])
    metrics=pd.concat([raw_metrics,fitted_metrics],ignore_index=True)
    metrics.to_csv(out/"heldout_seed43_metrics.csv",index=False)
    fitted_peak,source=peak_detection_metrics(test,rates,result["rate_se"],result["rate_supported"],result["lack_of_fit"])
    raw_se=np.sqrt(test["measurement_variance"][...,:-1]+test["measurement_variance"][...,1:])/np.diff(TIMES)
    raw_peak,_=peak_detection_metrics(test,test["raw_rates"],raw_se,_raw_selection(test,rate_z))
    pd.concat([raw_peak.assign(estimator="raw"),fitted_peak.assign(estimator="fitted")],ignore_index=True).to_csv(out/"heldout_seed43_peak_detection.csv",index=False)
    model_counts=pd.DataFrame({"family":np.repeat(test["metadata"].family.to_numpy(),20),"model":result["model"].ravel(),"mode":result["mode"].ravel(),"count_supported":test["count_supported"].ravel()}).groupby(["family","model","mode","count_supported"]).size().rename("n").reset_index()
    model_counts.to_csv(out/"heldout_seed43_model_counts.csv",index=False)
    test["metadata"].to_csv(out/"heldout_seed43_peak_truth.csv")
    np.savez_compressed(out/"heldout_seed43_arrays.npz",times=TIMES,truth=test["truth"],observed_values=test["values"],observed_mask=test["observed"],counts=test["counts"],exposure=test["exposure"],fitted=fitted,true_rates=test["true_rates"],raw_rates=test["raw_rates"],fitted_rates=rates,target_mask=test["target_mask"],true_mode=true_mode,estimated_mode=estimated_mode,core_mode=result["mode"],model=result["model"],rate_supported=result["rate_supported"],rate_se=result["rate_se"],cv_error=result["cv_error"],lack_of_fit=result["lack_of_fit"],source_active=source["source_active"],source_snr=source["source_snr"])
    summary=dict(source_sha256=source_hashes,calibration_seed=42,evaluation_seed=43,recheck_only=bool(frozen_only),current_seed42_metrics=calibration_file,n_peaks=len(test["metadata"]),n_celltypes=20,n_times=6,times_hpf=TIMES,rate_floor_cpm_per_hpf=RATE_FLOOR,selected_rate_z=rate_z,rate_z_candidates=list(rate_z_candidates),strong_signal_amplitude_min_cpm=5.,minimum_total_counts_per_trajectory=20,count_supported_trajectory_fraction=float(test["count_supported"].mean()),fitted_trajectory_fraction=float(result["fit_valid"].mean()),exposure_range=[float(test["exposure"].min()),float(test["exposure"].max())],penalty_candidates=list(penalties),selected_complexity_penalty=penalty,calibration_gates_satisfied=bool(best.passes_predeclared_gates),selection_objective=operating_point["selection"],raw_evaluation=_calibration_summary(raw_metrics,raw_peak),fitted_evaluation=_calibration_summary(fitted_metrics,fitted_peak),interpretation=["Independent computational seed, same simulator; not independent biological validation.","Poisson sampling omits biological overdispersion and cell-level dependence.","Narrow pulse and technical spike have exactly identical observations; this is a deliberate non-identifiability test.","No per-trajectory min/max normalization; values and rates remain in CPM and CPM/hpf.","Practical effect floors and empirical synthetic false-positive rates are not biological confidence levels."])
    (out/"benchmark_summary.json").write_text(json.dumps(_clean_json(summary),indent=2,allow_nan=False)+"\n")
    _plot_outputs(out,test,result,metrics,calibration)
    print(json.dumps(_clean_json(summary),indent=2,allow_nan=False),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--n-per-family",type=int,default=120)
    parser.add_argument("--penalties",type=float,nargs="+",default=[0.,.02,.05,.1,.2])
    parser.add_argument("--rate-z-candidates",type=float,nargs="+",default=[3.,3.5,4.])
    parser.add_argument("--frozen-only",action="store_true",help="Recheck only the frozen .2/z3.5 setting; do not repeat parameter selection")
    args=parser.parse_args()
    run_benchmark(args.output_dir,args.n_per_family,args.penalties,args.rate_z_candidates,args.frozen_only)


if __name__ == "__main__":
    main()
