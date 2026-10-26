#!/usr/bin/env python3
"""Audit the explicitly post-primary, selected-training/tuning sensitivity."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import time

import h5py
import numpy as np
import pandas as pd

import ct_shared_weight_forecast as base
import ct_shared_weight_selected_refit as selected
import ct_shared_weight_forecast_audit as audit

OUT = base.ROOT/'independent_audit/selected_refit'


def future_invariance(d, cohort, origin, half, fold):
    original = selected.run_outer_fold_cohort(d, half, origin, fold, cohort)
    changed = copy.deepcopy(d)
    future = np.arange(6)>origin
    rng = np.random.default_rng(audit.SEED+100+origin)
    for h in ('a','b'):
        changed['observed'][h][:,:,future] = rng.uniform(100,10000,changed['observed'][h][:,:,future].shape)
        changed['selection_observed'][h][:,:,future] = rng.uniform(100,10000,changed['selection_observed'][h][:,:,future].shape)
        changed['selection_counts'][h][:,:,future] = rng.integers(1000,100000,changed['selection_counts'][h][:,:,future].shape)
    changed['selection_ncells'][:,future]=0
    changed['raw']=base.build_raw_blocks(changed)
    altered=selected.run_outer_fold_cohort(changed,half,origin,fold,cohort)
    for name in ('train_indices','test_indices'):
        assert np.array_equal(original[name],altered[name])
    for kind in ('scores','masks'):
        for name in original['selection'][kind]:
            assert np.array_equal(original['selection'][kind][name],altered['selection'][kind][name])
    assert audit.native(original['selection']['audit'])==audit.native(altered['selection']['audit'])
    for model in base.MODELS:
        for key in ('prediction','unshrunk_prediction'):
            assert np.array_equal(original['models'][model][key],altered['models'][model][key]),(cohort,model,key)
        for key in ('fit','gate','tuning'):
            assert audit.native(original['models'][model][key])==audit.native(altered['models'][model][key]),(cohort,model,key)
    return dict(cohort=cohort,origin_hpf=float(d['stages'][origin]),half=half,fold=fold,
                all_six_models_bitwise_invariant=True,selection_and_tuning_bitwise_invariant=True)


def selected_membership_and_comparison(d, cohort, directory):
    checks=[];comparison_rows=[]
    with h5py.File(directory/'predictions.h5','r') as new,h5py.File(base.OUT/'predictions.h5','r') as old:
        assert new.attrs['training_and_tuning_cohort']==cohort
        assert bool(new.attrs['post_primary_exploratory'])
        for half in ('a','b'):
            assert np.array_equal(new[f'half_{half}/target_rate'][:],old[f'half_{half}/target_rate'][:])
            for name in old[f'half_{half}/cohorts']:
                assert np.array_equal(new[f'half_{half}/cohorts/{name}'][:],old[f'half_{half}/cohorts/{name}'][:])
            for t in base.ORIGINS:
                for fold in range(5):
                    outertrain=d['fold']!=fold
                    selection=base.selector(d,half,t,outertrain)
                    expected=np.flatnonzero(outertrain&selection['masks'][cohort])
                    saved=new[f'fits/half_{half}/origin_{t}/fold_{fold}/train_indices'][:]
                    assert np.array_equal(saved,expected)
                    for model in base.MODELS:
                        tune=json.loads((directory/f'tuning_half_{half}_origin_{t}_fold_{fold}_{model}.json').read_text())
                        assert tune['training_and_tuning_cohort']==cohort
                        assert len(tune['inner_details'])==3
                        for item in tune['inner_details']:
                            inner=item['inner_fold'];latest=t-1
                            innertrain=outertrain&(d['inner_fold']!=inner)
                            innerval=outertrain&(d['inner_fold']==inner)
                            sel=base.selector(d,half,latest,innertrain)
                            expected_train=np.flatnonzero(innertrain&sel['masks'][cohort])
                            expected_val=np.flatnonzero(innerval&sel['masks'][cohort])
                            assert len(expected_train)==item['n_train_peaks']
                            assert len(expected_val)==item['n_validation_peaks']
                            assert item['validation_interval']==latest
                            assert item['training_intervals']==(list(range(1,latest)) or [latest])
                            assert item['training_cohort']==cohort
                            assert audit.native(sel['audit'])==item['selection_audit']
                    checks.append(dict(half=half,origin=t,fold=fold,n_selected_training_peaks=len(saved)))
        metrics=pd.read_csv(directory/'metrics.csv')
        recomputed=[]
        for row in metrics.itertuples():
            ti=list(base.ORIGINS).index(row.origin)
            mask=old[f'half_{row.half}/cohorts/{cohort}'][:,ti]
            other='b' if row.half=='a' else 'a';a=d['observed'][other][d['target_local_indices']]
            truth=(a[:,:,row.origin+1]-a[:,:,row.origin])/(d['stages'][row.origin+1]-d['stages'][row.origin])
            pred=old[f'half_{row.half}/predictions/{row.model}'][:,:,ti].astype(float)
            sse=float(np.sum((pred[mask]-truth[mask])**2))
            assert np.isclose(sse,row.primary_same_model_rate_sse,rtol=1e-12,atol=1e-8)
            recomputed.append(dict(half=row.half,origin=row.origin,model=row.model,sse=sse))
        comparison=pd.read_csv(directory/'fit_strategy_comparison.csv')
        for row in comparison.itertuples():
            q=metrics[metrics.model==row.model]
            if str(row.interval)!='pooled':q=q[q.origin==int(row.interval)]
            new_sse=float(q[row.error].sum())
            if row.error=='rate_sse':old_sse=float(q.primary_same_model_rate_sse.sum())
            else:old_sse=float(np.sum(q.primary_same_model_rate_sse*(q.end_hpf-q.start_hpf)**2))
            assert np.isclose(new_sse,row.selected_training_sse,rtol=1e-12,atol=1e-8)
            assert np.isclose(old_sse,row.all_training_same_model_sse,rtol=1e-12,atol=1e-8)
            gain=1-new_sse/old_sse
            assert np.isclose(gain,row.improvement_selected_training_and_tuning_vs_all,atol=2e-12,rtol=2e-12)
            assert row.low<=row.high
            comparison_rows.append(dict(interval=str(row.interval),model=row.model,error=row.error,recomputed_gain=gain))
    return dict(n_outer_selected_training_sets_checked=len(checks),
                n_inner_selected_training_validation_sets_checked=len(checks)*len(base.MODELS)*3,
                outer_masks_and_targets_bitwise_match_primary=True,
                n_same_model_comparison_rows_checked=len(comparison_rows),
                n_primary_error_rows_checked=len(recomputed),
                interpretation='Joint selected training AND selected tuning, post-primary exploratory comparison, not independent confirmation.',
                training_counts=checks)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.time();report={'status':'running','cohorts':{}}
    paths=[Path(base.__file__),Path(selected.__file__),base.OUT/'predictions.h5',base.SELECT,audit.SOURCE,audit.EDGES]
    before=[audit.signature(p) for p in paths]
    d=base.load_data();source=audit.load_source();real,null,edges,_=audit.independent_graphs(source)
    manual,_=audit.manual_raw_blocks(d,real,null,edges)
    for cohort,origin,half,fold in [('total_variance_top20',3,'a',0),('gini_top20',4,'b',1)]:
        directory=selected.OUTROOT/cohort;out=OUT/cohort;out.mkdir(parents=True,exist_ok=True)
        adapter=SimpleNamespace(OUT=directory,MODELS=base.MODELS,ORIGINS=base.ORIGINS)
        result={}
        result['future_perturbation']=future_invariance(d,cohort,origin,half,fold)
        result['saved_outputs']=audit.saved_output_audit(adapter,d,manual,out)
        result['selected_membership_and_primary_comparison']=selected_membership_and_comparison(d,cohort,directory)
        report['cohorts'][cohort]=result
        audit.save_json(OUT/'selected_refit_independent_validation.json',report)
        print('SELECTED AUDIT PASSED',cohort,flush=True)
    assert before==[audit.signature(p) for p in paths]
    report.update(status='passed',seconds=time.time()-started,sources_unchanged=True,sources=before,
                  selected_model_sha256=audit.sha256(selected.__file__),
                  primary_model_sha256=audit.sha256(base.__file__),
                  script_sha256=audit.sha256(__file__),
                  primary_harness_sha256=audit.sha256(audit.__file__))
    audit.save_json(OUT/'selected_refit_independent_validation.json',report)
    print('SELECTED REFIT AUDIT PASSED',round(report['seconds'],1),'seconds',flush=True)


if __name__=='__main__':main()
