#!/usr/bin/env python3
"""Held-out full-celltype-space shared-flow and count-noise controls.

An observation is a genomic peak. kNN is recomputed independently at each start
stage, with no temporal edges and no held-out target access. Exact Euclidean
neighbors use the complete fixed 21-celltype state. `observed` uses state and
change from the same count draw; `independent` uses state A and change B.
Poisson draws describe conditional count sampling, not biological replication.
"""
from pathlib import Path
import argparse, json, time, hashlib, sys
import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
OUT=BASE/'outputs/ct_flow_noise_v1'
SAMPLE=BASE/'outputs/state_trajectory_atlas_v1/sample_data.h5'
SOURCE=BASE/'outputs/trajectory_velocity_v1/trajectory_fits.h5'
SEED=20261021
EPS=1e-10


def decode(x):
    return np.asarray([v.decode() if isinstance(v,bytes) else str(v) for v in x])


def unit(x):
    n=np.linalg.norm(x,axis=-1,keepdims=True)
    return np.divide(x,n,out=np.zeros_like(x,dtype=float),where=n>EPS)


def cosine(a,b):
    out=np.sum(unit(a)*unit(b),axis=-1)
    out[(np.linalg.norm(a,axis=-1)<=EPS)|(np.linalg.norm(b,axis=-1)<=EPS)]=np.nan
    return out


def finite_mean(x):
    x=np.asarray(x);return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan


def transform(x,name):
    return np.log1p(x) if name=='log1p' else x.copy()


def jsonwrite(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def load_sample():
    with h5py.File(SAMPLE) as f:
        mask=f['condition_valid'][:].all(axis=1)
        assert np.array_equal(mask,f['common_celltypes'][:])
        d={k:f[k][:] for k in ['source_indices','train_mask','test_mask','stages']}
        d['peak_ids']=f['peak_ids'].asstr()[:];d['celltypes']=f['celltypes'].asstr()[:][mask]
        d['celltype_indices']=np.flatnonzero(mask)
        for k in ['observed_cpm','fitted_cpm','rate_supported','rates','rate_se']:
            d[k]=np.asarray(f[k][:][:,:,mask],dtype=float)
        d['support_count']=d['rate_supported'].sum(axis=2)
        d['snr']=np.linalg.norm(d['rates'],axis=2)/np.maximum(np.linalg.norm(d['rate_se'],axis=2),EPS)
    with h5py.File(SOURCE) as f:
        d['coverage']=f['coverage'][:][mask,:].T
    d['train']=np.flatnonzero(d['train_mask']);d['test']=np.flatnonzero(d['test_mask'])
    assert len(d['train'])==20000 and len(d['test'])==5000 and len(d['celltypes'])==21
    assert not np.intersect1d(d['train'],d['test']).size
    assert np.isfinite(d['observed_cpm']).all() and np.isfinite(d['fitted_cpm']).all()
    return d


def summarize(y,predictions,metadata,mask,coherence=None,truth=None):
    y=y[mask];rows=[]
    norm2=np.sum(y*y,axis=1)
    mean_pred=predictions['interval_mean'][mask]
    baseline_err=np.sum((y-mean_pred)**2,axis=1)
    ridge_err=np.sum((y-predictions['ridge'][mask])**2,axis=1)
    for name,fullpred in predictions.items():
        pred=fullpred[mask];err=np.sum((y-pred)**2,axis=1)
        cos=cosine(y,pred)
        row={**metadata,'method':name,'n_peaks':len(y),
             'mean_squared_vector_error':float(np.mean(err)),
             'mean_target_squared_norm':float(np.mean(norm2)),
             'improvement_vs_zero':float(1-err.sum()/norm2.sum()) if norm2.sum()>EPS else np.nan,
             'improvement_vs_interval_mean':float(1-err.sum()/baseline_err.sum()) if baseline_err.sum()>EPS else np.nan,
             'improvement_vs_ridge':float(1-err.sum()/ridge_err.sum()) if ridge_err.sum()>EPS else np.nan,
             'mean_cosine':finite_mean(cos),'median_cosine':float(np.nanmedian(cos)) if np.isfinite(cos).any() else np.nan,
             'positive_cosine_fraction':float(np.mean(cos[np.isfinite(cos)]>0)) if np.isfinite(cos).any() else np.nan,
             'n_nonzero_target_and_prediction':int(np.isfinite(cos).sum()),
             'speed_spearman':float(spearmanr(np.linalg.norm(y,axis=1),np.linalg.norm(pred,axis=1)).statistic) if np.std(np.linalg.norm(pred,axis=1))>EPS else np.nan}
        if coherence is not None and name.startswith('knn_'):
            cc=coherence[int(name.split('_')[1])]
            for k,v in cc.items():row[k]=finite_mean(v[mask])
        if truth is not None:
            ty=truth[mask];terr=np.sum((pred-ty)**2,axis=1);tn=np.sum(ty*ty,axis=1)
            row['truth_error']=float(terr.mean());row['truth_improvement_vs_zero']=float(1-terr.sum()/tn.sum()) if tn.sum()>EPS else np.nan
            row['truth_mean_cosine']=finite_mean(cosine(ty,pred))
        # Peak resampling quantifies sampling sensitivity only; neighboring peaks
        # and cis-related peaks are not independent biological replicates.
        if metadata['dataset'] in ['real','cell_split'] and name=='knn_30':
            rng=np.random.default_rng(SEED+int(metadata['interval_index']))
            rr=[];cc=[]
            for _ in range(200):
                ix=rng.integers(len(y),size=len(y));den=baseline_err[ix].sum()
                rr.append(1-err[ix].sum()/den if den>EPS else np.nan)
                cc.append(finite_mean(cos[ix]))
            row['sampling_bootstrap_improvement_low'],row['sampling_bootstrap_improvement_high']=np.nanpercentile(rr,[2.5,97.5])
            row['sampling_bootstrap_cosine_low'],row['sampling_bootstrap_cosine_high']=np.nanpercentile(cc,[2.5,97.5])
        rows.append(row)
    return rows


def write_array(g,name,a):
    if name in g:del g[name]
    g.create_dataset(name,data=a,compression='gzip',shuffle=True)


def evaluate_dataset(d,dataset,replicate,state_cpm,targets,rows,vectors=None,truth_cpm=None,geometry_suffix='',transforms=('log1p','linear')):
    """targets maps target name to a separate or reused peak×stage×CT array."""
    tr,te=d['train'],d['test'];dt=np.diff(d['stages'])
    for trans in transforms:
        state_transform='log1p' if trans=='logstate_linear_rate' else trans
        target_transform='linear' if trans=='logstate_linear_rate' else trans
        state=transform(state_cpm,state_transform)
        target_rates={name:np.diff(transform(a,target_transform),axis=1)/dt[None,:,None] for name,a in targets.items()}
        truth_rates=None if truth_cpm is None else np.diff(transform(truth_cpm,target_transform),axis=1)/dt[None,:,None]
        for t in range(5):
            started=time.time();xtr=state[tr,t,:];xte=state[te,t,:]
            nn=NearestNeighbors(n_neighbors=60,algorithm='kd_tree',n_jobs=2).fit(xtr)
            distances,neighbors=nn.kneighbors(xte)
            assert not np.intersect1d(tr[neighbors.ravel()],te).size
            # Fixed random-neighbor baseline has the same averaging count.
            rng=np.random.default_rng(SEED+100+t)
            random_neighbors=rng.integers(len(tr),size=(len(te),30))
            scaler=StandardScaler().fit(xtr)
            ztr=scaler.transform(xtr);zte=scaler.transform(xte)
            for target_name,rates in target_rates.items():
                train_y=rates[tr,t,:];test_y=rates[te,t,:]
                predictions={'zero':np.zeros_like(test_y),
                             'interval_mean':np.broadcast_to(train_y.mean(axis=0),test_y.shape),
                             'random_30':train_y[random_neighbors].mean(axis=1),
                             'ridge':Ridge(alpha=100.0,fit_intercept=True).fit(ztr,train_y).predict(zte)}
                yn=train_y[neighbors]
                coherence={}
                for k in [15,30,60]:
                    predictions[f'knn_{k}']=yn[:,:k].mean(axis=1)
                    u=unit(yn[:,:k]);nonzero=np.linalg.norm(yn[:,:k],axis=2)>EPS
                    denom=nonzero.sum(axis=1)
                    umean=np.divide(u.sum(axis=1),denom[:,None],out=np.zeros((len(te),len(d['celltypes']))),where=denom[:,None]>0)
                    dots=np.sum(u*unit(test_y)[:,None,:],axis=2)
                    mean_dot=np.divide((dots*nonzero).sum(axis=1),denom,out=np.full(len(te),np.nan),where=denom>0)
                    opposite=np.divide(((dots<0)&nonzero).sum(axis=1),denom,out=np.full(len(te),np.nan),where=denom>0)
                    query_nonzero=np.linalg.norm(test_y,axis=1)>EPS
                    mean_dot[~query_nonzero]=np.nan;opposite[~query_nonzero]=np.nan
                    coherence[k]={'neighbor_direction_resultant':np.linalg.norm(umean,axis=1),
                                  'query_neighbor_cosine':mean_dot,'opposing_neighbor_fraction':opposite}
                meta={'dataset':dataset,'replicate':replicate,'transform':trans,'target':target_name,
                      'geometry':geometry_suffix or 'same_draw' if target_name=='observed' else geometry_suffix or 'observed_state',
                      'interval_index':t,'start_hpf':d['stages'][t],'end_hpf':d['stages'][t+1]}
                strata={'all':np.ones(len(te),bool)}
                if dataset in ['real','cell_split']:
                    support=d['support_count'][te,t]
                    strata.update(no_supported_ct=support==0,any_supported_ct=support>0,at_least_3_supported_ct=support>=3,
                                  any_support_snr2=(support>0)&(d['snr'][te,t]>=2))
                for stratum,mask in strata.items():
                    if mask.any():rows.extend(summarize(test_y,predictions,{**meta,'stratum':stratum},mask,coherence,None if truth_rates is None else truth_rates[te,t]))
                if vectors is not None:
                    group=f'{dataset}/{trans}/{target_name}/interval_{int(d["stages"][t])}_{int(d["stages"][t+1])}'
                    g=vectors.require_group(group)
                    write_array(g,'target_rate',test_y.astype('f4'))
                    for method,pred in predictions.items():write_array(g,'prediction_'+method,pred.astype('f4'))
                    for key,arr in coherence[30].items():write_array(g,key,arr.astype('f4'))
                    write_array(g,'neighbor_sample_indices',tr[neighbors].astype('i4'))
                    write_array(g,'neighbor_distances',distances.astype('f4'))
                    write_array(g,'prediction_cosine',cosine(test_y,predictions['knn_30']).astype('f4'))
                    write_array(g,'target_minus_prediction_squared_error',np.sum((test_y-predictions['knn_30'])**2,axis=1).astype('f4'))
            print(f'{dataset} {replicate} {trans} {d["stages"][t]:g}->{d["stages"][t+1]:g} {time.time()-started:.1f}s',flush=True)


def generate_count_draws(truth,coverage,seed):
    rng=np.random.default_rng(seed)
    effective=coverage/1e6
    return rng.poisson(truth*effective[None,:,:])/effective[None,:,:]


def controls(d):
    # Exposure-weighted constant latent rate: a count MLE across all stages.
    coverage=d['coverage'];base=np.sum(d['observed_cpm']*coverage[None,:,:],axis=1)/coverage.sum(axis=0)[None,:]
    stable=np.repeat(base[:,None,:],6,axis=1)
    # State-dependent planted programs: the dominant celltype determines the
    # mode, while only a peak's three most accessible CTs change. Parameters
    # fixed before inspecting real-data results; all peaks remain in evaluation.
    mode=np.argmax(base,axis=1)%4
    u=(d['stages']-d['stages'][0])/(d['stages'][-1]-d['stages'][0])
    patterns=np.vstack([.8*u,-.6*u,.8*np.sin(np.pi*u),-.6*np.sin(np.pi*u)])
    targeted=np.zeros_like(base);np.put_along_axis(targeted,np.argsort(base,axis=1)[:,-3:],1,axis=1)
    planted=stable*(1+patterns[mode,:,None]*targeted[:,None,:])
    assert np.min(planted)>=0
    return stable,planted,mode


def validate(d,stable,planted,manifest):
    # Direct calculation checks independent of the metric aggregation code.
    assert np.array_equal(np.diff(stable,axis=1),np.zeros_like(stable[:,:5]))
    dt=np.diff(d['stages']);rng=np.random.default_rng(991)
    x=rng.normal(size=(200,21));y=2*x+1
    tr=np.arange(150);te=np.arange(150,200)
    nn=NearestNeighbors(n_neighbors=15,algorithm='kd_tree').fit(x[tr]);dist,idx=nn.kneighbors(x[te])
    direct=np.argsort(np.sum((x[te,None,:]-x[None,tr,:])**2,axis=2),axis=1)[:,:15]
    assert np.array_equal(idx,direct)
    assert np.allclose(cosine(y,y),1)
    assert np.allclose(cosine(y,-y),-1)
    assert np.allclose(np.linalg.norm(unit(y),axis=1),1)
    assert np.all(np.isnan(cosine(y,np.zeros_like(y))))
    probe=np.arange(6,dtype=float)[None,:,None]*np.ones((3,1,21))
    assert np.allclose(np.diff(probe,axis=1)/dt[None,:,None],np.broadcast_to(1/dt[None,:,None],(3,5,21)))
    # Independent draws share latent rates, not realized counts.
    a=generate_count_draws(stable,d['coverage'],991);b=generate_count_draws(stable,d['coverage'],992)
    assert not np.array_equal(a,b)
    observed_integer=d['observed_cpm']*d['coverage'][None,:,:]/1e6
    max_rounding=float(np.max(np.abs(observed_integer-np.round(observed_integer))))
    manifest.update(validation={'exact_neighbor_reference_passed':True,'cosine_sign_checks_passed':True,
                                'nonuniform_time_division_passed':True,'stable_latent_rates_exactly_zero':True,
                                'independent_count_draws_differ':True,'max_reconstructed_count_rounding_error':max_rounding})
    assert max_rounding<0.01


def plots_and_report(metrics_path=OUT/'metrics.csv'):
    df=pd.read_csv(metrics_path)
    main=df[(df.stratum=='all')&(df.method=='knn_30')]
    intervals=['10→12','12→14','14→16','16→19','19→24'];x=np.arange(5)
    colors={'observed':'#2766a6','fitted':'#d87521','independent':'#228753'}
    fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for col,trans in enumerate(['log1p','linear']):
        for row,metric in enumerate(['improvement_vs_interval_mean','mean_cosine']):
            ax=axs[row,col]
            for target in ['observed','fitted']:
                dd=main[(main.dataset=='real')&(main['transform']==trans)&(main.target==target)].sort_values('interval_index')
                ax.plot(x,dd[metric],'-o',c=colors[target],label='Real '+('observed, same draw' if target=='observed' else 'retrospective fitted target'))
            for target,ls,c in [('observed','--','#888888'),('independent',':','#1c7b55')]:
                dd=main[(main.dataset=='constant_null')&(main['transform']==trans)&(main.target==target)]
                pivot=dd.pivot(index='replicate',columns='interval_index',values=metric)
                ax.plot(x,pivot.mean(axis=0),ls,c=c,label='Constant-latent null: '+('same draw' if target=='observed' else 'independent draw'))
                ax.fill_between(x,pivot.min(axis=0),pivot.max(axis=0),color=c,alpha=.13)
            ax.axhline(0,c='black',lw=.7);ax.set_xticks(x,intervals);ax.set_xlabel('Developmental interval (hpf)')
            ax.set_title(trans+' full 21-celltype space');ax.set_ylabel('Prediction improvement versus interval mean' if row==0 else 'Mean prediction–target cosine')
            if row==1:ax.set_ylim(-.15,1)
    axs[0,0].legend(fontsize=8,loc='best');fig.suptitle('Does a shared local change predict held-out peaks?\n20,000 training / 5,000 held-out whole peaks; k=30, interval-specific fields',fontsize=14)
    fig.savefig(OUT/'01_shared_flow_vs_count_noise.png',dpi=180);fig.savefig(OUT/'01_shared_flow_vs_count_noise.pdf');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for col,trans in enumerate(['log1p','linear']):
        for target,ls,c in [('observed','-','#2766a6'),('independent','--','#228753')]:
            dd=main[(main.dataset=='planted')&(main['transform']==trans)&(main.target==target)].sort_values('interval_index')
            axs[0,col].plot(x,dd.truth_improvement_vs_zero,ls+'o',c=c,label='Noisy target: '+('same draw' if target=='observed' else 'independent draw'))
            axs[1,col].plot(x,dd.truth_mean_cosine,ls+'o',c=c,label='Noisy target: '+('same draw' if target=='observed' else 'independent draw'))
        dd=main[(main.dataset=='planted')&(main['transform']==trans)&(main.target=='truth')].sort_values('interval_index')
        axs[0,col].plot(x,dd.truth_improvement_vs_zero,':o',c='#6b4b9a',label='True rates available to training neighbors')
        axs[1,col].plot(x,dd.truth_mean_cosine,':o',c='#6b4b9a',label='True rates available to training neighbors')
        for row in range(2):
            ax=axs[row,col];ax.axhline(0,c='black',lw=.7);ax.set_xticks(x,intervals);ax.set_title(trans+' space');ax.set_xlabel('Developmental interval (hpf)')
        axs[0,col].set_ylabel('Recovery of planted true rates versus zero');axs[1,col].set_ylabel('Mean cosine with planted true direction');axs[1,col].set_ylim(-.15,1)
    axs[0,0].legend(fontsize=8);fig.suptitle('Power control: known opening, closing, transient peak and trough\nActual depths; modes fixed from baseline dominant celltype; change in top three celltypes',fontsize=13)
    fig.savefig(OUT/'02_planted_program_recovery.png',dpi=180);fig.savefig(OUT/'02_planted_program_recovery.pdf');plt.close(fig)
    real=df[(df.dataset=='real')&(df.target=='observed')&(df.stratum=='all')]
    fig,axs=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for ax,trans in zip(axs,['log1p','linear']):
        for method in ['knn_15','knn_30','knn_60','ridge','random_30','zero']:
            dd=real[(real['transform']==trans)&(real.method==method)].sort_values('interval_index')
            ax.plot(x,dd.improvement_vs_interval_mean,'-o',label=method)
        ax.axhline(0,c='black',lw=.7);ax.set_xticks(x,intervals);ax.set_title(trans);ax.set_ylabel('Improvement versus interval mean');ax.set_xlabel('Interval (hpf)')
    axs[0].legend(fontsize=8,ncol=2);fig.suptitle('Prediction baselines and prespecified neighbor sensitivity\nObserved same-draw target: apparent predictability may contain regression to the mean')
    fig.savefig(OUT/'03_prediction_baselines.png',dpi=180);fig.savefig(OUT/'03_prediction_baselines.pdf');plt.close(fig)
    keep=['dataset','replicate','transform','target','start_hpf','end_hpf','n_peaks','improvement_vs_interval_mean','improvement_vs_ridge','mean_cosine','neighbor_direction_resultant','opposing_neighbor_fraction','truth_improvement_vs_zero','truth_mean_cosine']
    main[[k for k in keep if k in main]].to_csv(OUT/'key_results.csv',index=False)
    text='''<!doctype html><html><head><meta charset="utf-8"><title>Shared chromatin flow and count-noise controls</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;line-height:1.5}img{width:100%}table{font-size:12px;border-collapse:collapse}td,th{padding:5px;border:1px solid #ddd}.note{background:#eef4f7;padding:16px}</style></head><body><h1>Shared chromatin flow: held-out prediction and count-noise controls</h1><p>Each point is a genomic peak, with its state represented by the full fixed 21-celltype accessibility vector at a stage. A separate predictor is built for each of five intervals; peaks are split as whole identities (20,000 training, 5,000 held out). k=30 is primary; 15 and 60 are prespecified sensitivities. Geometry is computed only from the current observed state.</p><div class="note"><b>Interpretation guardrail:</b> the velocity includes minus the current measurement. Selecting neighbors by that same measurement can predict regression to the mean even when the underlying accessibility is constant. Compare real observations with the same-depth constant-latent null, and use independent-draw or independent-cell state/target measurements for stronger evidence. Fitted trajectories are retrospective and borrow all six stages; they are not forecasts.</div><h2>Shared-flow test</h2><img src="01_shared_flow_vs_count_noise.png"><p>Shading spans three fixed-seed null realizations and is not a confidence interval. Positive improvement means lower squared vector error than the training interval-mean predictor; zero means equal error. Cosine excludes exactly zero vectors and its denominator is reported in the tables.</p><h2>Known dynamics control</h2><img src="02_planted_program_recovery.png"><p>The planted programs depend on the dominant baseline celltype and alter each peak’s top three celltypes. This checks recoverability for this particular signal strength and geometry; it is not a comprehensive power guarantee.</p><h2>Baseline comparison</h2><img src="03_prediction_baselines.png"><p>The ridge model uses feature standardization learned only on training peaks, fixed alpha=100, and an intercept. kNN uses unscaled Euclidean distance in log1p(CPM) or CPM, separately. No held-out target is used to choose neighbors or tune parameters.</p><h2>Scope and limitations</h2><ul><li>CPM rates use the actual intervals of 2, 2, 2, 3, and 5 hours. Log-space differences are transformed-state rates, not absolute chromatin molecules per hour.</li><li>Constant-latent controls use exposure-weighted accessibility across all stages and actual pseudobulk total counts. Poisson sampling omits biological variation, overdispersion and compositional constraints. Different depths also induce log-transformation bias.</li><li>Support strata use the existing trajectory analysis; no peaks are discarded from the unconditional results. Fit support is conditional count support, not biological replication.</li><li>Peak bootstrap intervals describe sampling sensitivity; nearby or co-regulated peaks are not independent biological replicates.</li><li>Similar-state prediction can reflect shared celltype composition or technical effects. Stronger evidence requires agreement in independent cell subsets or biological replicates.</li></ul><h2>Primary unconditional results</h2>'''
    r=main[(main.dataset=='real')][[k for k in keep if k in main]]
    text+=r.to_html(index=False,float_format=lambda x:f'{x:.3f}')+'<p>See metrics.csv for every baseline, null realization, and support stratum; flow_vectors.h5 stores held-out target/prediction vectors and neighbor identities.</p></body></html>'
    (OUT/'report.html').write_text(text)



def fit_existing_model(cpm,coverage,stages):
    sys.path.insert(0,str(BASE.parents[1]/'scripts/chrom_velo'))
    from trajectory_modes import fit_trajectories
    counts=np.round(cpm*coverage[None,:,:]/1e6)
    exact=counts*1e6/coverage[None,:,:]
    vals=exact.transpose(0,2,1);exp=coverage.T
    variance=(counts.transpose(0,2,1)+.5)*(1e6/exp)**2
    with h5py.File(SOURCE) as f:params=json.loads(f.attrs['parameters_json'])
    result=fit_trajectories(vals,stages,observed=np.ones(exp.shape,bool),weights=exp,measurement_variance=variance,**params)
    return result,params


def refit_null():
    d=load_sample();stable,planted,mode=controls(d)
    a=generate_count_draws(stable,d['coverage'],SEED+1000)
    b=generate_count_draws(stable,d['coverage'],SEED+1001)
    real,params=fit_existing_model(d['observed_cpm'],d['coverage'],d['stages'])
    error=float(np.max(np.abs(real['fitted'].transpose(0,2,1)-d['fitted_cpm'])))
    assert np.allclose(real['fitted'].transpose(0,2,1),d['fitted_cpm'],atol=2e-4,rtol=2e-5),error
    fitted_a,_=fit_existing_model(a,d['coverage'],d['stages'])
    fitted_b,_=fit_existing_model(b,d['coverage'],d['stages'])
    rows=pd.read_csv(OUT/'metrics.csv').query("dataset != 'constant_null_refit'").to_dict('records')
    evaluate_dataset(d,'constant_null_refit',0,a,{'fitted':fitted_a['fitted'].transpose(0,2,1),'independent_fitted':fitted_b['fitted'].transpose(0,2,1)},rows,truth_cpm=stable)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    with h5py.File(OUT/'constant_null_trajectory_refit.h5','w') as f:
        for key,res in [('same_draw',fitted_a),('independent_draw',fitted_b)]:
            g=f.create_group(key)
            for name in ['fitted','rates','rate_se','rate_supported','model','mode']:
                write_array(g,name,res[name])
        f.attrs['axis_order']='peak,celltype,stage or interval'
        f.attrs['parameters_json']=json.dumps(params)
        f.attrs['seed_a']=SEED+1000;f.attrs['seed_b']=SEED+1001
        f.create_dataset('peak_ids',data=d['peak_ids'],dtype=h5py.string_dtype())
    j={'same_cohort_as_real':True,'train_peaks':20000,'held_out_peaks':5000,'null_replicates':1,
       'parameters':params,'real_refit_max_abs_difference':error,
       'validation':'Identical original fitting function, parameters, exposure weights, count-derived variance, and peak cohort; real fitted states independently reconstructed from integer counts.',
       'interpretation':'One fixed-seed full-pipeline count-noise diagnostic, not a null distribution or biological confidence.'}
    jsonwrite(OUT/'trajectory_refit_validation.json',j)
    plots_and_report();split_plots();refit_plot()


def refit_plot():
    df=pd.read_csv(OUT/'metrics.csv');df=df[(df.stratum=='all')&(df.method=='knn_30')]
    if not (df.dataset=='constant_null_refit').any():return
    fig,axs=plt.subplots(2,2,figsize=(13,8.5),constrained_layout=True);x=np.arange(5)
    for col,trans in enumerate(['log1p','linear']):
        for row,metric in enumerate(['improvement_vs_interval_mean','mean_cosine']):
            ax=axs[row,col]
            for dataset,target,label,color,style in [('real','fitted','Real retrospective fitted','#d87521','-'),('constant_null_refit','fitted','Constant-latent refit, same draw','#777777','--'),('constant_null_refit','independent_fitted','Constant-latent refit, independent draw','#228753',':')]:
                q=df[(df.dataset==dataset)&(df.target==target)&(df['transform']==trans)].sort_values('interval_index')
                if len(q):ax.plot(x,q[metric],style+'o',color=color,label=label)
            ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,['10→12','12→14','14→16','16→19','19→24']);ax.set_title(trans);ax.set_xlabel('Interval (hpf)')
            ax.set_ylabel('Improvement versus interval mean' if row==0 else 'Mean prediction–target cosine')
    axs[0,0].legend(fontsize=8);fig.suptitle('Identical six-stage fitting applied to real and constant-latent data\nOne fixed-seed null realization; full 20,000 / 5,000 peak split',fontsize=13)
    fig.savefig(OUT/'05_full_trajectory_refit_null.png',dpi=180);fig.savefig(OUT/'05_full_trajectory_refit_null.pdf');plt.close(fig)
    p=OUT/'report.html';s=p.read_text();s=s.replace('</body>', '<h2>Full trajectory-fitting noise control</h2><img src="05_full_trajectory_refit_null.png"><p>The identical original six-stage model, parameters, count-derived variances and coverage weights were applied to one constant-latent Poisson realization for all 25,056 peaks. Original real fitted states were independently reconstructed from integer counts as a numerical check. This comparison matches the retrospective fitting step; the one realization does not establish a null distribution.</p></body>');p.write_text(s)


def split_plots():
    df=pd.read_csv(OUT/'metrics.csv');df=df[(df.stratum=='all')&(df.method=='knn_30')]
    if not (df.dataset=='cell_split').any():return
    fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);x=np.arange(5)
    for col,trans in enumerate(['log1p','linear']):
        for row,metric in enumerate(['improvement_vs_interval_mean','mean_cosine']):
            ax=axs[row,col]
            for target,label,color,ls in [('Astate_Bchange','A state → independent B change','#228753','-'),('Bstate_Achange','B state → independent A change','#7856a2','-'),('Astate_Achange','A state → same A change','#2766a6','--')]:
                q=df[(df.dataset=='cell_split')&(df['transform']==trans)&(df.target==target)].sort_values('interval_index')
                if len(q):ax.plot(x,q[metric],ls+'o',color=color,label=label)
            for target,color,ls in [('Astate_Bchange','#333333',':'),('Astate_Achange','#a0a0a0','--')]:
                q=df[(df.dataset=='cell_split_null')&(df['transform']==trans)&(df.target==target)]
                if len(q):
                    pp=q.pivot(index='replicate',columns='interval_index',values=metric)
                    ax.plot(x,pp.mean(axis=0),ls,color=color,label='Matched-depth null: '+('independent' if target=='Astate_Bchange' else 'same draw'))
                    ax.fill_between(x,pp.min(axis=0),pp.max(axis=0),color=color,alpha=.12)
            ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,['10→12','12→14','14→16','16→19','19→24']);ax.set_title(trans);ax.set_xlabel('Interval (hpf)')
            ax.set_ylabel('Improvement versus interval mean' if row==0 else 'Mean prediction–target cosine')
    axs[0,0].legend(fontsize=8);fig.suptitle('Independent cell halves: does shared flow survive measurement separation?\nSame whole-peak split and fixed k=30; each half uses its own actual pseudobulk coverage',fontsize=13)
    fig.savefig(OUT/'04_independent_cell_split_flow.png',dpi=180);fig.savefig(OUT/'04_independent_cell_split_flow.pdf');plt.close(fig)
    p=OUT/'report.html';s=p.read_text();s=s.replace('</body>', '<h2>Independent cell halves</h2><img src="04_independent_cell_split_flow.png"><p>Cells were split within celltype and stage (and within dataset at 16 hpf). Neighbor selection uses one half, and training/test change vectors use the other. Each half is normalized by its own actual coverage. Matched Poisson controls use those two half-coverages. These are sampling-stability controls, not biological replicates; stage and batch are largely confounded.</p></body>');p.write_text(s)


def strong_planted():
    d=load_sample();stable,planted,mode=controls(d)
    multiplier=np.divide(planted,stable,out=np.ones_like(planted),where=stable>0)
    strong=stable*multiplier**3
    assert np.isfinite(strong).all() and np.min(strong)>=0
    assert np.max(strong.sum(axis=0))<1e6
    a=generate_count_draws(strong,d['coverage'],SEED+4000);b=generate_count_draws(strong,d['coverage'],SEED+4001)
    rows=pd.read_csv(OUT/'metrics.csv').query("dataset not in ['planted_strong','planted_strong_noiseless']").to_dict('records')
    evaluate_dataset(d,'planted_strong',0,a,{'observed':a,'independent':b,'truth':strong},rows,truth_cpm=strong)
    evaluate_dataset(d,'planted_strong_noiseless',0,strong,{'truth':strong},rows,truth_cpm=strong)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    desc={'purpose':'Additional stronger power sensitivity after the moderate control showed weak recovery; no real-data tuning or selection.',
          'formula':'strong = baseline * (moderate / baseline)^3, with zero baseline retained at zero',
          'minimum_multiplier':float(multiplier.min()**3),'maximum_multiplier':float(multiplier.max()**3),
          'maximum_sample_total_cpm_per_condition':float(np.max(strong.sum(axis=0))),
          'same_peak_cohort_and_modes_and_top3_celltype_targets':True,'geometry_and_predictor_parameters_unchanged':True,
          'oracle_control':'planted_strong_noiseless uses noiseless state and true change; planted_strong/truth uses noisy state but true training changes'}
    jsonwrite(OUT/'strong_planted_control.json',desc)
    np.savez_compressed(OUT/'strong_planted_truth.npz',truth_cpm=strong.astype('f4'),mode=mode)
    strong_plot()


def strong_plot():
    df=pd.read_csv(OUT/'metrics.csv');df=df[(df.stratum=='all')&(df.method=='knn_30')]
    if not (df.dataset=='planted_strong').any():return
    fig,axs=plt.subplots(2,2,figsize=(13,8.5),constrained_layout=True);x=np.arange(5)
    for col,trans in enumerate(['log1p','linear']):
        for row,metric in enumerate(['truth_improvement_vs_zero','truth_mean_cosine']):
            ax=axs[row,col]
            for ds,target,label,c,ls in [('planted','independent','Moderate signal, independent noisy changes','#777777','--'),('planted_strong','independent','Stronger signal, independent noisy changes','#228753','-'),('planted_strong','truth','Stronger signal, true changes / noisy state','#b77e26',':'),('planted_strong_noiseless','truth','Stronger signal, noiseless state and changes','#7856a2','-')]:
                q=df[(df.dataset==ds)&(df.target==target)&(df['transform']==trans)].sort_values('interval_index')
                ax.plot(x,q[metric],ls+'o',color=c,label=label)
            ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,['10→12','12→14','14→16','16→19','19→24']);ax.set_title(trans);ax.set_xlabel('Interval (hpf)')
            ax.set_ylabel('Recovery of true rate versus zero' if row==0 else 'Mean cosine with true direction')
    axs[0,0].legend(fontsize=8);fig.suptitle('Power sensitivity: increase planted signal while keeping real depths\nOriginal multipliers cubed; maximum opening 5.832×, minimum closing 0.064×',fontsize=13)
    fig.savefig(OUT/'07_stronger_planted_power_control.png',dpi=180);fig.savefig(OUT/'07_stronger_planted_power_control.pdf');plt.close(fig)


def effect_bins():
    d=load_sample();path=OUT/'cell_split_input.h5'
    with h5py.File(path) as f:
        cts=f['celltypes'].asstr()[:];ix=np.array([np.where(cts==c)[0][0] for c in d['celltypes']])
        aa=f['observed_cpm_a'][:][:,:,ix];bb=f['observed_cpm_b'][:][:,:,ix]
    rows=[];thresholds=[];te=d['test'];tr=d['train'];dt=np.diff(d['stages'])
    with h5py.File(OUT/'flow_vectors.h5','a') as f:
        for trans in ['log1p','linear','logstate_linear_rate']:
            ar=np.diff(transform(aa,trans),axis=1)/dt[None,:,None]
            br=np.diff(transform(bb,trans),axis=1)/dt[None,:,None]
            ag=f.require_group('cell_split_agreement/'+trans)
            write_array(ag,'observed_change_cosine',cosine(ar[te],br[te]).astype('f4'))
            prediction_a=[];prediction_b=[]
            for t in range(5):
                interval=f'interval_{int(d["stages"][t])}_{int(d["stages"][t+1])}'
                g=f['cell_split/'+trans+'/Astate_Bchange/'+interval]
                reverse=f['cell_split/'+trans+'/Bstate_Achange/'+interval]
                prediction_a.append(g['prediction_knn_30'][:]);prediction_b.append(reverse['prediction_knn_30'][:])
                pred={key.removeprefix('prediction_'):g[key][:].astype(float) for key in g if key.startswith('prediction_') and key!='prediction_cosine'}
                train_effect=np.linalg.norm(ar[tr,t,:],axis=1)
                edges=np.quantile(train_effect,[0,.5,.8,.95,1]);edges[0]=-np.inf;edges[-1]=np.inf
                bin_id=np.digitize(np.linalg.norm(ar[te,t,:],axis=1),edges[1:-1],right=False)
                write_array(ag,f'half_A_effect_bin_interval_{t}',bin_id.astype('i1'))
                total_error=np.sum(br[te,t,:]**2)
                for k,label in enumerate(['0–50%','50–80%','80–95%','95–100%']):
                    mask=bin_id==k
                    meta={'dataset':'cell_split_effect_bins','replicate':0,'transform':trans,'target':'Astate_Bchange',
                          'interval_index':t,'start_hpf':d['stages'][t],'end_hpf':d['stages'][t+1],'stratum':label,
                          'n_peaks':int(mask.sum()),'fraction_target_squared_norm':float(np.sum(br[te,t,:][mask]**2)/total_error),
                          'training_A_lower_threshold':float(edges[k]),'training_A_upper_threshold':float(edges[k+1])}
                    rows.extend(summarize(br[te,t,:],pred,meta,mask))
                    thresholds.append({k:v for k,v in meta.items() if k!='dataset'})
            pa=np.stack(prediction_a,axis=1);pb=np.stack(prediction_b,axis=1)
            write_array(ag,'prediction_change_cosine',cosine(pa,pb).astype('f4'))
            write_array(ag,'prediction_positive_in_both_halves',((cosine(pa,br[te])>0)&(cosine(pb,ar[te])>0)))
    pd.DataFrame(rows).to_csv(OUT/'independent_change_effect_bins.csv',index=False)
    pd.DataFrame(thresholds).to_csv(OUT/'effect_bin_definitions.csv',index=False)
    jsonwrite(OUT/'effect_bin_method.json',{'bin_thresholds':'Training-peak half-A rate L2 norm quantiles [0,0.5,0.8,0.95,1], independently within interval and transform',
                                         'held_out_bin_assignment':'Held-out half-A rate norm; half-B outcomes never used for defining bins',
                                         'interpretation':'Conditional exploratory comparison; half-A measured magnitude contains noise. Large norm in log space differs from large norm in CPM.'})
    q=pd.DataFrame(rows);q=q[q.method=='knn_30'];fig,axs=plt.subplots(2,3,figsize=(16,9),constrained_layout=True);x=np.arange(5)
    colors=['#9d9d9d','#5b94bf','#b7802c','#ad3860']
    for row,trans in enumerate(['log1p','linear']):
        for j,(metric,title) in enumerate([('improvement_vs_interval_mean','Prediction improvement'),('mean_cosine','Direction agreement'),('fraction_target_squared_norm','Share of target squared magnitude')]):
            ax=axs[row,j]
            for label,c in zip(['0–50%','50–80%','80–95%','95–100%'],colors):
                z=q[(q['transform']==trans)&(q.stratum==label)].sort_values('interval_index');ax.plot(x,z[metric],'-o',color=c,label=label)
            ax.axhline(0,color='black',lw=.7);ax.set_title(trans+': '+title);ax.set_xticks(x,['10→12','12→14','14→16','16→19','19→24']);ax.set_xlabel('Interval (hpf)')
            if j==2:ax.set_ylim(0,1)
    axs[0,0].legend(title='Half-A change magnitude bin',fontsize=8,title_fontsize=9);fig.suptitle('Are independent predictions concentrated in a minority of large changes?\nThresholds from training half A; held-out bin assignment uses half A; evaluation uses half B only',fontsize=13)
    fig.savefig(OUT/'06_independent_effect_size_bins.png',dpi=180);fig.savefig(OUT/'06_independent_effect_size_bins.pdf');plt.close(fig)


def hybrid():
    d=load_sample()
    with h5py.File(OUT/'cell_split_input.h5') as f:
        ids=f['peak_ids'].asstr()[:];assert np.array_equal(ids,d['peak_ids'])
        cts=f['celltypes'].asstr()[:];ix=np.array([np.where(cts==c)[0][0] for c in d['celltypes']])
        a=f['observed_cpm_a'][:][:,:,ix];b=f['observed_cpm_b'][:][:,:,ix]
        ca=f['coverage_a'][:];cb=f['coverage_b'][:]
        if ca.shape==(len(cts),6):ca=ca.T;cb=cb.T
        ca=ca[:,ix];cb=cb[:,ix]
    rows=pd.read_csv(OUT/'metrics.csv').query("transform != 'logstate_linear_rate'").to_dict('records')
    with h5py.File(OUT/'flow_vectors.h5','a') as vectors:
        evaluate_dataset(d,'cell_split',0,a,{'Astate_Bchange':b,'Astate_Achange':a},rows,vectors,geometry_suffix='log1p_CPM_state__linear_CPM_per_hpf_target',transforms=['logstate_linear_rate'])
        evaluate_dataset(d,'cell_split',1,b,{'Bstate_Achange':a},rows,vectors,geometry_suffix='log1p_CPM_state__linear_CPM_per_hpf_target',transforms=['logstate_linear_rate'])
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    base=np.sum(a*ca[None,:,:]+b*cb[None,:,:],axis=1)/np.sum(ca+cb,axis=0)[None,:]
    stable=np.repeat(base[:,None,:],6,axis=1)
    for rep in range(3):
        na=generate_count_draws(stable,ca,SEED+3000+2*rep);nb=generate_count_draws(stable,cb,SEED+3001+2*rep)
        evaluate_dataset(d,'cell_split_null',rep,na,{'Astate_Bchange':nb,'Astate_Achange':na},rows,geometry_suffix='log1p_CPM_state__linear_CPM_per_hpf_target',transforms=['logstate_linear_rate'])
        pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    jsonwrite(OUT/'hybrid_geometry_method.json',{'state_geometry':'Euclidean distance in full21-celltype log1p(CPM) current-stage state',
                                             'target_units':'Linear CPM per hpf; diff(linear CPM)/actual developmental time gap',
                                             'target_and_geometry_separated':True,'no_parameter_retuning':True,
                                             'purpose':'Test log-state neighbor geometry without log-rate Jensen/depth bias'})
    hybrid_plot()


def hybrid_plot():
    df=pd.read_csv(OUT/'metrics.csv');df=df[(df.stratum=='all')&(df.target=='Astate_Bchange')]
    if not (df['transform']=='logstate_linear_rate').any():return
    fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);x=np.arange(5)
    for col,method in enumerate(['knn_30','ridge']):
        for row,metric in enumerate(['improvement_vs_interval_mean','mean_cosine']):
            ax=axs[row,col]
            for trans,c,label in [('linear','#275f9a','Linear CPM state → CPM/hpf'),('logstate_linear_rate','#a76f25','Log1p(CPM) state → CPM/hpf')]:
                q=df[(df.dataset=='cell_split')&(df.method==method)&(df['transform']==trans)].sort_values('interval_index');ax.plot(x,q[metric],'-o',color=c,label=label)
                q=df[(df.dataset=='cell_split_null')&(df.method==method)&(df['transform']==trans)];pp=q.pivot(index='replicate',columns='interval_index',values=metric)
                ax.plot(x,pp.mean(axis=0),'--',color=c,label='Matched null: '+label.split(' → ')[0]);ax.fill_between(x,pp.min(axis=0),pp.max(axis=0),color=c,alpha=.12)
            ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,['10→12','12→14','14→16','16→19','19→24']);ax.set_title(method);ax.set_xlabel('Interval (hpf)')
            ax.set_ylabel('Improvement versus interval mean' if row==0 else 'Mean prediction–target cosine')
    axs[0,0].legend(fontsize=8);fig.suptitle('Separate neighbor geometry from biological change units\nIndependent A-state / B-change; all targets remain linear CPM per hpf',fontsize=13)
    fig.savefig(OUT/'08_log_geometry_linear_rates.png',dpi=180);fig.savefig(OUT/'08_log_geometry_linear_rates.pdf');plt.close(fig)


def spatial_map():
    d=load_sample();atlas=BASE/'outputs/ct_peak_atlas_v1'
    with h5py.File(atlas/'aligned_features.h5') as f:
        assert np.array_equal(f['peak_ids'].asstr()[:],d['peak_ids'])
        assert np.array_equal(f['source_indices'][:],d['source_indices'])
        assert np.array_equal(f['train_mask'][:],d['train_mask'])
        genes=f['nearest_gene'].asstr()[:] if isinstance(f['nearest_gene'],h5py.Dataset) else np.repeat('',len(d['peak_ids']))
    coords=np.load(atlas/'coordinates/ct_log_2d.npz')['umap'];assert coords.shape==(len(d['peak_ids']),2)
    te=d['test'];xy=coords[te];records=[];scores=[];raw=[];bins=[]
    with h5py.File(OUT/'flow_vectors.h5') as f:
        agreement=f['cell_split_agreement/logstate_linear_rate']
        for t in range(5):
            interval=f'interval_{int(d["stages"][t])}_{int(d["stages"][t+1])}'
            ab=f['cell_split/logstate_linear_rate/Astate_Bchange/'+interval]
            ba=f['cell_split/logstate_linear_rate/Bstate_Achange/'+interval]
            c_ab=ab['prediction_cosine'][:];c_ba=ba['prediction_cosine'][:]
            pair=np.stack([c_ab,c_ba],axis=1)
            # Require both reversal evaluations to exist; do not silently use
            # only whichever half happened to have a nonzero prediction.
            score=np.mean(pair,axis=1)
            direct=agreement['observed_change_cosine'][:,t]
            bid=agreement[f'half_A_effect_bin_interval_{t}'][:]
            scores.append(score);raw.append(direct);bins.append(bid)
            for q,si in enumerate(te):
                records.append({'sample_index':int(si),'source_index':int(d['source_indices'][si]),'peak_id':d['peak_ids'][si],
                                'nearest_gene':genes[si],'start_hpf':d['stages'][t],'end_hpf':d['stages'][t+1],
                                'ct_log_umap1':float(xy[q,0]),'ct_log_umap2':float(xy[q,1]),
                                'prediction_change_cosine_Astate_Bchange':float(c_ab[q]),'prediction_change_cosine_Bstate_Achange':float(c_ba[q]),
                                'mean_prediction_change_cosine':float(score[q]),'direct_change_cosine_A_B':float(direct[q]),
                                'half_A_effect_bin':int(bid[q]),'top5_percent_training_A_threshold':bool(bid[q]==3),
                                'supported_ct_count_from_full_counts':int(d['support_count'][si,t])})
    table=pd.DataFrame(records);table.to_csv(OUT/'ct_atlas_spatial_agreement.csv.gz',index=False)
    fig,axs=plt.subplots(3,5,figsize=(18,11),constrained_layout=True)
    for t in range(5):
        for row in range(3):
            ax=axs[row,t];ax.scatter(coords[:,0],coords[:,1],s=.35,color='#d8d8d8',alpha=.45,rasterized=True,linewidths=0)
            valid=np.isfinite(scores[t] if row<2 else raw[t])
            if row==1:valid&=bins[t]==3
            value=scores[t] if row<2 else raw[t]
            im=ax.scatter(xy[valid,0],xy[valid,1],c=value[valid],cmap='PiYG',vmin=-1,vmax=1,s=3 if row!=1 else 12,alpha=.85,linewidths=0,rasterized=True)
            ax.set_title(f'{int(d["stages"][t])}→{int(d["stages"][t+1])} hpf | n={valid.sum():,}',fontsize=10);ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect('equal','box')
            for spine in ax.spines.values():spine.set_visible(False)
        axs[0,0].set_ylabel('Prediction–change agreement\nAll held-out peaks',fontsize=11)
        axs[1,0].set_ylabel('Prediction–change agreement\nTop 5% half-A magnitude',fontsize=11)
        axs[2,0].set_ylabel('Direct A/B change agreement\nAll held-out peaks',fontsize=11)
    cb=fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.65,pad=.01);cb.set_label('Directional agreement cosine; diagnostic, not calibrated confidence')
    fig.suptitle('Where do independent changes agree on the new celltype-based peak UMAP?\nFixed whole-course log-CPM atlas; interval prediction uses 21 CTs, log-state neighbors and linear CPM/hpf rates',fontsize=14)
    fig.supxlabel('Grey background: 25,056 peaks. Colored points: 5,000 held-out peaks; top bins contain 238–259. Colors show agreement, not opening/closing.\nPrediction–change agreement averages both cell-half reversals; no global stream or root direction is imposed. Large absolute changes are enriched for abundant/broad peaks.',fontsize=10)
    fig.savefig(OUT/'09_ct_atlas_independent_agreement.png',dpi=180);fig.savefig(OUT/'09_ct_atlas_independent_agreement.pdf');plt.close(fig)
    summary=table.groupby(['start_hpf','end_hpf','top5_percent_training_A_threshold']).agg(n_peaks=('peak_id','size'),mean_prediction_agreement=('mean_prediction_change_cosine','mean'),mean_direct_change_agreement=('direct_change_cosine_A_B','mean')).reset_index()
    summary.to_csv(OUT/'ct_atlas_spatial_agreement_summary.csv',index=False)
    jsonwrite(OUT/'spatial_map_validation.json',{'peak_ids_exact_match':True,'source_indices_exact_match':True,'train_mask_exact_match':True,
                                             'atlas_geometry':'Fixed peak×190 celltype-time pseudobulk log1p(CPM) atlas; all stages in each peak profile',
                                             'field_geometry':'Independent current-stage21-CT log1p(CPM) state; target21-CT linear CPM/hpf change',
                                             'top5_percent':'Thresholds fixed by training-half-A rate norms; held-out bins assigned using halfA; no B outcomes in selection',
                                             'agreement_interpretation':'Per-peak noisy diagnostic, not calibrated confidence or proof of biological replication',
                                             'plotted_points':5000,'background_points':25056})


def finalize_artifacts():
    import scipy, sklearn, platform
    df=pd.read_csv(OUT/'metrics.csv')
    main=df[(df.dataset=='cell_split')&(df.stratum=='all')&(df.target=='Astate_Bchange')&(df.method=='knn_30')]
    view=main[main['transform'].isin(['linear','logstate_linear_rate'])][['transform','start_hpf','end_hpf','n_peaks','improvement_vs_interval_mean','mean_cosine']].copy()
    view['improvement_vs_interval_mean_percent']=100*view.pop('improvement_vs_interval_mean')
    view.to_csv(OUT/'primary_independent_cell_results.csv',index=False)
    q=df[(df.dataset=='cell_split')&(df.target=='Astate_Bchange')&(df.method.isin(['knn_30','ridge']))&(df['transform'].isin(['linear','logstate_linear_rate']))]
    q.to_csv(OUT/'independent_cell_support_strata.csv',index=False)
    report=OUT/'report.html';html=report.read_text()
    summary='<div class="note"><h2>What this analysis supports</h2><p>Much of the attractive same-measurement directional field can be recreated by constant-accessibility count noise. Independent cell halves preserve a smaller predictive signal in linear CPM rates, concentrated in a minority of large changes. Log1p(CPM) neighbor geometry retains that signal when the target stays in CPM/hpf. A universal root-to-branch stream is not established.</p><p>The top approximately5% of half-A change magnitudes account for59–66% of half-B squared change magnitude, but these peaks are also highly accessible: median11–12 CPM versus0.44 in the remaining peaks. Roughly44–50% are prior broad-reference peaks. This is evidence for reproducible large changes, not evidence by itself for emerging celltype-specific programs.</p><p>The stronger planted control is recoverable at the actual depths, whereas moderate signals are harder. Weak observed coherence therefore does not prove the absence of biological dynamics.</p></div>'
    html=html.replace('<h2>Shared-flow test</h2>',summary+'<h2>Shared-flow test</h2>')
    extra='<h2>Independent-cell primary results</h2>'+view.to_html(index=False,float_format=lambda x:f'{x:.3f}')
    extra+='<p>Each row evaluates5,000 held-out peaks with fixed k=30. Rates remain21-dimensional and interval-specific. The hybrid condition uses log1p(CPM) for neighbor selection and linear CPM/hpf for target changes.</p>'
    for title,name,caption in [
        ('Concentration in large changes','06_independent_effect_size_bins.png','Bin thresholds use training-half-A rate magnitude; held-out assignments use halfA only, and outcomes use halfB. Selection does not use the B outcome. These magnitudes still contain count noise.'),
        ('Stronger known-dynamics power sensitivity','07_stronger_planted_power_control.png','After weak recovery of the moderate planted control, the original nonnegative multipliers were cubed, giving maximum opening5.832× and minimum closing0.064×. Predictor settings, affected celltypes, modes and depths were unchanged. This is an additional power sensitivity, not tuning to improve real-data outcomes.'),
        ('Log geometry with linear rates','08_log_geometry_linear_rates.png','The same independent cell halves and matched half-depth Poisson nulls separate neighbor geometry from the units of accessibility change.'),
        ('Location on the new fixed peak UMAP','09_ct_atlas_independent_agreement.png','This fixed atlas contains one point per peak and uses the full measured celltype-time profile. The prediction test instead uses the complete21-celltype state at each interval start. Colors show noisy directional agreement, not calibrated confidence or opening/closing. Large-change peaks concentrate in an abundant, broad-accessibility region. The top-change bins are selected from half A, whereas these spatial colors average A→B and B→A prediction–change agreement. The reverse A outcome therefore participated in bin selection, so the colored top-bin maps are descriptive diagnostics. The separate half-B effect-bin evaluation and its 59–66% magnitude-share statistic use outcomes independent of bin selection.')]:
        if (OUT/name).exists():extra+=f'<h2>{title}</h2><img src="{name}"><p>{caption}</p>'
    extra+='<h2>Additional limits and numerical validation</h2><p>The support-stratum labels come from the original pooled-count trajectory fit, so those strata are descriptive; the unconditional independent-cell analysis is primary. Randomly held-out peaks may share genomic neighborhoods with training peaks. Neither peak resampling nor random cell halves establishes biological replication; stage and batch remain largely confounded. CPM captures relative accessibility, and subtype-composition changes can contribute to a pseudobulk trajectory.</p><p>Independent numerical checks reconstructed65 interval/model groups from source arrays, checked train/test separation, exact neighbor distances, all k15/30/60 means and prediction metrics. The maximum metric difference was4.3×10⁻9. The identical original trajectory fitter reproduced real fitted states within6.4×10⁻6 CPM before being applied to the constant-latent control.</p><h2>Files and reproducibility</h2><ul><li>metrics.csv: all methods, intervals, controls and support strata.</li><li>flow_vectors.h5: held-out target/prediction vectors, neighbors, alignment IDs and cross-half agreement.</li><li>ct_atlas_spatial_agreement.csv.gz: per-peak spatial diagnostic values.</li><li>independent_change_effect_bins.csv: effect-bin outcomes; effect_bin_definitions.csv: thresholds.</li><li>analysis_manifest.json and control-specific JSON files: fixed settings, validation and limitations.</li><li>ct_flow_noise.py: implementation. Run the base analysis, --refit-null, --only-split --split-h5 cell_split_input.h5, --strong-planted, --hybrid, --effect-bins, --spatial-map, --plots-only, and --finalize in that order to regenerate this package.</li></ul>'
    html=html.replace('</body>',extra+'</body>')
    for before,after in [('approximately5%','approximately 5%'),('for59–66%','for 59–66%'),('median11–12','median 11–12'),('versus0.44','versus 0.44'),('Roughly44–50%','Roughly 44–50%'),('evaluates5,000','evaluates 5,000'),('remain21-dimensional','remain 21-dimensional'),('complete21-celltype','complete 21-celltype'),('opening5.832','opening 5.832'),('closing0.064','closing 0.064'),('reconstructed65','reconstructed 65'),('was4.3','was 4.3'),('within6.4','within 6.4'),('halfA','half A'),('halfB','half B')]:html=html.replace(before,after)
    report.write_text(html)
    with h5py.File(OUT/'flow_vectors.h5','a') as f:
        for dataset in ['real','cell_split']:
            for trans in f[dataset]:
                for target in f[dataset][trans]:
                    for interval in f[dataset][trans][target]:
                        g=f[dataset][trans][target][interval]
                        g.attrs['target_units']='log1p(CPM) per hpf' if trans=='log1p' else 'CPM per hpf'
                        g.attrs['neighbor_geometry']='Euclidean CPM' if trans=='linear' else 'Euclidean log1p(CPM)'
                        g.attrs['interpretation']='Held-out cross-peak prediction; agreement is diagnostic, not calibrated confidence.'
    jsonwrite(OUT/'software_versions.json',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'h5py':h5py.__version__,'matplotlib':matplotlib.__version__})
    jsonwrite(OUT/'completion_manifest.json',{'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'figures':[x.name for x in sorted(OUT.glob('[0-9][0-9]_*.png'))],
                                           'primary_rates':'LinearCPM/hpf, with linear-state and log1p-state geometries reported separately',
                                           'primary_evaluation':'Unconditional independent cell-half cross-prediction;20,000 training /5,000 held-out whole peaks',
                                           'source_inputs_unchanged_by_this_script':True,'cicero_smoothing_included':False})

def run(split_path=None,only_split=False):
    OUT.mkdir(parents=True,exist_ok=True);d=load_sample()
    rows=[] if not only_split else pd.read_csv(OUT/'metrics.csv').query("dataset not in ['cell_split', 'cell_split_null']").to_dict('records')
    manifest={'seed':SEED,'train_peaks':len(d['train']),'held_out_peaks':len(d['test']),'celltypes':d['celltypes'].tolist(),
              'stages':d['stages'].tolist(),'primary_k':30,'sensitivity_k':[15,60],'ridge_alpha':100.0,
              'null_replicates':3,'bootstrap_replicates':200,'bootstrap_scope':'Sampling sensitivity of held-out peak identities, not biological replicate confidence',
              'null_model':'Independent Poisson counts at actual CT-stage total count exposure; constant per-CT latent CPM estimated by exposure-weighted across-stage average',
              'target_scope':'Observed and retrospective fitted targets analyzed separately; no held-out target used for fitting predictions',
              'source_sample':str(SAMPLE),'source_coverage':str(SOURCE),'cell_split_file':str(split_path) if split_path else None}
    stable,planted,mode=controls(d);validate(d,stable,planted,manifest)
    jsonwrite(OUT/'analysis_manifest.json',manifest)
    if not only_split:
        with h5py.File(OUT/'flow_vectors.h5','w') as vectors:
            vectors.attrs['axis_order']='held-out peak, celltype; interval encoded in group name'
            vectors.attrs['interpretation']='Cross-peak held-out prediction. Same-draw measurement coupling and retrospective fitted targets require controls.'
            for k,a in [('sample_indices',d['test']),('source_indices',d['source_indices'][d['test']]),('stages',d['stages']),('celltype_indices',d['celltype_indices']),('support_count',d['support_count'][d['test']]),('vector_snr',d['snr'][d['test']])]:write_array(vectors,k,a)
            vectors.create_dataset('peak_ids',data=d['peak_ids'][d['test']],dtype=h5py.string_dtype())
            vectors.create_dataset('celltypes',data=d['celltypes'],dtype=h5py.string_dtype())
            evaluate_dataset(d,'real',0,d['observed_cpm'],{'observed':d['observed_cpm'],'fitted':d['fitted_cpm']},rows,vectors)
            vectors.flush()
        pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
        for rep in range(3):
            a=generate_count_draws(stable,d['coverage'],SEED+1000+2*rep)
            b=generate_count_draws(stable,d['coverage'],SEED+1001+2*rep)
            evaluate_dataset(d,'constant_null',rep,a,{'observed':a,'independent':b},rows,truth_cpm=stable)
            pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
        a=generate_count_draws(planted,d['coverage'],SEED+2000);b=generate_count_draws(planted,d['coverage'],SEED+2001)
        evaluate_dataset(d,'planted',0,a,{'observed':a,'independent':b,'truth':planted},rows,truth_cpm=planted)
        pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
        np.savez_compressed(OUT/'synthetic_truth.npz',constant_cpm=stable.astype('f4'),planted_cpm=planted.astype('f4'),mode=mode,coverage=d['coverage'],sample_indices=np.arange(len(stable)))
    if split_path:
        with h5py.File(split_path) as f:
            ids=f['peak_ids'].asstr()[:];assert np.array_equal(ids,d['peak_ids'])
            assert np.array_equal(f['stages'][:],d['stages'])
            cts=f['celltypes'].asstr()[:];ix=np.array([np.where(cts==c)[0][0] for c in d['celltypes']])
            a=f['observed_cpm_a'][:][:,:,ix];b=f['observed_cpm_b'][:][:,:,ix]
            assert np.isfinite(a).all() and np.isfinite(b).all()
            ca=f['coverage_a'][:];cb=f['coverage_b'][:]
            if ca.shape==(len(cts),6):ca=ca.T;cb=cb.T
            assert ca.shape==(6,len(cts)) and cb.shape==ca.shape
            ca=ca[:,ix];cb=cb[:,ix]
            assert np.isfinite(ca).all() and np.min(ca)>0 and np.min(cb)>0
        with h5py.File(OUT/'flow_vectors.h5','a') as vectors:
            evaluate_dataset(d,'cell_split',0,a,{'Astate_Bchange':b,'Astate_Achange':a},rows,vectors,geometry_suffix='independent_cell_split_A_state')
            evaluate_dataset(d,'cell_split',1,b,{'Bstate_Achange':a},rows,vectors,geometry_suffix='independent_cell_split_B_state')
        pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
        split_base=np.sum(a*ca[None,:,:]+b*cb[None,:,:],axis=1)/np.sum(ca+cb,axis=0)[None,:]
        split_stable=np.repeat(split_base[:,None,:],6,axis=1)
        for rep in range(3):
            na=generate_count_draws(split_stable,ca,SEED+3000+2*rep)
            nb=generate_count_draws(split_stable,cb,SEED+3001+2*rep)
            evaluate_dataset(d,'cell_split_null',rep,na,{'Astate_Bchange':nb,'Astate_Achange':na},rows,geometry_suffix='Poisson_state_A_actual_half_coverage')
            pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    plots_and_report()
    split_plots()
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split-h5',type=Path);p.add_argument('--only-split',action='store_true');p.add_argument('--plots-only',action='store_true');p.add_argument('--refit-null',action='store_true');p.add_argument('--strong-planted',action='store_true');p.add_argument('--effect-bins',action='store_true');p.add_argument('--hybrid',action='store_true');p.add_argument('--spatial-map',action='store_true');p.add_argument('--finalize',action='store_true');a=p.parse_args()
    if a.plots_only:
        plots_and_report();split_plots();refit_plot();strong_plot();hybrid_plot()
    elif a.refit_null:refit_null()
    elif a.strong_planted:strong_planted()
    elif a.effect_bins:effect_bins()
    elif a.hybrid:hybrid()
    elif a.spatial_map:spatial_map()
    elif a.finalize:finalize_artifacts()
    else:run(a.split_h5,a.only_split)
