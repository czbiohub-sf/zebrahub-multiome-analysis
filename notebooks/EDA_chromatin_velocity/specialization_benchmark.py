"""Calibrate and evaluate specialization paths and time-guided arrows on fixed peak maps.

Synthetic truth is used only for calibration/evaluation, never field construction.
Seeds 42/43 calibrate settings, 44 is development inspection, and 45 is final evaluation.
A scalar specificity rate does not determine a unique vector field: the temporal
arrows explicitly use the structural gradient as a chosen display direction.
"""
import argparse
import json
import sys
from pathlib import Path
import warnings

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'scripts'/'chrom_velo'))
from specialization_synthetic import make_specialization_dataset
from specialization_flow import (temporal_specificity, specificity_by_stage,
    aggregate_specificity, build_state_graph, structural_flow, rate_guided_vectors, bidirectional_rate_guidance)
from temporal_velocity import STAGES_HPF, temporal_rates
from specialization_viewer import export_viewer

PALETTE=['#d6a329','#3b82f6','#14b8a6','#ef7b45','#9860ce']
TIMES=STAGES_HPF
METHOD="entropy"
WEIGHTING="duration"


def cube(x):
    return np.asarray(x,dtype=float).reshape(-1,20,6)


def replicates(a):
    keys=sorted([k for k in a.layers if k.startswith('sampling_rep')],key=lambda k:int(k.split('rep')[-1]))
    return np.stack([cube(a.layers[k]) for k in keys])


def estimate(a, pseudocount=0., shrinkage=0., bias_correct=False, signal_floor=.02, method=None, matched_loo=False):
    reps=replicates(a);r=len(reps)
    opts=dict(pseudocount=pseudocount,shrinkage=shrinkage,signal_floor=signal_floor,method=method or METHOD)
    base=temporal_specificity(reps.mean(0),TIMES,**opts)
    loo=[temporal_specificity((reps.sum(0)-reps[i])/(r-1),TIMES,**opts) for i in range(r)]
    scores=np.stack([d['score'] for d in loo]);rates=np.stack([d['interval_rate'] for d in loo])
    score=base['score'].copy();rate=base['interval_rate'].copy()
    # Delete-one jackknife corrects nonlinear entropy bias. Negative corrected
    # specificity is clipped to its theoretical bound; intervals are recomputed.
    if bias_correct:
        score=np.clip(r*score-(r-1)*np.mean(scores,axis=0),0.,1.)
        first=np.stack([d['paired_start'] for d in loo])
        last=np.stack([d['paired_end'] for d in loo])
        paired_start=np.clip(r*base['paired_start']-(r-1)*first.mean(0),0.,1.)
        paired_end=np.clip(r*base['paired_end']-(r-1)*last.mean(0),0.,1.)
        rate=(paired_end-paired_start)/np.diff(TIMES)
    score_se=np.sqrt((r-1)/r*np.sum((scores-scores.mean(0))**2,axis=0))
    rate_se=np.sqrt((r-1)/r*np.sum((rates-rates.mean(0))**2,axis=0))
    valid=base['valid'] & np.all(np.stack([d['valid'] for d in loo]),axis=0)
    interval_valid=base['interval_valid'] & np.all(np.stack([d['interval_valid'] for d in loo]),axis=0)
    stability_scores=scores
    if bias_correct and matched_loo:
        stability_scores=[]
        for i in range(r):
            reduced=np.delete(reps,i,axis=0);n=len(reduced)
            inner=[specificity_by_stage((reduced.sum(0)-reduced[j])/(n-1),**opts)['score'] for j in range(n)]
            stability_scores.append(np.clip(n*scores[i]-(n-1)*np.mean(inner,axis=0),0.,1.))
        stability_scores=np.asarray(stability_scores)
    return dict(score=score,rate=rate,score_se=score_se,rate_se=rate_se,
                valid=valid,interval_valid=interval_valid,loo_scores=stability_scores)


def truth(a):
    t=temporal_specificity(cube(a.layers['truth_state']),TIMES,signal_floor=.02,method=METHOD)
    return dict(score=t['score'],rate=t['interval_rate'],valid=t['valid'],interval_valid=t['interval_valid'])


def macro_error(estimate_values,truth_values,valid,programs):
    errors=[]
    for p in np.unique(programs):
        mask=valid & (programs[:,None]==p)
        if mask.any(): errors.append(np.sqrt(np.mean((estimate_values[mask]-truth_values[mask])**2)))
    return float(np.mean(errors)) if errors else float('nan')


def entropy_sweep(datasets,output):
    rows=[];cache={}
    for seed,a in datasets.items():
        t=truth(a);programs=a.obs.true_program.to_numpy()
        for pseudo in [0.,.001,.01]:
            for shrink in [0.,.05]:
                for correction in [False,True]:
                    opts=dict(pseudocount=pseudo,shrinkage=shrink,bias_correct=correction)
                    e=estimate(a,**opts);key=(seed,pseudo,shrink,correction);cache[key]=e
                    rows.append(dict(seed=seed,**opts,
                        specificity_macro_rmse=macro_error(e['score'],t['score'],e['valid']&t['valid'],programs),
                        rate_macro_rmse=macro_error(e['rate'],t['rate'],e['interval_valid']&t['interval_valid'],programs)))
    df=pd.DataFrame(rows);df.to_csv(output/'entropy_sweep.csv',index=False)
    # Selection sees only calibration/development seeds. Equal program weighting
    # prevents the large hierarchy from overwhelming the null controls.
    selected=df.groupby(['pseudocount','shrinkage','bias_correct']).rate_macro_rmse.mean().idxmin()
    opts=dict(pseudocount=float(selected[0]),shrinkage=float(selected[1]),bias_correct=bool(selected[2]),method=METHOD)
    return opts,df,{s:cache[(s,*selected)] for s in datasets}


def rate_diagnostics(a,e,t,z=2.57,floor=1e-4):
    valid=e['interval_valid']&t['interval_valid']
    active=valid & (np.abs(t['rate'])>=5e-4)
    null=valid & (np.abs(t['rate'])<1e-10)
    reliable=valid & (np.abs(e['rate'])>np.maximum(z*e['rate_se'],floor))
    selected=active&reliable
    class_fp=[]
    for program in a.obs.true_program.unique():
        mask=null & (a.obs.true_program.to_numpy()[:,None]==program)
        if mask.any():class_fp.append(float(reliable[mask].mean()))
    return dict(worst_null_program_false_positive=max(class_fp,default=0.),
        null_false_positive=float(reliable[null].mean()) if null.any() else 0.,
        active_recall=float(reliable[active].mean()) if active.any() else 0.,
        selected_sign_accuracy=float((np.sign(e['rate'][selected])==np.sign(t['rate'][selected])).mean()) if selected.any() else 0.,
        n_null=int(null.sum()),n_active=int(active.sum()),n_selected=int(selected.sum())),reliable


def uncertainty_sweep(datasets,estimates,output):
    rows=[]
    for z in [2.57,3.,4.]:
        for floor in [0.,1e-4,3e-4,1e-3]:
            for seed,a in datasets.items():
                m,_=rate_diagnostics(a,estimates[seed],truth(a),z,floor)
                rows.append(dict(seed=seed,z=z,rate_floor=floor,**m))
    df=pd.DataFrame(rows);df.to_csv(output/'uncertainty_sweep.csv',index=False)
    summary=df.groupby(['z','rate_floor']).agg(null_false_positive=('worst_null_program_false_positive','max'),
        active_recall=('active_recall','mean'),selected_sign_accuracy=('selected_sign_accuracy','min')).reset_index()
    good=summary[(summary.null_false_positive<=.05)&(summary.selected_sign_accuracy>=.95)]
    if len(good): best=good.sort_values(['active_recall','null_false_positive'],ascending=[False,True]).iloc[0]
    else: best=summary.sort_values(['null_false_positive','selected_sign_accuracy'],ascending=[True,False]).iloc[0]
    return dict(z=float(best.z),rate_floor=float(best.rate_floor)),df


def prepare_pca(a):
    a.obsm['X_pca']=PCA(n_components=30,svd_solver='arpack',random_state=42).fit_transform(a.X)


def aggregate(e,weighting):
    return aggregate_specificity(e['score'],TIMES,valid=e['valid'],weighting=weighting)['aggregate']


def aggregate_se(e,weighting):
    values=np.stack([aggregate_specificity(s,TIMES,valid=e['valid'],weighting=weighting)['aggregate'] for s in e['loo_scores']])
    r=len(values)
    return np.sqrt((r-1)/r*np.sum((values-values.mean(0))**2,axis=0))


def edge_metrics(a,field,true_score):
    p=field['probabilities'].tocoo();n=len(a)
    h=a.obs.is_hierarchy.to_numpy();latent=a.obs.true_specialization.to_numpy(float)
    dyn=h & (a.obs.true_program.to_numpy()!='constitutive_core')
    keep=h[p.row]&h[p.col]&np.isfinite(latent[p.row])&np.isfinite(latent[p.col])
    delta=latent[p.col]-latent[p.row]
    weights=p.data[keep];d=delta[keep]
    truth_delta=true_score[p.col]-true_score[p.row]
    finite=np.isfinite(truth_delta)
    return dict(structural_coverage=float((field['support_count'][dyn]>0).mean()),
        weighted_latent_forward=float(np.sum(weights[d>1e-8])/max(np.sum(weights),1e-12)),
        weighted_true_specificity_forward=float(np.sum(p.data[finite&(truth_delta>1e-8)])/max(np.sum(p.data[finite]),1e-12)),
        median_support=float(np.median(field['support_count'][dyn])),n_edges=int(p.nnz))


def arrow_sweep(datasets,estimates,output):
    rows=[]
    for seed,a in datasets.items():
        prepare_pca(a);t=truth(a)
        for k in [15,30,60]:
            g=build_state_graph(a.obsm['X_pca'],n_neighbors=k)
            for weighting in ['equal','duration','late','terminal']:
                score=aggregate(estimates[seed],weighting);true_score=aggregate(t,weighting)
                for tolerance in [0.,1e-4,5e-4,1e-3]:
                    for power in [.5,1.,2.]:
                        f=structural_flow(g,score,{},edge_tolerance=tolerance,gradient_power=power)
                        rows.append(dict(seed=seed,n_neighbors=k,weighting=weighting,
                            edge_tolerance=tolerance,gradient_power=power,**edge_metrics(a,f,true_score)))
    df=pd.DataFrame(rows);df.to_csv(output/'arrow_sweep.csv',index=False)
    # Weighting is an explicit biological question: duration summarizes the
    # whole trajectory; terminal asks about late-program restriction. Compare
    # alternatives without pretending these are estimates of the same quantity.
    group=['n_neighbors','edge_tolerance','gradient_power']
    m=df[df.weighting==WEIGHTING].groupby(group).mean(numeric_only=True).reset_index()
    # Penalize missing branches, not just wrong orientation on a few easy edges.
    m['selection_score']=m.weighted_true_specificity_forward-.2*(1-m.structural_coverage)
    best=m.sort_values(['selection_score','n_neighbors'],ascending=[False,True]).iloc[0]
    return dict(n_neighbors=int(best.n_neighbors),edge_tolerance=float(best.edge_tolerance),
                gradient_power=float(best.gradient_power),weighting=WEIGHTING),df



def edge_margin_sweep(datasets,estimates,params,output):
    rows=[]
    for seed,a in datasets.items():
        score=aggregate(estimates[seed],params['weighting'])
        se=aggregate_se(estimates[seed],params['weighting'])
        target=aggregate(truth(a),params['weighting'])
        g=build_state_graph(a.obsm['X_pca'],n_neighbors=params['n_neighbors'])
        for z in [0.,.5,1.,1.5,2.]:
            f=structural_flow(g,score,{},edge_tolerance=params['edge_tolerance'],
                gradient_power=params['gradient_power'],score_uncertainty=se,edge_confidence_z=z)
            metrics=edge_metrics(a,f,target)
            flat=a.obs.true_program.to_numpy()=='uniform_opening'
            metrics['uniform_opening_structural_fraction']=float((f['support_count'][flat]>0).mean())
            rows.append(dict(seed=seed,edge_confidence_z=z,**metrics))
    table=pd.DataFrame(rows);table.to_csv(output/'edge_uncertainty_sweep.csv',index=False)
    m=table.groupby('edge_confidence_z').mean(numeric_only=True)
    objective=m.weighted_true_specificity_forward-.2*(1-m.structural_coverage)-.1*m.uniform_opening_structural_fraction
    return float(objective.idxmax()),table


def embed_fixed(a):
    if 'X_pca' not in a.obsm: prepare_pca(a)
    # One embedding for every parameter comparison and every stage.
    sc.pp.neighbors(a,n_neighbors=30,use_rep='X_pca',random_state=42)
    for dim in [3,2]:
        key=f'X_pca_init{dim}d';a.obsm[key]=a.obsm['X_pca'][:,:dim].copy()
        sc.tl.umap(a,n_components=dim,min_dist=.4,random_state=42,
                   init_pos=key,gamma=.5,maxiter=1000)
        a.obsm[f'X_umap{dim}d']=a.obsm['X_umap'].copy()


def choose_arrows(coords,vectors,mask,max_arrows=180):
    idx=np.flatnonzero(mask & np.isfinite(vectors).all(1) & (np.linalg.norm(vectors,axis=1)>1e-12))
    if len(idx)<=max_arrows:return idx
    # Farthest-point sampling changes display density only, not the field.
    selected=[int(idx[np.argmax(np.linalg.norm(vectors[idx],axis=1))])]
    distance=np.full(len(idx),np.inf)
    for _ in range(max_arrows-1):
        distance=np.minimum(distance,np.sum((coords[idx]-coords[selected[-1]])**2,axis=1))
        j=int(np.argmax(distance))
        if distance[j]<1e-14:break
        selected.append(int(idx[j]))
    return np.asarray(selected)


def quiver_panel(ax,a,vectors,mask,color,title,dim=2,rate=False,max_arrows=150,color_limit=None):
    xy=a.obsm[f'X_umap{dim}d'];norm=np.linalg.norm(vectors,axis=1)
    if isinstance(color,str) and color=='family':
        c=np.asarray([PALETTE[0] if p=='constitutive_core' else PALETTE[int(f)+1] if int(f)>=0 else '#9ca3af'
                      for p,f in zip(a.obs.true_program,a.obs.true_family)])
        ax.scatter(*xy.T,c=c,s=3 if dim==2 else 2,alpha=.65,rasterized=True)
    else:
        vals=np.asarray(color);good=np.isfinite(vals)
        vmax=(color_limit if color_limit is not None else max(np.nanquantile(abs(vals),.98),1e-4)) if rate else 1.
        ax.scatter(*xy[~good].T,c='#b8bec7',s=3,alpha=.5,rasterized=True)
        artist=ax.scatter(*xy[good].T,c=vals[good],s=4,alpha=.8,cmap='RdBu_r' if rate else 'viridis',
            norm=TwoSlopeNorm(0,-vmax,vmax) if rate else Normalize(0,max(np.nanquantile(vals,.99),.02)),rasterized=True)
        plt.colorbar(artist,ax=ax,shrink=.65,label='Specificity / hpf' if rate else 'Specificity')
    ids=choose_arrows(xy,vectors,mask,max_arrows)
    direction=np.divide(vectors,norm[:,None],out=np.zeros_like(vectors),where=norm[:,None]>0)
    length=np.linalg.norm(np.ptp(xy,axis=0))*.018
    if dim==2:
        ax.quiver(*xy[ids].T,*(direction[ids]*length).T,color='#172b43',angles='xy',scale_units='xy',scale=1,
            width=.0024,headwidth=3.4,headlength=4.6,pivot='tail',zorder=4)
        ax.set_aspect('equal',adjustable='datalim')
        ax.set(xticks=[],yticks=[],xlabel='Peak UMAP 1',ylabel='Peak UMAP 2')
    else:
        ax.quiver(*xy[ids].T,*direction[ids].T,color='#172b43',length=length,normalize=False,
            arrow_length_ratio=.3,linewidth=.6)
        ax.set(xticks=[],yticks=[],zticks=[],xlabel='Peak UMAP 1',ylabel='Peak UMAP 2',zlabel='Peak UMAP 3')
        ax.view_init(elev=20,azim=35)
    ax.set_title(title+f'\n{len(ids)} displayed arrows',fontsize=10)


def savefig(fig,output,name):
    fig.savefig(output/f'{name}.png',dpi=160,bbox_inches='tight')
    fig.savefig(output/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def to_json(x):
    if isinstance(x,dict):return {k:to_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [to_json(v) for v in x]
    if isinstance(x,np.ndarray):return to_json(x.tolist())
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.bool_):return bool(x)
    return x


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=HERE/'outputs'/'specialization_flow_v3'/'entropy')
    parser.add_argument('--test-seed',type=int,default=45)
    parser.add_argument('--method',choices=['entropy','tau'],default='entropy')
    parser.add_argument('--weighting',choices=['equal','duration','late','terminal'],default='duration')
    args=parser.parse_args()
    global METHOD,WEIGHTING
    METHOD=args.method;WEIGHTING=args.weighting
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':9})
    datasets={}
    for seed in [42,43]:
        print(f'Generating calibration seed {seed}',flush=True)
        datasets[seed]=make_specialization_dataset(seed=seed)
    print('Calibrating entropy estimation',flush=True)
    entropy_params,entropy_table,estimates=entropy_sweep(datasets,output)
    print(f'Entropy parameters: {entropy_params}',flush=True)
    estimates={seed:estimate(a,**entropy_params,matched_loo=True) for seed,a in datasets.items()}
    uncertainty_params,uncertainty_table=uncertainty_sweep(datasets,estimates,output)
    print(f'Uncertainty parameters: {uncertainty_params}',flush=True)
    print('Comparing structural arrows, graph neighborhoods, and temporal weighting',flush=True)
    arrow_params,arrow_table=arrow_sweep(datasets,estimates,output)
    edge_z,_=edge_margin_sweep(datasets,estimates,arrow_params,output)
    arrow_params['edge_confidence_z']=edge_z
    print(f'Arrow parameters: {arrow_params}',flush=True)
    print(f'Generating held-out seed {args.test_seed}; parameters are now locked',flush=True)
    a=make_specialization_dataset(seed=args.test_seed);e=estimate(a,**entropy_params,matched_loo=True);t=truth(a)
    atlas_path=output.parent/f'fixed_atlas_seed{args.test_seed}.h5ad'
    if atlas_path.exists():
        old=ad.read_h5ad(atlas_path)
        assert old.obs_names.equals(a.obs_names) and np.array_equal(old.X,a.X)
        for key in old.obsm:a.obsm[key]=old.obsm[key].copy()
        for key in old.obsp:a.obsp[key]=old.obsp[key].copy()
        a.uns['neighbors']=old.uns['neighbors']
        del old
    else:
        prepare_pca(a);embed_fixed(a)
        a.write_h5ad(atlas_path,compression='gzip')
    score=aggregate(e,arrow_params['weighting']);true_score=aggregate(t,arrow_params['weighting'])
    g=build_state_graph(a.obsm['X_pca'],n_neighbors=arrow_params['n_neighbors'])
    embeddings={'umap2d':a.obsm['X_umap2d'],'umap3d':a.obsm['X_umap3d']}
    score_se=aggregate_se(e,arrow_params['weighting'])
    f=structural_flow(g,score,embeddings,edge_tolerance=arrow_params['edge_tolerance'],gradient_power=arrow_params['gradient_power'],
        score_uncertainty=score_se,edge_confidence_z=arrow_params['edge_confidence_z'])
    # A separate downhill field retains reversals at tips, where an uphill
    # field has no direction. Each uses actual neighboring peaks, not a negative
    # arrow extrapolated into a region without supporting neighbors.
    down=structural_flow(g,-score,embeddings,edge_tolerance=arrow_params['edge_tolerance'],gradient_power=arrow_params['gradient_power'],
        score_uncertainty=score_se,edge_confidence_z=arrow_params['edge_confidence_z'])
    # Conditional replicate agreement: embedding/neighbor graph held fixed.
    agreement={};coherence={};structural_masks={};down_masks={}
    for sign,field,masks in [(1,f,structural_masks),(-1,down,down_masks)]:
        for name,v in field['vectors'].items():
            norm=np.linalg.norm(v,axis=1);ref=np.divide(v,norm[:,None],out=np.zeros_like(v),where=norm[:,None]>0)
            cosines=[]
            for loo in e['loo_scores']:
                ss=aggregate_specificity(loo,TIMES,valid=e['valid'],weighting=arrow_params['weighting'])['aggregate']
                ff=structural_flow(g,sign*ss,{name:embeddings[name]},edge_tolerance=arrow_params['edge_tolerance'],gradient_power=arrow_params['gradient_power'],
                    score_uncertainty=score_se,edge_confidence_z=arrow_params['edge_confidence_z'])
                vv=ff['vectors'][name];nn=np.linalg.norm(vv,axis=1)
                cosines.append(np.sum(vv*ref,axis=1)/np.maximum(nn,1e-12))
            agree=np.mean(cosines,axis=0)
            masks[name]=(field['support_count']>=2)&(field['coherence'][name]>=.25)&(agree>=.8)
            if sign==1:
                agreement[name]=agree
                coherence[name]=field['coherence'][name]
    rate_metrics,reliable=rate_diagnostics(a,e,t,uncertainty_params['z'],uncertainty_params['rate_floor'])
    guided={};guided_masks={}
    for name in embeddings:
        guided[name]=np.stack([bidirectional_rate_guidance(f['vectors'][name],down['vectors'][name],e['rate'][:,j],
            valid=reliable[:,j],upward_valid=structural_masks[name],downward_valid=down_masks[name]) for j in range(5)],axis=1)
        guided_masks[name]=reliable&np.where(e['rate']>0,structural_masks[name][:,None],down_masks[name][:,None])
    # Storage separates signed rates, structural orientation, and display gates.
    a.obsp['specialization_state_graph']=g
    a.obsp['specialization_directed_graph']=f['probabilities']
    a.obsp['specialization_downhill_graph']=down['probabilities']
    for name in embeddings:
        a.obsm[f'specialization_{name}']=f['vectors'][name]
        a.obsm[f'specialization_downhill_{name}']=down['vectors'][name]
        a.obs[f'{name}_downhill_display']=down_masks[name]
        a.obsm[f'specialization_guided_{name}']=guided[name].reshape(len(a),-1)
        a.obs[f'{name}_coherence']=coherence[name]
        a.obs[f'{name}_leave_one_out_direction_stability']=agreement[name]
        a.obs[f'{name}_structural_display']=structural_masks[name]
    a.obsm['specificity']=e['score'];a.obsm['truth_specificity']=t['score']
    a.obsm['specificity_rate']=e['rate'];a.obsm['truth_specificity_rate']=t['rate']
    a.obsm['specificity_rate_se']=e['rate_se'];a.obsm['specificity_rate_reliable']=reliable
    a.obs['aggregate_specificity']=score
    a.obs['aggregate_specificity_se']=score_se
    a.obs['specialization_graph_support']=f['support_count']
    a.uns['specialization_parameters']=dict(entropy=entropy_params,arrows=arrow_params,uncertainty=uncertainty_params,
        graph_fixed_for_uncertainty=True,held_out_seed=args.test_seed,seed_roles=f'42 and 43 calibration; 44 development inspection; {args.test_seed} final evaluation',
        temporal_arrow_meaning='Signed specificity rate along an imposed structural specificity direction; not inferred transport')
    # Never use true hierarchy labels or true signs to filter displayed arrows.
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    quiver_panel(axes[0],a,f['vectors']['umap2d'],structural_masks['umap2d'],'family','Structural paths toward increasing specificity')
    quiver_panel(axes[1],a,f['vectors']['umap2d'],structural_masks['umap2d'],score,f'Same paths colored by {WEIGHTING} {METHOD} specificity')
    fig.suptitle(f'{METHOD.title()} / {WEIGHTING} weighting · Every dot is a peak · Arrows describe a chosen specificity orientation\nFixed UMAP; arrow lengths normalized for display')
    savefig(fig,output,'01_structural_arrows_2d')
    fig=plt.figure(figsize=(10,8),layout='constrained');ax=fig.add_subplot(111,projection='3d')
    quiver_panel(ax,a,f['vectors']['umap3d'],structural_masks['umap3d'],'family','Structural paths toward increasing specificity',dim=3)
    savefig(fig,output,'02_structural_arrows_3d')
    plot_temporal_arrows(a,e,t,guided,guided_masks,output)
    plot_diagnostics(a,e,t,entropy_table,arrow_table,output)
    plot_controls(a,e,t,output)
    plot_arrow_comparison(a,e,g,f,structural_masks,arrow_params,output)
    temporal_nulls(a,e,t,entropy_params,output)
    rowlist=[]
    for program in a.obs.true_program.unique():
        idx=a.obs.true_program.to_numpy()==program
        em={k:v[idx] for k,v in e.items() if k!='loo_scores'}
        tm={k:v[idx] for k,v in t.items()}
        metrics,_=rate_diagnostics(a[idx],em,tm,uncertainty_params['z'],uncertainty_params['rate_floor'])
        rowlist.append(dict(program=program,n_peaks=int(idx.sum()),**metrics))
    pd.DataFrame(rowlist).to_csv(output/'held_out_program_metrics.csv',index=False)
    report=dict(shape=list(a.shape),specificity_definition=METHOD,stage_weighting=WEIGHTING,entropy_parameters=entropy_params,arrow_parameters=arrow_params,
        uncertainty_parameters=uncertainty_params,held_out_seed=args.test_seed,
        held_out_rate_metrics=rate_metrics,held_out_edge_metrics=edge_metrics(a,f,true_score),
        structural_arrows_eligible_2d=int(structural_masks['umap2d'].sum()),
        structural_arrows_eligible_3d=int(structural_masks['umap3d'].sum()),
        guided_arrows_eligible_by_interval_2d=guided_masks['umap2d'].sum(0).tolist(),
        held_out_specificity_macro_rmse=macro_error(e['score'],t['score'],e['valid']&t['valid'],a.obs.true_program.to_numpy()),
        held_out_rate_macro_rmse=macro_error(e['rate'],t['rate'],e['interval_valid']&t['interval_valid'],a.obs.true_program.to_numpy()))
    (output/'run_summary.json').write_text(json.dumps(to_json(report),indent=2)+'\n')
    a.write_h5ad(output/'specialization_flow.h5ad',compression='gzip')
    a.obs.to_csv(output/'peak_metadata.csv');a.var.to_csv(output/'conditions.csv')
    export_payload(a,e,t,f,guided,structural_masks,guided_masks,coherence,output)
    print(json.dumps(to_json(report),indent=2),flush=True)
    print(f'Outputs: {output.resolve()}',flush=True)


def shared_rate_limit(*arrays):
    values=np.concatenate([np.asarray(x).ravel() for x in arrays])
    return max(float(np.quantile(np.abs(values[np.isfinite(values)]),.97)),1e-5)


def plot_temporal_arrows(a,e,t,guided,guided_masks,output):
    rate_limit=shared_rate_limit(e['rate'],t['rate'])
    fig,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained')
    for j,ax in enumerate(axes.flat[:5]):
        quiver_panel(ax,a,guided['umap2d'][:,j],guided_masks['umap2d'][:,j],e['rate'][:,j],f'{TIMES[j]:g} → {TIMES[j+1]:g} hpf',rate=True,max_arrows=110,color_limit=rate_limit)
    axes.flat[-1].axis('off');axes.flat[-1].text(.05,.9,'Temporal specificity changes\n\nOutward: increasing specificity\nInward: decreasing specificity\n\nOnly supported, reliable rates shown.\nDirection follows structural paths.\nShared color scale across intervals.\nLength is normalized; not physical speed.\n\nA stationary reference need not emit\ntemporal arrows.',va='top',transform=axes.flat[-1].transAxes)
    fig.suptitle('Specificity change along atlas paths · held-out synthetic peaks')
    savefig(fig,output,'03_temporal_guided_arrows_2d')


def plot_diagnostics(a,e,t,entropy_table,arrow_table,output):
    fig,axes=plt.subplots(1,3,figsize=(17,4.8),layout='constrained')
    group=entropy_table.groupby(['pseudocount','shrinkage','bias_correct']).rate_macro_rmse.mean().sort_values()
    axes[0].barh(np.arange(len(group)),group.values,color='#4575a5')
    axes[0].set_yticks(np.arange(len(group)),[f'pc={p:g}, shrink={s:g}, JK={int(j)}' for p,s,j in group.index],fontsize=7)
    axes[0].invert_yaxis();axes[0].set(xlabel='Mean program RMSE (specificity / hpf)',title='Specificity estimation: calibration seeds 42, 43')
    for k,col in [(15,'#3b82f6'),(30,'#14b8a6'),(60,'#9860ce')]:
        d=arrow_table[(arrow_table.n_neighbors==k)&(arrow_table.weighting==WEIGHTING)]
        axes[1].scatter(d.structural_coverage,d.weighted_true_specificity_forward,s=15,c=col,alpha=.55,label=f'k={k}')
    axes[1].set(xlabel='Hierarchy peaks with an outgoing direction',ylabel='Weighted edges toward true score increase',title='Arrow tuning on the known hierarchy',xlim=(0,1.03),ylim=(0,1.03));axes[1].legend()
    table=pd.read_csv(output/'uncertainty_sweep.csv')
    axes[2].scatter(table.worst_null_program_false_positive,table.active_recall,c=table.z,cmap='viridis',s=25)
    axes[2].axvline(.05,color='#9c3c38',ls='--',lw=1)
    axes[2].set(xlabel='Worst stationary-program false-positive rate',ylabel='Recall for true |rate| ≥ 0.0005 / hpf',title='Uncertainty threshold trade-off',ylim=(0,1.03))
    savefig(fig,output,'04_parameter_calibration')


def plot_controls(a,e,t,output):
    programs=['constitutive_core','stable_celltype_specific','uniform_opening','lineage_opening','transient_lineage','low_signal']
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    rows=[]
    for p,ax in zip(programs,axes.flat):
        idx=np.flatnonzero(a.obs.true_program.to_numpy()==p)
        if not len(idx):continue
        # Representative by median true endpoint specificity; labels select an
        # explanatory example, not an estimator or arrow direction.
        i=int(idx[len(idx)//2]);valid=e['valid'][i]
        ax.plot(TIMES,t['score'][i],color='#155eaa',lw=2.5,label='True specificity')
        ax.errorbar(TIMES[valid],e['score'][i,valid],yerr=2*e['score_se'][i,valid],fmt='o-',color='#c04a32',lw=1.2,
            markersize=4,label='Estimate ± 2 jackknife noise estimates')
        ax.set(title=p.replace('_',' '),xlabel='Developmental time (hpf)',ylabel='Specificity',xticks=TIMES)
        finite_truth=t['score'][i][np.isfinite(t['score'][i])]
        ax.set_ylim(-.01,max(.04,float(finite_truth.max())*1.3 if len(finite_truth) else .04))
        if not valid.any():ax.text(.5,.5,'Below signal threshold\nNo specificity inference',ha='center',va='center',transform=ax.transAxes)
        rows.append(dict(peak=a.obs_names[i],program=p))
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Held-out controls: stable specificity, opening without specificity, and temporal reversal')
    savefig(fig,output,'05_ground_truth_controls')
    pd.DataFrame(rows).to_csv(output/'control_examples.csv',index=False)


def plot_arrow_comparison(a,e,g,selected,masks,params,output):
    naive=estimate(a,pseudocount=0.,shrinkage=0.,bias_correct=False)
    ns=aggregate(naive,'equal');ng=build_state_graph(a.obsm['X_pca'],n_neighbors=15)
    nf=structural_flow(ng,ns,{'umap2d':a.obsm['X_umap2d']},edge_tolerance=0.,gradient_power=1.)
    fig,axes=plt.subplots(1,3,figsize=(18,6),layout='constrained')
    quiver_panel(axes[0],a,nf['vectors']['umap2d'],nf['support_count']>0,'family','Initial: raw entropy, equal stages, k=15')
    quiver_panel(axes[1],a,selected['vectors']['umap2d'],selected['support_count']>0,'family','Selected numerical parameters; all supported arrows')
    quiver_panel(axes[2],a,selected['vectors']['umap2d'],masks['umap2d'],'family','Selected + coherence and leave-one-out stability')
    fig.suptitle('Iteration on one fixed peak UMAP · arrow positions and lengths use the same display rules')
    savefig(fig,output,'06_arrow_iteration_comparison')
    rows=[]
    for weighting in ['equal','duration','late','terminal']:
        f=structural_flow(g,aggregate(e,weighting),{'umap2d':a.obsm['X_umap2d']},
            edge_tolerance=params['edge_tolerance'],gradient_power=params['gradient_power'],score_uncertainty=aggregate_se(e,weighting),
            edge_confidence_z=params.get('edge_confidence_z',0.))
        rows.append(dict(weighting=weighting,**edge_metrics(a,f,aggregate(truth(a),weighting))))
    pd.DataFrame(rows).to_csv(output/'held_out_time_weight_sensitivity.csv',index=False)


def temporal_nulls(a,e,t,params,output):
    # The atlas and its chosen structural orientation are fixed in this test.
    # Permute only stage labels on the measured specificity trajectory.
    good=e['interval_valid']&t['interval_valid']&(np.abs(t['rate'])>=5e-4)
    target=np.where(good,t['rate'],0.)
    def alignment(r):
        r=np.where(good,r,0.)
        return float(np.sum(r*target)/max(np.linalg.norm(r)*np.linalg.norm(target),1e-12))
    observed=alignment(e['rate']);rng=np.random.default_rng(804)
    null=[]
    for _ in range(200):
        perm=rng.permutation(6)
        null.append(alignment(np.diff(e['score'][:,perm],axis=1)/np.diff(TIMES)))
    reversed_order=alignment(np.diff(e['score'][:,::-1],axis=1)/np.diff(TIMES))
    report=dict(observed_rate_alignment=observed,time_permuted_median_alignment=float(np.median(null)),
        time_permuted_95th_alignment=float(np.quantile(null,.95)),
        permutation_tail_fraction=float((1+np.sum(np.asarray(null)>=observed))/201),
        reversed_stage_order_alignment=reversed_order,
        caveat='Atlas/structural orientation fixed; permuting labels changes elapsed-time assignments. Full reversal is not a uniform negative of interval rates on irregular times.')
    (output/'time_order_controls.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    ax.hist(null,bins=25,color='#94a3b8',label='200 time-label permutations')
    ax.axvline(observed,color='#155eaa',lw=2,label='Correct stage order')
    ax.axvline(reversed_order,color='#c04a32',lw=2,label='Reversed stage order')
    ax.set(xlabel='Alignment with true interval specificity changes',ylabel='Permutations',title='Time ordering supports the scalar dynamics');ax.legend(fontsize=8)
    savefig(fig,output,'07_time_order_control')


def export_payload(a,e,t,f,guided,structural_masks,guided_masks,coherence,output):
    opening,valid,_=temporal_rates(a.X,a.var)
    def rounded(x):return np.round(np.asarray(x,dtype=float),7)
    def display(v,dim):
        length=np.linalg.norm(np.ptp(a.obsm[f'X_umap{dim}d'],axis=0))*.018
        norm=np.linalg.norm(v,axis=-1,keepdims=True)
        return rounded(np.divide(v,norm,out=np.zeros_like(v),where=norm>0)*length)
    fields=[dict(id='structural',label='Increasing specificity (reference orientation)',kind='structural',
        description=f'Supported neighbor paths toward greater {WEIGHTING} {METHOD} specificity. Direction is chosen by this score, not inferred developmental transport.',
        vectors2d=display(f['vectors']['umap2d'],2),vectors3d=display(f['vectors']['umap3d'],3),
        support=f['support_count'],coherence=np.minimum(coherence['umap2d'],coherence['umap3d']),
        arrow_mask=structural_masks['umap2d']&structural_masks['umap3d'],
        arrow_mask2d=structural_masks['umap2d'],arrow_mask3d=structural_masks['umap3d'],
        coherence2d=coherence['umap2d'],coherence3d=coherence['umap3d']),
        dict(id='temporal',label='Specificity change along atlas paths (guided)',kind='temporal',
        description='Positive specificity changes follow supported uphill neighbors; negative changes follow supported downhill neighbors. The scalar rate is measured, while the spatial direction is an explicit mapping assumption. Only reliable rates are displayed.',
        vectors2d=display(guided['umap2d'],2),vectors3d=display(guided['umap3d'],3),
        support=np.maximum(np.linalg.norm(guided['umap2d'],axis=2)>0,np.linalg.norm(guided['umap3d'],axis=2)>0).astype(int),
        arrow_mask=guided_masks['umap2d']&guided_masks['umap3d'],
        arrow_mask2d=guided_masks['umap2d'],arrow_mask3d=guided_masks['umap3d'])]
    payload=dict(title=f'Chromatin specialization: {METHOD} / {WEIGHTING}',default_dimension=3,
        notes=f'Using {METHOD} specificity and {WEIGHTING} structural weighting. Every dot is a fixed genomic peak. Structural arrows point toward greater specificity; temporal arrows map measured specificity changes onto supported paths. Arrow lengths are normalized for display, not physical speed; signed rates are shown separately. Gold identifies the constitutive reference. Simulation seed {int(a.uns["specialization_parameters"]["held_out_seed"])} was excluded from parameter selection.',
        celltypes=list(a.uns['synthetic']['celltypes']),stages=TIMES,
        peaks=[dict(peak=k,**row.to_dict()) for k,row in a.obs.iterrows()],
        umap2d=rounded(a.obsm['X_umap2d']),umap3d=rounded(a.obsm['X_umap3d']),
        observed=rounded(a.X),truth_state=rounded(a.layers['truth_state']),
        specificity=rounded(np.where(e['valid'],e['score'],np.nan)),
        truth_specificity=rounded(np.where(t['valid'],t['score'],np.nan)),
        specificity_rate=rounded(np.where(e['interval_valid'],e['rate'],np.nan)),
        truth_specificity_rate=rounded(np.where(t['interval_valid'],t['rate'],np.nan)),
        opening_rate=rounded(np.where(valid,opening,np.nan)),
        color_limits=dict(specificity_rate=shared_rate_limit(e['rate'],t['rate']),
            truth_specificity_rate=shared_rate_limit(e['rate'],t['rate']),opening_rate=shared_rate_limit(np.where(valid,opening,np.nan))),
        fields=fields,default_field='structural')
    payload=to_json(payload)
    (output/'explorer_data.json').write_text(json.dumps(payload,separators=(',',':'),allow_nan=False))
    export_viewer(payload,output/'specialization_arrow_explorer.html')


if __name__=='__main__':
    main()
