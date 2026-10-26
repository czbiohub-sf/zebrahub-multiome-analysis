"""Compare final methods on identical saved coordinates and validate exported arrows."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from specialization_benchmark import HERE, quiver_panel, savefig, to_json
from specialization_flow import aggregate_specificity, structural_flow, bidirectional_rate_guidance
from temporal_velocity import STAGES_HPF


def inspect(a):
    pars=a.uns['specialization_parameters']['arrows']
    target=aggregate_specificity(a.obsm['truth_specificity'],STAGES_HPF,weighting=pars['weighting'])['aggregate']
    embeddings={f'umap{d}d':a.obsm[f'X_umap{d}d'] for d in [2,3]}
    graph=a.obsp['specialization_state_graph']
    ref=structural_flow(graph,target,embeddings,edge_tolerance=pars['edge_tolerance'],gradient_power=pars['gradient_power'])
    refdown=structural_flow(graph,-target,embeddings,edge_tolerance=pars['edge_tolerance'],gradient_power=pars['gradient_power'])
    truth_rate=a.obsm['truth_specificity_rate']
    uniform=a.obs.true_program.to_numpy()=='uniform_opening'
    core=a.obs.true_program.to_numpy()=='constitutive_core'
    report=[]
    p=a.obsp['specialization_directed_graph'].tocoo()
    for dim in [2,3]:
        key=f'umap{dim}d';mask=a.obs[f'{key}_structural_display'].to_numpy()
        prediction=a.obsm[f'specialization_{key}'];rv=ref['vectors'][key]
        pn=np.linalg.norm(prediction,axis=1);rn=np.linalg.norm(rv,axis=1)
        valid=mask&(pn>0)&(rn>0)
        cos=np.sum(prediction*rv,axis=1)/np.maximum(pn*rn,1e-12)
        edges=mask[p.row];delta=target[p.col]-target[p.row]
        forward=np.sum(p.data[edges&(delta>1e-8)])/max(np.sum(p.data[edges]),1e-12)
        temporal=a.obsm[f'specialization_guided_{key}'].reshape(len(a),5,dim)
        references=np.stack([bidirectional_rate_guidance(ref['vectors'][key],refdown['vectors'][key],truth_rate[:,j],valid=np.isfinite(truth_rate[:,j])) for j in range(5)],axis=1)
        tn=np.linalg.norm(temporal,axis=2);tr=np.linalg.norm(references,axis=2)
        ok=(tn>0)&(tr>0)&(np.abs(truth_rate)>=5e-4)
        tc=np.sum(temporal*references,axis=2)/np.maximum(tn*tr,1e-12)
        report.append(dict(dimension=dim,displayed_source_score_forward=float(forward),
            median_structural_direction_cosine=float(np.median(cos[valid])),
            fraction_structural_cosine_positive=float((cos[valid]>0).mean()),
            n_structure_reference_supported=int(valid.sum()),
            uniform_opening_structural_fraction=float(mask[uniform].mean()),
            core_guided_fraction=float((tn[core]>0).mean()),
            uniform_opening_guided_fraction=float((tn[uniform]>0).mean()),
            median_guided_direction_cosine=float(np.median(tc[ok])),
            fraction_guided_cosine_positive=float((tc[ok]>0).mean()),
            n_guided_reference_supported=int(ok.sum())))
    return report


def validate_payload(a,path):
    data=json.loads(path.read_text());n=len(a)
    assert len(data['peaks'])==n
    assert [p['peak'] for p in data['peaks']]==list(a.obs_names)
    for dim in [2,3]:
        np.testing.assert_allclose(data[f'umap{dim}d'],a.obsm[f'X_umap{dim}d'],atol=1e-6)
        for field in data['fields']:
            v=np.asarray(field[f'vectors{dim}d']);mask=np.asarray(field.get(f'arrow_mask{dim}d',field['arrow_mask']),dtype=bool)
            if field['id']=='structural':raw=a.obsm[f'specialization_umap{dim}d']
            else:raw=a.obsm[f'specialization_guided_umap{dim}d'].reshape(n,5,dim)
            norm=np.linalg.norm(raw,axis=-1)
            scale=np.linalg.norm(np.ptp(a.obsm[f'X_umap{dim}d'],axis=0))*.018
            expected=np.divide(raw,norm[...,None],out=np.zeros_like(raw),where=norm[...,None]>0)*scale
            np.testing.assert_allclose(v,expected,atol=1e-6)
            assert np.isfinite(v[mask]).all()
    # Distinguish absent/low-signal values from genuine measured zeros.
    weak=a.obs.true_program.to_numpy()=='low_signal'
    assert all(all(v is None for v in data['specificity'][i]) for i in np.flatnonzero(weak))
    return {'ordered_peak_ids':True,'coordinates_match':True,'display_vectors_match':True,'weak_signal_masked':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=HERE/'outputs'/'specialization_flow_v3')
    output=parser.parse_args().output_dir
    datasets={m:ad.read_h5ad(output/m/'specialization_flow.h5ad') for m in ['entropy','tau']}
    reports={m:json.loads((output/m/'run_summary.json').read_text()) for m in datasets}
    for dim in [2,3]:np.testing.assert_array_equal(datasets['entropy'].obsm[f'X_umap{dim}d'],datasets['tau'].obsm[f'X_umap{dim}d'])
    assert all(r['held_out_seed']==45 for r in reports.values())
    fig,axes=plt.subplots(1,2,figsize=(14,6.5),layout='constrained')
    for ax,(name,a) in zip(axes,datasets.items()):
        mask=a.obs.umap2d_structural_display.to_numpy()
        title='Entropy: specificity averaged over 10–24 hpf' if name=='entropy' else 'Tau: restriction at 24 hpf'
        quiver_panel(ax,a,a.obsm['specialization_umap2d'],mask,'family',title,max_arrows=150)
    fig.suptitle('Same peaks and fixed UMAP; different definitions of specificity\nGold: always-accessible reference · Arrow length normalized for display')
    savefig(fig,output,'08_entropy_vs_tau_arrows_2d')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for name,color in [('entropy','#155eaa'),('tau','#c04a32')]:
        d=pd.read_csv(output/name/'edge_uncertainty_sweep.csv').groupby('edge_confidence_z').mean(numeric_only=True)
        axes[0].plot(d.index,d.weighted_true_specificity_forward,'o-',color=color,label=name)
        axes[1].plot(d.index,d.uniform_opening_structural_fraction,'o-',color=color,label=name)
    axes[0].set(xlabel='Neighbor-comparison noise margin',ylabel='Weighted edges toward true score increase',title='Accuracy on calibration seeds')
    axes[1].set(xlabel='Neighbor-comparison noise margin',ylabel='Uniform-opening controls with a structural arrow',title='Suppressing false structure on flat controls')
    for ax in axes:ax.legend();ax.set_ylim(-.02,1.02)
    savefig(fig,output,'09_edge_uncertainty_calibration')
    rows=[];validation={}
    for name,a in datasets.items():
        rows.extend([dict(method=name,**r) for r in inspect(a)])
        validation[name]=validate_payload(a,output/name/'explorer_data.json')
    pd.DataFrame(rows).to_csv(output/'final_projection_metrics.csv',index=False)
    (output/'artifact_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    summary=dict(final_seed=45,calibration_seeds=[42,43],development_seed=44,coordinate_comparison='identical',
        methods=reports,projection_metrics=rows,artifact_validation=validation)
    (output/'final_summary.json').write_text(json.dumps(to_json(summary),indent=2)+'\n')
    print(json.dumps(to_json(summary),indent=2))


if __name__=='__main__':main()
