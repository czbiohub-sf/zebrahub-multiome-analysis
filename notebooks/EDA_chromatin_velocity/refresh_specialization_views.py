"""Re-export presentation only from saved numerical results; never refit the atlas."""
import argparse
from pathlib import Path
import anndata as ad
import numpy as np
import specialization_benchmark as b


def refresh(folder):
    a=ad.read_h5ad(folder/'specialization_flow.h5ad')
    b.METHOD=a.uns['specialization_parameters']['entropy']['method']
    b.WEIGHTING=a.uns['specialization_parameters']['arrows']['weighting']
    e=dict(score=a.obsm['specificity'],rate=a.obsm['specificity_rate'],
        valid=np.isfinite(a.obsm['specificity']),interval_valid=np.isfinite(a.obsm['specificity_rate']))
    t=dict(score=a.obsm['truth_specificity'],rate=a.obsm['truth_specificity_rate'],
        valid=np.isfinite(a.obsm['truth_specificity']),interval_valid=np.isfinite(a.obsm['truth_specificity_rate']))
    f=dict(vectors={},support_count=a.obs.specialization_graph_support.to_numpy())
    guided={};masks={};guided_masks={};coherence={}
    for dim in [2,3]:
        key=f'umap{dim}d'
        f['vectors'][key]=a.obsm[f'specialization_{key}']
        guided[key]=a.obsm[f'specialization_guided_{key}'].reshape(len(a),5,dim)
        masks[key]=a.obs[f'{key}_structural_display'].to_numpy()
        guided_masks[key]=np.linalg.norm(guided[key],axis=2)>0
        coherence[key]=a.obs[f'{key}_coherence'].to_numpy()
    b.plot_temporal_arrows(a,e,t,guided,guided_masks,folder)
    b.export_payload(a,e,t,f,guided,masks,guided_masks,coherence,folder)
    print(f'Refreshed {folder}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=b.HERE/'outputs'/'specialization_flow_v3')
    root=p.parse_args().output_dir
    for name in ['entropy','tau']:refresh(root/name)
