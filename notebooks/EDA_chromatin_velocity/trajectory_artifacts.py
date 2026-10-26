"""Export the fixed-celltype AnnData and additional descriptive diagnostics."""
from pathlib import Path
import argparse,json,sys
import h5py,numpy as np,pandas as pd,anndata as ad
from scipy import sparse
HERE=Path(__file__).resolve().parent
from real_trajectory_velocity import OUT,DATA,PARAMS,PROJ,MODE_LABELS,log
from real_specialization_benchmark import read_graph,write_json,cosine


def export_anndata(out):
    with h5py.File(out/'trajectory_fits.h5','r') as f,h5py.File(DATA/'peaks_by_pb_annotated_master.h5ad','r') as atlas:
        common=f['common_celltypes'][:].astype(bool);ct=f['celltypes'].asstr()[:][common];times=f['stages'][:].astype(int)
        obs=pd.read_csv(out/'peak_metadata.csv.gz',index_col=0,keep_default_na=False)
        if not np.array_equal(obs.index.astype(str),f['peak_ids'].asstr()[:]):raise ValueError('Metadata peak order mismatch')
        obs['broad_reference']=f['broad_reference'][:].astype(bool)
        a=ad.AnnData(X=f['fitted_cpm'][:,:,0][:,common],obs=obs,var=pd.DataFrame(index=pd.Index(ct,name='celltype')))
        a.obsm['X_umap']=f['umap2d'][:];a.obsm['X_umap_2D']=f['umap2d'][:];a.obsm['X_umap_3D']=f['umap3d'][:]
        a.obsp['connectivities']=read_graph(atlas['obsp/connectivities'])
        a.obsm['trajectory_modes']=f['modes'][:][:,common]
        a.obsm['trajectory_lack_of_fit']=f['lack_of_fit'][:][:,common]
        for t,time in enumerate(times):
            a.layers[f'state_{time}hpf']=f['fitted_cpm'][:,:,t][:,common]
            a.layers[f'observed_{time}hpf']=f['observed_cpm'][:,:,t][:,common]
        for t,(left,right) in enumerate(zip(times[:-1],times[1:])):
            key=f'{left}_{right}hpf';g=f[f'projection/{t}']
            a.layers[f'velocity_{key}']=f['rates'][:,:,t][:,common]
            a.layers[f'conditional_se_{key}']=f['rate_se'][:,:,t][:,common]
            a.layers[f'change_supported_{key}']=f['rate_supported'][:,:,t][:,common]
            for d in ['2d','3d']:
                a.obsm[f'velocity_umap_{d}_{key}']=g[f'display_vectors/umap{d}'][:]
                a.obs[f'projected_{d}_{key}']=g[f'projection_valid/umap{d}'][:]
            for name in ['net_rate','velocity_rms','source_feature_count','source_snr','changing','active']:
                a.obs[f'{name}_{key}']=g[name][:]
            pg=g['probabilities'];a.obsp[f'transition_{key}']=sparse.csr_matrix((pg['data'][:],pg['indices'][:],pg['indptr'][:]),shape=tuple(pg.attrs['shape']))
        a.layers['velocity']=a.layers['velocity_10_12hpf'].copy()
        a.obsm['velocity_umap']=a.obsm['velocity_umap_2d_10_12hpf'].copy()
        a.uns['neighbors']={'connectivities_key':'connectivities','params':{'method':'published_peak_atlas_reused'}}
        a.uns['trajectory_velocity']={
            'reference_interval':'10-12 hpf: X, velocity layer and velocity_umap refer to this interval',
            'stages_hpf':times,'parameters_json':json.dumps(PARAMS),'projection_parameters_json':json.dumps(PROJ),
            'mode_labels_json':json.dumps(MODE_LABELS),'full_32_celltype_artifact':str((out/'trajectory_fits.h5').resolve()),
            'normalization':'linear CPM; unobserved/insufficient-time states are NaN; count/change/fit-quality gates are separate',
            'units':'state CPM; velocity CPM/hpf; embedding vectors dimensionless directional averages',
            'projection':'uncentered cosine on current-stage fitted celltype profiles and original atlas edges; no root, tau, entropy, or co-accessibility orientation',
            'uncertainty':'conditional Poisson count-noise approximation; not biological replicate confidence',
            'interpretation':'fixed genomic peaks; retrospective profile alignment, not peak movement, ancestry or causal enhancer interaction',
            'readme':'Full temporal fits and mode labels use all eligible times. Source SNR/count/fit quality and graph support gate arrows; lack of arrow does not establish stability. This custom projection preserves uniform opening across celltypes; recomputing with standard centered scVelo changes the estimand.'}
        log('Writing 21-celltype interval AnnData')
        a.write_h5ad(out/'trajectory_velocity.h5ad',compression='gzip',compression_opts=1)
    with h5py.File(out/'trajectory_velocity.h5ad','r') as check:
        assert len(check['obs/_index'])==640830
        assert len(check['var/celltype'])==21
        assert np.array_equal(check['obsm/X_umap'][:],a.obsm['X_umap'])
    log('AnnData export saved and reopened')


def diagnostics(out):
    rows=[];withingroup=[]
    with h5py.File(out/'trajectory_fits.h5','r') as f,h5py.File(DATA/'peaks_by_pb_annotated_master.h5ad','r') as atlas:
        broad=f['broad_reference'][:].astype(bool);graph=read_graph(atlas['obsp/connectivities']);degree=np.asarray(graph.sum(axis=1)).ravel()
        base_to_broad=np.divide(np.asarray(graph@broad.astype(float)).ravel(),degree,out=np.zeros(len(broad)),where=degree>0)
        times=f['stages'][:];common=f['common_celltypes'][:].astype(bool)
        for t in range(5):
            g=f[f'projection/{t}'];pv=g['projection_valid/umap2d'][:].astype(bool);changing=g['changing'][:].astype(bool)
            v=f['rates'][:,:,t][:,common];valid=np.isfinite(v)&(f['lack_of_fit'][:][:,common]<=PARAMS['lack_of_fit_threshold'])
            positive=(valid&(v>PARAMS['rate_floor'])).any(axis=1);negative=(valid&(v < -PARAMS['rate_floor'])).any(axis=1)
            mean=g['net_rate'][:];rms=g['velocity_rms'][:]
            baseline=g['baseline_vectors/umap2d'][:];baseline_norm=np.linalg.norm(baseline,axis=1)
            alignment=cosine(g['vectors/umap2d'][:],baseline);test=pv&(baseline_norm>.05)&np.isfinite(alignment)
            pg=g['probabilities'];prob=sparse.csr_matrix((pg['data'][:],pg['indices'][:],pg['indptr'][:]),shape=tuple(pg.attrs['shape']))
            to_broad=np.asarray(prob@broad.astype(float)).ravel()
            row=dict(interval=f'{times[t]:g}-{times[t+1]:g}',peaks=len(pv),supported_changing=int(changing.sum()),displayed_2d=int(pv.sum()),displayed_positive_net=int((pv&(mean>0)).sum()),displayed_negative_net=int((pv&(mean<0)).sum()),displayed_fitted_mixed_celltype_signs=int((pv&positive&negative).sum()),median_displayed_rms=float(np.median(rms[pv])),median_displayed_cts=float(np.median(g['source_feature_count'][:][pv])),geometry_comparable=int(test.sum()),median_baseline_direction_cosine=float(np.median(alignment[test])),fraction_baseline_direction_cosine_above_08=float(np.mean(alignment[test]>.8)))
            rows.append(row)
            for label,mask in [('broad_reference',broad),('other_peaks',~broad)]:
                selected=mask&pv
                withingroup.append(dict(interval=row['interval'],source_group=label,n_peaks=int(mask.sum()),displayed=int(selected.sum()),positive_net=int((selected&(mean>0)).sum()),negative_net=int((selected&(mean<0)).sum()),mean_probability_to_broad=float(np.mean(to_broad[selected])) if selected.any() else None,mean_unfiltered_graph_probability_to_broad=float(np.mean(base_to_broad[selected])) if selected.any() else None,median_net_rate=float(np.median(mean[selected])) if selected.any() else None))
        pd.DataFrame(rows).to_csv(out/'field_diagnostics.csv',index=False)
        pd.DataFrame(withingroup).to_csv(out/'broad_reference_connections.csv',index=False)
    log('Geometry, sign and broad-reference diagnostics saved')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--stage',choices=['export','diagnostics','all'],default='all');a=p.parse_args()
    if a.stage in ['export','all']:export_anndata(a.output_dir)
    if a.stage in ['diagnostics','all']:diagnostics(a.output_dir)
if __name__=='__main__':main()
