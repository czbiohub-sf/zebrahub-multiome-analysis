"""Fit six-stage accessibility and project root-free interval fields on the atlas.

Writes new artifacts only. The published peak coordinates/graph are reused.
Normalized linear CPM is the primary estimand; library coverage supplies fitting
precision weights and conditional count-noise diagnostics, not absolute units or
biological replicate confidence. No specificity score or root orients an edge.
"""
from pathlib import Path
import argparse,gc,json,sys,time,hashlib
import h5py,numpy as np,pandas as pd
from scipy import sparse
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'scripts/chrom_velo'))
from trajectory_modes import fit_trajectories,MODE_LABELS,MODEL_LABELS
from trajectory_classification import classify_supported_modes
from trajectory_projection import project_interval
from real_specialization import parse_condition_grid,to_condition_cube
from real_specialization_benchmark import read_frame,read_column,read_graph,write_json,cosine
DATA=HERE.parents[2]/'data/annotated_data/objects_v2'
OUT=HERE/'outputs/trajectory_velocity_v1'
PARAMS=dict(complexity_penalty=.2,rate_floor=.05,rate_z=3.5,lack_of_fit_threshold=4.,min_observations=4)
PROJ=dict(min_shared_features=5,min_shared_fraction=.7,min_cosine=.2,cosine_power=1.,min_neighbors=2,min_coherence=.25)


def log(s): print(time.strftime('%H:%M:%S'),s,flush=True)

def stat(p):
    s=p.stat();return dict(path=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def create(f,name,shape,dtype='f4'):
    chunks=(min(4096,shape[0]),)+tuple(1 if j==0 and len(shape)>2 else s for j,s in enumerate(shape[1:]))
    return f.create_dataset(name,shape=shape,dtype=dtype,chunks=chunks,compression='lzf')

def write_graph(group,g):
    group.attrs['shape']=g.shape
    for key,a in [('data',g.data.astype('f4')),('indices',g.indices),('indptr',g.indptr)]:group.create_dataset(key,data=a,compression='lzf')

def fit_real(out,block_size):
    state_path=DATA/'peaks_by_ct_tp_master.h5ad';atlas_path=DATA/'peaks_by_pb_annotated_master.h5ad'
    before={k:stat(v) for k,v in [('state',state_path),('atlas',atlas_path)]}
    with h5py.File(state_path,'r') as state,h5py.File(atlas_path,'r') as atlas,h5py.File(out/'trajectory_fits.h5','w') as f:
        var=read_frame(state['var']);grid=parse_condition_grid(var)
        obs=read_frame(atlas['obs'],['celltype','lineage','timepoint','leiden_coarse','nearest_gene','linked_gene','associated_gene','chrom','start','end','peak_type','peak_type_argelaguet','distance_to_tss'])
        ids=pd.Index(read_column(state['obs'],state['obs'].attrs['_index']).astype(str))
        if not ids.is_unique or not obs.index.is_unique or not ids.equals(obs.index):raise ValueError('Peak indices must match exactly')
        n=len(ids);C=len(grid['celltypes']);T=len(grid['stages']);valid=grid['present']&(grid['n_cells']>=20);common=valid.all(axis=1)
        f.create_dataset('peak_ids',data=np.asarray(ids,dtype=object),dtype=h5py.string_dtype())
        f.create_dataset('celltypes',data=np.asarray(grid['celltypes'],dtype=object),dtype=h5py.string_dtype())
        for key,a in [('stages',grid['stages']),('condition_valid',valid),('common_celltypes',common),('coverage',grid['total_coverage']),('column_index',grid['column_index'])]:f.create_dataset(key,data=a)
        for k,name in [('umap2d','X_umap_2D'),('umap3d','X_umap_3D')]:
            coords=atlas['obsm'][name][:]
            if not np.array_equal(coords,state['obsm'][name][:]):raise ValueError('Embedding mismatch')
            f.create_dataset(k,data=coords)
        old=HERE/'outputs/real_specialization_v1/real_specialization_flow.h5ad'
        with h5py.File(old,'r') as prior:
            if not np.array_equal(prior['obs/_index'].asstr()[:],np.asarray(ids)):raise ValueError('Reference peak order mismatch')
            broad=prior['obs/broad_reference'][:].astype(bool)
        f.create_dataset('broad_reference',data=broad)
        for name in ['observed_cpm','fitted_cpm']:create(f,name,(n,C,T))
        for name in ['rates','rate_se']:create(f,name,(n,C,T-1))
        create(f,'rate_supported',(n,C,T-1),'?')
        for name in ['modes','models','observed_count']:create(f,name,(n,C),'i1')
        create(f,'fit_valid',(n,C),'?')
        for name in ['cv_error','cv_improvement','lack_of_fit','raw_total']:create(f,name,(n,C))
        exposure=grid['total_coverage'];weights=np.where(valid,exposure,0.)
        scale=1e6/float(state['uns/common_scale_factor'][()]);max_error=0.
        for start in range(0,n,block_size):
            stop=min(n,start+block_size);sl=slice(start,stop)
            normalized=state['layers/normalized'][sl];raw=state['layers/sum'][sl]
            max_error=max(max_error,float(np.max(np.abs(raw*var['scale_factor'].to_numpy()[None,:]-normalized))))
            values=to_condition_cube(normalized*scale,grid);counts=to_condition_cube(raw,grid)
            variance=(counts+.5)*(1e6/exposure)**2
            result=fit_trajectories(values,grid['stages'],observed=valid,weights=weights,measurement_variance=variance,**PARAMS)
            total=np.nansum(np.where(valid[None,:,:],counts,np.nan),axis=-1)
            count_supported=total>=20
            modes=result['mode'].copy();modes[~count_supported]=0
            f['observed_cpm'][sl]=values;f['fitted_cpm'][sl]=result['fitted']
            f['rates'][sl]=result['rates'];f['rate_se'][sl]=result['rate_se']
            f['rate_supported'][sl]=result['rate_supported']&count_supported[:,:,None]
            f['modes'][sl]=modes;f['models'][sl]=result['model'];f['observed_count'][sl]=result['observed_count']
            f['fit_valid'][sl]=result['fit_valid']&count_supported;f['raw_total'][sl]=total
            for key in ['cv_error','cv_improvement','lack_of_fit']:f[key][sl]=result[key]
            if start==0 or stop==n or (start//block_size)%8==0:log(f'Fitted {stop:,}/{n:,} peaks × {C} celltypes')
        if max_error>1e-6:raise ValueError('Normalization identity changed')
        f.attrs['parameters_json']=json.dumps(PARAMS);f.attrs['mode_labels_json']=json.dumps(MODE_LABELS);f.attrs['model_labels_json']=json.dumps(MODEL_LABELS)
        f.attrs['normalization']='linear CPM; normalized layer times 1e6/common_scale_factor'
        f.attrs['uncertainty']='Conditional Poisson count-noise approximation; not biological replicate confidence; model-selection uncertainty not fully represented'
        f.attrs['primary_panel']='fixed celltypes with at least20 cells at all six stages'
        f.attrs['source_files_json']=json.dumps(before)
        f.attrs['normalization_max_abs_error']=max_error
        obs.to_csv(out/'peak_metadata.csv.gz',compression='gzip');var.to_csv(out/'input_conditions.csv')
        write_json(out/'input_provenance.json',dict(source_files=before,n_peaks=n,n_conditions=len(var),celltypes=grid['celltypes'],stages=grid['stages'],condition_valid=valid,common_celltypes=grid['celltypes'][common],missing_pairs=[(grid['celltypes'][c],grid['stages'][t]) for c,t in zip(*np.where(~grid['present']))],normalization_max_abs_error=max_error,peak_order_equal=True,umap_coordinates_equal=True))
    after={k:stat(v) for k,v in [('state',state_path),('atlas',atlas_path)]}
    if before!=after:raise ValueError('Source files changed')
    log('All real trajectories saved; source files unchanged')


def source_diagnostics(v,se,feature_valid,rate_support):
    finite=feature_valid&np.isfinite(v)&np.isfinite(se)
    vv=np.where(finite,v,0.);ee=np.where(finite,se,0.)
    magnitude=np.linalg.norm(vv,axis=1);noise=np.linalg.norm(ee,axis=1)
    snr=np.divide(magnitude,noise,out=np.zeros(len(v)),where=noise>0)
    active=(rate_support&finite).any(axis=1)&(snr>=2.)
    return active,snr


def refresh_modes(out):
    # Enables mode-only corrections without refitting or changing saved rates.
    with h5py.File(out/'trajectory_fits.h5','r+') as f:
        for start in range(0,len(f['peak_ids']),8192):
            sl=slice(start,start+8192)
            mode=classify_supported_modes(f['rates'][sl],f['rate_supported'][sl],
                quality_valid=f['lack_of_fit'][sl]<=PARAMS['lack_of_fit_threshold'],rate_floor=PARAMS['rate_floor'])
            mode[~f['fit_valid'][sl]]=0;f['modes'][sl]=mode
        f.attrs['mode_classification']='supported signs; incomplete support for a fitted reversal is uncertain; v2'


def project_real(out):
    refresh_modes(out)
    with h5py.File(DATA/'peaks_by_pb_annotated_master.h5ad','r') as atlas:graph=read_graph(atlas['obsp/connectivities'])
    with h5py.File(out/'trajectory_fits.h5','r+') as f:
        common=f['common_celltypes'][:].astype(bool);exposure=f['coverage'][:][common];T=f['stages'][:];n=len(f['peak_ids'])
        embeddings={k:f[k][:] for k in ['umap2d','umap3d']}
        rows=[];summaries=[]
        for key in ['projection','raw_projection']:
            if key in f:del f[key]
            f.create_group(key)
        for t in range(len(T)-1):
            x=f['fitted_cpm'][:,:,t][:,common];v=f['rates'][:,:,t][:,common];se=f['rate_se'][:,:,t][:,common]
            lof=f['lack_of_fit'][:][:,common];rate_support=f['rate_supported'][:,:,t][:,common]
            valid=np.isfinite(x)&np.isfinite(v)&np.isfinite(se)&(lof<=PARAMS['lack_of_fit_threshold'])
            active,snr=source_diagnostics(v,se,valid,rate_support)
            log(f'Project fitted {T[t]:g}→{T[t+1]:g}; {active.sum():,} supported-changing sources')
            result=project_interval(graph,x,v,feature_valid=valid,embeddings=embeddings,source_active=active,return_probabilities=True,**PROJ)
            group=f['projection'].create_group(str(t));group.create_dataset('source_snr',data=snr.astype('f4'),compression='lzf')
            store_projection(group,result)
            xraw=f['observed_cpm'][:,:,t][:,common];ynext=f['observed_cpm'][:,:,t+1][:,common]
            rawv=(ynext-xraw)/(T[t+1]-T[t])
            # Recover the count-noise variance from linear CPM and known exposure.
            rawvar0=(xraw*exposure[:,t]/1e6+.5)*(1e6/exposure[:,t])**2
            rawvar1=(ynext*exposure[:,t+1]/1e6+.5)*(1e6/exposure[:,t+1])**2
            rawse=np.sqrt(rawvar0+rawvar1)/(T[t+1]-T[t])
            rawvalid=np.isfinite(xraw)&np.isfinite(rawv)
            rawsupport=(np.abs(rawv)>np.maximum(PARAMS['rate_floor'],PARAMS['rate_z']*rawse))&(f['raw_total'][:][:,common]>=20)
            rawactive,rawsnr=source_diagnostics(rawv,rawse,rawvalid,rawsupport)
            log(f'Project raw {T[t]:g}→{T[t+1]:g}; {rawactive.sum():,} supported-changing sources')
            rawresult=project_interval(graph,xraw,rawv,feature_valid=rawvalid,embeddings=embeddings,source_active=rawactive,return_probabilities=False,**PROJ)
            rawgroup=f['raw_projection'].create_group(str(t));rawgroup.create_dataset('source_snr',data=rawsnr.astype('f4'),compression='lzf');store_projection(rawgroup,rawresult)
            for d in embeddings:
                comparable=result['projection_valid'][d]&rawresult['projection_valid'][d]
                cs=cosine(result['vectors'][d],rawresult['vectors'][d]);good=comparable&np.isfinite(cs)
                summaries.append(dict(interval=f'{T[t]:g}-{T[t+1]:g}',embedding=d,fitted_source_active=int(active.sum()),raw_source_active=int(rawactive.sum()),fitted_projected=int(result['projection_valid'][d].sum()),raw_projected=int(rawresult['projection_valid'][d].sum()),comparable=int(good.sum()),median_direction_cosine=float(np.median(cs[good])) if good.any() else None,fraction_direction_reversed=float(np.mean(cs[good]<0)) if good.any() else None))
            for kind,r in [('fitted',result),('raw',rawresult)]:
                row=dict(interval=f'{T[t]:g}-{T[t+1]:g}',kind=kind)
                for k in ['source_supported','stationary','changing','active','unprojectable']:row[k]=int(r[k].sum())
                rows.append(row)
            del result,rawresult;gc.collect()
        f.attrs['projection_parameters_json']=json.dumps(PROJ)
        f.attrs['source_active_rule']='At least one CT with supported change plus L2 fitted velocity / L2 conditional SE >=2; all fixed-panel CT dimensions retained where estimable and not lack-of-fit flagged'
    pd.DataFrame(rows).to_csv(out/'projection_coverage.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'raw_vs_fitted_projection.csv',index=False)
    log('Five fitted and raw interval fields saved')


def store_projection(g,r):
    for key,a in r.items():
        if key=='probabilities':
            if a is not None:write_graph(g.create_group(key),a)
        elif key=='metadata':g.attrs['metadata_json']=json.dumps(a)
        elif isinstance(a,dict):
            sub=g.create_group(key)
            for name,b in a.items():sub.create_dataset(name,data=np.asarray(b,dtype='?' if np.asarray(b).dtype==bool else 'f4'),compression='lzf')
        elif isinstance(a,np.ndarray):g.create_dataset(key,data=a,compression='lzf')
        else:g.attrs[key]=json.dumps(a)


def report_data(f,out):
    modes=f['modes'][:];obs=pd.read_csv(out/'peak_metadata.csv.gz',index_col=0,keep_default_na=False)
    data={k:f[k] for k in ['observed_cpm','fitted_cpm','fit_valid']}
    data.update(stages=f['stages'][:],celltypes=f['celltypes'].asstr()[:],peak_ids=f['peak_ids'].asstr()[:],umap2d=f['umap2d'][:],umap3d=f['umap3d'][:],condition_valid=f['condition_valid'][:],modes=modes,mode_labels=MODE_LABELS,obs=obs,broad_reference=f['broad_reference'][:],genomic_context=obs.peak_type_argelaguet.to_numpy(),fit_quality=f['lack_of_fit'])
    for name,prefix in [('fitted','projection'),('raw','raw_projection')]:
        for d in ['2d','3d']:
            data[f'{name}_vectors{d}']=np.stack([f[f'{prefix}/{t}/display_vectors/umap{d}'][:] for t in range(5)],axis=1)
            data[f'{name}_mask{d}']=np.stack([f[f'{prefix}/{t}/projection_valid/umap{d}'][:] for t in range(5)],axis=1)
    data['net_rate']=np.stack([f[f'projection/{t}/net_rate'][:] for t in range(5)],axis=1)
    data['rms_rate']=np.stack([f[f'projection/{t}/velocity_rms'][:] for t in range(5)],axis=1)
    common=f['common_celltypes'][:].astype(bool)
    data['no_supported_change']=np.stack([~f['rate_supported'][:,:,t][:,common].any(axis=1) for t in range(5)],axis=1)
    data['projection_support']=np.stack([f[f'projection/{t}/support_count'][:] for t in range(5)],axis=1)
    data['projection_coherence']=np.stack([f[f'projection/{t}/coherence/umap2d'][:] for t in range(5)],axis=1)
    return data


def write_summaries(out):
    with h5py.File(out/'trajectory_fits.h5','r') as f:
        modes=f['modes'][:];ct=f['celltypes'].asstr()[:];broad=f['broad_reference'][:].astype(bool)
        obs=pd.read_csv(out/'peak_metadata.csv.gz',index_col=0,keep_default_na=False)
        rows=[]
        for c,name in enumerate(ct):
            counts=np.bincount(modes[:,c],minlength=7)
            for code,count in enumerate(counts):rows.append(dict(celltype=name,mode=MODE_LABELS[code],peaks=int(count)))
        pd.DataFrame(rows).to_csv(out/'mode_counts_by_celltype.csv',index=False)
        rows=[]
        for t in range(5):
            g=f[f'projection/{t}'];valid=g['projection_valid/umap2d'][:];rate=g['net_rate'][:];rms=g['velocity_rms'][:];active=g['changing'][:]
            for field in ['leiden_coarse','peak_type_argelaguet']:
                for value,idx in obs.groupby(field).indices.items():
                    idx=np.asarray(idx);finite=np.isfinite(rate[idx]);
                    rows.append(dict(interval=t,group_type=field,group=value,peaks=len(idx),finite_rates=int(finite.sum()),source_active=int(active[idx].sum()),projected=int(valid[idx].sum()),mean_net_rate=float(np.nanmean(rate[idx])),mean_rms_rate=float(np.nanmean(rms[idx]))))
        pd.DataFrame(rows).to_csv(out/'cluster_and_genomic_summaries.csv',index=False)
        report=dict(n_peaks=len(modes),n_celltypes=len(ct),fixed_projection_celltypes=f['celltypes'].asstr()[:][f['common_celltypes'][:]].tolist(),mode_counts={MODE_LABELS[k]:int(np.sum(modes==k)) for k in MODE_LABELS},broad_reference_peaks=int(broad.sum()),parameters=PARAMS,projection_parameters=PROJ,
          interpretation='Retrospective accessibility-state alignment between fixed genomic peaks. No root/specificity orientation; no peak transport or biological replicate confidence.',broad_reference_intervals=[])
        for t in range(5):
            g=f[f'projection/{t}'];rms=g['velocity_rms'][:];pv=g['projection_valid/umap2d'][:];ac=g['changing'][:]
            report['broad_reference_intervals'].append(dict(interval=t,median_rms=float(np.nanmedian(rms[broad])),supported_changing=int(ac[broad].sum()),projected=int(pv[broad].sum())))
        write_json(out/'run_summary.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--block-size',type=int,default=8192);p.add_argument('--stage',choices=['fit','project','report','all'],default='all')
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if args.stage in ['fit','all']:fit_real(out,args.block_size)
    if args.stage in ['project','all']:project_real(out);write_summaries(out)
    if args.stage in ['report','all']:
        from trajectory_report import make_trajectory_report
        with h5py.File(out/'trajectory_fits.h5','r') as f:make_trajectory_report(report_data(f,out),out,make_html=True,make_streams=True)
    log('Requested stage completed')

if __name__=='__main__':main()
