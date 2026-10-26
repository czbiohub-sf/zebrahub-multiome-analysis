#!/usr/bin/env python3
"""Copy the existing biological fits and replace only CT-atlas projection fields.

All source files are read-only. A temporary copy of the previous AnnData is
updated through anndata.io.write_elem, audited, then renamed to the new output.
The 21-celltype linear-CPM layers remain bitwise identical to the source AnnData.
The whole-course atlas uses 190 CT-stage log1p(CPM) features and frozen 50-PC
geometry; candidate neighbors are rebuilt among all peaks in that geometry.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import shutil
import time
import anndata as ad
import h5py
import numpy as np
from scipy import sparse

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/ct_velocity_reprojection_v1'
PRIOR=HERE/'outputs/trajectory_velocity_v1/trajectory_velocity.h5ad'
ATLAS=HERE/'outputs/ct_peak_atlas_v1/full_ct_peak_atlas.h5'
UPDATED=OUT/'updated_projection.h5'
DESTINATION=OUT/'chromatin_velocity_ct_umap.h5ad'
COMPRESSION=dict(compression='gzip',compression_opts=1)


def log(value):print(time.strftime('%H:%M:%S'),value,flush=True)

def stat(path):
    path=Path(path);s=path.stat()
    return dict(path=str(path.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def read_graph(group):
    graph=sparse.csr_matrix((group['data'][:],group['indices'][:],group['indptr'][:]),shape=tuple(group.attrs['shape']))
    return graph


def replace_elem(handle,path,value):
    parent,key=path.rsplit('/',1) if '/' in path else ('',path)
    group=handle[parent] if parent else handle
    if key in group:del group[key]
    ad.io.write_elem(group,key,value,dataset_kwargs=COMPRESSION if not isinstance(value,dict) else {})


def dataset_digest(dataset,block_size=16384):
    """Hash numeric element bits, including NaN payloads, without holding layers."""
    h=hashlib.sha256()
    h.update(str(dataset.shape).encode());h.update(dataset.dtype.str.encode())
    if dataset.ndim==0:
        h.update(np.asarray(dataset[()]).tobytes())
    else:
        for start in range(0,len(dataset),block_size):
            value=np.ascontiguousarray(dataset[start:start+block_size])
            h.update(value.view(np.uint8))
    return h.hexdigest()


def export():
    if not UPDATED.exists():raise FileNotFoundError('The completed updated_projection.h5 is required; do not read the pending producer file')
    OUT.mkdir(parents=True,exist_ok=True)
    before={name:stat(path) for name,path in [('prior_anndata',PRIOR),('ct_atlas',ATLAS),('updated_projection',UPDATED)]}
    temporary=DESTINATION.with_suffix('.pending.h5ad')
    if temporary.exists():raise FileExistsError(f'Review or remove an existing incomplete export before retrying: {temporary}')
    if DESTINATION.exists():raise FileExistsError(f'Export already exists; use --validate-only to audit it: {DESTINATION}')
    log('Copying previous AnnData to a new export')
    shutil.copyfile(PRIOR,temporary)
    mappings=[];graph_mappings=[];updated_obs=[]
    with h5py.File(PRIOR,'r') as prior,h5py.File(ATLAS,'r') as atlas,h5py.File(UPDATED,'r') as source,h5py.File(temporary,'r+') as target:
        old_ids=prior['obs'][prior['obs'].attrs['_index']].asstr()[:]
        ids=source['peak_ids'].asstr()[:]
        if not np.array_equal(old_ids,ids) or not np.array_equal(ids,atlas['peak_ids'].asstr()[:]):raise ValueError('Peak ordering differs between source files')
        if len(np.unique(ids))!=len(ids):raise ValueError('Peak IDs must be unique')
        celltypes=source['projection_celltypes'].asstr()[:]
        if not np.array_equal(prior['var'][prior['var'].attrs['_index']].asstr()[:],celltypes):raise ValueError('Projection celltypes differ from immutable velocity columns')
        times=source['stages'][:]
        for name in ['umap2d','umap3d']:
            if not np.array_equal(source[name][:],atlas[name][:]):raise ValueError('Projection and atlas coordinates differ')
        def transfer(destination,kind,origin):
            handles={'prior':prior,'atlas':atlas,'updated':source}
            replace_elem(target,destination,handles[kind][origin][:])
            mappings.append(dict(destination=destination,source_kind=kind,source_path=origin))
            if destination.startswith('obs/'):updated_obs.append(destination.split('/',1)[1])
        for old,new in [('X_umap_2D','X_umap_published_2D'),('X_umap_3D','X_umap_published_3D')]:transfer('obsm/'+new,'prior','obsm/'+old)
        for key,origin in [('X_umap','umap2d'),('X_umap_2D','umap2d'),('X_umap_3D','umap3d')]:transfer('obsm/'+key,'atlas',origin)
        transfer('obsm/X_pca','atlas','pca')
        graph=read_graph(source['graph']);replace_elem(target,'obsp/connectivities',graph)
        graph_mappings.append(dict(destination='obsp/connectivities',source_path='graph'))
        del graph;gc.collect()
        for t,(left,right) in enumerate(zip(times[:-1],times[1:])):
            interval=f'{int(left)}_{int(right)}hpf';prefix=f'projection/{t}'
            for dim in ['2d','3d']:
                transfer(f'obsm/velocity_umap_{dim}_{interval}','updated',f'{prefix}/display_vectors/umap{dim}')
                transfer(f'obsm/velocity_umap_unmasked_{dim}_{interval}','updated',f'{prefix}/vectors/umap{dim}')
                transfer(f'obs/projected_{dim}_{interval}','updated',f'{prefix}/projection_valid/umap{dim}')
                transfer(f'obs/projection_coherence_{dim}_{interval}','updated',f'{prefix}/coherence/umap{dim}')
            required_fields=['net_rate','velocity_rms','source_feature_count','source_snr','changing','active']
            for field in required_fields+['source_supported','stationary','unprojectable','support_count','support_fraction','mean_cosine']:
                if field in required_fields and f'{prefix}/{field}' not in source:
                    raise KeyError(f'Required updated projection diagnostic missing: {prefix}/{field}')
                if f'{prefix}/{field}' in source:transfer(f'obs/{field}_{interval}','updated',f'{prefix}/{field}')
            probability=read_graph(source[f'{prefix}/probabilities'])
            destination=f'obsp/transition_{interval}'
            replace_elem(target,destination,probability)
            graph_mappings.append(dict(destination=destination,source_path=f'{prefix}/probabilities'))
            del probability;gc.collect()
            log(f'Exported new CT-atlas interval {interval}')
        transfer('obsm/velocity_umap','updated','projection/0/display_vectors/umap2d')
        columns=list(target['obs'].attrs['column-order'])
        for key in updated_obs:
            if key not in columns:columns.append(key)
        target['obs'].attrs['column-order']=np.asarray(columns,dtype=h5py.string_dtype())
        graph_manifest=json.loads(source.attrs['graph_manifest_json'])
        neighbors=dict(connectivities_key='connectivities',params=dict(method='custom_directed_affinity',algorithm='PyNNDescent',n_neighbors=30,metric='euclidean',use_rep='X_pca',n_pcs=50,random_state=int(graph_manifest['seed'])),
                       directed=True,interpretation='All-peak candidate graph in frozen whole-course log1p(CPM) PCA geometry; this graph is used for velocity correspondence, not a refitted UMAP graph')
        replace_elem(target,'uns/neighbors',neighbors)
        old_metadata=ad.io.read_elem(prior['uns/trajectory_velocity'])
        old_metadata.update(
            projection='Uncentered cosine between current-stage 21-celltype linear CPM state differences and unchanged linear CPM/hour rates; candidate edges are new full-peak k30 neighbors in frozen 50-PC whole-course log1p(CPM) geometry.',
            projection_parameters_json=str(source.attrs['projection_parameters_json']),
            interpretation='Fixed genomic peaks; retrospective profile correspondence with no imposed root or orientation. No peak movement, lineage, mechanistic kinetics or causal enhancer interaction is inferred.',
            uncertainty='Stored support and conditional count-noise diagnostics are unchanged; not biological replicate confidence, and model-selection uncertainty is not fully represented.',
            readme='X, layers[velocity] and obsm[velocity_umap] refer to10-12hpf. All state, rate, conditional-SE and support layers are copied bitwise from the prior AnnData. Each interval has a separate graph, projected vector and display mask. X_pca describes whole-course atlas geometry; X describes current-stage linear CPM. The hybrid correspondence is not a derivative/Jacobian projection through the whole-course UMAP. Standard centered scVelo recomputation changes the estimand.',
            whole_course_atlas_features='190 observed celltype×stage log1p(CPM) features; PCA fit on20000 training peaks, 50components, no whitening; frozen model projects all640830 peaks.',
            original_coordinates='obsm[X_umap_published_2D] and obsm[X_umap_published_3D] preserve prior atlas coordinates; primary UMAP and all velocity fields use the new CT atlas.',
            geometry_units='UMAP coordinates arbitrary; embedding vectors are dimensionless probability-weighted unit directions, not rates in physical UMAP units.',
            high_dimensional_rate_units='CPM/hour; time intervals use actual developmental hours.',
            source_projection=str(UPDATED.resolve()),source_atlas=str(ATLAS.resolve()),source_prior_anndata=str(PRIOR.resolve()),
            no_refitting=True,no_forced_root=True,coaccessibility_smoothing=False,
        )
        replace_elem(target,'uns/trajectory_velocity',old_metadata)
        export_meta=dict(source_files_json=json.dumps(before),graph_manifest_json=json.dumps(graph_manifest),
                         array_mapping_json=json.dumps(mappings),graph_mapping_json=json.dumps(graph_mappings),
                         copied_high_dimensional_arrays='X and every layer unchanged, including velocity aliases, observed/fitted CPM, conditional standard errors, and change-support flags.',
                         source_active_rule=str(source.attrs['source_active_rule']),
                         fixed_projection_celltypes=celltypes,stage_times_hpf=times,
                         coordinate_only_control=str((OUT/'reused_projection.h5').resolve()),
                         coordinate_only_control_note='Separate control file only; old transition probabilities are not the primary transitions in this export.',
                         exporter=str(Path(__file__).resolve()),exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        replace_elem(target,'uns/ct_umap_reprojection',export_meta)
        target.flush()
    assert before=={name:stat(path) for name,path in [('prior_anndata',PRIOR),('ct_atlas',ATLAS),('updated_projection',UPDATED)]},'A read-only source changed during export'
    validation=validate(temporary,expected_before=before)
    temporary.replace(DESTINATION)
    validation['destination']=stat(DESTINATION)
    validation['export_completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    (OUT/'export_validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
    log(f'Export validated: {DESTINATION}')
    return validation


def validate(path=DESTINATION,expected_before=None):
    before={name:stat(p) for name,p in [('prior_anndata',PRIOR),('ct_atlas',ATLAS),('updated_projection',UPDATED)]}
    if expected_before is not None:assert expected_before==before
    immutable=[];array_checks=[];graph_checks=[]
    with h5py.File(PRIOR,'r') as prior,h5py.File(ATLAS,'r') as atlas,h5py.File(UPDATED,'r') as source,h5py.File(path,'r') as target:
        metadata=ad.io.read_elem(target['uns/ct_umap_reprojection'])
        mappings=json.loads(metadata['array_mapping_json']);graph_mappings=json.loads(metadata['graph_mapping_json'])
        ids=source['peak_ids'].asstr()[:];obs_index=target['obs'].attrs['_index'];var_index=target['var'].attrs['_index']
        assert np.array_equal(target['obs'][obs_index].asstr()[:],ids)
        assert np.array_equal(ids,prior['obs'][prior['obs'].attrs['_index']].asstr()[:])
        assert np.array_equal(ids,atlas['peak_ids'].asstr()[:])
        celltypes=target['var'][var_index].asstr()[:]
        assert np.array_equal(celltypes,source['projection_celltypes'].asstr()[:])
        assert np.array_equal(celltypes,prior['var'][prior['var'].attrs['_index']].asstr()[:])
        assert set(target['layers'])==set(prior['layers'])
        immutable_paths=['X']+['layers/'+key for key in sorted(prior['layers'])]+['obsm/trajectory_modes','obsm/trajectory_lack_of_fit']
        for key in immutable_paths:
            a,b=prior[key],target[key]
            assert a.shape==b.shape and a.dtype==b.dtype,(key,'shape or dtype')
            digest_a=dataset_digest(a);digest_b=dataset_digest(b)
            assert digest_a==digest_b,(key,'changed element bits')
            immutable.append(dict(path=key,shape=list(a.shape),dtype=str(a.dtype),sha256=digest_a))
        log(f'Bitwise preservation passed for {len(immutable)} high-dimensional arrays')
        handles={'prior':prior,'atlas':atlas,'updated':source}
        for mapping in mappings:
            original=handles[mapping['source_kind']][mapping['source_path']];copied=target[mapping['destination']]
            assert original.shape==copied.shape and original.dtype==copied.dtype,mapping
            assert dataset_digest(original)==dataset_digest(copied),mapping
            array_checks.append(mapping)
        for mapping in graph_mappings:
            original=source[mapping['source_path']];copied=target[mapping['destination']]
            assert tuple(original.attrs['shape'])==tuple(copied.attrs['shape'])==(len(ids),len(ids))
            hashes={}
            for field in ['data','indices','indptr']:
                assert original[field].dtype==copied[field].dtype,(mapping,field,'dtype')
                hashes[field]=dataset_digest(copied[field])
                assert dataset_digest(original[field])==hashes[field],(mapping,field,'changed CSR entries')
            graph=ad.io.read_elem(copied)
            assert sparse.isspmatrix_csr(graph)
            assert np.isfinite(graph.data).all() and (graph.data>0).all()
            if mapping['destination']=='obsp/connectivities':
                assert np.all(np.diff(graph.indptr)==30)
                assert not np.any(graph.diagonal())
            else:
                interval=int(mapping['source_path'].split('/')[1]);active=source[f'projection/{interval}/active'][:].astype(bool)
                sums=np.asarray(graph.sum(axis=1)).ravel()
                assert np.allclose(sums[active],1.,rtol=1e-6,atol=1e-6)
                assert np.all(sums[~active]==0.)
            graph_checks.append(dict(**mapping,nnz=int(graph.nnz),array_sha256=hashes))
            del graph;gc.collect()
        column_names=list(target['obs'].attrs['column-order'])
        assert len(column_names)==len(set(column_names))
        for name in prior['obs'].attrs['column-order']:assert name in column_names
        for mapping in mappings:
            if mapping['destination'].startswith('obs/'):assert mapping['destination'].split('/',1)[1] in column_names
        neighbors=ad.io.read_elem(target['uns/neighbors']);velocity_meta=ad.io.read_elem(target['uns/trajectory_velocity'])
        assert neighbors['params']['n_neighbors']==30 and neighbors['params']['use_rep']=='X_pca' and neighbors['directed']
        assert velocity_meta['no_refitting'] and velocity_meta['no_forced_root']
        assert 'not biological replicate confidence' in velocity_meta['uncertainty']
    backed=ad.read_h5ad(path,backed='r')
    try:
        assert backed.shape==(len(ids),len(celltypes))
        assert np.array_equal(backed.obs_names.to_numpy(),ids)
        assert np.array_equal(backed.var_names.to_numpy(),celltypes)
        assert np.array_equal(backed.obsm['X_umap'],backed.obsm['X_umap_2D'])
        assert np.array_equal(backed.obsm['velocity_umap'],backed.obsm['velocity_umap_2d_10_12hpf'])
        assert backed.obsm['X_pca'].shape==(len(ids),50)
        assert sparse.isspmatrix_csr(backed.obsp['connectivities'])
        for t,key in enumerate(['10_12hpf','12_14hpf','14_16hpf','16_19hpf','19_24hpf']):
            for dim in ['2d','3d']:
                mask=backed.obs[f'projected_{dim}_{key}'].to_numpy(dtype=bool)
                arrows=backed.obsm[f'velocity_umap_{dim}_{key}']
                assert np.isfinite(arrows).all() and np.all(arrows[~mask]==0)
    finally:backed.file.close();del backed;gc.collect()
    after={name:stat(p) for name,p in [('prior_anndata',PRIOR),('ct_atlas',ATLAS),('updated_projection',UPDATED)]}
    assert before==after,'Read-only sources changed during validation'
    return dict(status='passed',n_peaks=len(ids),n_projection_celltypes=len(celltypes),source_files=before,
                source_files_unchanged=True,ordered_peak_ids_exact=True,ordered_celltypes_exact=True,
                high_dimensional_arrays_bitwise_unchanged=immutable,
                new_arrays_exact=array_checks,new_graphs_exact=graph_checks,
                all_projection_masks_and_active_fields_updated=True,masked_vectors_zero=True,
                new_neighbor_graph_30_nonself_candidates=True,transition_row_sums_match_active_fields=True,
                anndata_backed_read_passed=True,updated_metadata_verified=True,
                interpretation='Export integrity checks; no additional biological or kinetic inference is implied.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    result=validate() if args.validate_only else export()
    if args.validate_only:
        result['destination']=stat(DESTINATION)
        (OUT/'export_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ['status','n_peaks','n_projection_celltypes','source_files_unchanged','anndata_backed_read_passed']},indent=2))

if __name__=='__main__':main()
