"""Prepare fixed-graph reference distances for celltype-specific activation analysis.

No UMAP is recomputed. Existing Leiden regions are descriptive atlas regions;
neither they nor nearest-reference distances establish biological branches.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import h5py,numpy as np,pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'scripts/chrom_velo'))
from activation_geometry import reference_distances,region_graph_totals,select_reference_core
from real_specialization_benchmark import read_frame,read_graph,write_json
DATA=HERE.parents[2]/'data/annotated_data/objects_v2'
DEFAULT_OUT=HERE/'outputs/activation_progression_v1'
FITS=HERE/'outputs/trajectory_velocity_v1/trajectory_fits.h5'


def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)

def quantiles(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(zip(['min','q25','median','q75','q95','max'],np.quantile(x,[0,.25,.5,.75,.95,1]).tolist())) if len(x) else {}

def digest(x):return hashlib.sha256(np.ascontiguousarray(x).view('u1')).hexdigest()

def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    atlas_path=DATA/'peaks_by_pb_annotated_master.h5ad'
    with h5py.File(atlas_path,'r') as atlas,h5py.File(FITS,'r') as f:
        obs=read_frame(atlas['obs'],['leiden_coarse','celltype','lineage','timepoint'])
        ids=f['peak_ids'].asstr()[:];reference=f['broad_reference'][:].astype(bool)
        if not np.array_equal(obs.index.to_numpy(),ids) or not obs.index.is_unique:raise ValueError('Peak order mismatch')
        coords={k:f[k][:] for k in ['umap2d','umap3d']}
        for k,ak in [('umap2d','X_umap_2D'),('umap3d','X_umap_3D')]:
            if not np.array_equal(coords[k],atlas[f'obsm/{ak}'][:]):raise ValueError('Coordinates changed')
        graph=read_graph(atlas['obsp/connectivities'])
    region_labels=np.asarray(sorted(obs.leiden_coarse.astype(str).unique(),key=lambda x:(0,int(x)) if x.isdigit() else (1,x)))
    label_to_code={x:i for i,x in enumerate(region_labels)}
    region_code=np.asarray([label_to_code[x] for x in obs.leiden_coarse.astype(str)],dtype='i2')
    graph_hash={k:digest(getattr(graph,k)) for k in ['indptr','indices','data']}
    log(f'Distance calculation: {len(ids):,} peaks, {graph.nnz:,} directed entries, {reference.sum():,} reference peaks')
    d=reference_distances(graph,reference)
    if graph_hash!={k:digest(getattr(graph,k)) for k in ['indptr','indices','data']}:raise RuntimeError('Graph mutated')
    log(f'Distances computed: {d["n_components"]} components; reachable {np.isfinite(d["distance_hops"]).sum():,}')
    edge_counts,edge_weights=region_graph_totals(graph,region_code,len(region_labels))
    if int(edge_counts.sum())!=graph.nnz or not np.isclose(edge_weights.sum(),graph.data.sum(dtype=float)):raise RuntimeError('Region edge accounting mismatch')
    region_rows=[]
    for code,label in enumerate(region_labels):
        m=region_code==code;root=reference&m;n=int(m.sum());nr=int(root.sum());r=dict(region_code=code,region=label,peaks=n,broad_reference_peaks=nr,broad_reference_fraction=nr/n,reference_share=nr/reference.sum(),reachable_peaks=int(np.isfinite(d['distance_hops'][m]).sum()))
        for k in ['distance_hops','distance_weighted']:
            for q,v in quantiles(d[k][m]).items():r[k+'_'+q]=v
        for k,xy in coords.items():
            for dim,val in enumerate(xy[m].mean(axis=0)):r[f'{k}_centroid_{dim}']=float(val)
        for field in ['celltype','lineage','timepoint']:
            values=obs.loc[m,field].astype(str).value_counts();r[f'modal_{field}']=str(values.index[0]);r[f'modal_{field}_fraction']=float(values.iloc[0]/n)
        r['within_region_directed_edge_count']=int(edge_counts[code,code]);r['outgoing_cross_region_directed_edge_count']=int(edge_counts[code].sum()-edge_counts[code,code])
        region_rows.append(r)
    regions=pd.DataFrame(region_rows);regions.to_csv(out/'geometry_regions.csv',index=False)
    edge_rows=[dict(source_code=i,target_code=j,source_region=region_labels[i],target_region=region_labels[j],directed_edge_count=int(edge_counts[i,j]),affinity_sum=float(edge_weights[i,j])) for i,j in zip(*np.where(edge_counts))]
    pd.DataFrame(edge_rows).to_csv(out/'geometry_region_edges.csv',index=False)
    component_sizes=np.bincount(d['component']);root_counts=d['component_reference_count']
    pd.DataFrame(dict(component=np.arange(d['n_components']),peaks=component_sizes,broad_reference_peaks=root_counts,reachable=root_counts>0)).to_csv(out/'geometry_components.csv',index=False)
    p=regions.reference_share.to_numpy();nonzero=p>0
    entropy=float(-(p[nonzero]*np.log(p[nonzero])).sum());effective_regions=float(np.exp(entropy))
    finite=np.isfinite(d['distance_hops']);notroot=finite&~reference
    rho=float(spearmanr(d['distance_hops'][notroot],d['distance_weighted'][notroot]).statistic)
    umap_span={}
    for k,xy in coords.items():
        full=np.quantile(xy,[.05,.95],axis=0);rr=np.quantile(xy[reference],[.05,.95],axis=0)
        umap_span[k]=dict(atlas_p05_p95=full.tolist(),reference_p05_p95=rr.tolist(),reference_to_atlas_width_ratio=((rr[1]-rr[0])/(full[1]-full[0])).tolist())
    summary=dict(n_peaks=len(ids),n_regions=len(region_labels),reference_peaks=int(reference.sum()),reference_fraction=float(reference.mean()),regions_with_reference=int((regions.broad_reference_peaks>0).sum()),regions_with_at_least_10_reference=int((regions.broad_reference_peaks>=10).sum()),effective_reference_regions=effective_regions,largest_region_reference_share=float(p.max()),n_components=d['n_components'],largest_component_peaks=int(component_sizes.max()),components_with_reference=int(np.sum(root_counts>0)),reachable_peaks=int(finite.sum()),unreachable_peaks=int((~finite).sum()),hop_distance=quantiles(d['distance_hops']),weighted_distance=quantiles(d['distance_weighted']),nonreference_hop_weighted_spearman=rho,reference_umap_span=umap_span,original_graph_directed_entries=int(graph.nnz),original_graph_weight_min=float(graph.data.min()),original_graph_weight_max=float(graph.data.max()),graph_hashes_sha256=graph_hash,source_atlas=str(atlas_path.resolve()),source_fits=str(FITS.resolve()),peak_ids_equal=True,umap_coordinates_equal=True,original_graph_unchanged=True,edge_cost='1-log(max(original affinity,1e-6))',distance_scope='Nearest broad-reference peak on the original undirected atlas graph; +inf means no reference in that connected component',region_scope='Existing leiden_coarse labels are descriptive atlas regions, not inferred anatomical branches',interpretation='Reference distance is a test coordinate, not developmental time. Multiple broad peaks are deliberately retained; distributed references weaken a single-root interpretation. No paths, edge directions, or root-to-branch biological progression are imposed.')
    write_json(out/'geometry_summary.json',summary)
    with h5py.File(out/'geometry.h5','w') as f:
        for k,a in [('peak_ids',ids),('region_labels',region_labels)]:f.create_dataset(k,data=np.asarray(a,dtype=object),dtype=h5py.string_dtype())
        for k,a in [('distance_hops',d['distance_hops'].astype('f4')),('distance_weighted',d['distance_weighted'].astype('f4')),('component',d['component'].astype('i4')),('region_code',region_code),('broad_reference',reference),('region_edge_count',edge_counts),('region_edge_affinity',edge_weights),*coords.items()]:f.create_dataset(k,data=a,compression='lzf')
        f.attrs['summary_json']=json.dumps(summary,allow_nan=False)
    plot_geometry(out,coords['umap2d'],reference,d,regions)
    log(f'Wrote {out}/geometry.h5 and diagnostics')
    print(json.dumps({k:summary[k] for k in ['n_peaks','reference_peaks','regions_with_reference','regions_with_at_least_10_reference','effective_reference_regions','largest_region_reference_share','n_components','unreachable_peaks','hop_distance','nonreference_hop_weighted_spearman']},indent=2),flush=True)


def plot_geometry(out,xy,reference,d,regions):
    # Fixed seed and identical background subset preserve visual comparability.
    rng=np.random.default_rng(749);sample=np.sort(rng.choice(len(xy),min(len(xy),90000),replace=False))
    roots=np.flatnonzero(reference);fig,axs=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    ax=axs[0,0];ax.scatter(*xy[sample].T,s=.5,c='#ddd',rasterized=True);ax.scatter(*xy[roots].T,s=1,c='#118865',rasterized=True);ax.set_title(f'Broad-accessibility reference: {len(roots):,} peaks')
    for ax,key,title in [(axs[0,1],'distance_hops','Nearest-reference graph distance: hops'),(axs[1,0],'distance_weighted','Nearest-reference graph distance: affinity cost')]:
        good=np.isfinite(d[key][sample]);ss=sample[good];v=d[key][ss];p=ax.scatter(*xy[ss].T,c=v,s=.65,cmap='viridis',vmin=0,vmax=np.quantile(v,.99),rasterized=True);fig.colorbar(p,ax=ax,label='Graph distance');ax.set_title(title)
    for ax in [axs[0,0],axs[0,1],axs[1,0]]:ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel('Published peak UMAP')
    ax=axs[1,1];ax.bar(regions.region.astype(str),regions.broad_reference_fraction*100,color='#118865');ax.set_xlabel('Existing Leiden atlas region');ax.set_ylabel('Broad-reference peaks within region (%)');ax.tick_params(axis='x',rotation=90);ax.set_title('Reference distribution across existing regions')
    fig.suptitle('Atlas geometry audit — reference distance is not developmental time',fontsize=15)
    fig.savefig(out/'geometry_diagnostics.png',dpi=170);fig.savefig(out/'geometry_diagnostics.pdf');plt.close(fig)



def prepare_core(out):
    # Separate artifact: the primary all-reference geometry remains immutable.
    with h5py.File(out/'geometry.h5','r') as f:
        ids=f['peak_ids'].asstr()[:];reference=f['broad_reference'][:].astype(bool)
        code=f['region_code'][:];labels=f['region_labels'].asstr()[:]
        primary_hops=f['distance_hops'][:];primary_weighted=f['distance_weighted'][:]
        xy=f['umap2d'][:]
    core,selected=select_reference_core(reference,code,coverage=.9)
    with h5py.File(DATA/'peaks_by_pb_annotated_master.h5ad','r') as atlas:
        obs=read_frame(atlas['obs'],['leiden_coarse'])
        if not np.array_equal(ids,obs.index.to_numpy()):raise ValueError('Atlas peak order mismatch')
        graph=read_graph(atlas['obsp/connectivities'])
    d=reference_distances(graph,core)
    finite=np.isfinite(primary_hops)&np.isfinite(d['distance_hops']);nonroot=finite&~reference
    summary=dict(reference_definition='Smallest set of existing Leiden regions containing at least 90% of existing broad-reference peaks; ties by region code. No temporal ordering enters selection.',selected_regions=labels[selected].tolist(),selected_region_codes=selected.tolist(),core_reference_peaks=int(core.sum()),original_reference_peaks=int(reference.sum()),retained_reference_fraction=float(core.sum()/reference.sum()),hops=quantiles(d['distance_hops']),weighted=quantiles(d['distance_weighted']),fraction_peaks_distance_increases=float(np.mean(d['distance_hops'][finite]>primary_hops[finite])),mean_hop_increase=float(np.mean(d['distance_hops'][finite]-primary_hops[finite])),max_hop_increase=float(np.max(d['distance_hops'][finite]-primary_hops[finite])),primary_core_hop_spearman_nonreference=float(spearmanr(primary_hops[nonroot],d['distance_hops'][nonroot]).statistic),primary_core_weighted_spearman_nonreference=float(spearmanr(primary_weighted[nonroot],d['distance_weighted'][nonroot]).statistic),interpretation='Primary distance uses every broad reference, including rare references outside the main concentrations. These rare references can reset nearest-reference distance in otherwise distant regions. This separate sensitivity uses the largest reference concentrations, without establishing developmental roots or anatomical branches.')
    if np.any(d['distance_hops'][finite]<primary_hops[finite]):raise RuntimeError('Subset reference cannot decrease shortest distance')
    with h5py.File(out/'geometry_core_reference.h5','w') as f:
        f.create_dataset('peak_ids',data=np.asarray(ids,dtype=object),dtype=h5py.string_dtype())
        f.create_dataset('core_reference',data=core,compression='lzf')
        f.create_dataset('distance_core_hops',data=d['distance_hops'].astype('f4'),compression='lzf')
        f.create_dataset('distance_core_weighted',data=d['distance_weighted'].astype('f4'),compression='lzf')
        f.attrs['summary_json']=json.dumps(summary,allow_nan=False)
    write_json(out/'geometry_core_reference_summary.json',summary)
    rng=np.random.default_rng(749);sample=np.sort(rng.choice(len(xy),min(len(xy),90000),replace=False))
    fig,axs=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    axs[0].scatter(*xy[sample].T,s=.5,c='#ddd');axs[0].scatter(*xy[core].T,s=1,c='#118865');axs[0].set_title('Core reference in regions '+', '.join(labels[selected]))
    for ax,v,title in [(axs[1],d['distance_hops'],'Core-reference hop distance'),(axs[2],d['distance_hops']-primary_hops,'Distance increase from removing rare references')]:
        scat=ax.scatter(*xy[sample].T,c=v[sample],s=.65,cmap='viridis',rasterized=True);fig.colorbar(scat,ax=ax,label='Graph hops');ax.set_title(title)
    for ax in axs:ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel('Published peak UMAP')
    fig.suptitle('Reference sensitivity — selection uses reference counts, not temporal order')
    fig.savefig(out/'geometry_core_reference.png',dpi=170);plt.close(fig)
    log('Core-reference sensitivity saved separately; primary geometry unchanged')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=DEFAULT_OUT);p.add_argument('--core-reference-only',action='store_true');a=p.parse_args();prepare_core(a.output_dir) if a.core_reference_only else prepare(a.output_dir)
