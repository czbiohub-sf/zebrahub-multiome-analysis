"""Offline paired peak-state trajectories and fixed-atlas context.

All states share a fitted coordinate system. Lines join a locus's measured or
retrospectively fitted snapshots; their screen lengths are not biological rates.
This module only selects display examples and renders saved coordinates.
"""
from __future__ import annotations
import argparse, base64, gzip, hashlib, json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE=Path(__file__).resolve().parent
DEFAULT=HERE/'outputs/state_trajectory_atlas_v1'
PANELS={'primary_21':'21 celltypes · 10–24 hpf','matched_21':'21 celltypes · 12–24 hpf','optic_24':'24 celltypes, including optic cup · 12–24 hpf'}


def decode(a):
    return np.asarray([x.decode() if isinstance(x,bytes) else str(x) for x in a])


def packed(a):
    dtype='uint8' if np.asarray(a).dtype.kind=='b' else 'float32'
    a=np.ascontiguousarray(a,dtype=dtype)
    return {'dtype':dtype,'shape':list(a.shape),'gzip_base64':base64.b64encode(gzip.compress(a.tobytes(),mtime=0,compresslevel=6)).decode()}


def tau(a):
    maximum=np.max(a,axis=-1)
    return np.divide(np.sum(1-np.divide(a,maximum[...,None],out=np.zeros_like(a),where=maximum[...,None]>0),axis=-1),a.shape[-1]-1,out=np.full(maximum.shape,np.nan),where=maximum>0)


def load_sample(out):
    with h5py.File(out/'sample_data.h5') as f:
        data={k:f[k][:] for k in f if isinstance(f[k],h5py.Dataset)}
    data['peak_ids']=decode(data['peak_ids']);data['celltypes']=decode(data['celltypes'])
    metadata=pd.read_csv(out/'sample_metadata.csv').fillna('')
    if len(metadata)!=len(data['peak_ids']):raise ValueError('Metadata/sample row count mismatch')
    for name in ['peak_id','peak','peak_ids']:
        if name in metadata and not np.array_equal(metadata[name].astype(str),data['peak_ids']):raise ValueError('Metadata peak order mismatch')
    return data,metadata


def panel_indices(group):
    if 'stage_indices' not in group: group=group.parent
    for key in ['celltype_indices','feature_indices']:
        if key in group:c=group[key][:].astype(int);break
    else:raise KeyError('Missing celltype/feature indices')
    return group['stage_indices'][:].astype(int),c


def example_indices(data,feature_indices):
    """Descriptive, deliberately selected examples; never prevalence estimates."""
    a=data['fitted_cpm'][:,:,feature_indices].astype(float)
    observed=data['observed_cpm'][:,:,feature_indices].astype(float)
    mean=np.mean(a,axis=2);specific=tau(a);net=mean[:,-1]-mean[:,0]
    total=np.sum(np.abs(np.diff(mean,axis=1)),axis=1)
    support=np.any(data['rate_supported'][:,:,feature_indices],axis=(1,2))
    finite=np.isfinite(a).all(axis=(1,2))&np.isfinite(observed).all(axis=(1,2))
    broad=(np.nanmax(specific,axis=1)<.5)&(np.min(mean,axis=1)>np.quantile(np.min(mean,axis=1),.75))
    categories=[('Net opening',net,support&(net>0)),('Net closing',-net,support&(net<0)),('Specificity gain',specific[:,-1]-specific[:,0],support&(specific[:,-1]>specific[:,0])),('Transient',total-np.abs(net),support&(np.abs(net)<.4*np.maximum(total,1e-9))),('Stable broad',-np.ptp(mean,axis=1)/np.maximum(np.mean(mean,axis=1),1e-9),broad)]
    used=set();rows=[]
    for category,score,valid in categories:
        ids=np.flatnonzero(finite&valid&np.isfinite(score));ids=ids[np.argsort(score[ids])[::-1]]
        chosen=next((int(i) for i in ids if int(i) not in used),None)
        if chosen is None:continue
        used.add(chosen);rows.append({'category':category,'sample_index':chosen,'peak_id':data['peak_ids'][chosen],'selection_score':float(score[chosen]),'fitted_net_mean_cpm':float(net[chosen]),'fitted_total_mean_change_cpm':float(total[chosen]),'specificity_change':float(specific[chosen,-1]-specific[chosen,0]),'supported_ct_interval_count':int(data['rate_supported'][chosen,:,feature_indices].sum())})
    return pd.DataFrame(rows)


def get_examples(out, data, cts):
    path=out/'representative_peaks.csv'
    if path.exists():
        examples=pd.read_csv(path)
        if 'label' in examples: examples=examples.rename(columns={'label':'category'})
        return examples
    return example_indices(data,cts)


def render_static(out):
    data,meta=load_sample(out)
    with h5py.File(out/'state_pca.h5') as f:
        g=f['primary_21']['log1p'];times,cts=panel_indices(g)
    examples=get_examples(out,data,cts);examples.to_csv(out/'viewer_selected_examples.csv',index=False)
    with h5py.File(out/'state_umap.h5') as f:
        if 'primary_21' not in f:return
        for dim in [2,3]:
            g=f['primary_21'];coords=g[f'observed_umap{dim}d'][:];fit=g[f'fitted_umap{dim}d'][:]
            rng=np.random.default_rng(20261021);bg=np.sort(rng.choice(len(coords),min(3500,len(coords)),replace=False))
            fig=plt.figure(figsize=(18,10));axes=[]
            for j,(_,row) in enumerate(examples.head(6).iterrows()):
                ax=fig.add_subplot(2,3,j+1,projection='3d' if dim==3 else None);axes.append(ax)
                p=int(row.sample_index);b=coords[bg].reshape(-1,dim)
                if dim==3:
                    both=np.concatenate([coords[p],fit[p]],axis=0);center=(np.min(both,axis=0)+np.max(both,axis=0))/2;span=max(float(np.max(np.ptp(both,axis=0)))*1.55,.65)
                    b=b[np.all((b>=center-span/2)&(b<=center+span/2),axis=1)]
                if dim==2:
                    ax.scatter(*b.T,c='#aebbc6',s=.5,alpha=.18,rasterized=True)
                    ax.plot(*coords[p].T,c='#4a5662',lw=1,alpha=.7,label='Observed')
                    ax.plot(*fit[p].T,c='#a65925',lw=2,ls='--',label='Fitted')
                    art=ax.scatter(*coords[p].T,c=data['stages'][times],cmap='viridis',norm=Normalize(10,24),s=55,edgecolor='white',linewidth=.7,zorder=4)
                    for t in range(len(times)-1):ax.annotate('',xy=coords[p,t+1],xytext=coords[p,t],arrowprops={'arrowstyle':'->','lw':1,'color':'#364b60'})
                    ax.scatter(*fit[p].T,c=data['stages'][times],cmap='viridis',norm=Normalize(10,24),s=30,marker='D',edgecolor='#a65925',linewidth=.6,zorder=3)
                    ax.set_aspect('equal',adjustable='datalim')
                else:
                    ax.scatter(*b.T,c='#b6c2ca',s=.45,alpha=.12,rasterized=True,depthshade=False)
                    ax.plot(*coords[p].T,c='#4a5662',lw=1,label='Observed');ax.plot(*fit[p].T,c='#a65925',lw=2,ls='--',label='Fitted')
                    art=ax.scatter(*coords[p].T,c=data['stages'][times],cmap='viridis',norm=Normalize(10,24),s=28,depthshade=False)
                    ax.scatter(*fit[p].T,c=data['stages'][times],cmap='viridis',norm=Normalize(10,24),s=22,marker='D',edgecolor='#a65925',linewidth=.5,depthshade=False)
                    for t in range(len(times)-1):ax.quiver(*coords[p,t],*(coords[p,t+1]-coords[p,t]),color='#364b60',lw=.8,arrow_length_ratio=.15)
                    ax.set_zlabel('Joint state UMAP 3',fontsize=8)
                both=np.concatenate([coords[p],fit[p]],axis=0);center=(np.min(both,axis=0)+np.max(both,axis=0))/2;span=max(float(np.max(np.ptp(both,axis=0)))*1.55,.65)
                ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]-span/2,center[1]+span/2)
                if dim==3:ax.set_zlim(center[2]-span/2,center[2]+span/2)
                gene=next((str(meta.iloc[p][k]) for k in ['nearest_gene','linked_gene','associated_gene'] if k in meta and str(meta.iloc[p][k]) not in ['', 'nan']), '')
                ax.set_title(f'{row.category}\n{data["peak_ids"][p]}'+(f' · {gene}' if gene else ''),fontsize=9)
                ax.set_xlabel('Joint state UMAP 1',fontsize=8);ax.set_ylabel('Joint state UMAP 2',fontsize=8)
                ax.tick_params(labelsize=7)
            fig.suptitle(f'Same-peak trajectories · local views of the joint {dim}D UMAP',fontsize=18)
            fig.tight_layout(rect=[0,.07,.94,.93]);cax=fig.add_axes([.958,.30,.012,.40]);fig.colorbar(art,cax=cax,ticks=data['stages'],label='Stage (hpf)')
            from matplotlib.lines import Line2D
            fig.legend(handles=[Line2D([0],[0],color='#4a5662',marker='o',label='Observed snapshots'),Line2D([0],[0],color='#a65925',ls='--',marker='D',label='Retrospectively fitted snapshots')],loc='upper center',bbox_to_anchor=(.48,.955),ncol=2,frameon=False,fontsize=10)
            fig.text(.5,.012,'Each panel is a local zoom of the same fixed map; zoom levels differ. Arrows connect observed snapshots; elapsed intervals are 2, 2, 2, 3 and 5 hours.\nScreen lengths are not accessibility rates. Full-map context is in figure 12 and the viewer. Profiles were selected before inspecting their UMAP paths.',ha='center',fontsize=9)
            stem=f'11_selected_state_trajectories_{dim}d';fig.savefig(out/(stem+'.png'),dpi=160,facecolor='white');fig.savefig(out/(stem+'.pdf'),facecolor='white');plt.close(fig)
    with h5py.File(out/'state_umap.h5') as f:
        g=f['primary_21'];xy=g['observed_umap2d'][:]
    levels=np.log1p(np.mean(data['observed_cpm'][:,:,cts],axis=2))
    limit=float(np.nanquantile(levels,.99));norm=Normalize(0,limit)
    extent=np.array([np.nanmin(xy,axis=(0,1)),np.nanmax(xy,axis=(0,1))]);pad=.04*(extent[1]-extent[0])
    fig,axes=plt.subplots(2,3,figsize=(17,10))
    for j,ax in enumerate(axes.flat):
        art=ax.scatter(*xy[:,j].T,c=levels[:,j],s=1.1,cmap='viridis',norm=norm,alpha=.7,linewidths=0,rasterized=True)
        ax.set(title=f'{int(data["stages"][times[j]])} hpf',xlim=(extent[0,0]-pad[0],extent[1,0]+pad[0]),ylim=(extent[0,1]-pad[1],extent[1,1]+pad[1]),xlabel='Joint state UMAP 1',ylabel='Joint state UMAP 2');ax.set_aspect('equal')
    fig.subplots_adjust(top=.91,bottom=.09,right=.9,wspace=.22,hspace=.25)
    cax=fig.add_axes([.92,.22,.012,.56]);fig.colorbar(art,cax=cax,label='log(1 + mean relative CPM across 21 celltypes)')
    fig.suptitle('Measured accessibility states over time · one fixed joint UMAP',fontsize=18)
    fig.text(.5,.022,'Each panel shows the same sampled peaks at one stage, in coordinates learned jointly from observed states. Colors share one scale.\nA point is a peak × stage state; this display does not impose a root or a direction of travel.',ha='center',fontsize=9)
    fig.savefig(out/'12_joint_state_umap_stage_maps.png',dpi=160,facecolor='white');fig.savefig(out/'12_joint_state_umap_stage_maps.pdf',facecolor='white');plt.close(fig)
    return examples


def optic_examples(out, render=False):
    """Choose peaks solely by optic-cup curves, before consulting UMAP geometry."""
    data,meta=load_sample(out);c=list(data['celltypes']).index('optic_cup');ts=np.arange(1,6);times=data['stages'][ts]
    a=data['fitted_cpm'][:,ts,c];rates=np.diff(a,axis=1)/np.diff(times);support=data['rate_supported'][:,1:,c].astype(bool)
    valid=data['test_mask'].astype(bool)&data['fit_valid'][:,c].astype(bool)&(data['lack_of_fit'][:,c]<=4)&np.isfinite(a).all(axis=1)
    up=np.any(support&(rates>0),axis=1);down=np.any(support&(rates<0),axis=1)
    classes={'Opening':valid&up&np.all(rates>=-1e-6,axis=1),'Closing':valid&down&np.all(rates<=1e-6,axis=1),
             'Transient peak':valid&up&down&(np.argmax(a,axis=1)>0)&(np.argmax(a,axis=1)<4),
             'Transient trough':valid&up&down&(np.argmin(a,axis=1)>0)&(np.argmin(a,axis=1)<4)}
    amount=np.sum(np.abs(np.diff(a,axis=1)),axis=1);rows=[]
    for label,mask in classes.items():
        candidates=np.flatnonzero(mask)
        if not len(candidates):continue
        candidates=candidates[np.argsort(amount[candidates],kind='stable')];i=int(candidates[len(candidates)//2])
        row={'label':'Optic cup · '+label.lower(),'class':label,'sample_index':i,'peak_id':data['peak_ids'][i],
             'nearest_gene':str(meta.iloc[i]['nearest_gene']),'focal_celltype':'optic_cup','panel':'optic_24','eligible_heldout_peaks':len(candidates),
             'selection':'Middle-ranked total fitted absolute change among held-out peaks meeting the stated shape and conditional support criteria; no UMAP coordinates used.',
             'fitted_total_change_cpm':float(amount[i]),'lack_of_fit':float(data['lack_of_fit'][i,c]),'supported_intervals':int(support[i].sum())}
        for j,t in enumerate(times):row[f'observed_{int(t)}hpf']=float(data['observed_cpm'][i,ts[j],c]);row[f'fitted_{int(t)}hpf']=float(a[i,j])
        for j in range(4):row[f'supported_{int(times[j])}_{int(times[j+1])}']=bool(support[i,j]);row[f'fitted_rate_{int(times[j])}_{int(times[j+1])}']=float(rates[i,j])
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(out/'optic_cup_state_examples.csv',index=False)
    definition={'window_hpf':times.tolist(),'source_feature':'optic_cup','heldout_only':True,'fit_quality':'fit_valid and lack_of_fit <= 4','opening':'All fitted interval rates nonnegative (numerical tolerance 1e-6) and at least one supported positive interval','closing':'All fitted interval rates nonpositive and at least one supported negative interval','transient':'Interior fitted maximum/minimum, with at least one supported positive and negative interval','representative_choice':'Middle-ranked total absolute fitted CPM change within each eligible class','class_counts':{k:int(v.sum()) for k,v in classes.items()},'excluded_10hpf':'Optic cup at 10 hpf is observed but below the 20-cell condition threshold; excluded from this figure, rates, and the 24-celltype embedding. Never filled with zero.'}
    (out/'optic_cup_state_example_selection.json').write_text(json.dumps(definition,indent=2))
    if not render:return table
    with h5py.File(out/'state_umap.h5') as f:
        if 'optic_24' not in f:raise ValueError('Optic panel is not yet available')
        xy=f['optic_24']['observed_umap2d'][:];fitxy=f['optic_24']['fitted_umap2d'][:]
    bg=np.sort(np.random.default_rng(20261021).choice(len(xy),2500,replace=False));b=xy[bg].reshape(-1,2)
    fig,axes=plt.subplots(len(table),2,figsize=(16,3.8*len(table)),gridspec_kw={'width_ratios':[1,1.2]})
    for rownum,(_,row) in enumerate(table.iterrows()):
        i=int(row.sample_index);left,right=axes[rownum];left.scatter(*b.T,s=.55,c='#abb9c6',alpha=.18,rasterized=True)
        left.plot(*xy[i].T,color='#31556d',lw=1.2);left.plot(*fitxy[i].T,color='#b36730',ls='--',lw=2)
        stage_art=left.scatter(*xy[i].T,c=times,cmap='viridis',norm=Normalize(12,24),s=38,edgecolor='white',linewidth=.5,zorder=4)
        for j in range(4):
            if np.linalg.norm(xy[i,j+1]-xy[i,j])>1e-10:left.annotate('',xy=xy[i,j+1],xytext=xy[i,j],arrowprops={'arrowstyle':'->','color':'#31556d','lw':1})
        left.scatter(*fitxy[i].T,c=times,cmap='viridis',norm=Normalize(12,24),s=25,marker='D',edgecolor='#b36730',linewidth=.5)
        both=np.concatenate([xy[i],fitxy[i]],axis=0);center=(np.min(both,axis=0)+np.max(both,axis=0))/2;span=max(float(np.max(np.ptp(both,axis=0)))*1.55,.65)
        left.set_xlim(center[0]-span/2,center[0]+span/2);left.set_ylim(center[1]-span/2,center[1]+span/2)
        left.set(title=f'{row.peak_id} · {row.nearest_gene}',xlabel='Joint state UMAP 1',ylabel='Joint state UMAP 2');left.set_aspect('equal',adjustable='datalim')
        observed=data['observed_cpm'][i,ts,c];right.plot(times,observed,'-',color='#31556d',lw=1,label='Observed optic-cup CPM');right.scatter(times,observed,c=times,cmap='viridis',norm=Normalize(12,24),s=35,zorder=4)
        right.plot(times,a[i],'s--',color='#b36730',lw=1.8,ms=4,label='Retrospectively fitted CPM')
        for j in range(4):
            if support[i,j]:right.axvspan(times[j],times[j+1],color='#cfdfb9' if rates[i,j]>0 else '#c7dbec',alpha=.35,zorder=0)
        labels=[f'{int(times[j])}–{int(times[j+1])}: '+('opening' if rates[i,j]>0 else 'closing') for j in range(4) if support[i,j]]
        right.text(.02,.97,'Supported: '+', '.join(labels),transform=right.transAxes,va='top',fontsize=8)
        right.set(title=f'{row["class"]} · {int(row.eligible_heldout_peaks)} eligible held-out peaks',xlabel='Stage (hpf)',ylabel='Optic-cup relative accessibility (CPM)',xticks=times,ylim=(0,max(np.max(observed),np.max(a[i]))*1.24))
        right.grid(alpha=.12)
    fig.suptitle('Optic-cup programs · same-peak trajectories and measured accessibility',fontsize=18)
    fig.tight_layout(rect=[0,.06,.93,.92],h_pad=3)
    cax=fig.add_axes([.955,.26,.012,.49]);fig.colorbar(stage_art,cax=cax,ticks=times,label='Stage (hpf)')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color='#31556d',marker='o',label='Observed snapshots'),Line2D([0],[0],color='#b36730',ls='--',marker='D',label='Retrospectively fitted snapshots')],loc='upper center',bbox_to_anchor=(.5,.955),ncol=2,frameon=False)
    fig.text(.5,.012,'Left: local views of one joint 24-celltype map; zoom levels differ. Point colors match the actual stages in the optic-cup curves. Screen lengths are qualitative.\nRight: optic cup alone; green/blue shading marks supported fitted opening/closing. Selection used curve shapes, not UMAP paths. Optic-cup 10 hpf is excluded because it has fewer than 20 cells.',ha='center',fontsize=9)
    fig.savefig(out/'13_optic_cup_state_trajectories.png',dpi=160,facecolor='white');fig.savefig(out/'13_optic_cup_state_trajectories.pdf',facecolor='white');plt.close(fig)
    return table


def export_viewer(out,count=8000):
    from plotly.offline import get_plotlyjs
    data,meta=load_sample(out);n=len(data['peak_ids']);rng=np.random.default_rng(20261021)
    with h5py.File(out/'state_pca.h5') as f:_,cts=panel_indices(f['primary_21']['log1p'])
    examples=get_examples(out,data,cts)
    optic_path=out/'optic_cup_state_examples.csv'
    if optic_path.exists(): examples=pd.concat([examples,pd.read_csv(optic_path).rename(columns={'label':'category'})],ignore_index=True)
    forced=np.flatnonzero(~data['train_mask'].astype(bool)&~data['test_mask'].astype(bool))
    ids=np.unique(np.r_[rng.choice(n,min(count,n),replace=False),forced,examples.sample_index.to_numpy(int)])
    bg=np.sort(rng.choice(len(ids),min(2500,len(ids)),replace=False))
    arrays={'observed':data['observed_cpm'][ids],'fitted':data['fitted_cpm'][ids],'support':data['rate_supported'][ids].astype(bool),'condition':data['condition_valid'].astype(bool),'fitvalid':data['fit_valid'][ids].astype(bool),'lof':data['lack_of_fit'][ids],'existing2d':data.get('existing_umap2d',data.get('existingumap2d',data.get('umap2d')))[ids],'existing3d':data.get('existing_umap3d',data.get('existingumap3d',data.get('umap3d')))[ids]}
    panel_defs={}
    with h5py.File(out/'state_pca.h5') as f:
        for panel in PANELS:
            if panel not in f:continue
            g=f[panel]['log1p'];ts,cs=panel_indices(g)
            panel_defs[panel]={'label':PANELS[panel],'stages':ts.tolist(),'features':cs.tolist(),'umap':False}
            for transform in ['log1p','linear']:
                if transform not in f[panel]:continue
                gg=f[panel][transform]
                for source in ['observed','fitted']:arrays[f'{panel}_{transform}_{source}_pca']=gg[f'{source}_pca'][ids,:,:3]
    if (out/'state_umap.h5').exists():
        with h5py.File(out/'state_umap.h5') as f:
            for panel in panel_defs:
                if panel not in f:continue
                g=f[panel]
                if not all(f'{source}_umap{dim}d' in g for source in ['observed','fitted'] for dim in [2,3]):continue
                panel_defs[panel]['umap']=True
                for source in ['observed','fitted']:
                    for dim in [2,3]:arrays[f'{panel}_log1p_{source}_umap{dim}d']=g[f'{source}_umap{dim}d'][ids]
    peaks=[]
    for i in ids:
        p={'peak':str(data['peak_ids'][i]),'sample_index':int(i),'split':'training' if data['train_mask'][i] else 'held out' if data['test_mask'][i] else 'marker extra'}
        for col in ['nearest_gene','linked_gene','associated_gene','leiden_coarse','peak_type_argelaguet']:
            if col in meta:p[col]=str(meta.iloc[i][col])
        peaks.append(p)
    examples_export=[{'label':row.category,'index':int(np.flatnonzero(ids==row.sample_index)[0]),'focal_celltype':str(row.get('focal_celltype','')),'panel':str(row.get('panel','primary_21')) if pd.notna(row.get('panel','primary_21')) else 'primary_21'} for _,row in examples.iterrows()]
    diagnostics=pd.read_csv(out/'pca_preservation.csv');diagnostics=diagnostics[(diagnostics['split']=='test')&(diagnostics.estimator=='observed')&(diagnostics.components.isin([2,3]))]
    umap_diagnostics=json.loads((out/'umap_full_diagnostics.json').read_text()) if (out/'umap_full_diagnostics.json').exists() else {}
    payload={'umap_diagnostics':umap_diagnostics,'pca_diagnostics':diagnostics.to_dict(orient='records'),'n_sample':n,'n_display':len(ids),'stages':data['stages'].tolist(),'celltypes':data['celltypes'].tolist(),'panels':panel_defs,'peaks':peaks,'background':bg.tolist(),'examples':examples_export,'arrays':{k:packed(v) for k,v in arrays.items()}}
    template=(HERE/'state_atlas_viewer_template.html').read_text()
    body=template.replace('__PLOTLY__',get_plotlyjs()).replace('__PAYLOAD__',json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/'))
    path=out/'state_trajectory_explorer.html';path.write_text(body)
    np.save(out/'viewer_sample_indices.npy',ids)
    manifest={'file':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'searchable_peaks':len(ids),'background_peaks':len(bg),'full_sample_peaks':n,'panel_definitions':panel_defs,'examples':examples_export,'array_shapes':{k:list(v.shape) for k,v in arrays.items()},'sampling':'Uniform whole peaks plus all marker extras and deliberately selected illustrative examples; display sample only.'}
    (out/'state_viewer_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:manifest[k] for k in ['file','bytes','searchable_peaks','background_peaks']},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--stage',choices=['viewer','figures','optic-select','optic','all'],default='all');p.add_argument('--display-peaks',type=int,default=8000);args=p.parse_args()
    if args.stage=='optic-select':optic_examples(args.out)
    if args.stage=='optic':optic_examples(args.out,render=True)
    if args.stage in ['figures','all']:render_static(args.out)
    if args.stage in ['viewer','all']:export_viewer(args.out,args.display_peaks)

if __name__=='__main__':main()
