"""Streamline views of saved real peak-atlas directions; no model refitting.

Uses Matplotlib's streamplot, the renderer used by scVelo's embedding stream
function, with an explicit grid that retains unsupported peaks in its support
calculation. Original and depth-matched analyses remain paired sensitivity views.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import matplotlib.patheffects as pe

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'scripts/chrom_velo'))
from streamline_field import build_stream_grid
from streamline_paths import draw_supported_streams

DEFAULT_OUT = HERE / 'outputs/real_specialization_streams_v1'
PARAMS = dict(grid_size=180, n_neighbors=50, bandwidth_cells=.75,
              radius_cells=1.5, proximity_cells=.8, min_neighbors=5,
              min_effective_neighbors=3., min_support_fraction=.2, min_coherence=.35)
DRAW = dict(density=3., linewidth=.8, arrowsize=.8, minlength=.005,
            maxlength=.65, integration_direction='both', broken_streamlines=True)
TIMES = [10, 12, 14, 16, 19, 24]


def log(s): print(time.strftime('%H:%M:%S'), s, flush=True)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def fingerprint(path):
    s=path.stat();return dict(path=str(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def load_fields(path):
    with h5py.File(path, 'r') as f:
        xy=f['obsm/X_umap2d'][:].astype(float)
        fields={}
        for method in ['tau','entropy']:
            fields[f'{method}_structural']=dict(v=f[f'obsm/{method}_up_umap2d'][:],
                mask=f[f'obs/{method}_up_umap2d_display'][:].astype(bool),
                color=f[f'obsm/{method}_specificity'][:,-1],method=method,interval=-1)
            guided=f[f'obsm/{method}_guided_umap2d'][:].reshape(len(xy),5,2)
            rates=f[f'obsm/{method}_rate'][:]
            for t in range(5):
                fields[f'{method}_t{t}']=dict(v=guided[:,t],mask=np.linalg.norm(guided[:,t],axis=1)>0,
                    color=rates[:,t],method=method,interval=t)
    return dict(xy=xy,fields=fields)


def agreement(xy, field, grid):
    ids=np.flatnonzero(field['mask'] & np.isfinite(field['v']).all(axis=1) & (np.linalg.norm(field['v'],axis=1)>0))
    if len(ids)>50000: ids=np.sort(np.random.default_rng(1729).choice(ids,50000,replace=False))
    q=xy[ids][:,::-1]
    uv=np.column_stack([RegularGridInterpolator((grid['y'],grid['x']),grid[k],bounds_error=False,fill_value=np.nan)(q) for k in ['u','v']])
    good=np.isfinite(uv).all(axis=1)&(np.linalg.norm(uv,axis=1)>1e-12)
    native=field['v'][ids][good].astype(float); shown=uv[good]
    cos=np.sum(native*shown,axis=1)/(np.linalg.norm(native,axis=1)*np.linalg.norm(shown,axis=1)) if good.any() else np.array([])
    return dict(sampled_eligible_peaks=int(len(ids)),sampled_peaks_in_supported_grid_cells=int(good.sum()),
        fraction_sample_in_supported_grid_cells=float(good.mean()) if len(good) else 0.,
        median_native_direction_cosine=float(np.median(cos)) if len(cos) else None,
        fraction_native_direction_cosine_ge08=float(np.mean(cos>=.8)) if len(cos) else None,
        fraction_native_direction_reversed=float(np.mean(cos<0)) if len(cos) else None,
        interpretation='Display fidelity among supported interpolation cells; in-sample diagnostic, not held-out biological validation')


class Report:
    def __init__(self,out,models):
        self.out=out;self.models=models;self.cache={};self.rows=[];self.panels=[]
        (out/'grids').mkdir(parents=True,exist_ok=True);(out/'paths').mkdir(exist_ok=True)
        self.background=np.sort(np.random.default_rng(42).choice(len(models['matched']['xy']),120000,replace=False))

    def grid(self,model,key,smooth=.75):
        token=f'{model}_{key}_smooth{smooth:g}'
        if token not in self.cache:
            field=self.models[model]['fields'][key];xy=self.models[model]['xy']
            pars={**PARAMS,'bandwidth_cells':smooth}
            g=build_stream_grid(xy,field['v'],valid=field['mask'],**pars)
            fidelity=agreement(xy,field,g)
            metadata={**g['metadata'],**fidelity,'model':model,'field':key,
                'source_vectors_sha256':hashlib.sha256(np.ascontiguousarray(field['v']).tobytes()).hexdigest(),
                'source_mask_sha256':hashlib.sha256(np.ascontiguousarray(field['mask']).tobytes()).hexdigest()}
            np.savez_compressed(self.out/'grids'/f'{token}.npz',
                **{k:v for k,v in g.items() if k!='metadata'},metadata_json=json.dumps(metadata))
            self.rows.append(metadata);g['token']=token;self.cache[token]=g
            log(f"{token}: {g['metadata']['supported_grid_nodes']:,} supported nodes; native cosine {fidelity['median_native_direction_cosine']}")
        return self.cache[token]

    def panel(self,ax,model,key,title,figure,panel,smooth=.75,rate_limit=None):
        data=self.models[model];xy=data['xy'];field=data['fields'][key]
        g=self.grid(model,key,smooth)
        ids=self.background;values=field['color'];good=np.isfinite(values[ids])
        ax.scatter(*xy[ids[~good]].T,s=.5,c='#c9ced4',alpha=.35,linewidths=0,rasterized=True,zorder=0)
        norm=Normalize(0,1) if field['interval']<0 else TwoSlopeNorm(0,-rate_limit,rate_limit)
        artist=ax.scatter(*xy[ids[good]].T,s=.6,c=values[ids[good]],cmap='viridis' if field['interval']<0 else 'RdBu_r',
            norm=norm,alpha=.58,linewidths=0,rasterized=True,zorder=1)
        streams=draw_supported_streams(ax,g,color='#152c40',zorder=4,**DRAW)
        streams['lines'].set_path_effects([pe.Stroke(linewidth=1.65,foreground='white',alpha=.7),pe.Normal()])
        segments=streams['paths']
        offsets=np.r_[0,np.cumsum([len(s) for s in segments])]
        points=np.concatenate(segments) if segments else np.empty((0,2))
        pathfile=f'paths/{figure}_panel{panel}.npz'
        arrowheads=np.array([patch.get_xy()[:3] for patch in streams['arrows']]).reshape(-1,3,2)
        np.savez_compressed(self.out/pathfile,points=points,offsets=offsets,arrowheads=arrowheads,
            grid_file=f'grids/{g["token"]}.npz',model=model,field=key)
        self.panels.append(dict(figure=figure,panel=panel,model=model,field=key,
            grid_file=f'grids/{g["token"]}.npz',path_file=pathfile,paths=len(segments),vertices=len(points),
            rendering_audit=streams['audit']))
        ax.set(xlim=(g['x'][0],g['x'][-1]),ylim=(g['y'][0],g['y'][-1]),
               aspect='equal',xticks=[],yticks=[],xlabel='Peak UMAP 1',ylabel='Peak UMAP 2',title=title)
        plt.colorbar(artist,ax=ax,shrink=.68,pad=.02,label='Specificity' if field['interval']<0 else 'Specificity change / hpf')
        return len(segments)

    def save(self,fig,name):
        fig.savefig(self.out/f'{name}.png',dpi=190,bbox_inches='tight')
        fig.savefig(self.out/f'{name}.pdf',dpi=160,bbox_inches='tight');plt.close(fig)


def limit(fields,method):
    values=np.concatenate([f['color'][np.isfinite(f['color'])] for k,f in fields.items() if k.startswith(method+'_t')])
    return float(np.quantile(np.abs(values),.98))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    sources={name:HERE/f'outputs/{folder}/real_specialization_flow.h5ad' for name,folder in
        [('original','real_specialization_v1'),('matched','real_specialization_depth_matched_v1')]}
    before={k:fingerprint(v) for k,v in sources.items()}
    log('Loading saved directions and scalars; no velocity inference or UMAP recomputation')
    models={k:load_fields(v) for k,v in sources.items()}
    np.testing.assert_array_equal(models['original']['xy'],models['matched']['xy'])
    r=Report(out,models)
    name='01_matched_structural_streams';fig,axes=plt.subplots(1,2,figsize=(16,8))
    for i,method in enumerate(['tau','entropy']):
        r.panel(axes[i],'matched',f'{method}_structural',f'{method.capitalize()} · 24 hpf specificity',name,i)
    fig.suptitle('Structural reference streamlines · existing real peak UMAP',fontsize=16)
    fig.text(.5,.925,'Chosen orientation toward greater specificity; curves do not represent developmental motion.',ha='center',fontsize=11,color='#844224')
    fig.text(.5,.018,'Depth-matched sensitivity view. All 640,830 peaks inform grid support; 120,000 points shown.\nStreams stop at unsupported or conflicting grid regions. Curve length and width do not encode biological speed.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.91]);r.save(fig,name)
    for model,method,name in [('matched','entropy','02_matched_temporal_streams_entropy'),
                              ('matched','tau','05_matched_temporal_streams_tau'),
                              ('original','entropy','06_original_temporal_streams_entropy')]:
        fig,axes=plt.subplots(2,3,figsize=(18,12));lim=limit(models[model]['fields'],method)
        for t,ax in enumerate(axes.flat[:5]):
            r.panel(ax,model,f'{method}_t{t}',f'{TIMES[t]} → {TIMES[t+1]} hpf',name,t,rate_limit=lim)
        ax=axes.flat[5];ax.axis('off')
        ax.text(.02,.88,('Depth-matched sensitivity view' if model=='matched' else 'Original-depth sensitivity view')+
            '\n\nRed: increasing specificity\nBlue: decreasing specificity\nGray: unsupported scalar estimate\n\nStreams summarize local directions.\nSparse/conflicting regions are masked.\nEach panel is a separate interval field.\n\nCurves are not peak trajectories.\nLength and width are not biological speed.\nOpening/closing is a separate quantity.'+
            ('\n\nTemporal vectors retain the ≥4/5\nthinning-sign display filter.' if model=='matched' else '\n\nStrong depth sensitivity is documented;\ninterpret alongside matched results.'),
            va='top',fontsize=12,transform=ax.transAxes)
        fig.suptitle(f'{method.capitalize()} specificity change · {model} depth · local streamline display',fontsize=16)
        fig.text(.5,.025,'Per-peak directions are unchanged. Stream smoothing is a display choice, not a new velocity estimate.',ha='center',fontsize=10)
        fig.tight_layout(rect=[0,.055,1,.95]);r.save(fig,name)
    name='03_original_vs_matched_streams';fig,axes=plt.subplots(2,2,figsize=(16,15))
    values=np.r_[models['original']['fields']['entropy_t4']['color'],models['matched']['fields']['entropy_t4']['color']]
    lim=float(np.quantile(np.abs(values[np.isfinite(values)]),.98))
    for row,model in enumerate(['original','matched']):
        r.panel(axes[row,0],model,'entropy_structural',f'{model.capitalize()} depth · structural reference',name,2*row)
        r.panel(axes[row,1],model,'entropy_t4',f'{model.capitalize()} depth · 19 → 24 hpf',name,2*row+1,rate_limit=lim)
    fig.suptitle('Paired depth sensitivity views · entropy · identical grid and smoothing settings',fontsize=15)
    fig.text(.5,.025,'Coverage differs because the saved support filters differ. Temporal color limits are shared.\nMatching addresses a sampling mechanism; neither panel is calibrated absolute accessibility or biological ground truth.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.055,1,.955]);r.save(fig,name)
    name='04_stream_smoothing_sensitivity';fig,axes=plt.subplots(1,3,figsize=(21,7))
    for i,smooth in enumerate([.5,.75,1.]):
        r.panel(axes[i],'matched','entropy_structural',f'Gaussian bandwidth: {smooth:g} grid cells'+(' · default' if smooth==.75 else ''),name,i,smooth=smooth)
    fig.suptitle('Display smoothing sensitivity · same saved entropy structural field',fontsize=16)
    fig.text(.5,.025,'No model refitting. Neighbor radius, support filters, streamline density and color limits are fixed.\nThe middle setting is a declared display default; no biological optimization objective was used.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.075,1,.945]);r.save(fig,name)
    name='07_structural_stream_detail';fig,axes=plt.subplots(1,2,figsize=(16,8))
    g=r.grid('matched','entropy_structural')
    valid=g['valid'];cells=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
    components,ncomponents=label(cells)
    if ncomponents:
        sizes=np.bincount(components.ravel());sizes[0]=0
        iy,ix=np.where(components==sizes.argmax())
        xlo,xhi=g['x'][ix.min()],g['x'][ix.max()+1]
        ylo,yhi=g['y'][iy.min()],g['y'][iy.max()+1]
        pad=.12*max(xhi-xlo,yhi-ylo)
        bounds=(xlo-pad,xhi+pad,ylo-pad,yhi+pad)
    else:
        bounds=(g['x'][0],g['x'][-1],g['y'][0],g['y'][-1])
    for i,title in enumerate(['Atlas context','Detail: largest connected supported region']):
        r.panel(axes[i],'matched','entropy_structural',title,name,i)
    axes[0].add_patch(Rectangle((bounds[0],bounds[2]),bounds[1]-bounds[0],bounds[3]-bounds[2],
                               fill=False,ec='#cf6e21',lw=1.4,zorder=6))
    axes[1].set(xlim=bounds[:2],ylim=bounds[2:])
    fig.suptitle('Local streamlines toward greater celltype specificity · 24 hpf reference',fontsize=15)
    fig.text(.5,.025,'Same saved directions, smoothing and paths in both panels. The box is chosen by supported-region size.\nLocal mean directions; these curves do not represent developmental time or opening/closing.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.075,1,.945]);r.save(fig,name)
    after={k:fingerprint(v) for k,v in sources.items()};assert before==after
    pd.DataFrame(r.rows).to_csv(out/'grid_diagnostics.csv',index=False)
    write_json(out/'streamplot_manifest.json',dict(parameters=PARAMS,drawing=DRAW,
        smoothing_comparison=[.5,.75,1.],seeding='Deterministic supported interpolation-cell centres, at most 2000',backend='Matplotlib streamplot candidates (also used by scVelo), clipped to supported cells and checked for local forward direction',
        model_refit=False,source_files=before,source_files_unchanged=True,panels=r.panels,
        fixed_reference_stage_hpf=24,temporal_intervals_hpf=list(zip(TIMES[:-1],TIMES[1:])),
        interpretation='Static streamline integration of locally averaged unit vectors, not a time-evolving trajectory or an absolute velocity',
        limitations='Masks protect visible low-density/unsupported regions. UMAP overlap can still mix distinct high-dimensional neighborhoods; smoothing reduces detail and can change local directions.'))
    log('All streamline figures and numerical grids saved')


if __name__=='__main__':main()
