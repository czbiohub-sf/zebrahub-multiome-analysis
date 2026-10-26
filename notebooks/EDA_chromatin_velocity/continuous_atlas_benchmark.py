# %% [markdown]
# # A constitutive core and continuous peak-profile branches
# Each point is a peak; 120 features are celltype x time accessibility.
# UMAP uses measured states only. Simulated specialization is not developmental time.

# %% Imports
import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import trustworthiness

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'scripts'/'chrom_velo'))
from continuous_accessibility import ContinuousConfig,make_continuous,evaluate_hierarchy
from synthetic_accessibility import make_dataset,CELLTYPES
from temporal_velocity import STAGES_HPF,temporal_rates,cis_weights,couple_rates

PALETTE=['#d6a329','#3b82f6','#14b8a6','#ef7b45','#9860ce']


def embed(a,seed=42):
    sc.pp.pca(a,n_comps=30,svd_solver='arpack',random_state=seed)
    sc.pp.neighbors(a,n_neighbors=30,use_rep='X_pca',random_state=seed)
    # PCA initialization comes entirely from measured accessibility. It helps
    # preserve the connected shape without prescribing any branch coordinates.
    for ndim in [3,2]:
        key=f'X_pca_init{ndim}d'
        a.obsm[key]=a.obsm['X_pca'][:,:ndim].copy()
        sc.tl.umap(a,n_components=ndim,min_dist=0.4,random_state=seed,
                   init_pos=key,gamma=0.5,maxiter=1000)
        if ndim==3:
            a.obsm['X_umap3d']=a.obsm['X_umap'].copy()
    a.uns['atlas_settings']=dict(n_neighbors=30,min_dist=0.4,n_pcs=30,seed=seed,
        initialization='PCA',repulsion_strength=0.5,n_epochs=1000,
        input='Measured accessibility only; no manual coordinates or graph bridges')


def connectivity(a,label):
    rows=[]
    for k in [5,15,30]:
        g=kneighbors_graph(a.obsm['X_pca'],k,mode='connectivity',include_self=False,n_jobs=1)
        n,labels=connected_components(g,directed=False)
        sizes=np.bincount(labels)
        rows.append(dict(model=label,n_peaks=len(a),k=k,n_components=n,
                         largest_component_fraction=float(sizes.max()/len(a))))
    return rows


def add_roots_to_discrete(discrete,roots):
    n=len(roots)
    obs=discrete.obs[['true_program','primary_celltype','chrom']].copy()
    extra=roots.obs[['true_program','primary_celltype','chrom']].copy()
    extra.index=['constitutive_'+x for x in extra.index]
    obs=pd.concat([obs,extra])
    a=ad.AnnData(np.concatenate([discrete.X,roots.X]),obs=obs,var=discrete.var.copy())
    a.layers['truth_state']=np.concatenate([discrete.layers['truth_state'],roots.layers['truth_state']])
    a.obs['true_family']=[CELLTYPES.index(c)//5 for c in a.obs.primary_celltype]
    a.obs.iloc[-n:,a.obs.columns.get_loc('true_family')]=-1
    return a


def colors(a):
    family=a.obs.true_family.to_numpy().astype(int)
    return [PALETTE[i+1] for i in family]


def savefig(fig,output,name):
    fig.savefig(output/f'{name}.png',dpi=160,bbox_inches='tight')
    fig.savefig(output/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def plot_comparison(models,output):
    fig=plt.figure(figsize=(16,5.5),layout='constrained')
    interactive=make_subplots(rows=1,cols=len(models),specs=[[{'type':'scene'}]*len(models)],
                              subplot_titles=[title for title,_ in models])
    for panel,(title,a) in enumerate(models):
        xyz=a.obsm['X_umap3d'];col=colors(a)
        core=(a.obs.true_family.to_numpy()==-1)
        ax=fig.add_subplot(1,len(models),panel+1,projection='3d')
        ax.scatter(*xyz[~core].T,c=np.asarray(col)[~core],s=3,alpha=0.65,rasterized=True)
        if core.any():
            ax.scatter(*xyz[core].T,color=PALETTE[0],s=9,alpha=0.9,rasterized=True)
        ax.set(title=f'{title}\n{len(a):,} peaks',xticks=[],yticks=[],zticks=[],xlabel='UMAP 1',ylabel='UMAP 2',zlabel='UMAP 3')
        ax.view_init(elev=20,azim=35)
        for family in sorted(a.obs.true_family.unique()):
            idx=np.flatnonzero(a.obs.true_family.to_numpy()==family)
            label='Always-accessible core' if family==-1 else f'Family {family+1}'
            interactive.add_trace(go.Scatter3d(x=xyz[idx,0],y=xyz[idx,1],z=xyz[idx,2],mode='markers',
                name=label,legendgroup=label,showlegend=panel==1,
                marker=dict(size=4 if family==-1 else 2,color=PALETTE[int(family)+1],opacity=0.85),
                text=a.obs_names[idx],hovertemplate='%{text}<extra>'+label+'</extra>'),row=1,col=panel+1)
    fig.suptitle('Gold = always-accessible peaks. Continuous intermediate profiles connect the branches.\nSame PCA / UMAP settings; each view fitted independently.')
    savefig(fig,output,'01_continuity_comparison')
    interactive.update_layout(title='Each dot is a peak · Gold: constitutive accessibility · Colors: synthetic celltype families',
                               height=700,margin=dict(t=90,b=40,l=0,r=0))
    interactive.update_scenes(aspectmode='data')
    interactive.write_html(output/'continuity_comparison_3d.html',include_plotlyjs=True)


def plot_profiles(a,output):
    s=a.obs.true_specialization.to_numpy()
    family=a.obs.true_family.to_numpy()
    subtype=a.obs.true_celltype_index.to_numpy()
    selection=[0]
    selection += [int(np.argmin(np.where((family==0)&(a.obs.true_program.to_numpy()=='family_bridge'),abs(s-goal),np.inf))) for goal in [0.3,0.7,0.99]]
    selection += [int(np.argmin(np.where((subtype==0)&(a.obs.true_program.to_numpy()=='celltype_branch'),abs(s-goal),np.inf))) for goal in [1.5,1.99]]
    dense=np.linspace(10,24,141)
    means,_=evaluate_hierarchy(a.obs.iloc[selection],dense)
    fig,axes=plt.subplots(1,6,figsize=(18,3.5),layout='constrained')
    for ax,i,values in zip(axes,selection,means):
        for c,color in [(0,PALETTE[1]),(1,PALETTE[4]),(5,PALETTE[3])]:
            ax.plot(dense,values[c],color=color,label=CELLTYPES[c],lw=2)
        ax.set(title=f'{a.obs.true_program.iloc[i].replace("_"," ")}\ns={s[i]:.2f}',ylim=(0,1),xlabel='Time (hpf)')
    axes[0].set_ylabel('True accessibility')
    axes[0].legend(fontsize=8)
    fig.suptitle('Different peaks along the connected profile family; s is specialization, not time')
    savefig(fig,output,'02_root_to_specific_profiles')
    return selection


def attach_rates(a):
    v,mask,_=temporal_rates(a.X,a.var,min_cells=20)
    w=cis_weights(a.obsp['coaccessibility'],a.obs.chrom)
    final,_=couple_rates(v,mask,w,alpha=0.6)
    a.layers['velocity_temporal']=v.astype('float32')
    a.layers['velocity']=final.astype('float32')
    a.layers['velocity_valid']=mask
    a.var['rate_supported']=mask.all(axis=0)


def export_continuous_explorer(a,output):
    def rounded(x):
        return np.round(np.asarray(x,dtype=float),6).tolist()
    # Discrete colorbar with one entry for the core and four for families.
    scale=[]
    for i,color in enumerate(PALETTE):
        scale.extend([[i/5,color],[(i+1)/5,color]])
    annotations=dict(
        family=dict(label='Core and branch family',values=(a.obs.true_family.to_numpy()+1).tolist(),
            min=-0.5,max=4.5,colorscale=scale,tickvals=list(range(5)),
            ticktext=['Always accessible','Family 1','Family 2','Family 3','Family 4'],
            note='Gold peaks are accessible in all celltypes at every stage. Colored branches contain progressively more selective profiles.'),
        specialization=dict(label='Profile specialization',values=a.obs.true_specialization.to_list(),
            min=0,max=2,colorscale='Viridis',
            note='0 = constitutive core; 1 = family-specific; 2 = celltype-specific. This indexes different peak profiles, not developmental time.'),
    )
    peaks=[dict(peak=name,**row.to_dict()) for name,row in a.obs.iterrows()]
    data=dict(title='A continuous peak atlas with an always-accessible core',
        note='Each dot is a peak. Gold marks the constitutive core. Intermediate accessibility profiles connect it to family and celltype branches. Click a peak to inspect its time course.',
        footnote='The connected family of profiles is a simulation assumption. Its constitutive core has zero true temporal velocity and is not evidence of a developmental ancestor. UMAP uses measured profiles only; no coordinates or graph connections were manually drawn.',
        default_dimension=3,default_metric='family',emphasize_core=True,
        annotations=annotations,celltypes=list(CELLTYPES),stages=STAGES_HPF.tolist(),peaks=peaks,
        examples=[int(i) for i in a.uns['explorer_examples']],
        observed=rounded(a.X),umap2d=rounded(a.obsm['X_umap']),umap3d=rounded(a.obsm['X_umap3d']),
        valid=a.var.rate_supported.to_list(),
        rate_limit=float(np.quantile(abs(a.layers['truth_velocity'][a.layers['truth_velocity_valid']]),0.99)))
    for key in ['truth_state','truth_velocity','velocity_temporal','velocity']:
        data[key]=rounded(a.layers[key])
    text=json.dumps(data,separators=(',',':'),allow_nan=False)
    (output/'explorer_data.json').write_text(text)
    html=(HERE/'synthetic_explorer_template.html').read_text().replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',text).replace('__APP__',(HERE/'synthetic_explorer.js').read_text())
    (output/'continuous_peak_explorer.html').write_text(html)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=HERE/'outputs'/'synthetic_20ct_continuous_v2')
    args=parser.parse_args()
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':9})
    old=HERE/'outputs'/'synthetic_20ct_v1'/'synthetic_noisy.h5ad'
    discrete=ad.read_h5ad(old) if old.exists() else make_dataset()
    discrete.obs['true_family']=[CELLTYPES.index(c)//5 for c in discrete.obs.primary_celltype]
    continuous=make_continuous()
    roots=continuous[continuous.obs.true_program=='constitutive_core'].copy()
    plus=add_roots_to_discrete(discrete,roots)
    models=[('Original discrete programs',discrete),('Add only constitutive peaks',plus),('Constitutive core + continuous bridges',continuous)]
    metrics=[]
    for title,a in models:
        print(f'Embedding {title}: {a.shape}',flush=True)
        embed(a)
        metrics.extend(connectivity(a,title))
    # Paired removal control: preserve the full-model PCA feature basis, remove
    # family bridges, and rebuild kNN. No drawing/layout adjustments are made.
    without=continuous[continuous.obs.true_program!='family_bridge'].copy()
    metrics.extend(connectivity(without,'Continuous model with family bridges removed'))
    metrics=pd.DataFrame(metrics)
    metrics.to_csv(output/'connectivity_metrics.csv',index=False)
    print(metrics.to_string(index=False),flush=True)
    attach_rates(continuous)
    sample=np.random.default_rng(42).choice(len(continuous),1000,replace=False)
    trust=float(trustworthiness(continuous.obsm['X_pca'][sample],continuous.obsm['X_umap3d'][sample],n_neighbors=15))
    plot_comparison(models,output)
    examples=plot_profiles(continuous,output)
    continuous.uns['explorer_examples']=examples
    continuous.write_h5ad(output/'synthetic_continuous.h5ad',compression='gzip')
    export_continuous_explorer(continuous,output)
    plus.write_h5ad(output/'discrete_plus_constitutive.h5ad',compression='gzip')
    continuous.obs.to_csv(output/'peak_metadata.csv')
    continuous.var.to_csv(output/'conditions.csv')
    root=continuous.obs.true_program.to_numpy()=='constitutive_core'
    report=dict(shape=list(continuous.shape),n_constitutive=int(root.sum()),n_family_bridges=int((continuous.obs.true_program=='family_bridge').sum()),
        n_celltype_branches=int((continuous.obs.true_program=='celltype_branch').sum()),
        root_max_abs_true_rate=float(abs(continuous.layers['truth_velocity'][root]).max()),
        sampled_3d_trustworthiness_k15=trust,atlas_settings=continuous.uns['atlas_settings'],
        root_meaning=continuous.uns['synthetic']['root_meaning'])
    (output/'run_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    print(f'Outputs: {output.resolve()}',flush=True)


if __name__=='__main__':
    main()
