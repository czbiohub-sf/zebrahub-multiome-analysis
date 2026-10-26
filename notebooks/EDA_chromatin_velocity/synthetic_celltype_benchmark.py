# %% [markdown]
# # A peak atlas with 20 celltypes x 6 developmental stages
# Each observation is one peak. No single-cell matrices are simulated.
# This experiment validates rates within celltypes before interpreting map arrows.
# Run: python notebooks/EDA_chromatin_velocity/synthetic_celltype_benchmark.py

# %% Imports
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'scripts' / 'chrom_velo'))
from synthetic_accessibility import SyntheticConfig, PROGRAMS, CELLTYPES, make_dataset, evaluate_truth
from temporal_velocity import STAGES_HPF, temporal_rates, cis_weights, couple_rates


METHODS = ['temporal', 'coupled_imperfect', 'coupled_oracle', 'coupled_rewired', 'pure_coupled_imperfect']


def estimate(a):
    v, mask, _ = temporal_rates(a.X, a.var, min_cells=20)
    graph = a.obsp['coaccessibility']
    rng = np.random.default_rng(a.uns['synthetic']['seed'] + 902)
    perm = np.arange(len(a))
    for c in a.obs.chrom.unique():
        idx = np.flatnonzero(a.obs.chrom.to_numpy() == c)
        perm[idx] = rng.permutation(idx)
    graphs = dict(imperfect=graph, oracle=a.obsp['true_cis_candidates'], rewired=graph[perm][:,perm])
    estimates = {'temporal': v}
    for name, g in graphs.items():
        w = cis_weights(g, a.obs.chrom, threshold=0.35)
        estimates[f'coupled_{name}'], _ = couple_rates(v, mask, w, alpha=0.6)
        if name == 'imperfect':
            estimates['pure_coupled_imperfect'], _ = couple_rates(v, mask, w, alpha=0.)
    a.layers['velocity_temporal'] = v.astype('float32')
    a.layers['velocity'] = estimates['coupled_imperfect'].astype('float32')
    a.layers['velocity_valid'] = mask
    a.var['rate_supported'] = mask.all(axis=0)
    a.obs['cis_degree_above_035'] = np.asarray((graph > 0.35).sum(1)).ravel()
    return estimates, mask


def score_rates(a, estimates, mask):
    truth = a.layers['truth_velocity']
    rows = []
    strata = [('all', 'all', np.ones(a.shape, dtype=bool))]
    strata += [('program', p, np.broadcast_to((a.obs.true_program.to_numpy() == p)[:,None],a.shape)) for p in PROGRAMS]
    strata += [('celltype', c, np.broadcast_to((a.var.celltype.to_numpy() == c)[None,:],a.shape)) for c in CELLTYPES]
    for method, v in estimates.items():
        for kind, label, subset in strata:
            valid = mask & subset
            dynamic = valid & (abs(truth) > 0.01)
            static = valid & (abs(truth) < 1e-7)
            def mean(values, selected):
                return float(np.mean(values[selected])) if selected.any() else np.nan
            rows.append(dict(seed=a.uns['synthetic']['seed'], scenario=a.uns['synthetic']['scenario'],
                method=method, group_kind=kind, group=label,
                n_valid=int(valid.sum()), n_dynamic=int(dynamic.sum()), n_static=int(static.sum()),
                rmse=float(np.sqrt(mean((v-truth)**2,valid))),
                dynamic_rmse=float(np.sqrt(mean((v-truth)**2,dynamic))),
                dynamic_sign_accuracy=mean(np.sign(v)==np.sign(truth),dynamic),
                static_false_change_fraction=mean(abs(v)>0.01,static)))
    return rows


def atlas(a, seed):
    # Only measured accessibility enters PCA and neighborhoods. True labels and
    # regulatory graph are excluded. Both embeddings share the measured-state kNN.
    sc.pp.pca(a, n_comps=30, svd_solver='arpack', random_state=seed)
    sc.pp.neighbors(a, n_neighbors=30, use_rep='X_pca', random_state=seed)
    sc.tl.umap(a, n_components=2, min_dist=0.4, random_state=seed)
    xy = a.obsm['X_umap'].copy()
    sc.tl.umap(a, n_components=3, min_dist=0.4, random_state=seed)
    a.obsm['X_umap3d'] = a.obsm['X_umap'].copy()
    a.obsm['X_umap'] = xy


def savefig(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def example_peaks(a):
    # One explicit example per program; prefer family-shared when available.
    result = []
    for p in PROGRAMS:
        indices = np.flatnonzero(a.obs.true_program.to_numpy() == p)
        family = indices[a.obs.true_scope.iloc[indices].to_numpy() == 'family']
        result.append(int(family[0] if len(family) else indices[0]))
    return result


def figures(a, metrics, output):
    example = example_peaks(a)
    dense = np.linspace(10.,24.,141)
    latent, _ = evaluate_truth(a.obs, a.obsm['true_celltype_loadings'], a.obsm['true_celltype_lags_hpf'],
                               a.obsm['true_celltype_baseline'], dense)
    fig, axes = plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for ax, i in zip(axes.flat, example):
        row = a.obs.iloc[i]
        primary = CELLTYPES.index(row.primary_celltype)
        partner = CELLTYPES.index(row.partner_celltype)
        for c in range(20):
            ax.plot(dense,latent[i,c],color='#cbd5e1',lw=0.8,zorder=1)
        for c, color in [(primary,'#2563eb'), (partner,'#d97706')]:
            ax.plot(dense,latent[i,c],color=color,lw=2,label=CELLTYPES[c],zorder=2)
            ax.scatter(STAGES_HPF,a.X[i,c*6:(c+1)*6],color=color,s=12,zorder=3)
        ax.set(title=row.true_program.replace('_',' '),xlabel='Time (hpf)',ylabel='Accessibility',ylim=(0,1))
        ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Each panel = one peak; lines = celltypes; points = measured means (3 sampling replicates)')
    savefig(fig,output,'01_example_peak_trajectories')
    del latent

    # Sample the entire 120-column matrix without hiding its celltype structure.
    groups = [np.flatnonzero(a.obs.true_program.to_numpy()==p)[::max(1,int((a.obs.true_program==p).sum())//25)][:25] for p in PROGRAMS]
    indices = np.concatenate(groups)
    centers = np.cumsum([0] + [len(g) for g in groups[:-1]]) + np.array([len(g) for g in groups])/2 - 0.5
    fig, axes = plt.subplots(2,1,figsize=(16,8),constrained_layout=True)
    for ax, matrix, label in zip(axes,[a.layers['truth_state'],a.X],['Noise-free truth','Measured accessibility']):
        image=ax.imshow(matrix[indices],aspect='auto',interpolation='nearest',cmap='viridis',vmin=0,vmax=1)
        ax.set_xticks(np.arange(20)*6+2.5,CELLTYPES,rotation=60,fontsize=8)
        ax.set_yticks(centers,[p.replace('_',' ') for p in PROGRAMS],fontsize=9)
        ax.set(title=label,xlabel='Celltype blocks; each contains 10, 12, 14, 16, 19, 24 hpf',ylabel='Peaks grouped by known program')
        for x in np.arange(1,20)*6-0.5: ax.axvline(x,color='white',lw=0.4,alpha=0.6)
    fig.colorbar(image,ax=axes,label='Accessibility score',shrink=0.7)
    savefig(fig,output,'02_peaks_by_120_conditions')

    # The same peak atlas, conditioned on different celltype/interval queries.
    coords = a.obsm['X_umap']
    i = example[5]
    primary,partner = a.obs.primary_celltype.iloc[i],a.obs.partner_celltype.iloc[i]
    cells=[primary,partner]
    fig,axes=plt.subplots(2,5,figsize=(17,7),constrained_layout=True)
    bound=float(np.quantile(abs(a.layers['truth_velocity'][a.layers['truth_velocity_valid']]),0.99))
    for r,c in enumerate(cells):
        for j,t in enumerate(STAGES_HPF[:-1]):
            k=a.var_names.get_loc(f'{c}_{t:g}hpf')
            image=axes[r,j].scatter(*coords.T,c=a.layers['truth_velocity'][:,k],s=3,cmap='coolwarm',vmin=-bound,vmax=bound,rasterized=True)
            axes[r,j].set(title=f'{c} · {t:g}→{STAGES_HPF[j+1]:g} hpf',xticks=[],yticks=[])
    fig.colorbar(image,ax=axes,label='True accessibility change / hpf',shrink=0.8)
    fig.suptitle('Each dot = one peak. Fixed positions, different celltype-specific opening/closing fields.')
    savefig(fig,output,'03_celltype_time_rate_maps')

    summary=metrics[(metrics.group_kind=='all') & (metrics.scenario=='noisy')]
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,col,title in zip(axes,['dynamic_rmse','dynamic_sign_accuracy','static_false_change_fraction'],
                           ['Error on truly dynamic entries','Opening/closing sign accuracy','False changes in truly static entries']):
        g=summary.groupby('method')[col].agg(['mean','std']).reindex(METHODS)
        ax.bar(range(len(g)),g['mean'],yerr=g['std'].fillna(0),color=['#64748b','#c04a32','#2563eb','#d97706','#8b5cf6'],capsize=3)
        ax.set_xticks(range(len(g)),['Temporal','Mixed cis','Oracle cis','Rewired cis','Pure cis'],rotation=35,ha='right')
        ax.set(title=title)
    savefig(fig,output,'04_rate_recovery_benchmark')


def export_explorer(a, output):
    def rounded(array):
        return np.round(np.asarray(array,dtype=float),6).tolist()
    peaks=[]
    for i,(name,row) in enumerate(a.obs.iterrows()):
        peaks.append(dict(peak=name,**row.to_dict(),
            loadings=a.obsm['true_celltype_loadings'][i].astype(float).tolist(),
            lags=a.obsm['true_celltype_lags_hpf'][i].astype(float).tolist(),
            baseline=a.obsm['true_celltype_baseline'][i].astype(float).tolist(),
            partner_index=CELLTYPES.index(row.partner_celltype)))
    data=dict(celltypes=list(CELLTYPES),stages=STAGES_HPF.tolist(),peaks=peaks,
              examples=example_peaks(a),observed=rounded(a.X),
              umap2d=rounded(a.obsm['X_umap']),umap3d=rounded(a.obsm['X_umap3d']),
              valid=a.var.rate_supported.to_list(),
              rate_limit=float(np.quantile(abs(a.layers['truth_velocity'][a.layers['truth_velocity_valid']]),0.99)))
    for key in ['truth_state','truth_velocity','velocity_temporal','velocity']:
        data[key]=rounded(a.layers[key])
    text=json.dumps(data,separators=(',',':'),allow_nan=False)
    # Separate JSON supports validation and alternative viewers; HTML is standalone.
    (output/'explorer_data.json').write_text(text)
    html=(HERE/'synthetic_explorer_template.html').read_text().replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',text).replace('__APP__',(HERE/'synthetic_explorer.js').read_text())
    (output/'synthetic_celltypes_explorer.html').write_text(html)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=HERE/'outputs'/'synthetic_20ct_v1')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--benchmark-seeds',type=int,default=3)
    parser.add_argument('--n-modules',type=int,default=80)
    parser.add_argument('--peaks-per-module',type=int,default=30)
    args=parser.parse_args()
    if args.benchmark_seeds<1:
        parser.error('--benchmark-seeds must be positive')
    config=SyntheticConfig(seed=args.seed,n_modules=args.n_modules,peaks_per_module=args.peaks_per_module)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':10})
    all_scores=[]
    main_noisy=None
    scenario_summaries={}
    for seed in range(args.seed,args.seed+args.benchmark_seeds):
        for scenario in ['clean','noisy','stress']:
            print(f'Generating seed={seed} scenario={scenario}',flush=True)
            a=make_dataset(replace(config,seed=seed),scenario)
            estimates,mask=estimate(a)
            all_scores.extend(score_rates(a,estimates,mask))
            if seed==args.seed:
                scenario_summaries[scenario]=dict(shape=list(a.shape),valid_rate_columns=int(mask.all(0).sum()),
                    missing_conditions=a.uns['synthetic']['missing_conditions'],
                    low_count_conditions=a.var_names[a.var.n_cells<20].tolist(),
                    graph=a.uns['synthetic']['graph'])
                if scenario=='noisy':
                    main_noisy=a
                else:
                    a.write_h5ad(args.output_dir/f'synthetic_{scenario}.h5ad',compression='gzip')
            del estimates
    metrics=pd.DataFrame(all_scores)
    metrics.to_csv(args.output_dir/'validation_metrics.csv',index=False)
    print('Computing measured-state PCA, neighbors, 2D/3D UMAP; truth is excluded.',flush=True)
    atlas(main_noisy,args.seed)
    main_noisy.write_h5ad(args.output_dir/'synthetic_noisy.h5ad',compression='gzip')
    main_noisy.var.to_csv(args.output_dir/'conditions.csv')
    main_noisy.obs.to_csv(args.output_dir/'peak_metadata.csv')
    figures(main_noisy,metrics,args.output_dir)
    export_explorer(main_noisy,args.output_dir)
    aggregate=metrics[metrics.group_kind=='all'].groupby(['scenario','method'])[['rmse','dynamic_rmse','dynamic_sign_accuracy','static_false_change_fraction']].mean()
    aggregate.to_csv(args.output_dir/'benchmark_summary.csv')
    report=dict(seed=args.seed,n_seeds=args.benchmark_seeds,scenarios=scenario_summaries,
        programs=list(PROGRAMS),family_structure='4 synthetic families of 5 celltypes',
        measurement_model='Independent beta sampling replicates; concentration = n_cells * depth * precision',
        truth_and_projection='Rates evaluated against interval-average truth; no ground-truth peak-to-peak motion assumed',
        versions=dict(numpy=np.__version__,anndata=ad.__version__,scanpy=sc.__version__))
    (args.output_dir/'run_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(aggregate.round(5).to_string(),flush=True)
    print(f'Outputs: {args.output_dir.resolve()}',flush=True)


if __name__=='__main__':
    main()
