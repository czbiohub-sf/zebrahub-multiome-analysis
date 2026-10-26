#!/usr/bin/env python3
"""Known-state controls for the joint peak-stage accessibility-state atlas.

The synthetic atlas is an independent geometry/measurement sanity check. It is
not evidence that real peaks share these trajectories or calibrated uncertainty.
"""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import h5py
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / 'outputs/state_trajectory_atlas_v1'
TIMES = np.array([10, 12, 14, 16, 19, 24.], dtype=float)
GROUPS = ['stable_broad', 'uniform_opening', 'uniform_closing',
          'increasing_specificity', 'transient_peak', 'transient_trough']
COLORS = ['#62676b', '#0072b2', '#d55e00', '#009e73', '#cc79a7', '#b79c00']

def specificity(x):
    """One minus normalized Shannon entropy; separate from accessibility level."""
    p = x / np.maximum(x.sum(axis=-1, keepdims=True), 1e-300)
    terms = np.zeros_like(p)
    np.log(p, out=terms, where=p > 0)
    return 1 + (p * terms).sum(axis=-1) / np.log(x.shape[-1])

def generate(n_each=120, features=21, seed=20261020):
    rng = np.random.default_rng(seed)
    n = n_each * len(GROUPS)
    true = np.empty((n, 6, features))
    labels = np.repeat(np.arange(len(GROUPS)), n_each)
    progress = (TIMES-TIMES[0])/(TIMES[-1]-TIMES[0])
    pulse = np.exp(-.5*((TIMES-16)/2.8)**2)
    for i, group in enumerate(labels):
        baseline = rng.uniform(1.8, 3.2)
        amplitude = rng.uniform(1.5, 3.)
        if group == 0:
            state = np.full((6, features), baseline)
        elif group == 1:
            state = np.tile((baseline + amplitude*progress)[:, None], (1, features))
        elif group == 2:
            state = np.tile((baseline + amplitude*(1-progress))[:, None], (1, features))
        elif group == 3:
            # Constant mean accessibility while the profile becomes selective.
            target = np.full(features, .1)
            favored = np.arange(3)  # One interpretable program, not random branch labels.
            target[favored] = (features-.1*(features-3))/3
            state = baseline*((1-progress[:, None])+progress[:, None]*target)
        elif group == 4:
            state = np.tile((baseline + amplitude*pulse)[:, None], (1, features))
        else:
            state = np.tile((baseline + amplitude*(1-pulse))[:, None], (1, features))
        true[i] = state
    # Known stage-specific effective depths. Normalize each count by its own
    # depth; remaining motion in stable peaks is conditional sampling noise.
    depth = 12*np.array([.5, .8, 1., .7, 1.3, 1.8])
    count = rng.poisson(true*depth[None, :, None])
    observed = count/depth[None, :, None]
    training = np.ones(n, bool)
    for g in range(len(GROUPS)):
        ix = np.flatnonzero(labels == g)
        training[rng.choice(ix, size=n_each//5, replace=False)] = False
    return true, observed, labels, training, depth

def main(out=OUT):
    out.mkdir(exist_ok=True, parents=True)
    true, observed, groups, training, depth = generate()
    # Learn a single rotation over the entire developmental window, and apply
    # that frozen rotation to exact truth and independent noise realizations.
    pca = PCA(n_components=true.shape[-1], svd_solver='full')
    pca.fit(np.log1p(observed[training]).reshape(-1, true.shape[-1]))
    true_pc = pca.transform(np.log1p(true).reshape(-1, true.shape[-1])).reshape(true.shape)
    observed_pc = pca.transform(np.log1p(observed).reshape(-1, true.shape[-1])).reshape(true.shape)
    hi_delta = np.diff(np.log1p(true), axis=1)
    pc_delta = np.diff(true_pc, axis=1)
    err = np.max(np.abs(np.linalg.norm(hi_delta, axis=-1)-np.linalg.norm(pc_delta, axis=-1)))
    stable = groups == 0
    stable_motion = np.linalg.norm(np.diff(true_pc[stable], axis=1), axis=-1)
    # Tests distinguish opening from increasing specificity.
    mean_true = true.mean(axis=-1)
    spec_true = specificity(true)
    uniform = (groups == 1) | (groups == 2)
    specific = groups == 3
    assertions = {
        'full_pc_displacement_max_abs_error': float(err),
        'true_stable_max_displacement': float(stable_motion.max()),
        'uniform_open_close_specificity_max_abs_change': float(np.abs(np.diff(spec_true[uniform], axis=1)).max()),
        'specificity_program_mean_accessibility_max_abs_change': float(np.abs(np.diff(mean_true[specific], axis=1)).max()),
        'specificity_program_median_entropy_specificity_increase': float(np.median(spec_true[specific, -1]-spec_true[specific, 0])),
        'stable_observed_median_interval_displacement': float(np.median(np.linalg.norm(np.diff(observed_pc[stable], axis=1), axis=-1))),
        'training_peaks': int(training.sum()), 'held_out_peaks': int((~training).sum()),
        'total_peaks': int(len(true)), 'n_features': int(true.shape[-1]),
        'stages': TIMES.tolist(), 'stage_effective_depth': depth.tolist(),
        'display_pc3_training_variance_fraction': float(pca.explained_variance_ratio_[:3].sum()),
    }
    assert err < 1e-10 and stable_motion.max() < 1e-10
    assert assertions['uniform_open_close_specificity_max_abs_change'] < 1e-12
    assert assertions['specificity_program_mean_accessibility_max_abs_change'] < 1e-12
    assert assertions['specificity_program_median_entropy_specificity_increase'] > .3
    rows = []
    for g, name in enumerate(GROUPS):
        mask = groups == g
        for ti, t in enumerate(TIMES):
            rows.append(dict(program=name, stage_hpf=t,
                             mean_true_accessibility=float(mean_true[mask, ti].mean()),
                             mean_observed_accessibility=float(observed[mask, ti].mean()),
                             mean_true_specificity=float(spec_true[mask, ti].mean()),
                             mean_observed_specificity=float(specificity(observed[mask, ti]).mean())))
    pd.DataFrame(rows).to_csv(out/'synthetic_program_stage_summary.csv', index=False)
    assertions['interpretation'] = ('Exact stable states stay stationary; count sampling alone produces nonzero observed trajectories. '
        'Uniform opening/closing changes level without specificity; increasing specificity can leave mean level unchanged. '
        'These are known synthetic controls, not uncertainty estimates for real data.')
    (out/'synthetic_controls_validation.json').write_text(json.dumps(assertions, indent=2))
    with h5py.File(out/'synthetic_states.h5', 'w') as f:
        for name, arr in [('true_accessibility', true), ('observed_accessibility', observed), ('true_pca', true_pc),
                          ('observed_pca', observed_pc), ('training', training), ('program', groups), ('stages', TIMES)]:
            f.create_dataset(name, data=arr, compression='gzip')
        f.create_dataset('program_names', data=np.asarray(GROUPS, dtype=h5py.string_dtype()))
        f.attrs['representation'] = 'Joint PCA of log1p(observed), all 21 components, fitted on whole training peaks.'
        f.attrs['seed'] = 20261020

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(15, 10.5))
    axes = [fig.add_subplot(2, 3, i+1) for i in [0, 1, 2, 3, 4]]
    ax3 = fig.add_subplot(2, 3, 6, projection='3d')
    ax_level, ax_spec, ax_motion, ax_path, ax_noise = axes
    for g, (name, color) in enumerate(zip(GROUPS, COLORS)):
        label = name.replace('_', ' ')
        mask = groups == g
        ax_level.plot(TIMES, mean_true[mask].mean(axis=0), '-o', ms=3, color=color, label=label)
        ax_spec.plot(TIMES, spec_true[mask].mean(axis=0), '-o', ms=3, color=color)
        med = np.median(np.linalg.norm(np.diff(observed_pc[mask], axis=1), axis=-1)/np.diff(TIMES), axis=0)
        truth = np.median(np.linalg.norm(pc_delta[mask], axis=-1)/np.diff(TIMES), axis=0)
        ax_motion.plot((TIMES[:-1]+TIMES[1:])/2, truth, '-o', color=color, ms=3)
        ax_motion.plot((TIMES[:-1]+TIMES[1:])/2, med, '--', color=color, alpha=.65)
        # One held-out example per mode, actual identity-linked trajectory.
        ix = np.flatnonzero(mask & ~training)[0]
        path = true_pc[ix]
        ax_path.plot(path[:, 0], path[:, 1], '-o', color=color, ms=4)
        for a, b in zip(path[:-1], path[1:]):
            if np.linalg.norm(b[:2]-a[:2]) > 1e-8:
                ax_path.annotate('', b[:2], a[:2], arrowprops={'arrowstyle':'->', 'color':color, 'lw':1})
        ax3.plot(path[:, 0], path[:, 1], path[:, 2], '-o', color=color, ms=3)
    ix = np.flatnonzero(stable & ~training)[0]
    ax_noise.plot(observed_pc[ix, :, 0], observed_pc[ix, :, 1], '-o', color=COLORS[0], label='Observed counts')
    ax_noise.scatter(true_pc[ix, 0, 0], true_pc[ix, 0, 1], s=90, marker='*', color='#009e73', zorder=3, label='Exact stable state')
    for k, t in enumerate(TIMES):
        ax_noise.annotate(str(int(t)), observed_pc[ix, k, :2], xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax_level.set(title='A. True accessibility level', xlabel='Stage (hpf)', ylabel='Mean across celltypes (simulation units)')
    ax_spec.set(title='B. True celltype specificity', xlabel='Stage (hpf)', ylabel='1 − normalized Shannon entropy', ylim=(-.03, 1))
    ax_motion.set(title='C. True and sampling-noisy movement', xlabel='Interval midpoint (hpf)', ylabel='Full-PC displacement / hour')
    ax_motion.text(.02, .97, 'Solid: truth; dashed: observed-count median', transform=ax_motion.transAxes, va='top', fontsize=8)
    ax_path.set(title='D. Same-peak paths in shared PCA', xlabel='PC 1', ylabel='PC 2')
    ax_noise.set(title='E. A stable peak can appear to move', xlabel='PC 1', ylabel='PC 2')
    ax_noise.legend(fontsize=8, loc='best')
    ax3.set(title='F. Same paths in shared 3D PCA', xlabel='PC 1', ylabel='PC 2', zlabel='PC 3')
    handles, labels = ax_level.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, .075), fontsize=9)
    fig.suptitle('Known-state controls: accessibility change, specialization, and sampling noise', fontsize=15, y=.98)
    fig.text(.04, .025, 'Six stages; 21 celltype features; 720 peaks. One joint PCA fitted on observed training states; held-out examples shown.\n'
             'All 21 PCs preserve transformed distances. Exact stable states have zero motion; noisy observations do not.\n'
             'The narrowing of a profile and its overall accessibility level are separate quantities. Synthetic paths are not evidence of real biological flow.', fontsize=10)
    fig.subplots_adjust(left=.065, right=.97, top=.9, bottom=.22, wspace=.34, hspace=.36)
    fig.savefig(out/'00_synthetic_state_controls.png', dpi=180)
    fig.savefig(out/'00_synthetic_state_controls.pdf')
    plt.close(fig)
    print(json.dumps(assertions, indent=2))


def umap_controls(out=OUT):
    """Small independent joint-UMAP control, deduplicated in state coordinates."""
    import umap
    from scipy.stats import spearmanr
    with h5py.File(out/'synthetic_states.h5', 'r') as f:
        true_pc=f['true_pca'][:]
        obs_pc=f['observed_pca'][:]
        training=f['training'][:]
        programs=f['program'][:]
    n, nt, nc = true_pc.shape
    train_states=np.unique(obs_pc[training].reshape(-1,nc), axis=0)
    query, inverse=np.unique(np.concatenate([obs_pc.reshape(-1,nc),true_pc.reshape(-1,nc)]), axis=0, return_inverse=True)
    lookup={row.tobytes():i for i,row in enumerate(train_states)}
    match=np.array([lookup.get(row.tobytes(),-1) for row in query])
    validation={}
    coords={}
    for dim in [2,3]:
        model=umap.UMAP(n_components=dim, n_neighbors=30, min_dist=.3,
                        metric='euclidean', random_state=20261020,
                        transform_seed=20261020, n_jobs=1)
        train_emb=model.fit_transform(train_states)
        qe=np.empty((len(query),dim), dtype=np.float32)
        found=match>=0
        qe[found]=train_emb[match[found]]
        qe[~found]=model.transform(query[~found])
        expanded=qe[inverse]
        obs=expanded[:n*nt].reshape(n,nt,dim)
        true=expanded[n*nt:].reshape(n,nt,dim)
        coords[dim]=(obs,true)
        stable_move=np.linalg.norm(np.diff(true[programs==0],axis=1),axis=-1)
        assert stable_move.max()==0
        hi=np.linalg.norm(np.diff(true_pc,axis=1),axis=-1)
        lo=np.linalg.norm(np.diff(true,axis=1),axis=-1)
        ok=np.isfinite(hi)&(hi>1e-9)
        validation[f'umap{dim}d']={
            'exact_stable_max_displacement':float(stable_move.max()),
            'nonzero_truth_interval_length_spearman':float(spearmanr(hi[ok],lo[ok]).statistic),
            'identical_states_share_one_coordinate':True,
            'n_distinct_training_states':len(train_states),
            'n_distinct_joint_query_states':len(query)}
    validation['interpretation']='Exact duplicates receive exactly identical coordinates by a state-vector cache. UMAP interval lengths can distort true state distances; coordinates are visualization, not calibrated rate units.'
    (out/'synthetic_umap_validation.json').write_text(json.dumps(validation,indent=2))
    with h5py.File(out/'synthetic_states.h5','a') as f:
        for dim,(obs,true) in coords.items():
            for name,arr in [(f'observed_umap{dim}d',obs),(f'true_umap{dim}d',true)]:
                if name in f: del f[name]
                f.create_dataset(name,data=arr,compression='gzip')
    fig=plt.figure(figsize=(13,6))
    a=fig.add_subplot(1,2,1)
    b=fig.add_subplot(1,2,2,projection='3d')
    for g,(name,color) in enumerate(zip(GROUPS,COLORS)):
        ix=np.flatnonzero((programs==g)&~training)[0]
        for ax,dim in [(a,2),(b,3)]:
            xy=coords[dim][1][ix]
            if dim==2:
                ax.plot(*xy.T,'-o',color=color,ms=4,label=name.replace('_',' '))
                for aa,bb in zip(xy[:-1],xy[1:]):
                    if np.linalg.norm(bb-aa)>1e-8:
                        ax.annotate('',bb,aa,arrowprops={'arrowstyle':'->','color':color,'lw':1})
            else: ax.plot(*xy.T,'-o',color=color,ms=4)
    a.set(title='Shared 2D UMAP: known same-peak paths',xlabel='UMAP 1',ylabel='UMAP 2')
    b.set(title='Shared 3D UMAP',xlabel='UMAP 1',ylabel='UMAP 2',zlabel='UMAP 3')
    handles,labels=a.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.10))
    fig.text(.04,.035,'Independent synthetic atlas, fitted on noisy training states; exact true held-out paths shown. Stable states have zero plotted motion.\n'
             'Uniform opening and closing may traverse the same accessibility states in opposite temporal directions. UMAP lengths are not rates.',fontsize=10)
    fig.subplots_adjust(top=.9,bottom=.27,wspace=.25)
    fig.savefig(out/'00b_synthetic_joint_umap_controls.png',dpi=180)
    fig.savefig(out/'00b_synthetic_joint_umap_controls.pdf')
    plt.close(fig)
    print(json.dumps(validation,indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--umap-only', action='store_true')
    args=parser.parse_args()
    if args.umap_only: umap_controls(args.out)
    else: main(args.out)
