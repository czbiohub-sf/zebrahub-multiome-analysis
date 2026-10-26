"""Compare original and depth-matched specificity on one fixed peak UMAP.

Both maps use exactly the same sampled, commonly supported peaks and color scale.
The histogram uses every commonly supported peak; the support panel uses the full
atlas. This is a visualization of sensitivity to exposure matching, not biological
confidence or a claim that the matched estimate is unbiased.
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'outputs/real_specialization_depth_matched_v1'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=DEFAULT)
    p.add_argument('--display-peaks', type=int, default=120000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    path = args.output_dir/'real_specialization_flow.h5ad'
    with h5py.File(path, 'r') as f:
        coords = f['obsm/X_umap2d'][:]
        old = f['obsm/entropy_uncorrected_rate'][:]
        matched = f['obsm/entropy_rate'][:]
    assert old.shape == matched.shape == (len(coords),5)
    old_valid, matched_valid = np.isfinite(old), np.isfinite(matched)
    if np.any(matched_valid & ~old_valid):
        raise AssertionError('Supported thinned counts should also pass original-count support')
    common = old_valid[:,-1] & matched_valid[:,-1]
    if not common.any():
        raise ValueError('No shared late-interval support')
    original_late, matched_late = old[common,-1], matched[common,-1]
    scale = float(np.quantile(np.abs(np.r_[original_late,matched_late]),.99))
    scale = max(scale,1e-6)
    sampled = np.sort(np.random.default_rng(args.seed).choice(len(coords),min(len(coords),args.display_peaks),replace=False))
    displayed = sampled[common[sampled]]
    n = len(coords)
    fig = plt.figure(figsize=(14,11),constrained_layout=True)
    gs = fig.add_gridspec(2,2,height_ratios=[1.4,1])
    ax0,ax1 = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])
    for ax, values, title in [(ax0,old[:,-1],'Original-depth entropy rate'),
                              (ax1,matched[:,-1],'Depth-matched entropy rate')]:
        ax.scatter(coords[sampled,0],coords[sampled,1],s=.4,c='#dddddd',alpha=.5,rasterized=True)
        dots=ax.scatter(coords[displayed,0],coords[displayed,1],s=.75,c=values[displayed],
                        cmap='coolwarm',vmin=-scale,vmax=scale,alpha=.8,rasterized=True)
        ax.set_title(title,fontsize=13)
        ax.set_xlabel('Existing peak UMAP 1');ax.set_ylabel('Existing peak UMAP 2')
        ax.set_aspect('equal',adjustable='box')
        ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(dots,ax=[ax0,ax1],shrink=.65,label='Entropy specificity change / hpf; shared color scale')
    hist = fig.add_subplot(gs[1,0])
    limit = float(np.max(np.abs(np.r_[original_late,matched_late])))
    bins=np.linspace(-limit,limit,121)
    hist.hist(original_late,bins=bins,density=True,histtype='step',lw=1.8,color='#b44935',label='Original depth')
    hist.hist(matched_late,bins=bins,density=True,histtype='step',lw=1.8,color='#2677a5',label='Depth matched')
    hist.axvline(0,color='#555555',lw=.8,ls=':')
    hist.axvline(np.median(original_late),color='#b44935',lw=1.3,ls='--')
    hist.axvline(np.median(matched_late),color='#2677a5',lw=1.3,ls='--')
    hist.set_title(f'All {common.sum():,} commonly supported peaks')
    hist.set_xlabel('19→24 hpf entropy specificity rate / hpf');hist.set_ylabel('Density')
    hist.legend(frameon=False,fontsize=9,loc='upper left')
    positive_old=float(np.mean(original_late>0));positive_matched=float(np.mean(matched_late>0))
    hist.text(.98,.95,f'Positive: {positive_old:.1%} → {positive_matched:.1%}\n'
              f'Median: {np.median(original_late):.4f} → {np.median(matched_late):.4f}/hpf',
              transform=hist.transAxes,ha='right',va='top',fontsize=9)
    support = fig.add_subplot(gs[1,1])
    positions=np.arange(5)
    first,second=old_valid.sum(0),matched_valid.sum(0)
    support.bar(positions-.18,first/n*100,width=.36,color='#b44935',label='Original depth')
    support.bar(positions+.18,second/n*100,width=.36,color='#2677a5',label='Depth matched')
    for j,value in enumerate(second):
        support.text(j+.18,value/n*100+1,f'{value/n:.1%}',ha='center',fontsize=8)
    support.set_xticks(positions,['10→12','12→14','14→16','16→19','19→24'])
    support.set_ylim(0,109);support.set_xlabel('Interval, hpf');support.set_ylabel('Supported peaks, % of full atlas')
    support.set_title('Count-support loss after five-thinning gate')
    support.legend(frameon=False,fontsize=9,loc='upper right')
    fig.suptitle('Temporal specificity is sensitive to sequencing depth\n'
                 f'19→24 hpf; same UMAP and common support ({len(displayed):,} colored display points)',fontsize=15)
    fig.supxlabel('Measured CPM and within-celltype opening rates remain unchanged. Matching uses five count thinnings; no biological confidence intervals.\n'
                  'Gray points are outside common late support. Color saturation is the pooled 99th percentile; histogram includes all values.',fontsize=9)
    for ext in ('png','pdf'):
        fig.savefig(args.output_dir/f'07_original_vs_depth_matched_entropy.{ext}',dpi=200)
    plt.close(fig)
    report=dict(source=str(path),n_atlas_peaks=n,interval_hpf=[19,24],display_background_peaks=len(sampled),
                display_common_supported_peaks=len(displayed),common_supported_peaks=int(common.sum()),
                color_scale_symmetric_limit_per_hpf=scale,color_scale_rule='Pooled 99th percentile of absolute original and matched rates on common support',
                histogram_scope='All commonly supported peaks, full observed range',
                original_supported_by_interval=first.tolist(),matched_supported_by_interval=second.tolist(),
                original_late_median_common=float(np.median(original_late)),matched_late_median_common=float(np.median(matched_late)),
                original_late_positive_fraction_common=positive_old,matched_late_positive_fraction_common=positive_matched,
                note='Positive fractions include all finite rates on common support, without an effect-size threshold. Matching is a sensitivity control, not a claim of unbiased biological velocity.')
    (args.output_dir/'depth_comparison_figure.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
