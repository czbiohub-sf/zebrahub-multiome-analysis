"""Scientific display layer for fitted peak-accessibility trajectories.

This module does not fit trajectories, construct neighbors, or orient a root.
``make_trajectory_report(data, output_dir)`` accepts NumPy arrays or sliceable
HDF5 datasets; large accessibility cubes are read only for selected curves.
Existing UMAP coordinates and supplied masks are used without recomputation.
Observed CPM must retain measured low-support values; only truly absent values
should be NaN. condition_valid controls derived celltype rate display, not raw
curve visibility. Fitted states can exist even when fit_valid is false.

Required data keys
------------------
stages: (T,), celltypes: (C,), peak_ids: (N,), umap2d: (N, 2)
observed_cpm, fitted_cpm: (N, C, T); condition_valid: (C, T)
modes: integer (N, C); mode_labels: {integer: label}
fit_valid: (N, C), dynamic count/observation support, not fit quality
net_rate: (N, T-1), the signed fitted mean CPM/hpf rate across a fixed CT panel
fitted_vectors2d, raw_vectors2d: (N, T-1, 2)
fitted_mask2d, raw_mask2d: bool (N, T-1)

Optional keys include umap3d, fitted_vectors3d/raw_vectors3d and their masks,
obs (DataFrame), broad_reference (N,), fit_quality (N,C), selected_celltypes,
genomic_context (N,), context_vectors2d/context_mask2d, context_label,
notes (list[str]), projection_celltypes, and rate_label. The context comparison
is omitted explicitly when no genomic context or coupled field is provided.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
import textwrap

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from real_specialization_report import choose_arrows, clean


_BASE_COLORS = ['#b66226', '#2c73a2', '#8d5ca6', '#238f85', '#b75269', '#91813c', '#57735d', '#6d6d6d']


def _array(value, dtype=None):
    return np.asarray(value, dtype=dtype)


def _take(value, rows):
    # h5py point selections become slow for tens of thousands of irregular rows.
    # Read bounded contiguous blocks instead; sampling remains display-only.
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    ordered = rows[order]
    if hasattr(value, 'chunks') and hasattr(value, 'file') and len(rows) > 1000:
        result = np.empty((len(rows),) + value.shape[1:], dtype=value.dtype)
        block_size = 4096
        blocks = ordered // block_size
        for block in np.unique(blocks):
            positions = np.flatnonzero(blocks == block)
            start = int(block * block_size)
            stop = min(start + block_size, len(value))
            result[positions] = value[start:stop][ordered[positions] - start]
    else:
        result = np.asarray(value[ordered])
    return result[np.argsort(order)]


def _save(fig, out, stem, caption):
    fig.text(.5, .012, caption, ha='center', va='bottom', fontsize=8.4, color='#44515e')
    fig.savefig(out / f'{stem}.png', dpi=190, bbox_inches='tight', facecolor='white')
    fig.savefig(out / f'{stem}.pdf', dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return {'stem': stem, 'png': f'{stem}.png', 'pdf': f'{stem}.pdf', 'caption': caption}


def _limit(values):
    values = np.asarray(values)
    finite = np.abs(values[np.isfinite(values)])
    return max(float(np.quantile(finite, .98)), 1e-6) if finite.size else 1.


def _mode_specs(data):
    source = data['mode_labels']
    if not isinstance(source, dict):
        source = dict(enumerate(source))
    specs = []
    for i, (code, label) in enumerate(sorted(source.items(), key=lambda item: int(item[0]))):
        label = str(label).replace('_', ' ')
        low = label.lower()
        if any(word in low for word in ('absent', 'insufficient', 'unobserved', 'no data')):
            color = '#e2e5e9'
        elif any(word in low for word in ('uncertain', 'unresolved', 'ambiguous', 'poor fit')):
            color = '#b7a8c5'
        elif any(word in low for word in ('stable', 'constant', 'stationary', 'flat')):
            color = '#69737e'
        elif any(word in low for word in ('opening', 'increasing', 'rising')):
            color = '#c55e2d'
        elif any(word in low for word in ('closing', 'decreasing', 'falling')):
            color = '#2676a8'
        elif any(word in low for word in ('pulse', 'peak')):
            color = '#8d5ca6'
        elif any(word in low for word in ('dip', 'valley', 'trough')):
            color = '#238f85'
        else:
            color = _BASE_COLORS[i % len(_BASE_COLORS)]
        specs.append((int(code), label, color))
    return specs


def _background(data, budget=120000):
    xy = _array(data['umap2d'])
    pool = np.flatnonzero(np.isfinite(xy).all(axis=1))
    rng = np.random.default_rng(1729)
    return np.sort(rng.choice(pool, min(len(pool), budget), replace=False))


def _draw_map(ax, data, rows, values, title, vectors=None, mask=None,
              arrow_ids=None, limit=None, budget=200):
    xy = _array(data['umap2d'])
    values = _array(values)
    finite = np.isfinite(values[rows])
    ax.scatter(*xy[rows[~finite]].T, s=.8, c='#c5cad0', alpha=.55, linewidths=0, rasterized=True)
    art = ax.scatter(*xy[rows[finite]].T, s=.7, c=values[rows[finite]],
                     cmap='RdBu_r', norm=TwoSlopeNorm(0, -(limit or _limit(values)), limit or _limit(values)),
                     alpha=.75, linewidths=0, rasterized=True)
    chosen = np.empty(0, dtype=int)
    if vectors is not None:
        vectors, mask = _array(vectors), _array(mask, bool)
        valid = mask & np.isfinite(vectors).all(axis=1) & (np.linalg.norm(vectors, axis=1) > 1e-12)
        chosen = choose_arrows(xy, vectors, valid, budget=budget) if arrow_ids is None else np.asarray(arrow_ids)[valid[np.asarray(arrow_ids)]]
        v = vectors[chosen]
        lengths = np.linalg.norm(v, axis=1, keepdims=True)
        shown = v / np.maximum(lengths, 1e-12) * np.linalg.norm(np.ptp(xy, axis=0)) * .016
        ax.quiver(*xy[chosen].T, *shown.T, angles='xy', scale_units='xy', scale=1,
                  color='#1a2637', width=.00175, headwidth=3.8, headlength=5.1, zorder=4)
    ax.set_aspect('equal')
    ax.set(xticks=[], yticks=[], xlabel='Peak UMAP 1', ylabel='Peak UMAP 2')
    ax.set_title(title + (f'\n{len(chosen):,} displayed directions' if vectors is not None else ''), fontsize=10)
    return art, chosen


def _interval_label(data, t):
    times = _array(data['stages'])
    return f'{times[t]:g} → {times[t + 1]:g} hpf'


def _notes(data):
    notes = list(data.get('notes', []))
    default = ['Each point is a fixed genomic peak.',
               'Red: relative opening; blue: relative closing. Gray: unsupported rate.',
               'Arrows project stage-specific accessibility-profile changes onto existing atlas neighbors.',
               'No root is imposed. A direction toward another peak does not imply peak transport.',
               'Arrow lengths are normalized for display, not physical speeds.',
               'Shared color limits across all intervals; extreme colors saturate.',
               'Net rates can cancel opposing changes across celltypes; inspect the celltype curves.']
    return default + notes


def _temporal_figure(data, out, background, limit):
    rates = _array(data['net_rate'])
    vectors = _array(data['fitted_vectors2d'])
    masks = _array(data['fitted_mask2d'], bool)
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 11))
    for t, ax in enumerate(axes.flat[:rates.shape[1]]):
        art, _ = _draw_map(ax, data, background, rates[:, t], _interval_label(data, t), vectors[:, t], masks[:, t], limit=limit)
    axes.flat[-1].axis('off')
    text = '\n\n'.join(textwrap.fill(x, 45) for x in _notes(data))
    axes.flat[-1].text(.03, .97, text, va='top', fontsize=10, transform=axes.flat[-1].transAxes)
    cax = fig.add_axes([.72, .12, .22, .015])
    fig.colorbar(art, cax=cax, orientation='horizontal', label=data.get('rate_label', 'Fitted net relative accessibility rate (CPM / hpf)'))
    fig.suptitle('Fitted accessibility change on the existing peak atlas', fontsize=15, y=.98)
    fig.subplots_adjust(top=.92, bottom=.07, wspace=.22, hspace=.25)
    return _save(fig, out, '01_fitted_accessibility_projection', f'Full model: {len(rates):,} peaks; {len(background):,} background points shown. Fixed celltype panel for projection and rate averaging. No biological confidence intervals inferred.')


def _fit_raw_figure(data, out, background, limit):
    rates = _array(data['net_rate'])
    fv, rv = _array(data['fitted_vectors2d']), _array(data['raw_vectors2d'])
    fm, rm = _array(data['fitted_mask2d'], bool), _array(data['raw_mask2d'], bool)
    intervals = np.unique(np.linspace(0, rates.shape[1] - 1, 3).round().astype(int))
    fig, axes = plt.subplots(2, len(intervals), figsize=(17.5, 10.5), squeeze=False)
    audit = []
    for col, t in enumerate(intervals):
        common = fm[:, t] & rm[:, t]
        common &= (np.linalg.norm(fv[:, t], axis=1) > 1e-12) & (np.linalg.norm(rv[:, t], axis=1) > 1e-12)
        ids = choose_arrows(_array(data['umap2d']), fv[:, t], common, budget=190)
        for row, (name, v, mask) in enumerate([('Raw adjacent-stage differences', rv, rm), ('Fitted adjacent-stage differences', fv, fm)]):
            art, _ = _draw_map(axes[row, col], data, background, rates[:, t], name + '\n' + _interval_label(data, t), v[:, t], mask[:, t], arrow_ids=ids, limit=limit)
        a, b = fv[common, t], rv[common, t]
        cosine = np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
        audit.append({'interval': int(t), 'common_supported_peaks': int(common.sum()), 'displayed_origins': int(len(ids)), 'median_direction_cosine': float(np.median(cosine)) if cosine.size else None})
    fig.suptitle('Raw versus fitted profile projection · identical displayed origins', fontsize=15)
    fig.subplots_adjust(top=.89, bottom=.105, hspace=.35, wspace=.22)
    cax = fig.add_axes([.35, .062, .30, .012])
    fig.colorbar(art, cax=cax, orientation='horizontal', label='Identical background colors: fitted net CPM / hpf')
    item = _save(fig, out, '02_raw_vs_fitted_projection', 'Only origins supported in both fields are compared here; coverage is reported separately. Same normalized arrow length, coordinates, and rate colors in both rows.')
    item['comparison'] = audit
    return item


def _mode_figure(data, out):
    modes = _array(data['modes'])
    celltypes = [str(x) for x in data['celltypes']]
    specs = _mode_specs(data)
    fractions = np.stack([(modes == code).mean(axis=0) for code, _, _ in specs])
    fig, (left, right) = plt.subplots(1, 2, figsize=(17, max(10, len(celltypes) * .31)), gridspec_kw={'width_ratios': [1, 2.2]})
    overall = fractions.mean(axis=1)
    left.barh(np.arange(len(specs)), overall * 100, color=[x[2] for x in specs])
    left.set(yticks=np.arange(len(specs)), yticklabels=[x[1] for x in specs], xlabel='All peak × celltype pairs (%)', title='Atlas-wide mode composition')
    left.invert_yaxis()
    for i, value in enumerate(overall):
        left.text(value * 100 + .3, i, f'{value:.1%}', va='center', fontsize=9)
    left.set_xlim(0, max(10, overall.max() * 115))
    cumulative = np.zeros(len(celltypes))
    for (code, label, color), fraction in zip(specs, fractions):
        right.barh(np.arange(len(celltypes)), fraction * 100, left=cumulative * 100, color=color, label=label, height=.8)
        cumulative += fraction
    right.set(yticks=np.arange(len(celltypes)), yticklabels=[x.replace('_', ' ') for x in celltypes], xlabel='All peaks within each celltype (%)', xlim=(0, 100), title='Trajectory modes by celltype')
    right.invert_yaxis()
    fig.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in specs], loc='lower center', ncol=min(4, len(specs)), frameon=False, bbox_to_anchor=(.5, .035))
    fig.suptitle('Accessibility trajectory modes · all celltypes retained', fontsize=15)
    fig.tight_layout(rect=[0, .13, 1, .95])
    item = _save(fig, out, '03_trajectory_modes_by_celltype', 'Denominators include every peak × celltype pair. Uncertain and insufficient/absent observations are separate from stable accessibility.')
    item['mode_fraction_by_celltype'] = {label: fraction.tolist() for (_, label, _), fraction in zip(specs, fractions)}
    item['celltypes'] = celltypes
    return item


def _representatives(data, max_examples=8):
    modes = _array(data['modes'])
    valid = _array(data['fit_valid'], bool)
    celltypes = [str(x) for x in data['celltypes']]
    requested = list(data.get('selected_celltypes', ['hematopoietic_vasculature', 'NMPs', 'enteric_neurons']))
    preferred = [celltypes.index(x) for x in requested if x in celltypes]
    preferred += [x for x in range(len(celltypes)) if x not in preferred]
    rows = []
    for code, label, color in _mode_specs(data):
        candidates = modes == code
        # One deterministic representative per class, from a named CT if possible.
        counts = candidates.sum(axis=0)
        named = [celltypes.index(x) for x in requested if x in celltypes]
        shift = max(code - 2, 0) % max(len(named), 1)
        rotated = named[shift:] + named[:shift]
        priority = rotated + [c for c in preferred if c not in rotated]
        ct = next((c for c in priority if counts[c] > 0), None)
        if ct is None:
            continue
        pool = np.flatnonzero(candidates[:, ct])
        # Uniform deterministic subset avoids selecting the most extreme effect.
        sampled = pool[np.linspace(0, len(pool) - 1, min(len(pool), 401)).round().astype(int)]
        curves = _take(data['observed_cpm'], sampled)[:, ct]
        finite = np.isfinite(curves)
        mean = np.divide(np.nansum(curves, axis=1), finite.sum(axis=1), out=np.full(len(sampled), np.nan), where=finite.sum(axis=1) > 0)
        if np.isfinite(mean).any():
            target = np.nanmedian(mean)
            pick = int(np.nanargmin(np.abs(mean - target)))
        else:
            pick = len(sampled) // 2
        rows.append({'peak_index': int(sampled[pick]), 'celltype_index': int(ct), 'mode': label, 'mode_code': int(code), 'color': color, 'fit_valid': bool(valid[sampled[pick], ct]), 'candidate_count': int(len(pool))})
    return rows[:max_examples]


def _curve_figure(data, out):
    selected = _representatives(data)
    if not selected:
        return None
    ncols = 4
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.6 + .8), squeeze=False)
    times = _array(data['stages'])
    valid = _array(data['condition_valid'], bool)
    celltypes = [str(x) for x in data['celltypes']]
    for ax, item in zip(axes.flat, selected):
        peak, ct = item['peak_index'], item['celltype_index']
        obs = np.asarray(data['observed_cpm'][peak, ct], dtype=float).copy()
        fit = np.asarray(data['fitted_cpm'][peak, ct], dtype=float).copy()
        # Retain all measured raw CPM, including low-cell groups; only missing
        # values supplied as NaN disappear. Support marks are separate.
        if item['fit_valid']:
            ax.plot(times, fit, color=item['color'], lw=2, marker='.', label='Fitted stage values')
        ax.plot(times, obs, 'o', color='#182f46', ms=5, label='Observed CPM', zorder=4)
        unsupported = times[~np.isfinite(obs)]
        if len(unsupported):
            ax.plot(unsupported, np.full(len(unsupported), .035), 'x', color='#8a8f96', transform=ax.get_xaxis_transform(), label='Missing observation')
        pid = str(data['peak_ids'][peak])
        ax.set_title(item['mode'] + '\n' + celltypes[ct].replace('_', ' '), fontsize=10)
        ax.set(xlabel='Stage (hpf)', ylabel='Relative accessibility (CPM)', xticks=times)
        ax.text(.02, .98, pid, transform=ax.transAxes, fontsize=7.2, va='top', wrap=True)
        ax.margins(y=.22)
        ax.set_ylim(bottom=0)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes.flat[len(selected):]:
        ax.axis('off')
    handles = [plt.Line2D([], [], marker='o', linestyle='', color='#182f46', label='Observed CPM'), plt.Line2D([], [], color='#69737e', lw=2, label='Fitted stage values'), plt.Line2D([], [], marker='x', linestyle='', color='#8a8f96', label='Missing observation')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(.5, .045), frameon=False)
    fig.suptitle('Representative within-celltype accessibility curves', fontsize=15)
    fig.tight_layout(rect=[0, .14, 1, .94])
    item = _save(fig, out, '04_representative_accessibility_curves', 'Representative selection is deterministic: a median-accessibility example within each mode and preferred celltype. Lines connect fitted stage values. Raw low-support observations remain visible; six stages do not resolve fine kinetics.')
    item['selected_pairs'] = selected
    return item


def _context_figure(data, out, background, limit):
    rates = _array(data['net_rate'])
    if 'context_vectors2d' in data and 'context_mask2d' in data:
        base = _array(data['fitted_vectors2d'])
        coupled = _array(data['context_vectors2d'])
        bm = _array(data['fitted_mask2d'], bool)
        cm = _array(data['context_mask2d'], bool)
        intervals = [0, rates.shape[1] - 1]
        fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
        for col, t in enumerate(intervals):
            ids = choose_arrows(_array(data['umap2d']), base[:, t], bm[:, t] & cm[:, t], budget=180)
            for row, (label, vec, mask) in enumerate([('Fitted profile projection', base, bm), (data.get('context_label', 'Genomic-context variant'), coupled, cm)]):
                art, _ = _draw_map(axes[row, col], data, background, rates[:, t], label + '\n' + _interval_label(data, t), vec[:, t], mask[:, t], arrow_ids=ids, limit=limit)
        fig.suptitle('Genomic context comparison · identical displayed origins', fontsize=15)
        fig.subplots_adjust(top=.89, bottom=.09, hspace=.35, wspace=.18)
        return _save(fig, out, '05_genomic_context_comparison', data.get('context_note', 'Both rows retain the same peak coordinates and rate colors. Genomic context changes only the supplied comparison field; it does not establish causal regulation.'))
    if 'genomic_context' not in data:
        return None
    labels = np.asarray(data['genomic_context']).astype(str)
    names, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)[:8]
    names, counts = names[order], counts[order]
    masks = _array(data['fitted_mask2d'], bool)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={'width_ratios': [1.2, 1]})
    modes = _array(data['modes'])
    specs = _mode_specs(data)
    cumulative = np.zeros(len(names))
    for code, label, color in specs:
        fractions = np.array([(modes[labels == name] == code).mean() for name in names])
        axes[0].barh(np.arange(len(names)), fractions * 100, left=cumulative * 100, color=color, label=label)
        cumulative += fractions
    axes[0].set(yticks=np.arange(len(names)), yticklabels=[f'{name} (n={count:,})' for name, count in zip(names, counts)],
                xlabel='Peak × celltype pairs within annotation (%)', xlim=(0, 100), title='Trajectory modes by genomic annotation')
    axes[0].invert_yaxis()
    for name, count in zip(names, counts):
        keep = labels == name
        axes[1].plot(_array(data['stages'])[:-1], masks[keep].mean(axis=0), 'o-', label=name)
    axes[1].set(xlabel='Interval start (hpf)', ylabel='Fraction with a supported projected direction', title='Projection coverage by annotation', ylim=(0, 1))
    axes[1].legend(frameon=False, fontsize=8)
    fig.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in specs], loc='lower center', ncol=min(4, len(specs)), frameon=False, bbox_to_anchor=(.5, .035))
    fig.suptitle('Genomic context · descriptive annotation comparison', fontsize=15)
    fig.tight_layout(rect=[0, .17, 1, .93])
    return _save(fig, out, '05_genomic_context_comparison', 'Annotation groups are descriptive. Association with a nearby gene or genomic class does not establish a regulatory target or causal effect.')


def _stationary_figure(data, out):
    rates = _array(data['net_rate'])
    fm = _array(data['fitted_mask2d'], bool)
    rm = _array(data['raw_mask2d'], bool)
    stages = _array(data['stages'])[:-1]
    broad = _array(data.get('broad_reference', np.zeros(len(rates), bool)), bool)
    rms = _array(data['rms_rate']) if 'rms_rate' in data else None
    no_change = _array(data['no_supported_change'], bool) if 'no_supported_change' in data else None
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    axes[0].plot(stages, rm.mean(axis=0), 'o-', label='Raw profile', color='#8f9ba8')
    axes[0].plot(stages, fm.mean(axis=0), 'o-', label='Fitted profile', color='#255b85')
    axes[0].set(xlabel='Interval start (hpf)', ylabel='Fraction of all peaks', title='Supported projected directions', ylim=(0, 1))
    axes[0].legend(frameon=False)
    if broad.any():
        for mask, label, color in [(~broad, 'Other peaks', '#8293a6'), (broad, f'Broad reference (n={broad.sum():,})', '#ba922e')]:
            if not mask.any():
                continue
            values = rms[mask] if rms is not None else np.abs(rates[mask])
            count = np.isfinite(values).sum(axis=1)
            absolute = np.divide(np.nansum(values, axis=1), count, out=np.full(len(values), np.nan), where=count > 0)
            absolute = absolute[np.isfinite(absolute)]
            axes[1].hist(np.log10(absolute + 1e-6), bins=50, histtype='step', density=True, lw=1.5, label=label, color=color)
            fraction = no_change[mask].mean(axis=0) if no_change is not None else fm[mask].mean(axis=0)
            axes[2].plot(stages, fraction, 'o-', label=label, color=color)
        axes[1].set(xlabel='log10(mean RMS CPM / hpf + 10⁻⁶)' if rms is not None else 'log10(mean |net CPM / hpf| + 10⁻⁶)', ylabel='Density', title='Fitted rate magnitude across celltypes' if rms is not None else 'Net-rate magnitude (can conceal cancellation)')
        axes[2].set(xlabel='Interval start (hpf)', ylabel='Fraction of all group peaks', title='No count-supported CT change' if no_change is not None else 'Supported projected directions', ylim=(0, 1))
        for ax in axes[1:]:
            ax.legend(frameon=False, fontsize=8)
    else:
        supported = np.isfinite(rates)
        axes[1].plot(stages, supported.mean(axis=0), 'o-', color='#8d5ca6')
        axes[1].set(xlabel='Interval start (hpf)', ylabel='Fraction of peaks', title='Supported net accessibility rates', ylim=(0, 1))
        axes[2].axis('off')
        axes[2].text(.05, .9, 'No broad-reference set was supplied.\nNo peak is designated as a root.\n\nStable, unsupported, and uncertain\ntrajectories remain distinct.', va='top', transform=axes[2].transAxes, fontsize=11)
    fig.suptitle('Broad-reference activity and projection support', fontsize=15)
    fig.tight_layout(rect=[0, .12, 1, .92])
    item = _save(fig, out, '06_stationary_reference_and_support', 'No supported change does not prove stability. No arrow can also reflect an unprojectable direction. RMS avoids signed CT cancellation; broad-reference membership does not orient the field.')
    item['broad_reference_peaks'] = int(broad.sum())
    item['fitted_arrow_fraction'] = fm.mean(axis=0).tolist()
    item['raw_arrow_fraction'] = rm.mean(axis=0).tolist()
    item['broad_rate_magnitude_uses_rms'] = rms is not None
    if no_change is not None and broad.any():
        item['broad_no_supported_change_fraction'] = no_change[broad].mean(axis=0).tolist()
    return item


def _validate(data):
    xy = _array(data['umap2d'])
    n, c, t = len(xy), len(data['celltypes']), len(data['stages'])
    expected = {'umap2d': (n, 2), 'observed_cpm': (n, c, t), 'fitted_cpm': (n, c, t),
                'condition_valid': (c, t), 'modes': (n, c), 'fit_valid': (n, c),
                'net_rate': (n, t - 1), 'fitted_vectors2d': (n, t - 1, 2),
                'raw_vectors2d': (n, t - 1, 2), 'fitted_mask2d': (n, t - 1), 'raw_mask2d': (n, t - 1)}
    for key, shape in expected.items():
        if np.shape(data[key]) != shape:
            raise ValueError(f'{key}: expected {shape}, got {np.shape(data[key])}')
    if t != 6:
        raise ValueError('The report layout requires the six developmental stages')
    if len(data['peak_ids']) != n or not np.isfinite(xy).all():
        raise ValueError('Peak identifiers and finite existing coordinates must align')
    if not np.all(np.diff(_array(data['stages'])) > 0):
        raise ValueError('Stage hours must strictly increase')
    labels = {x[0] for x in _mode_specs(data)}
    unknown = set(np.unique(data['modes'])) - labels
    if unknown:
        raise ValueError(f'Unlabelled trajectory-mode codes: {unknown}')


def make_trajectory_report(data, output_dir, make_html=True, make_streams=True):
    """Write six concise scientific figure pairs and a machine-readable manifest.

    Large cubes remain on disk except selected curves / HTML display rows.
    Missing genomic context omits figure 05 with an explicit manifest reason.
    """
    _validate(data)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    background = _background(data)
    limit = float(data.get('rate_color_limit', _limit(data['net_rate'])))
    builders = [_temporal_figure, _fit_raw_figure, _mode_figure, _curve_figure, _context_figure, _stationary_figure]
    artifacts = []
    for build in builders:
        if build in (_temporal_figure, _fit_raw_figure, _context_figure):
            item = build(data, out, background, limit)
        else:
            item = build(data, out)
        if item:
            artifacts.append(item)
    result = {'peak_count': len(data['peak_ids']), 'celltypes': [str(x) for x in data['celltypes']],
              'stages': _array(data['stages']).tolist(), 'background_peaks': len(background),
              'rate_color_limit': limit, 'rate_color_limit_policy': 'shared across intervals; saturation at fixed limit',
              'figures': artifacts, 'notes': _notes(data), 'model_refit': False, 'root_imposed': False,
              'genomic_context_omitted': not ('genomic_context' in data or 'context_vectors2d' in data)}
    if make_streams:
        item = _stream_figure(data, out, background, limit)
        result['figures'].append(item)
    item = _three_dimensional_figure(data, out, background, limit)
    if item is not None:
        result['figures'].append(item)
    if make_html:
        result['html'] = export_trajectory_viewer(data, out / 'trajectory_peak_explorer.html')
    (out / 'trajectory_report_manifest.json').write_text(json.dumps(clean(result), indent=2, allow_nan=False) + '\n')
    return result




def _three_dimensional_figure(data, out, background, limit):
    """Three fixed camera views of existing supported 3D interval projections."""
    if not all(key in data for key in ['umap3d', 'fitted_vectors3d', 'fitted_mask3d']):
        return None
    xyz = _array(data['umap3d'])
    vectors = _array(data['fitted_vectors3d'])
    masks = _array(data['fitted_mask3d'], bool)
    rates = _array(data['net_rate'])
    rows = background[::3]
    intervals = np.unique(np.linspace(0, rates.shape[1] - 1, 3).round().astype(int))
    fig = plt.figure(figsize=(18, 7))
    norm = TwoSlopeNorm(0, -limit, limit)
    panels = []
    span = np.ptp(xyz, axis=0)
    for position, t in enumerate(intervals):
        ax = fig.add_subplot(1, len(intervals), position + 1, projection='3d')
        # Faint context remains visible when near-zero rate colors are white.
        ax.scatter(*xyz[rows].T, c='#8394a8', s=1.6, alpha=.24, linewidths=0, depthshade=False, rasterized=True)
        finite = np.isfinite(rates[rows, t])
        missing = rows[~finite]
        shown = rows[finite]
        ax.scatter(*xyz[missing].T, c='#c6cdd4', s=.55, alpha=.55, linewidths=0, depthshade=False, rasterized=True)
        art = ax.scatter(*xyz[shown].T, c=rates[shown, t], cmap='RdBu_r', norm=norm,
                         s=.75, alpha=.73, linewidths=0, depthshade=False, rasterized=True)
        ids = choose_arrows(xyz, vectors[:, t], masks[:, t], budget=145)
        vec = vectors[ids, t]
        vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12) * np.linalg.norm(span) * .019
        ax.quiver(*xyz[ids].T, *vec.T, color='#14283d', length=1, normalize=False,
                  arrow_length_ratio=.32, linewidth=.75)
        ax.view_init(elev=23, azim=-60)
        ax.set_box_aspect(span)
        ax.set(xticks=[], yticks=[], zticks=[], xlabel='Peak UMAP 1', ylabel='Peak UMAP 2', zlabel='Peak UMAP 3')
        ax.set_title(_interval_label(data, t) + f'\n{len(ids)} supported directions shown', fontsize=11, pad=14)
        panels.append({'interval': int(t), 'displayed_arrows': int(len(ids)), 'supported_peaks': int(masks[:, t].sum())})
    cax = fig.add_axes([.35, .115, .30, .018])
    fig.colorbar(art, cax=cax, orientation='horizontal', label='Fitted net relative accessibility rate (CPM / hpf)')
    fig.suptitle('Fitted accessibility projection on the existing 3D peak atlas', fontsize=16, y=.97)
    fig.text(.5, .92, 'Three representative intervals · identical camera and shared rate scale', ha='center', fontsize=10, color='#4b5f72')
    fig.subplots_adjust(top=.85, bottom=.19, left=.02, right=.94, wspace=.01)
    item = _save(fig, out, '08_3d_interval_projection', f'{len(rows):,} background peaks per panel; full model uses {len(xyz):,} peaks. No root or new projection is introduced. Faint gray underlay shows atlas context. Arrow lengths are normalized, not physical speed; rotate the HTML for other views.')
    item['panels'] = panels
    item['camera'] = {'elevation': 23, 'azimuth': -60}
    return item

def _stream_figure(data, out, background, limit):
    """Display-only local averaging with checked path and arrowhead geometry."""
    import hashlib
    import sys
    from matplotlib import patheffects
    from scipy.interpolate import RegularGridInterpolator
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/chrom_velo'))
    from streamline_field import build_stream_grid
    from streamline_paths import draw_supported_streams
    xy = _array(data['umap2d'])
    vectors = _array(data['fitted_vectors2d'])
    masks = _array(data['fitted_mask2d'], bool)
    rates = _array(data['net_rate'])
    artifact_dir = out / 'stream_display'
    artifact_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 11))
    panels = []
    for t, ax in enumerate(axes.flat[:rates.shape[1]]):
        art, _ = _draw_map(ax, data, background, rates[:, t], _interval_label(data, t), limit=limit)
        grid = build_stream_grid(xy, vectors[:, t], valid=masks[:, t])
        drawn = draw_supported_streams(ax, grid, color='#172c40', density=3, linewidth=.85, arrowsize=.8)
        drawn['lines'].set_path_effects([patheffects.Stroke(linewidth=1.7, foreground='white', alpha=.8), patheffects.Normal()])
        paths = drawn['paths']
        points = np.concatenate(paths) if paths else np.empty((0, 2))
        offsets = np.r_[0, np.cumsum([len(path) for path in paths])]
        heads = np.array([patch.get_xy()[:3] for patch in drawn['arrows']]).reshape(-1, 3, 2)
        # In-sample native-vector agreement diagnoses loss of minority directions.
        pool = np.flatnonzero(masks[:, t])
        ids = pool[np.linspace(0, len(pool) - 1, min(len(pool), 30000)).astype(int)] if len(pool) else pool
        uv = np.column_stack([RegularGridInterpolator((grid['y'], grid['x']), grid[k], bounds_error=False, fill_value=np.nan)(xy[ids][:, ::-1]) for k in ['u', 'v']])
        ok = np.isfinite(uv).all(axis=1) & (np.linalg.norm(uv, axis=1) > 1e-12)
        uv = uv[ok]
        native = vectors[ids[ok], t]
        cosine = np.sum(uv * native, axis=1) / np.maximum(np.linalg.norm(uv, axis=1) * np.linalg.norm(native, axis=1), 1e-12)
        metadata = {**grid['metadata'], 'interval': t, 'source_vectors_sha256': hashlib.sha256(np.ascontiguousarray(vectors[:, t]).tobytes()).hexdigest(), 'source_mask_sha256': hashlib.sha256(np.ascontiguousarray(masks[:, t]).tobytes()).hexdigest(), 'median_native_cosine': float(np.median(cosine)) if len(cosine) else None, 'native_opposing_fraction': float(np.mean(cosine < 0)) if len(cosine) else None}
        np.savez_compressed(artifact_dir / f't{t}_grid.npz', **{k: v for k, v in grid.items() if k != 'metadata'}, metadata_json=json.dumps(clean(metadata)))
        np.savez_compressed(artifact_dir / f't{t}_paths.npz', points=points, offsets=offsets, arrowheads=heads, grid_file=f't{t}_grid.npz')
        ax.set_title(_interval_label(data, t) + f'\n{len(paths)} supported local fragments', fontsize=10)
        panels.append({**metadata, 'path_audit': drawn['audit'], 'grid_file': f'stream_display/t{t}_grid.npz', 'path_file': f'stream_display/t{t}_paths.npz'})
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.03, .95, 'Local display averages only\n\nStreams follow supported averages of\nunit directions, not individual peaks.\n\nAveraging can suppress minority directions.\nUnsupported gaps are not bridged.\n\nNo peak transport or physical speed\nis inferred. Sparse coverage is retained.', va='top', fontsize=11, transform=axes.flat[-1].transAxes)
    cax = fig.add_axes([.72, .12, .22, .015])
    fig.colorbar(art, cax=cax, orientation='horizontal', label='Fitted net relative accessibility rate (CPM / hpf)')
    fig.suptitle('Fitted profile projection · supported local stream display', fontsize=15)
    fig.subplots_adjust(top=.92, bottom=.07, wspace=.22, hspace=.25)
    item = _save(fig, out, '07_supported_local_streams', 'Same existing peak coordinates and fitted vectors. Interpolation and clipped path geometry are display operations; the rates and projection model are unchanged.')
    item['panels'] = panels
    return item


def _encode_array(values, dtype='float32'):
    array = np.ascontiguousarray(values, dtype=dtype)
    return {'dtype': str(array.dtype), 'shape': list(array.shape), 'base64': base64.b64encode(array.tobytes()).decode('ascii')}


def export_trajectory_viewer(data, output_path):
    """Export an offline arrow viewer; display sampling never changes inference."""
    from plotly.offline import get_plotlyjs
    from real_specialization_viewer import select_display_peaks
    n = len(data['peak_ids'])
    obs = data.get('obs')
    annotations = None
    forced = np.zeros(n, bool)
    if obs is not None:
        if 'leiden_coarse' in obs:
            annotations = obs['leiden_coarse'].astype(str).to_numpy()
        for col in ['nearest_gene', 'linked_gene', 'associated_gene']:
            if col in obs:
                forced |= obs[col].astype(str).str.contains(r'\b(?:slc4a1a|cdx1a|phox2bb|gata1a|tal1|sox10)\b', regex=True).to_numpy()
    xyz = _array(data['umap3d']) if 'umap3d' in data else None
    ids, selection = select_display_peaks(_array(data['umap2d']), xyz, strata=annotations,
                                         include=np.flatnonzero(forced), max_points=int(data.get('viewer_points', 30000)), seed=1729)
    modes = _take(data['modes'], ids)
    observed = _take(data['observed_cpm'], ids)
    fitted = _take(data['fitted_cpm'], ids)
    peaks = []
    annotation_values = {col: obs[col].astype(str).to_numpy()[ids] for col in ['nearest_gene', 'linked_gene', 'leiden_coarse', 'peak_type_argelaguet'] if obs is not None and col in obs}
    for local, i in enumerate(ids):
        item = {'peak': str(data['peak_ids'][i]), 'global_index': int(i), 'forced': bool(forced[i])}
        for col, values in annotation_values.items():
            item[col] = str(values[local])
        peaks.append(item)
    arrays = {'xy': _array(data['umap2d'])[ids], 'observed': observed, 'fitted': fitted,
              'mode': modes, 'fitvalid': _take(data['fit_valid'], ids),
              'netrate': _take(data['net_rate'], ids), 'condition': _array(data['condition_valid'], bool)}
    if xyz is not None:
        arrays['xyz'] = xyz[ids]
    for prefix, source in [('fit', 'fitted'), ('raw', 'raw')]:
        for dim in [2, 3]:
            if f'{source}_vectors{dim}d' in data:
                arrays[f'{prefix}v{dim}'] = _take(data[f'{source}_vectors{dim}d'], ids)
                arrays[f'{prefix}m{dim}'] = _take(data[f'{source}_mask{dim}d'], ids)
    colors = {}
    for code, label, color in _mode_specs(data):
        colors[str(code)] = {'label': label, 'color': color}
    payload = {'total_peaks': n, 'display_peaks': len(ids), 'selection': selection,
               'stages': [float(x) for x in data['stages']], 'celltypes': [str(x) for x in data['celltypes']],
               'peaks': peaks, 'modes': colors, 'rate_limit': float(data.get('rate_color_limit', _limit(data['net_rate']))),
               'notes': _notes(data), 'arrays': {k: _encode_array(v, 'uint8' if k in ['mode', 'fitvalid', 'condition'] or k.startswith(('fitm', 'rawm')) else 'float32') for k, v in arrays.items()}}
    body = _VIEWER_HTML.replace('__PAYLOAD__', json.dumps(clean(payload), separators=(',', ':'), allow_nan=False).replace('</', '<\\/'))
    body = body.replace('__PLOTLY__', get_plotlyjs())
    path = Path(output_path)
    path.write_text(body, encoding='utf-8')
    return {'filename': path.name, 'bytes': path.stat().st_size, 'selection': selection,
            'notes': 'Self-contained. Search and curves include every exported peak. 3D background draws at most 6500 coverage points plus forced markers, selected peak, and arrow origins.'}


_VIEWER_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Peak accessibility trajectories</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf1f5;color:#203246;font:14px system-ui,sans-serif}header{padding:20px 28px;background:#fff;border-bottom:1px solid #d7e0e8}h1{font-size:25px;margin:0 0 7px}header p{margin:5px 0;max-width:140ch}.notice{color:#685024;font-size:13px}.controls{display:flex;flex-wrap:wrap;gap:13px;padding:15px 24px;background:#f8fafc;border-bottom:1px solid #dae2eb}.controls label{display:flex;flex-direction:column;gap:4px;font-size:12px;font-weight:600}select,input,button{font:inherit;padding:6px 8px;border:1px solid #bfcbd6;border-radius:5px;background:white;max-width:290px}input[type=range]{width:130px;padding:0}button{cursor:pointer;align-self:end}.layout{display:grid;grid-template-columns:minmax(650px,1.65fr) minmax(360px,1fr);gap:13px;padding:14px 20px}.card{background:white;border:1px solid #d9e2eb;border-radius:8px;overflow:hidden}#atlas{height:700px}#curve{height:335px}#heatmap{height:360px}.caption{padding:10px 17px;border-top:1px solid #e1e7ed;line-height:1.5;color:#46596c;font-size:12px}#status{padding:9px 24px;font-size:13px}.selected{padding:11px 17px;font-weight:600;font-size:12px;overflow-wrap:anywhere}#error{color:#9d2121;padding:10px}details{padding:13px 26px}summary{cursor:pointer;font-weight:600}#search{width:250px}@media(max-width:1050px){.layout{display:block}.card{margin-bottom:15px}#atlas{height:650px}}@media(max-width:650px){header,.controls{padding:13px}.layout{padding:8px}#atlas{height:530px}}
</style></head><body><header><h1>Peak accessibility trajectories</h1><p>Each point is a fixed genomic peak on the existing UMAP. Colors show signed relative accessibility change; arrows project stage-specific profile changes onto atlas neighbors.</p><p class="notice">No root is imposed. Arrows do not describe peak transport or physical speed. Raw measured CPM remains visible, including low-support observations; missing observations remain missing. Fits and projection support are computational diagnostics, not biological confidence intervals.</p></header>
<div class="controls"><label>Dimension<select id="dimension"><option value="2">2D</option><option value="3">3D · rotate with mouse</option></select></label><label>Projection<select id="field"><option value="fit">Fitted stage profiles</option><option value="raw">Raw stage differences</option></select></label><label>Developmental interval<select id="interval"></select></label><label>Celltype curves / colors<select id="celltype"></select></label><label>Point color<select id="color"><option value="net">Net fitted CPM / hpf</option><option value="ct_rate">Celltype fitted CPM / hpf</option><option value="mode">Celltype trajectory mode</option></select></label><label>Max arrows · display only<input id="density" type="range" min="0" max="350" step="25" value="175"></label><label>Arrow length · display only<input id="length" type="range" min="0.4" max="2.5" step="0.1" value="1.2"></label><label>Find exported peak or gene<input id="search" placeholder="slc4a1a or peak coordinate"></label><button id="find">Find</button><button id="save">Save atlas PNG</button></div>
<div id="status"></div><div id="error"></div><div class="layout"><section class="card"><div id="atlas"></div><div class="caption" id="map-caption"></div></section><section class="card"><div class="selected" id="selected"></div><div id="curve"></div><div id="heatmap"></div><div class="caption">Dots: observed CPM. Line: fitted values at the six stages; connecting segments do not imply resolved fine kinetics. All measured observations remain visible. Gray heatmap cells indicate missing observations. Mode labels separate insufficient, stable, directional, transient, and uncertain trajectories.</div></section></div><details><summary>Interpretation and display sampling</summary><div id="details"></div></details>
<script>__PLOTLY__</script><script>
const DATA=__PAYLOAD__;
const A={};for(const [k,x] of Object.entries(DATA.arrays)){const bytes=Uint8Array.from(atob(x.base64),c=>c.charCodeAt(0));A[k]=x.dtype==='uint8'?bytes:new Float32Array(bytes.buffer)}
const N=DATA.display_peaks,C=DATA.celltypes.length,T=DATA.stages.length;let selected=0,busy=false,renderPending=false,lastSearch='',lastHit=-1;
const $=id=>document.getElementById(id),finite=Number.isFinite;
function options(id,values){$(id).innerHTML=values.map(([v,s])=>`<option value="${v}">${s}</option>`).join('')}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
options('interval',DATA.stages.slice(0,-1).map((t,i)=>[i,`${t} → ${DATA.stages[i+1]} hpf`]));options('celltype',DATA.celltypes.map((c,i)=>[i,esc(c)]));
if(!A.xyz){$('dimension').value='2';$('dimension').disabled=true}
$('details').innerHTML='<ul>'+DATA.notes.map(n=>'<li>'+esc(n)+'</li>').join('')+'</ul><p>Model: '+DATA.total_peaks.toLocaleString()+' peaks. Searchable display subset: '+N.toLocaleString()+'. Coverage sampling is a display choice, not an abundance estimate. 3D uses a smaller background while retaining marker peaks, selected peak, and arrow origins.</p>';
const ranges={};for(const [name,d] of [['xy',2],['xyz',3]])if(A[name]){let lo=Array(d).fill(Infinity),hi=Array(d).fill(-Infinity);for(let i=0;i<N;i++)for(let k=0;k<d;k++){const v=A[name][i*d+k];lo[k]=Math.min(lo[k],v);hi[k]=Math.max(hi[k],v)}ranges[d]={lo,hi,span:hi.map((v,k)=>v-lo[k]),diag:Math.hypot(...hi.map((v,k)=>v-lo[k]))}}
const background3=[];if(A.xyz){const bins=new Map();for(let i=0;i<N;i++){const key=[0,1,2].map(k=>Math.min(29,Math.floor((A.xyz[i*3+k]-ranges[3].lo[k])/Math.max(ranges[3].span[k],1e-8)*30))).join(',');if(!bins.has(key))bins.set(key,i)}background3.push(...bins.values());if(background3.length>6500)background3.splice(6500);const used=new Set(background3);for(let i=0;background3.length<Math.min(6500,N)&&i<N;i++){let j=(i*7919)%N;if(!used.has(j)){background3.push(j);used.add(j)}}}
function state(){return {d:+$('dimension').value,field:$('field').value,t:+$('interval').value,c:+$('celltype').value,color:$('color').value,budget:+$('density').value,length:+$('length').value}}
function point(s,i){const a=s.d===3?A.xyz:A.xy;return Array.from(a.subarray(i*s.d,(i+1)*s.d))}
function curve(a,i,c){return Array.from(a.subarray((i*C+c)*T,(i*C+c+1)*T)).map(v=>finite(v)?v:null)}
function values(s){let out=new Array(N);for(let i=0;i<N;i++){if(s.color==='mode'){out[i]=A.mode[i*C+s.c];continue}if(s.color==='net'){out[i]=A.netrate[i*(T-1)+s.t];continue}let j=(i*C+s.c)*T+s.t;out[i]=A.fitvalid[i*C+s.c]&&A.condition[s.c*T+s.t]&&A.condition[s.c*T+s.t+1]?(A.fitted[j+1]-A.fitted[j])/(DATA.stages[s.t+1]-DATA.stages[s.t]):NaN}return out}
function ctLimit(c){let vals=[];for(let i=0;i<N;i++)if(A.fitvalid[i*C+c])for(let t=0;t<T-1;t++){const j=(i*C+c)*T+t;if(A.condition[c*T+t]&&A.condition[c*T+t+1]){let v=Math.abs((A.fitted[j+1]-A.fitted[j])/(DATA.stages[t+1]-DATA.stages[t]));if(finite(v))vals.push(v)}}vals.sort((a,b)=>a-b);return Math.max(vals[Math.floor(.98*(vals.length-1))]||0,1e-6)}
const limits={};function arrows(s){const v=A[s.field+'v'+s.d],m=A[s.field+'m'+s.d];if(!v||!m)return [];const bins=new Map();for(let i=0;i<N;i++){let j=(i*(T-1)+s.t)*s.d;if(!m[i*(T-1)+s.t]||!finite(v[j])||Math.hypot(...v.subarray(j,j+s.d))<1e-12)continue;const p=point(s,i);const key=p.map((x,k)=>Math.min(35,Math.floor((x-ranges[s.d].lo[k])/Math.max(ranges[s.d].span[k],1e-8)*36))).join(',');if(!bins.has(key))bins.set(key,i)}let ids=[...bins.values()];if(ids.length<=s.budget)return ids;if(!s.budget)return [];return Array.from({length:s.budget},(_,j)=>ids[Math.floor(j*ids.length/s.budget)])}
function arrowTraces(s,ids){let xs=[],ys=[],zs=[],cx=[],cy=[],cz=[],cu=[],cv=[],cw=[];const all=A[s.field+'v'+s.d],length=ranges[s.d].diag*.012*s.length;for(const i of ids){const p=point(s,i),j=(i*(T-1)+s.t)*s.d,v=Array.from(all.subarray(j,j+s.d)),norm=Math.hypot(...v);const q=p.map((x,k)=>x+v[k]/norm*length);xs.push(p[0],q[0],null);ys.push(p[1],q[1],null);if(s.d===3){zs.push(p[2],q[2],null);cx.push(q[0]);cy.push(q[1]);cz.push(q[2]);cu.push(v[0]/norm);cv.push(v[1]/norm);cw.push(v[2]/norm)}else{const ux=v[0]/norm,uy=v[1]/norm;for(const sign of [-1,1]){xs.push(q[0]-length*.26*ux+sign*length*.13*uy,q[0],null);ys.push(q[1]-length*.26*uy-sign*length*.13*ux,q[1],null)}}}const line={type:s.d===3?'scatter3d':'scattergl',mode:'lines',x:xs,y:ys,line:{color:'#17283d',width:s.d===3?3:1.5},hoverinfo:'skip',showlegend:false};if(s.d===3)line.z=zs;let result=[line];if(s.d===3&&ids.length)result.push({type:'cone',x:cx,y:cy,z:cz,u:cu,v:cv,w:cw,sizemode:'absolute',sizeref:length*.23,anchor:'tip',colorscale:[[0,'#17283d'],[1,'#17283d']],showscale:false,hoverinfo:'skip'});return result}
async function render(){if(busy){renderPending=true;return}busy=true;$('error').textContent='';try{const s=state(),val=values(s),aid=arrows(s);let ids=s.d===3?Array.from(new Set([...background3,...aid,selected,...DATA.peaks.flatMap((p,i)=>p.forced?[i]:[])])):Array.from({length:N},(_,i)=>i);const good=ids.filter(i=>finite(val[i])),missing=ids.filter(i=>!finite(val[i]));let lim=s.color==='ct_rate'?(limits[s.c]??(limits[s.c]=ctLimit(s.c))):DATA.rate_limit;function dots(ii,miss){const tr={type:s.d===3?'scatter3d':'scattergl',mode:'markers',x:ii.map(i=>point(s,i)[0]),y:ii.map(i=>point(s,i)[1]),customdata:ii,marker:{size:s.d===3?2:2.6,opacity:s.d===3?1:.72,color:miss?'#bfc6ce':s.color==='mode'?ii.map(i=>DATA.modes[String(val[i])]?.color||'#bfc6ce'):ii.map(i=>val[i]),line:{width:0}},text:ii.map(i=>DATA.peaks[i].peak),hovertemplate:'%{text}<extra></extra>',showlegend:false};if(s.d===3)tr.z=ii.map(i=>point(s,i)[2]);if(!miss&&s.color!=='mode')Object.assign(tr.marker,{colorscale:[[0,'#2166ac'],[.25,'#67a9cf'],[.5,'#f7f7f7'],[.75,'#ef8a62'],[1,'#b2182b']],cmin:-lim,cmax:lim,colorbar:{title:{text:'CPM / hpf'},thickness:13,len:.7}});return tr}const traces=[dots(missing,true),dots(good,false),...arrowTraces(s,aid)];const p=point(s,selected),highlight={type:s.d===3?'scatter3d':'scattergl',mode:'markers',x:[p[0]],y:[p[1]],marker:{size:8,color:'#f8d167',line:{color:'#35495e',width:2}},hoverinfo:'skip',showlegend:false};if(s.d===3)highlight.z=[p[2]];traces.push(highlight);let layout={margin:{l:35,r:45,t:40,b:65},annotations:[{xref:'paper',yref:'paper',x:.5,y:-.09,xanchor:'center',showarrow:false,text:'Fixed genomic peaks · no imposed root · normalized arrow lengths, not physical speed',font:{size:10,color:'#536575'}}],title:{text:`${s.field==='fit'?'Fitted':'Raw'} projection · ${DATA.stages[s.t]} → ${DATA.stages[s.t+1]} hpf`,font:{size:16}},paper_bgcolor:'#fff',plot_bgcolor:'#fff',uirevision:'existingpeak'+s.d,showlegend:false,xaxis:{title:'Peak UMAP 1',showgrid:false,zeroline:false},yaxis:{title:'Peak UMAP 2',showgrid:false,zeroline:false,scaleanchor:'x'},scene:{aspectmode:'data',xaxis:{title:'Peak UMAP 1'},yaxis:{title:'Peak UMAP 2'},zaxis:{title:'Peak UMAP 3'},camera:{eye:{x:1.4,y:1.4,z:1.1}}}};await Plotly.react('atlas',traces,layout,{responsive:true,displaylogo:false});$('status').textContent=`${DATA.total_peaks.toLocaleString()} model peaks · ${N.toLocaleString()} searchable exported peaks · ${ids.length.toLocaleString()} displayed points · ${aid.length} arrows shown`;$('map-caption').innerHTML=s.color==='mode'?Object.values(DATA.modes).map(m=>`<span style="color:${m.color}">■</span> ${esc(m.label)}`).join(' · ')+'<br>Arrows retain the fixed-panel projection when celltype colors change.':`Red = relative opening; blue = relative closing; gray = unsupported. Fixed scale across intervals: ±${lim.toPrecision(3)} CPM/hpf; colors saturate. ${s.color==='net'?'Equally weighted mean across eligible celltypes in the fixed projection panel.':'Selected celltype; fitted rate requires group and fit support.'} Arrows retain the fixed-panel projection when celltype colors change. Arrow length is a display control.`;await renderCurves(s)}catch(e){$('error').textContent=e.stack||String(e);console.error(e)}finally{busy=false;if(renderPending){renderPending=false;await render()}}}
async function renderCurves(s){const peak=DATA.peaks[selected],obs=curve(A.observed,selected,s.c),fit=curve(A.fitted,selected,s.c),mode=DATA.modes[String(A.mode[selected*C+s.c])];$('selected').textContent=peak.peak+' · '+(peak.nearest_gene||peak.linked_gene||'')+' · '+DATA.celltypes[s.c]+' · '+(mode?.label||'unlabelled');const traces=[{x:DATA.stages,y:obs,type:'scatter',mode:'markers',name:'Observed CPM',marker:{size:8,color:'#1d354d'}}];if(A.fitvalid[selected*C+s.c])traces.push({x:DATA.stages,y:fit,type:'scatter',mode:'lines+markers',name:'Fitted stage values',line:{color:mode?.color||'#ba713a',width:2},connectgaps:false});await Plotly.react('curve',traces,{margin:{l:58,r:20,t:30,b:45},xaxis:{title:'Stage (hpf)',tickvals:DATA.stages},yaxis:{title:'Relative accessibility (CPM)',rangemode:'tozero'},legend:{orientation:'h',y:1.15},uirevision:'curve'+selected+':'+s.c},{responsive:true,displaylogo:false});await Plotly.react('heatmap',[{type:'heatmap',x:DATA.stages,y:DATA.celltypes,z:DATA.celltypes.map((_,c)=>curve(A.observed,selected,c)),colorscale:'Viridis',colorbar:{title:'CPM',thickness:10},hovertemplate:'%{y}<br>%{x} hpf<br>%{z:.3g} CPM<extra></extra>',hoverongaps:false}],{margin:{l:135,r:40,t:30,b:40},title:{text:'Observed accessibility · all celltypes',font:{size:13}},xaxis:{title:'Stage (hpf)',tickvals:DATA.stages},yaxis:{tickfont:{size:8},automargin:true},plot_bgcolor:'#d5dbe1'},{responsive:true,displaylogo:false})}
for(const id of ['dimension','field','interval','celltype','color','density','length'])$(id).addEventListener('change',render);
$('find').onclick=()=>{const q=$('search').value.trim().toLowerCase();if(!q)return;const matches=DATA.peaks.flatMap((p,i)=>Object.values(p).some(x=>typeof x==='string'&&x.toLowerCase().includes(q))?[i]:[]);if(!matches.length){$('status').textContent='No matching peak in the exported display subset; the full atlas remains in the model artifact.';return}lastHit=q===lastSearch?(lastHit+1)%matches.length:0;lastSearch=q;selected=matches[lastHit];render()};$('search').onkeydown=e=>{if(e.key==='Enter')$('find').click()};$('save').onclick=()=>Plotly.downloadImage('atlas',{format:'png',filename:'peak_accessibility_projection',width:1600,height:1100});
(async()=>{await render();$('atlas').on('plotly_click',ev=>{const i=ev.points[0]?.customdata;if(Number.isInteger(i)){selected=i;render()}})})();
</script></body></html>'''
