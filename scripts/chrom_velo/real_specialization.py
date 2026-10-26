"""Align real pseudobulks and compute supported specificity/accessibility rates.

No synthetic replicates, uncertainty estimates, or ground truth are manufactured.
Specificity is calculated from nonnegative normalized accessibility (for example
CPM). Its default has no pseudocount or shrinkage and is scale invariant. Signed
accessibility rates retain the supplied normalization's units per hpf.
"""

import numpy as np
import pandas as pd

from specialization_flow import temporal_specificity
from temporal_velocity import STAGES_HPF, SOMITES_TO_HPF


def parse_condition_grid(var, celltype_col='annotation_ML_coarse', stage_col='dev_stage',
                         n_cells_col='n_cells', total_coverage_col='total_coverage',
                         stages=STAGES_HPF):
    """Strictly map explicit condition metadata to a celltype-by-stage grid.

    Stage labels are the six known somite labels, or numeric hpf. Celltype names
    are read from metadata, never split from strings containing underscores.
    Each condition must have a unique (celltype, stage) pair. Missing pairs have
    column_index=-1, present=False, and NaN metadata; they are never filled with
    zero accessibility. Celltypes use lexical order; stages use supplied order.
    """
    if not isinstance(var, pd.DataFrame) or len(var) < 1:
        raise ValueError('Condition metadata must be a nonempty DataFrame')
    required = [celltype_col, stage_col, n_cells_col, total_coverage_col]
    missing = [key for key in required if key not in var]
    if missing:
        raise ValueError(f'Missing required condition metadata columns: {missing}')
    stages = np.asarray(stages, dtype=float)
    if (stages.ndim != 1 or len(stages) < 2 or not np.isfinite(stages).all()
            or np.any(np.diff(stages) <= 0)):
        raise ValueError('Stages must be finite and strictly increasing')
    if var[celltype_col].isna().any() or var[stage_col].isna().any():
        raise ValueError('Celltype and stage metadata cannot be missing')
    celltype_labels = var[celltype_col].astype(str).to_numpy()
    if np.any(np.char.strip(celltype_labels.astype(str)) == ''):
        raise ValueError('Celltype names cannot be empty')
    times = []
    for label in var[stage_col]:
        if isinstance(label, str) and label in SOMITES_TO_HPF:
            time = SOMITES_TO_HPF[label]
        else:
            try:
                time = float(label)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Unrecognized developmental stage: {label}') from exc
        if not np.isfinite(time) or time not in stages:
            raise ValueError(f'Unscheduled developmental stage: {label}')
        times.append(time)
    times = np.asarray(times)
    pairs = list(zip(celltype_labels, times))
    if len(set(pairs)) != len(pairs):
        raise ValueError('Duplicate celltype-stage conditions; aggregate explicitly before analysis')
    n_cells_values = pd.to_numeric(var[n_cells_col], errors='raise').to_numpy(float)
    coverage_values = pd.to_numeric(var[total_coverage_col], errors='raise').to_numpy(float)
    if not np.isfinite(n_cells_values).all() or np.any(n_cells_values < 0):
        raise ValueError('n_cells must be finite and nonnegative')
    if not np.isfinite(coverage_values).all() or np.any(coverage_values <= 0):
        raise ValueError('total_coverage must be finite and positive for every present condition')
    celltypes = np.asarray(sorted(set(celltype_labels)), dtype=str)
    shape = (len(celltypes), len(stages))
    column_index = np.full(shape, -1, dtype=int)
    n_cells = np.full(shape, np.nan)
    total_coverage = np.full(shape, np.nan)
    ct_lookup = {c: i for i, c in enumerate(celltypes)}
    stage_lookup = {t: i for i, t in enumerate(stages)}
    for j, (c, t) in enumerate(pairs):
        idx = (ct_lookup[c], stage_lookup[t])
        column_index[idx] = j
        n_cells[idx] = n_cells_values[j]
        total_coverage[idx] = coverage_values[j]
    return dict(celltypes=celltypes, stages=stages.copy(), column_index=column_index,
                n_cells=n_cells, total_coverage=total_coverage, present=column_index >= 0,
                n_conditions=len(var), condition_names=var.index.astype(str).to_numpy())


def to_condition_cube(values, condition_grid):
    """Map peaks-by-present-conditions to peaks-by-celltypes-by-stages.

    Returns float64 for intermediate calculations. Missing columns remain NaN;
    measured zeros remain zero. The parent pipeline can process bounded row
    blocks, so this function never needs the entire peak atlas in memory.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != condition_grid['n_conditions']:
        raise ValueError('Values must have one column per parsed condition')
    columns = np.asarray(condition_grid['column_index'])
    cube = np.full((len(values), *columns.shape), np.nan, dtype=float)
    present = columns >= 0
    cube[:, present] = values[:, columns[present]]
    return cube


def _specificity_statistics(normalized, raw, observed, stages, minimum, shrinkage, method):
    """Gate each stage and paired comparison on its own observed raw totals."""
    result = temporal_specificity(normalized, stages, observed=observed,
                                  signal_floor=0., pseudocount=0.,
                                  shrinkage=shrinkage, method=method)
    raw_total = np.where(observed, raw, 0.).sum(axis=1)
    common = observed[:, :, :-1] & observed[:, :, 1:]
    paired_start_raw = np.where(common, raw[:, :, :-1], 0.).sum(axis=1)
    paired_end_raw = np.where(common, raw[:, :, 1:], 0.).sum(axis=1)
    valid = result['valid'] & (raw_total >= minimum)
    interval_valid = (result['interval_valid'] & (paired_start_raw >= minimum)
                      & (paired_end_raw >= minimum))
    # Endpoint positivity is already enforced inside temporal_specificity. A
    # stage with no normalized signal never becomes a zero-specificity reference.
    return dict(
        score=np.where(valid, result['score'], np.nan).astype('float32'),
        rate=np.where(interval_valid, result['interval_rate'], np.nan).astype('float32'),
        valid=valid,
        interval_valid=interval_valid,
        n_celltypes=result['n_celltypes'].astype('int16'),
        paired_celltypes=result['paired_celltypes'].astype('int16'),
        mean_accessibility=result['mean_accessibility'].astype('float32'),
        raw_total=raw_total.astype('float32'),
        paired_start_raw_total=paired_start_raw.astype('float32'),
        paired_end_raw_total=paired_end_raw.astype('float32'),
        paired_start=np.where(paired_start_raw >= minimum, result['paired_start'], np.nan).astype('float32'),
        paired_end=np.where(paired_end_raw >= minimum, result['paired_end'], np.nan).astype('float32'),
    )


def compute_block_statistics(normalized, raw_counts, condition_grid,
                             min_cells=20, min_total_counts=20, shrinkage=0.):
    """Compute fixed-support primary estimates and available-support sensitivity.

    Inputs are matching peaks-by-present-conditions blocks. Nonfinite entries
    are unsupported; finite negative values are rejected. ``normalized`` may be
    CPM or another explicitly declared comparable linear normalization, never
    separately standardized stage features. Raw counts supply evidence gates.

    ``fixed_common`` restricts to celltypes meeting min_cells at every scheduled
    stage. A peak-specific missing measurement excludes that celltype across all
    stages for that peak, preserving fixed support. ``available`` uses eligible
    celltypes separately at each stage and matched observed celltypes for each
    adjacent-stage rate. Its last score is the terminal available-support
    alternative. Both groups contain tau/entropy score, rate, support masks,
    celltype counts, and raw-count totals for each compared endpoint.

    A specificity stage needs summed raw counts >= min_total_counts, positive
    normalized signal, and >=2 observed celltypes. A specificity rate needs both
    paired endpoints to satisfy these requirements on the SAME celltypes.

    Celltype opening rates use all eligible condition pairs. They require
    combined endpoint raw counts >= min_total_counts, retaining supported
    zero-to-opening and closing-to-zero events. They do not require positive
    accessibility at both endpoints. No absent/low-cell-count stage is bridged.
    Numerical estimates are float32; masks are boolean and counts are int16.
    There are no confidence intervals, SEs, or invented replicate information.
    """
    if not np.isfinite(min_cells) or min_cells < 0:
        raise ValueError('min_cells must be finite and nonnegative')
    if not np.isfinite(min_total_counts) or min_total_counts < 0:
        raise ValueError('min_total_counts must be finite and nonnegative')
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError('shrinkage must lie in [0, 1]')
    normalized = np.asarray(normalized, dtype=float)
    raw_counts = np.asarray(raw_counts, dtype=float)
    if normalized.shape != raw_counts.shape:
        raise ValueError('Normalized accessibility and raw count blocks must match')
    if np.any(normalized[np.isfinite(normalized)] < 0) or np.any(raw_counts[np.isfinite(raw_counts)] < 0):
        raise ValueError('Normalized accessibility and raw counts must be nonnegative')
    a = to_condition_cube(normalized, condition_grid)
    raw = to_condition_cube(raw_counts, condition_grid)
    available_conditions = condition_grid['present'] & (condition_grid['n_cells'] >= min_cells)
    common_celltypes = available_conditions.all(axis=1)
    finite = np.isfinite(a) & np.isfinite(raw)
    available_observed = finite & available_conditions[None, :, :]
    complete_peak_celltypes = finite.all(axis=2) & common_celltypes[None, :]
    fixed_observed = np.broadcast_to(complete_peak_celltypes[:, :, None], a.shape)
    stages = condition_grid['stages']
    groups = {}
    for name, observed in [('fixed_common', fixed_observed), ('available', available_observed)]:
        groups[name] = {
            method: _specificity_statistics(a, raw, observed, stages, min_total_counts,
                                           shrinkage, method)
            for method in ('tau', 'entropy')
        }
    paired_observed = available_observed[:, :, :-1] & available_observed[:, :, 1:]
    opening_raw_total = np.where(paired_observed, raw[:, :, :-1] + raw[:, :, 1:], np.nan)
    opening_valid = paired_observed & (opening_raw_total >= min_total_counts)
    rate = (a[:, :, 1:] - a[:, :, :-1]) / np.diff(stages)
    groups.update(
        common_celltype_mask=common_celltypes,
        available_condition_mask=available_conditions,
        n_common=int(common_celltypes.sum()),
        observed_count=available_observed.sum(axis=1).astype('int16'),
        fixed_observed_count=complete_peak_celltypes.sum(axis=1).astype('int16'),
        opening_rate=np.where(opening_valid, rate, np.nan).astype('float32'),
        opening_valid=opening_valid,
        opening_raw_total=opening_raw_total.astype('float32'),
    )
    return groups
