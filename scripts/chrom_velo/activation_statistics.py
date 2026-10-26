"""Conditional temporal-order diagnostics for fixed peak-atlas summaries.

The six measured developmental snapshots are permuted together across every
celltype/region series. The atlas and observed snapshots stay fixed, preserving
cross-peak/celltype structure. Actual hours stay fixed, including unequal gaps.
These exact permutation fractions describe how unusual the observed ordering is
among 720 relabelings, conditional on the atlas and summaries. They are NOT
biological replicate p-values: developmental snapshots are not exchangeable
replicates, and this atlas was constructed from the same accessibility profiles.
"""
from __future__ import annotations

from itertools import permutations
import math
import numpy as np
from scipy.stats import spearmanr


def _actual_time_statistics(values, times, min_observations):
    """Equal-stage OLS slope and correlation with actual hours; NaNs stay absent."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    count = finite.sum(axis=-1)
    safe_count = np.maximum(count, 1)
    x_mean = np.sum(np.where(finite, times, 0.), axis=-1) / safe_count
    y_mean = np.sum(np.where(finite, values, 0.), axis=-1) / safe_count
    x = np.where(finite, times - x_mean[..., None], 0.)
    y = np.where(finite, values - y_mean[..., None], 0.)
    xx = np.sum(x * x, axis=-1)
    yy = np.sum(y * y, axis=-1)
    # Aggregated distance centers can differ only by floating-point roundoff
    # under uniform accessibility rescaling. Do not call that temporal structure.
    magnitude = np.maximum(1., np.max(np.where(finite, np.abs(values), 0.), axis=-1))
    numerical_constant = yy <= count * (64 * np.finfo(float).eps * magnitude) ** 2
    yy = np.where(numerical_constant, 0., yy)
    xy = np.where(numerical_constant, 0., np.sum(x * y, axis=-1))
    valid = (count >= min_observations) & (xx > 0)
    slope = np.divide(xy, xx, out=np.full_like(xy, np.nan), where=valid)
    denominator = np.sqrt(xx * yy)
    correlation = np.divide(xy, denominator, out=np.zeros_like(xy), where=denominator > 0)
    correlation = np.clip(correlation, -1., 1.)
    correlation[~valid] = np.nan
    return slope, correlation, count


def temporal_permutation_test(values, times, *, min_observations=4):
    """Enumerate a shared stage-label reference for one or many time series.

    Parameters
    ----------
    values : array [..., T]
        Prefer measured CPM-weighted graph-distance centers on a fixed peak set.
        The final dimension indexes developmental snapshots, not peak observations.
        Missing snapshots move with their values during each shared permutation.
    times : array [T]
        Strictly increasing measured hours. Unequal gaps are used in each OLS fit.
    min_observations : int
        Minimum finite snapshots per series. A fixed complete panel is preferred.

    Returns
    -------
    dict
        Per-series arrays retain the leading shape of ``values``. Slopes have
        input-value/hour units; correlation is unitless. All exact reference
        fractions include the observed permutation, with ties counted. The
        familywise two-sided reference uses the maximum |correlation| over the
        entire supplied family on each same permutation. Global statistics are
        the equal-series mean correlation, so no peak-independence is assumed.
        No multiple-testing family is silently split by celltype or region.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.ndim < 1 or times.ndim != 1 or values.shape[-1] != len(times):
        raise ValueError('values final dimension must match one-dimensional times')
    if not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('times must be finite and strictly increasing')
    if not 2 <= min_observations <= len(times):
        raise ValueError('min_observations must be between 2 and number of stages')
    if len(times) > 8:
        raise ValueError('Exact enumeration is restricted to at most eight stages')
    leading_shape = values.shape[:-1]
    flat = values.reshape(-1, len(times))
    orderings = np.array(list(permutations(range(len(times)))), dtype=np.int16)
    slope, observed, count = _actual_time_statistics(flat, times, min_observations)
    null = np.empty((len(orderings), len(flat)), dtype=float)
    null_slope = np.empty_like(null)
    for p, ordering in enumerate(orderings):
        null_slope[p], null[p], _ = _actual_time_statistics(flat[:, ordering], times, min_observations)
    valid = np.isfinite(observed)
    tolerance = 1e-12
    denominator = np.isfinite(null).sum(axis=0)
    def fraction(comparison):
        value = np.divide(comparison.sum(axis=0), denominator,
                          out=np.full(len(flat), np.nan), where=denominator > 0)
        value[~valid] = np.nan
        return value.reshape(leading_shape)
    outward = fraction(null >= observed[None, :] - tolerance)
    inward = fraction(null <= observed[None, :] + tolerance)
    two_sided = fraction(np.abs(null) >= np.abs(observed)[None, :] - tolerance)
    if valid.any():
        null_max = np.max(np.abs(null[:, valid]), axis=1)
        null_global = np.mean(null[:, valid], axis=1)
        global_observed = float(np.mean(observed[valid]))
        null_global_slope = np.mean(null_slope[:, valid], axis=1)
        global_slope = float(np.mean(slope[valid]))
        null_fraction_positive = np.mean(null[:, valid] > tolerance, axis=1)
        fraction_positive = float(np.mean(observed[valid] > tolerance))
        fwer = fraction(null_max[:, None] >= np.abs(observed)[None, :] - tolerance)
        global_result = dict(
            mean_correlation=global_observed, mean_slope=global_slope,
            mean_slope_p_outward=float(np.mean(null_global_slope >= global_slope - tolerance)),
            mean_slope_p_inward=float(np.mean(null_global_slope <= global_slope + tolerance)),
            mean_slope_p_two_sided=float(np.mean(np.abs(null_global_slope) >= abs(global_slope) - tolerance)),
            fraction_positive_p_outward=float(np.mean(null_fraction_positive >= fraction_positive - tolerance)),
            fraction_positive_p_inward=float(np.mean(null_fraction_positive <= fraction_positive + tolerance)),
            p_outward=float(np.mean(null_global >= global_observed - tolerance)),
            p_inward=float(np.mean(null_global <= global_observed + tolerance)),
            p_two_sided=float(np.mean(np.abs(null_global) >= abs(global_observed) - tolerance)),
            fraction_positive=fraction_positive,
            fraction_negative=float(np.mean(observed[valid] < -tolerance)),
            n_series=int(valid.sum()),
        )
    else:
        null_max = np.full(len(orderings), np.nan)
        null_global = np.full(len(orderings), np.nan)
        null_global_slope = np.full(len(orderings), np.nan)
        null_fraction_positive = np.full(len(orderings), np.nan)
        fwer = np.full(leading_shape, np.nan)
        global_result = dict(mean_correlation=np.nan, mean_slope=np.nan,
                             mean_slope_p_outward=np.nan, mean_slope_p_inward=np.nan, mean_slope_p_two_sided=np.nan,
                             fraction_positive_p_outward=np.nan, fraction_positive_p_inward=np.nan, p_outward=np.nan,
                             p_inward=np.nan, p_two_sided=np.nan,
                             fraction_positive=np.nan, fraction_negative=np.nan, n_series=0)
    return dict(
        slope=slope.reshape(leading_shape), correlation=observed.reshape(leading_shape),
        n_observed=count.reshape(leading_shape), p_outward=outward, p_inward=inward,
        p_two_sided=two_sided, p_fwer_two_sided=fwer,
        permutation_indices=orderings, n_permutations=math.factorial(len(times)),
        null_correlation=null.reshape((len(orderings),) + leading_shape),
        null_slope=null_slope.reshape((len(orderings),) + leading_shape),
        null_global_mean_slope=null_global_slope,
        null_global_fraction_positive=null_fraction_positive,
        null_global_mean_correlation=null_global,
        null_max_abs_correlation=null_max, global_result=global_result,
        reference_type='Exact shared stage-label ordering reference, conditional on fixed atlas and observed profiles; not biological replicate inference',
        family_size=int(valid.sum()),
    )


def activation_weight_sensitivity(distance, primary_timing, alternative_timing):
    """Describe paired activation-time sensitivity without claiming ground truth.

    Inputs are one-dimensional matched peaks (or fixed graph bins). Missing timing
    denotes no supported activation, not a zero-hour activation. Distance and timing
    correlations are calculated only on the SAME finite pair set, while separate
    coverage/overlap counts expose changes in selected membership.
    """
    distance, primary, alternative = [np.asarray(x, dtype=float) for x in
                                      (distance, primary_timing, alternative_timing)]
    if distance.ndim != 1 or distance.shape != primary.shape or distance.shape != alternative.shape:
        raise ValueError('distance and both timing arrays must be matching one-dimensional vectors')
    d_valid = np.isfinite(distance)
    p_valid = d_valid & np.isfinite(primary)
    a_valid = d_valid & np.isfinite(alternative)
    both = p_valid & a_valid
    union = p_valid | a_valid
    def correlation(x, y):
        if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return np.nan
        return float(spearmanr(x, y).statistic)
    p, a, d = primary[both], alternative[both], distance[both]
    delta = a - p
    return dict(
        n_total=len(distance), n_primary=int(p_valid.sum()), n_alternative=int(a_valid.sum()),
        n_paired=int(both.sum()), activation_overlap_jaccard=float(both.sum() / union.sum()) if union.any() else np.nan,
        primary_retention=float(both.sum() / p_valid.sum()) if p_valid.any() else np.nan,
        median_abs_timing_shift_hpf=float(np.median(np.abs(delta))) if len(delta) else np.nan,
        p90_abs_timing_shift_hpf=float(np.quantile(np.abs(delta), .9)) if len(delta) else np.nan,
        median_signed_timing_shift_hpf=float(np.median(delta)) if len(delta) else np.nan,
        timing_spearman=correlation(p, a),
        primary_distance_timing_spearman=correlation(d, p),
        alternative_distance_timing_spearman=correlation(d, a),
        scope='Paired finite timings on the same peaks; coverage and overlap reported separately; descriptive sensitivity only',
    )
