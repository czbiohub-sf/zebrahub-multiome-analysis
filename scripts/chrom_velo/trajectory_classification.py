"""Describe fitted interval signs without turning an unresolved reversal monotone.

The codes match trajectory_modes.MODE_LABELS. Classifications summarize the
supported signs of fitted accessibility changes; they do not establish temporal
causality or biological-replicate confidence.
"""
from __future__ import annotations

import numpy as np


def classify_supported_modes(rates, supported, quality_valid=None, rate_floor=.05):
    """Return an int8 mode for each trajectory shaped ``(..., intervals)``.

    Codes: 0 insufficient, 1 no supported change, 2 increasing, 3 decreasing,
    4 transient peak, 5 transient trough, 6 uncertain/complex. NaN intervals are
    unobserved, never bridged by a numerical derivative. A reversal visible in
    the fitted rates above ``rate_floor`` must have support for both signs to
    receive a transient label. A one-sided supported leg of such a reversal is
    uncertain, rather than a monotone change. Multiple fitted reversals are also
    uncertain whenever any change is supported. No supported change is a lack
    of evidence for change, not proof that a peak is constant.

    ``quality_valid`` applies to whole trajectories, e.g. a conditional
    lack-of-fit gate. Failed quality gives code 6 unless every rate is missing
    (code 0). Missing intervals can leave a reversal's timing unresolved.
    """
    rates = np.asarray(rates, dtype=float)
    supported = np.asarray(supported, dtype=bool)
    if rates.ndim < 1 or rates.shape[-1] < 1 or supported.shape != rates.shape:
        raise ValueError('rates and supported must have the same (..., intervals) shape')
    if not np.isfinite(rate_floor) or rate_floor < 0:
        raise ValueError('rate_floor must be finite and nonnegative')
    if np.isinf(rates).any():
        raise ValueError('Infinite rates are invalid; use NaN for missing intervals')
    shape = rates.shape[:-1]
    quality = np.ones(shape, bool) if quality_valid is None else np.broadcast_to(
        np.asarray(quality_valid, dtype=bool), shape)
    flat = rates.reshape(-1, rates.shape[-1])
    finite = np.isfinite(flat)
    actual = np.where(finite & (np.abs(flat) > rate_floor), np.sign(flat), 0.)
    observed_signs = np.where(supported.reshape(flat.shape), actual, 0.)
    has_positive = (observed_signs > 0).any(axis=1)
    has_negative = (observed_signs < 0).any(axis=1)
    any_supported = has_positive | has_negative
    mode = np.ones(len(flat), dtype=np.int8)
    mode[has_positive & ~has_negative] = 2
    mode[has_negative & ~has_positive] = 3
    actual_mixed = (actual > 0).any(axis=1) & (actual < 0).any(axis=1)
    partial_reversal = actual_mixed & (has_positive ^ has_negative)
    mode[partial_reversal] = 6
    for i in np.flatnonzero(actual_mixed & any_supported):
        full_sequence = actual[i, actual[i] != 0]
        if np.count_nonzero(np.diff(full_sequence)) > 1:
            mode[i] = 6
        elif has_positive[i] and has_negative[i]:
            mode[i] = 4 if full_sequence[0] > 0 else 5
    mode[~quality.reshape(-1)] = 6
    mode[~finite.any(axis=1)] = 0
    return mode.reshape(shape)
