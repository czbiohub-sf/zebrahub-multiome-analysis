"""Temporal specificity after matching read exposure within each celltype.

Independent binomial thinnings are Monte Carlo sensitivity draws, not biological
replicates. They equalize the total-coverage exposure across stages separately
for each celltype; they do not equalize celltypes or imply confidence intervals.
"""
import numpy as np

from real_specialization import to_condition_cube
from specialization_flow import temporal_specificity


def estimate_depth_matched_block(raw_counts, grid, global_start, seed=4242,
                                 n_thinnings=5, min_cells=20,
                                 min_total_counts=20):
    """Estimate tau/entropy and per-hpf rates on a fixed complete celltype panel.

    For each eligible celltype the target exposure is its minimum coverage over
    all scheduled stages. Each count is thinned independently with probability
    target/exposure. CPM uses the celltype's target exposure. A stage is reported
    only if EVERY draw has positive signal and >=min_total_counts pooled reads.
    Adjacent-stage rates are differences of the mean scores divided by actual
    stage separation; missing support is never bridged.

    Random streams use SeedSequence([seed, global_start, replicate_index]) and
    C-order arrays (peaks, eligible celltypes, stages). Reproducibility therefore
    requires recording the processing block size as well as seed and draw count.
    Sign agreement is the fraction of draw-specific rates with the displayed
    mean rate's sign, conditional on both endpoint stages passing every draw.
    It measures Monte Carlo sign stability only, not biological confidence.
    """
    for name, value in [('min_cells', min_cells), ('min_total_counts', min_total_counts)]:
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    for name, value in [('seed', seed), ('global_start', global_start), ('n_thinnings', n_thinnings)]:
        if not np.isscalar(value) or not np.isfinite(value) or int(value) != value or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if n_thinnings < 1:
        raise ValueError('n_thinnings must be at least one')
    seed, global_start, n_thinnings = int(seed), int(global_start), int(n_thinnings)
    raw_counts = np.asarray(raw_counts, dtype=float)
    if (not np.isfinite(raw_counts).all() or np.any(raw_counts < 0)
            or np.any(raw_counts != np.floor(raw_counts))
            or np.any(raw_counts >= np.iinfo(np.int64).max)):
        raise ValueError('Raw counts must be finite, nonnegative integers below int64 maximum')
    eligible = np.asarray(grid['present']) & (np.asarray(grid['n_cells']) >= min_cells)
    common = eligible.all(axis=1)
    if common.sum() < 2:
        raise ValueError('At least two celltypes must be eligible at every stage')
    raw = to_condition_cube(raw_counts, grid)[:, common, :].astype(np.int64)
    exposure = np.asarray(grid['total_coverage'], dtype=float)[common]
    if not np.isfinite(exposure).all() or np.any(exposure <= 0):
        raise ValueError('Eligible conditions need finite positive total coverage')
    target = exposure.min(axis=1)
    probability = target[:, None] / exposure
    stages = np.asarray(grid['stages'], dtype=float)
    dt = np.diff(stages)
    shape = (n_thinnings, len(raw), len(stages))
    totals = np.empty(shape, dtype=np.float64)
    scores = {method: np.empty(shape, dtype=np.float64) for method in ('tau', 'entropy')}
    valid = {method: np.empty(shape, dtype=bool) for method in scores}
    for rep in range(n_thinnings):
        rng = np.random.default_rng(np.random.SeedSequence([seed, global_start, rep]))
        draw = rng.binomial(raw, probability[None, :, :])
        totals[rep] = draw.sum(axis=1)
        cpm = draw.astype(float) * (1e6 / target[None, :, None])
        for method in scores:
            result = temporal_specificity(cpm, stages, signal_floor=0.,
                                          pseudocount=0., shrinkage=0., method=method)
            scores[method][rep] = result['score']
            valid[method][rep] = result['valid'] & (totals[rep] >= min_total_counts)
    output = {}
    for method in scores:
        stage_valid = valid[method].all(axis=0)
        # Zero-total draws have NaN scores; they invalidate the final stage.
        mean_score = np.mean(scores[method], axis=0)
        mean_score = np.where(stage_valid, mean_score, np.nan)
        interval_valid = stage_valid[:, :-1] & stage_valid[:, 1:]
        rate = np.diff(mean_score, axis=1) / dt
        draw_rates = np.diff(scores[method], axis=2) / dt
        agreement = np.mean(np.sign(draw_rates) == np.sign(rate)[None], axis=0)
        output[method] = dict(
            score=mean_score.astype('float32'), rate=rate.astype('float32'),
            valid=stage_valid, interval_valid=interval_valid,
            thinning_sign_agreement=np.where(interval_valid, agreement, np.nan).astype('float32'))
    output.update(mean_raw_total=totals.mean(axis=0).astype('float32'),
                  min_raw_total=totals.min(axis=0).astype('float32'),
                  common_celltype_mask=common, celltypes=np.asarray(grid['celltypes'])[common],
                  stages=stages.copy(), target_coverage=target.copy(),
                  thinning_probabilities=probability.copy(),
                  n_common=int(common.sum()), n_thinnings=n_thinnings,
                  seed=seed, global_start=global_start)
    return output
