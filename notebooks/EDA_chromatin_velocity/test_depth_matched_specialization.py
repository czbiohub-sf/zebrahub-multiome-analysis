"""Focused scientific invariants for matched-exposure specificity."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/chrom_velo'))
from depth_matched_specialization import estimate_depth_matched_block
from real_specialization import parse_condition_grid


TIMES = [10, 12, 14, 16, 19, 24]


def grid_for(coverage=None):
    if coverage is None:
        coverage = np.full((3, 6), 10000.)
    frame = pd.DataFrame([dict(annotation_ML_coarse=f'ct{c}', dev_stage=t,
                              n_cells=100, total_coverage=coverage[c, k])
                          for c in range(3) for k, t in enumerate(TIMES)])
    return parse_condition_grid(frame), frame


def manual(draw, target, method):
    x = draw / target[None, :, None]
    total = x.sum(axis=1)
    if method == 'tau':
        return np.sum(1 - x / x.max(axis=1)[:, None], axis=1) / (x.shape[1]-1)
    p = x / total[:, None]
    return 1 + np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0).sum(axis=1) / np.log(x.shape[1])


class DepthMatchedTests(unittest.TestCase):
    def test_equal_exposure_preserves_flat_and_specific_and_irregular_rates(self):
        grid, _ = grid_for()
        raw = np.full((3, 3, 6), 30.)
        raw[1, 1:] = 0
        raw[2, 1:, 1:] = 0
        result = estimate_depth_matched_block(raw.reshape(3, -1), grid, 0)
        for method in ('tau', 'entropy'):
            np.testing.assert_allclose(result[method]['score'][0], 0., atol=1e-7)
            np.testing.assert_allclose(result[method]['score'][1], 1.)
            np.testing.assert_allclose(result[method]['rate'][1], 0.)
            np.testing.assert_allclose(result[method]['rate'][2], [.5, 0, 0, 0, 0])
            np.testing.assert_allclose(result[method]['thinning_sign_agreement'], 1.)
        np.testing.assert_array_equal(result['thinning_probabilities'], 1.)


    def test_exact_seed_draws_independent_formula_and_all_draw_gate(self):
        coverage = np.array([[100, 200, 300, 400, 500, 1000],
                             [200, 200, 400, 600, 800, 1000],
                             [100, 500, 300, 200, 400, 1000.]])
        grid, _ = grid_for(coverage)
        raw = np.array([coverage / 5, coverage / 25, coverage / 100, coverage * 0]).astype(int)
        kwargs = dict(global_start=512, seed=19, n_thinnings=5, min_total_counts=20)
        out = estimate_depth_matched_block(raw.reshape(4, -1), grid, **kwargs)
        target = coverage.min(1)
        draws = np.stack([np.random.default_rng(np.random.SeedSequence([19, 512, r])).binomial(
            raw, (target[:, None] / coverage)[None]) for r in range(5)])
        totals = draws.sum(2)
        np.testing.assert_array_equal(out['min_raw_total'], totals.min(0))
        np.testing.assert_allclose(out['mean_raw_total'], totals.mean(0))
        assert np.any((totals.min(0) < 20) & (totals.max(0) >= 20))
        valid = (totals >= 20).all(0)
        for method in ('tau', 'entropy'):
            with np.errstate(invalid='ignore', divide='ignore'):
                scores = np.stack([manual(d, target, method) for d in draws])
            expected = np.where(valid, scores.mean(0), np.nan)
            np.testing.assert_allclose(out[method]['score'], expected, atol=1e-7, equal_nan=True)
            rate = np.diff(expected, axis=1) / np.diff(TIMES)
            np.testing.assert_allclose(out[method]['rate'], rate, atol=1e-7, equal_nan=True)
            agreement = np.mean(np.sign(np.diff(scores, axis=2)) == np.sign(rate)[None], axis=0)
            agreement[~(valid[:, :-1] & valid[:, 1:])] = np.nan
            np.testing.assert_allclose(out[method]['thinning_sign_agreement'], agreement, equal_nan=True)
        repeat = estimate_depth_matched_block(raw.reshape(4, -1), grid, **kwargs)
        np.testing.assert_array_equal(out['tau']['score'], repeat['tau']['score'])
        changed = estimate_depth_matched_block(raw.reshape(4, -1), grid, **(kwargs | {'global_start': 1024}))
        assert not np.array_equal(out['mean_raw_total'], changed['mean_raw_total'])


    def test_column_permutation_and_low_cell_panel_exclusion(self):
        grid, frame = grid_for(np.tile([100, 200, 300, 400, 500, 600.], (3, 1)))
        frame.loc[0, 'n_cells'] = 1
        grid = parse_condition_grid(frame)
        raw = np.arange(18)[None] * 20
        out = estimate_depth_matched_block(raw, grid, 0)
        permutation = np.random.default_rng(12).permutation(18)
        permuted_grid = parse_condition_grid(frame.iloc[permutation])
        other = estimate_depth_matched_block(raw[:, permutation], permuted_grid, 0)
        np.testing.assert_array_equal(out['common_celltype_mask'], [False, True, True])
        np.testing.assert_array_equal(out['tau']['score'], other['tau']['score'])
        missing_grid = parse_condition_grid(frame.drop(index=0))
        missing = estimate_depth_matched_block(raw[:, 1:], missing_grid, 0)
        np.testing.assert_array_equal(out['tau']['score'], missing['tau']['score'])


    def test_invalid_counts_rejected(self):
        grid, _ = grid_for()
        for bad in [-1, .5, np.nan, np.inf]:
            raw = np.full((1, 18), 30.)
            raw[0, 0] = bad
            with np.testing.assert_raises_regex(ValueError, 'Raw counts'):
                estimate_depth_matched_block(raw, grid, 0)


    def test_zero_signal_invalid_even_with_zero_count_gate(self):
        grid, _ = grid_for()
        result = estimate_depth_matched_block(np.zeros((2, 18)), grid, 0, min_total_counts=0)
        for method in ('tau', 'entropy'):
            assert not result[method]['valid'].any()
            assert not result[method]['interval_valid'].any()
            assert np.isnan(result[method]['thinning_sign_agreement']).all()


    def test_insufficient_fixed_panel_rejected(self):
        _, frame = grid_for()
        frame.loc[[0, 6], 'n_cells'] = 0
        grid = parse_condition_grid(frame)
        with np.testing.assert_raises_regex(ValueError, 'At least two'):
            estimate_depth_matched_block(np.ones((1, 18)), grid, 0)


    def test_verifier_recreates_nonconsecutive_and_partial_source_blocks(self):
        import tempfile
        import h5py
        from verify_real_specialization import check_depth_matched_arithmetic
        coverage = np.tile([100, 200, 300, 400, 500, 600.], (3, 1))
        grid, _ = grid_for(coverage)
        raw = np.random.default_rng(81).integers(10, 200, (10, 18))
        blocks = [estimate_depth_matched_block(raw[start:start+4], grid, start, seed=12)
                  for start in range(0, 10, 4)]
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory)/'saved.h5', 'w') as saved, h5py.File(Path(directory)/'source.h5', 'w') as source:
                source.create_dataset('layers/sum', data=raw)
                info = saved.create_group('uns/real_specialization/depth_matching')
                for key, value in dict(seed=12, n_thinnings=5, block_size=4,
                                       min_total_counts=20, min_cells=20).items():
                    info.create_dataset(key, data=value)
                for key in ['target_coverage', 'thinning_probabilities']:
                    info.create_dataset(key, data=blocks[0][key])
                info.create_dataset('celltypes', data=blocks[0]['celltypes'].astype(object), dtype=h5py.string_dtype())
                for key in ['mean_raw_total', 'min_raw_total']:
                    saved.create_dataset('obsm/depth_matched_'+key, data=np.concatenate([b[key] for b in blocks]))
                for method in ['tau', 'entropy']:
                    for result_key, stored_key in [('score', 'specificity'), ('rate', 'rate'),
                                                    ('thinning_sign_agreement', 'thinning_sign_agreement')]:
                        saved.create_dataset('obsm/'+method+'_'+stored_key,
                                             data=np.concatenate([b[method][result_key] for b in blocks]))
                checks = {}
                check_depth_matched_arithmetic(saved, source, grid, np.array([0, 4, 9]), checks)
                assert checks['independent_depth_matched_arithmetic']['passed']
                assert checks['independent_depth_matched_arithmetic']['recreated_source_rows'] == 10


if __name__ == '__main__':
    unittest.main()
