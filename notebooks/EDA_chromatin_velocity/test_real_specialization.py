"""Alignment and support invariants for real peak-by-pseudobulk statistics."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'chrom_velo'))
from real_specialization import parse_condition_grid, to_condition_cube, compute_block_statistics


STAGES = ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites']


def metadata():
    rows = [(c, t, 100, 10000.) for c in ['a_cell_type', 'b_cell_type', 'c_cell_type'] for t in STAGES]
    return pd.DataFrame(rows, columns=['annotation_ML_coarse', 'dev_stage', 'n_cells', 'total_coverage'])


class RealSpecializationTests(unittest.TestCase):
    def test_grid_uses_explicit_metadata_and_irregular_stage_mapping(self):
        grid = parse_condition_grid(metadata())
        np.testing.assert_array_equal(grid['stages'], [10, 12, 14, 16, 19, 24])
        self.assertEqual(grid['celltypes'].tolist(), ['a_cell_type', 'b_cell_type', 'c_cell_type'])
        np.testing.assert_array_equal(grid['column_index'], np.arange(18).reshape(3, 6))

    def test_column_permutation_preserves_every_estimate(self):
        var = metadata(); grid = parse_condition_grid(var)
        rng = np.random.default_rng(21)
        normalized = rng.uniform(1, 50, (3, 18)); raw = rng.integers(10, 200, (3, 18))
        original = compute_block_statistics(normalized, raw, grid)
        order = rng.permutation(18)
        permuted = compute_block_statistics(normalized[:, order], raw[:, order], parse_condition_grid(var.iloc[order]))
        for support in ['fixed_common', 'available']:
            for method in ['tau', 'entropy']:
                for key in original[support][method]:
                    np.testing.assert_allclose(original[support][method][key], permuted[support][method][key], equal_nan=True)
        np.testing.assert_allclose(original['opening_rate'], permuted['opening_rate'], equal_nan=True)

    def test_absent_column_is_nan_while_observed_zero_is_retained(self):
        var = metadata(); keep = np.arange(17)
        grid = parse_condition_grid(var.iloc[keep])
        values = np.ones((1, 17)); values[0, 0] = 0.
        cube = to_condition_cube(values, grid)
        self.assertEqual(cube[0, 0, 0], 0.)
        self.assertTrue(np.isnan(cube[0, 2, 5]))
        self.assertFalse(grid['present'][2, 5])
        full = np.repeat(np.array([[10., 5., 0.]])[:, :, None], 6, axis=2).reshape(1, 18)
        counts = np.repeat(np.array([[10., 10., 0.]])[:, :, None], 6, axis=2).reshape(1, 18)
        complete = compute_block_statistics(full, counts, parse_condition_grid(var))
        missing = compute_block_statistics(full[:, keep], counts[:, keep], grid)
        self.assertAlmostEqual(complete['available']['tau']['score'][0, -1], .75)
        self.assertAlmostEqual(missing['available']['tau']['score'][0, -1], .5)

    def test_matched_celltypes_prevent_a_missing_condition_creating_rate(self):
        var = metadata(); keep = np.array([j for j in range(18) if j != 12])
        values = np.repeat(np.array([[10., 5., 10.]])[:, :, None], 6, axis=2).reshape(1, 18)
        result = compute_block_statistics(values[:, keep], np.full((1, 17), 20.), parse_condition_grid(var.iloc[keep]))
        tau = result['available']['tau']
        self.assertNotEqual(tau['score'][0, 0], tau['score'][0, 1])
        np.testing.assert_array_equal(tau['rate'], 0.)
        np.testing.assert_array_equal(tau['paired_celltypes'], [[2, 3, 3, 3, 3]])
        self.assertEqual(result['n_common'], 2)
        np.testing.assert_array_equal(result['fixed_common']['tau']['rate'], 0.)

    def test_paired_count_gate_cannot_borrow_counts_from_excluded_celltypes(self):
        var = metadata(); keep = np.array([j for j in range(18) if j != 12])
        values = np.ones((1, 17))
        full_counts = np.repeat(np.array([[3., 3., 100.]])[:, :, None], 6, axis=2).reshape(1, 18)
        result = compute_block_statistics(values, full_counts[:, keep], parse_condition_grid(var.iloc[keep]))
        tau = result['available']['tau']
        self.assertFalse(tau['interval_valid'][0, 0])
        self.assertTrue(tau['valid'][0, 1])
        self.assertEqual(tau['paired_end_raw_total'][0, 0], 6.)
        self.assertTrue(np.isnan(tau['rate'][0, 0]))
        self.assertFalse(result['fixed_common']['tau']['valid'].any())

    def test_opening_from_zero_uses_combined_counts_and_irregular_time(self):
        grid = parse_condition_grid(metadata())
        values = np.zeros((1, 18)); raw = np.zeros_like(values)
        values[0, :6] = [0., 8., 8., 8., 8., 28.]
        raw[0, :6] = [0., 20., 20., 20., 20., 20.]
        result = compute_block_statistics(values, raw, grid)
        self.assertTrue(result['opening_valid'][0, 0, 0])
        self.assertAlmostEqual(result['opening_rate'][0, 0, 0], 4.)
        self.assertAlmostEqual(result['opening_rate'][0, 0, -1], 4.)
        self.assertFalse(result['available']['tau']['interval_valid'][0, 0])
        self.assertTrue(np.isnan(result['available']['tau']['rate'][0, 0]))

    def test_uniform_opening_does_not_create_specialization(self):
        values = np.tile([1., 2., 3., 4., 5., 6.], (1, 3))
        result = compute_block_statistics(values, np.full_like(values, 20.), parse_condition_grid(metadata()))
        for support in ['fixed_common', 'available']:
            for method in ['tau', 'entropy']:
                np.testing.assert_allclose(result[support][method]['score'], 0., atol=1e-7)
                np.testing.assert_allclose(result[support][method]['rate'], 0., atol=1e-7)
        self.assertTrue((result['opening_rate'] > 0).all())

    def test_min_cells_removes_endpoints_without_bridging(self):
        var = metadata(); var.loc[1, 'n_cells'] = 30
        values = np.full((1, 18), 10.)
        loose = compute_block_statistics(values, values, parse_condition_grid(var), min_cells=20)
        strict = compute_block_statistics(values, values, parse_condition_grid(var), min_cells=50)
        self.assertEqual(loose['n_common'], 3)
        self.assertEqual(strict['n_common'], 2)
        self.assertFalse(strict['opening_valid'][0, 0, :2].any())
        self.assertTrue(strict['opening_valid'][0, 0, 2:].all())

    def test_nonfinite_peak_entry_preserves_fixed_support_and_stays_unsupported(self):
        values = np.full((1, 18), 20.); raw = values.copy(); values[0, 1] = np.nan
        result = compute_block_statistics(values, raw, parse_condition_grid(metadata()))
        self.assertEqual(result['n_common'], 3)
        self.assertEqual(result['fixed_observed_count'][0], 2)
        np.testing.assert_array_equal(result['fixed_common']['tau']['n_celltypes'], [[2] * 6])
        self.assertFalse(result['opening_valid'][0, 0, :2].any())

    def test_no_signal_remains_invalid_even_when_count_gate_disabled(self):
        values = np.zeros((1, 18))
        result = compute_block_statistics(values, values, parse_condition_grid(metadata()), min_total_counts=0)
        self.assertFalse(result['available']['tau']['valid'].any())
        self.assertFalse(result['fixed_common']['entropy']['interval_valid'].any())

    def test_scale_invariance_shrinkage_and_float32_outputs(self):
        values = np.repeat(np.array([[20., 4., 1.]])[:, :, None], 6, axis=2).reshape(1, 18)
        counts = np.full_like(values, 20.)
        grid = parse_condition_grid(metadata())
        original = compute_block_statistics(values, counts, grid, shrinkage=.05)
        scaled = compute_block_statistics(values * 100, counts, grid, shrinkage=.05)
        for method in ['tau', 'entropy']:
            np.testing.assert_allclose(original['fixed_common'][method]['score'], scaled['fixed_common'][method]['score'])
            self.assertEqual(original['fixed_common'][method]['score'].dtype, np.dtype('float32'))
        self.assertEqual(original['opening_rate'].dtype, np.dtype('float32'))

    def test_duplicates_unknown_stages_and_invalid_values_rejected(self):
        var = metadata()
        with self.assertRaises(ValueError):
            parse_condition_grid(pd.concat([var, var.iloc[[0]]]))
        bad = var.copy(); bad.loc[0, 'dev_stage'] = 'unknown'
        with self.assertRaises(ValueError):
            parse_condition_grid(bad)
        bad = var.copy(); bad.loc[0, 'total_coverage'] = 0.
        with self.assertRaises(ValueError):
            parse_condition_grid(bad)
        with self.assertRaises(ValueError):
            compute_block_statistics(-np.ones((1, 18)), np.ones((1, 18)), parse_condition_grid(var))


if __name__ == '__main__':
    unittest.main()
