"""Focused tests for conditional shared-stage references and timing sensitivity."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/chrom_velo'))
from activation_statistics import temporal_permutation_test, activation_weight_sensitivity

T = np.array([10., 12., 14., 16., 19., 24.])

class ActivationStatisticsTests(unittest.TestCase):
    def test_exact_actual_time_slope_and_reference(self):
        result = temporal_permutation_test(np.array([3. + .7 * T, 8. - .2 * T]), T)
        np.testing.assert_allclose(result['slope'], [.7, -.2], atol=1e-12)
        np.testing.assert_allclose(result['correlation'], [1., -1.], atol=1e-12)
        self.assertEqual(result['n_permutations'], 720)
        self.assertEqual(result['p_outward'][0], 1/720)
        self.assertEqual(result['p_inward'][1], 1/720)
        np.testing.assert_allclose(result['null_correlation'][:, 0], -result['null_correlation'][:, 1], atol=1e-12)
        self.assertGreater(result['global_result']['p_two_sided'], .99)

    def test_global_slope_and_fraction_positive_null(self):
        result = temporal_permutation_test(np.array([3. + .7 * T, 2. + .2 * T]), T)
        self.assertAlmostEqual(result['global_result']['mean_slope'], .45)
        self.assertEqual(result['global_result']['mean_slope_p_outward'], 1/720)
        self.assertEqual(result['global_result']['fraction_positive'], 1.)
        self.assertGreater(result['global_result']['fraction_positive_p_outward'], .4)
        np.testing.assert_allclose(result['null_global_mean_slope'], result['null_slope'].mean(axis=1))
        np.testing.assert_allclose(result['null_global_fraction_positive'], (result['null_correlation'] > 1e-12).mean(axis=1))

    def test_not_equal_index_time(self):
        result = temporal_permutation_test(np.arange(6.), T)
        self.assertLess(float(result['correlation']), 1.)
        self.assertAlmostEqual(float(result['slope']), np.polyfit(T, np.arange(6.), 1)[0])

    def test_constants_and_missing_are_not_fabricated_change(self):
        y = np.array([[5., 5., 5., 5., 5., 5.], [1., np.nan, 3., np.nan, np.nan, 6.]])
        result = temporal_permutation_test(y, T)
        self.assertEqual(result['correlation'][0], 0.)
        self.assertEqual(result['p_outward'][0], 1.)
        self.assertTrue(np.isnan(result['slope'][1]))
        self.assertTrue(np.isnan(result['null_correlation'][:, 1]).all())
        self.assertEqual(result['family_size'], 1)

    def test_missing_snapshots_move_together_and_global_scale_invariant(self):
        y = np.array([[1., np.nan, 4., 8., 10., 11.], [2., np.nan, 8., 16., 20., 22.]])
        result = temporal_permutation_test(y, T)
        np.testing.assert_allclose(result['null_correlation'][:, 0], result['null_correlation'][:, 1])
        np.testing.assert_array_equal(result['n_observed'], [5, 5])
        transformed = temporal_permutation_test(y * 13. + 71., T)
        np.testing.assert_allclose(result['null_correlation'], transformed['null_correlation'], atol=1e-12)
        np.testing.assert_array_equal(result['p_two_sided'], transformed['p_two_sided'])

    def test_familywise_reference_and_multidimensional_shape(self):
        y = np.random.default_rng(1).normal(size=(3, 4, 6))
        result = temporal_permutation_test(y, T)
        self.assertEqual(result['null_correlation'].shape, (720, 3, 4))
        self.assertTrue((result['p_fwer_two_sided'] >= result['p_two_sided']).all())
        self.assertEqual(result['family_size'], 12)
        np.testing.assert_array_equal(result['permutation_indices'][0], np.arange(6))

    def test_all_insufficient(self):
        result = temporal_permutation_test(np.full((2, 6), np.nan), T)
        self.assertTrue(np.isnan(result['p_fwer_two_sided']).all())
        self.assertEqual(result['global_result']['n_series'], 0)

    def test_uniform_opening_preserves_mass_center(self):
        distance = np.linspace(0, 10, 101)
        profile = np.exp(-distance / 3) + .2
        accessibility = profile[:, None] * np.array([.13, .47, 1.13, 2.7, 3.9, 8.2])[None, :]
        centers = (distance[:, None] * accessibility).sum(axis=0) / accessibility.sum(axis=0)
        result = temporal_permutation_test(centers, T)
        self.assertEqual(float(result['slope']), 0.)
        self.assertEqual(float(result['correlation']), 0.)
        self.assertEqual(float(result['p_two_sided']), 1.)

    def test_opposing_activation_waves_cancel_global_but_remain_local(self):
        distance = np.linspace(0, 10, 201)
        positions = 2. + .4 * (T - 10)
        waves = np.exp(-((distance[None, :, None] - np.stack([positions, 10 - positions])[:, None, :]) ** 2) / 1.5)
        centers = (distance[None, :, None] * waves).sum(axis=1) / waves.sum(axis=1)
        result = temporal_permutation_test(centers, T)
        self.assertGreater(result['correlation'][0], .99)
        self.assertLess(result['correlation'][1], -.99)
        self.assertEqual(result['p_outward'][0], 1/720)
        self.assertEqual(result['p_inward'][1], 1/720)
        self.assertAlmostEqual(result['global_result']['mean_correlation'], 0., places=12)
        self.assertEqual(result['global_result']['p_two_sided'], 1.)

    def test_sensitivity_uses_paired_support_and_reports_loss(self):
        result = activation_weight_sensitivity(np.arange(5.), [10, 12, 14, 16, np.nan], [11, 13, 15, np.nan, 23])
        self.assertEqual(result['n_primary'], 4)
        self.assertEqual(result['n_alternative'], 4)
        self.assertEqual(result['n_paired'], 3)
        self.assertEqual(result['activation_overlap_jaccard'], 3/5)
        self.assertEqual(result['median_abs_timing_shift_hpf'], 1.)
        self.assertEqual(result['timing_spearman'], 1.)
        self.assertEqual(result['primary_distance_timing_spearman'], 1.)

if __name__ == '__main__':
    unittest.main()
