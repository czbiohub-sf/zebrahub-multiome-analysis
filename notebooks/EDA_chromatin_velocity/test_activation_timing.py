"""Check that descriptive event timing is not assigned to unresolvable profiles."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from activation_progression import timing_metrics
T = np.array([10., 12., 14., 16., 19., 24.])

class ActivationTimingTests(unittest.TestCase):
    def test_constant_rate_has_no_artificial_event_age(self):
        rates = np.array([[1., 1., 1., 1., 1.], [-2., -2., -2., -2., -2.]])
        result = timing_metrics(rates, np.ones_like(rates, bool), T, np.ones(2, bool))
        self.assertFalse(result['opening_resolved'].any())
        self.assertFalse(result['closing_resolved'].any())
        self.assertTrue(np.isnan(result['opening_center_hpf']).all())
        self.assertTrue(np.isnan(result['closing_center_hpf']).all())

    def test_event_age_weights_increment_not_interval_count(self):
        rates = np.array([[1., 1., 1., 2., 2.]])
        result = timing_metrics(rates, np.ones_like(rates, bool), T, np.ones(1, bool))
        expected = np.sum(rates[0] * np.diff(T) * (T[:-1] + T[1:]) / 2) / np.sum(rates[0] * np.diff(T))
        self.assertTrue(result['opening_resolved'][0])
        self.assertAlmostEqual(result['opening_center_hpf'][0], expected)
        self.assertFalse(result['closing_resolved'][0])

    def test_transient_can_have_both_opening_and_closing(self):
        rates = np.array([[1., 2., 0., -1., -.6]])
        result = timing_metrics(rates, np.ones_like(rates, bool), T, np.ones(1, bool))
        self.assertTrue(result['opening_resolved'][0])
        self.assertTrue(result['closing_resolved'][0])
        self.assertLess(result['opening_center_hpf'][0], result['closing_center_hpf'][0])
        self.assertAlmostEqual(result['opening_change_cpm'][0], result['closing_change_cpm'][0])

    def test_missing_unsupported_and_low_quality_stay_unresolved(self):
        rates = np.array([[1., 2., np.nan, 0., 0.], [1., 2., 0., 0., 0.], [1., 2., 0., 0., 0.]])
        support = np.ones_like(rates, bool); support[1] = False
        result = timing_metrics(rates, support, T, np.array([True, True, False]))
        self.assertFalse(result['opening_resolved'].any())
        self.assertTrue(np.isnan(result['opening_center_hpf']).all())

    def test_supported_trajectory_uses_all_fitted_positive_mass(self):
        rates = np.array([[.1, .1, 0., 0., 1.]])
        support = np.array([[False, False, False, False, True]])
        result = timing_metrics(rates, support, T, np.ones(1, bool))
        expected = (.2*11 + .2*13 + 5*21.5) / 5.4
        self.assertAlmostEqual(result['opening_center_hpf'][0], expected)

if __name__ == '__main__':
    unittest.main()
