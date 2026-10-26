"""Independent conditional-variance checks for nonnegative fitted trajectories."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/chrom_velo'))
from trajectory_modes import fit_trajectories


class ClippingVarianceTests(unittest.TestCase):
    times = np.array([10., 12., 14., 16., 19., 24.])
    weights = np.array([2.2, 3.3, 2.4, 2.8, 2., .5])

    def independent_variances(self, values, variance, model):
        """Use explicit per-interval row operators, independent of vectorized code."""
        q = (self.times - self.times[0]) / np.ptp(self.times)
        if model == 0:
            design = np.ones((len(q), 1))
        elif model == 1:
            design = np.column_stack([np.ones(len(q)), q])
        else:
            knot = [14., 16.][model - 2]
            design = np.column_stack([
                np.ones(len(q)), q,
                np.maximum(q - (knot - self.times[0]) / np.ptp(self.times), 0.)])
        weight_matrix = np.diag(self.weights / self.weights.sum())
        hat = design @ np.linalg.solve(design.T @ weight_matrix @ design,
                                       design.T @ weight_matrix)
        linear_prediction = hat @ values
        active = linear_prediction > 0.
        linear_variance, clipped_variance = [], []
        for j, dt in enumerate(np.diff(self.times)):
            linear_row = (hat[j + 1] - hat[j]) / dt
            clipped_row = (active[j + 1] * hat[j + 1] - active[j] * hat[j]) / dt
            linear_variance.append(np.dot(linear_row ** 2, variance))
            clipped_variance.append(np.dot(clipped_row ** 2, variance))
        return linear_prediction, np.asarray(linear_variance), np.asarray(clipped_variance)

    def test_boundary_rate_se_covers_active_set_and_linear_operators(self):
        for y in (np.array([0., 0., 0., 1., 2., 3.]),
                  np.array([3., 2., 1., .01, .02, 0.])):
            with self.subTest(values=y.tolist()):
                variance = .05 * (y + .5)
                result = fit_trajectories(
                    y, self.times, weights=self.weights,
                    measurement_variance=variance, complexity_penalty=.2,
                    rate_z=3.5)
                predicted, linear, clipped = self.independent_variances(
                    y, variance, int(result['model']))
                np.testing.assert_allclose(
                    result['rate_se'] ** 2, np.maximum(linear, clipped),
                    rtol=2e-12, atol=2e-14)
                np.testing.assert_allclose(
                    result['rates'], np.diff(np.maximum(predicted, 0.)) / np.diff(self.times),
                    rtol=2e-12, atol=2e-14)
                if y[-1] == 3.:
                    self.assertEqual(int(result['model']), 1)
                    self.assertLess(predicted[0], 0.)
                    self.assertGreater(predicted[1], 0.)
                    self.assertGreater(clipped[0], 2. * linear[0])

    def test_positive_fit_keeps_linear_se_and_accessibility_amplitude(self):
        y = 2. + .4 * (self.times - self.times[0])
        variance = np.array([.2, .4, .3, .5, .2, .6])
        result = fit_trajectories(
            y, self.times, weights=self.weights, measurement_variance=variance,
            complexity_penalty=.2, rate_z=3.5)
        predicted, linear, clipped = self.independent_variances(
            y, variance, int(result['model']))
        self.assertTrue(np.all(predicted > 0.))
        np.testing.assert_allclose(clipped, linear, rtol=2e-12, atol=2e-14)
        np.testing.assert_allclose(result['rate_se'] ** 2, linear, rtol=2e-12, atol=2e-14)
        scale = 7.
        scaled = fit_trajectories(
            scale * y, self.times, weights=self.weights,
            measurement_variance=scale ** 2 * variance,
            complexity_penalty=.2, rate_z=3.5)
        np.testing.assert_allclose(scaled['fitted'], scale * result['fitted'], rtol=2e-12)
        np.testing.assert_allclose(scaled['rates'], scale * result['rates'], rtol=2e-12)
        np.testing.assert_allclose(scaled['rate_se'], scale * result['rate_se'], rtol=2e-12)


if __name__ == '__main__':
    unittest.main()
