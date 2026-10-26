"""Scientific display invariants for support-aware streamline interpolation.

These tests concern local interpolation in the plotted plane. An empty-space
mask is not a guarantee about high-dimensional atlas graph components.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/chrom_velo'))
from streamline_field import build_stream_grid


def square_points(n=45):
    x, y = np.meshgrid(np.linspace(-1., 1., n), np.linspace(-1., 1., n))
    return np.column_stack((x.ravel(), y.ravel()))


def assert_same_field(first, second):
    np.testing.assert_array_equal(first['valid'], second['valid'])
    for key in ['x', 'y', 'u', 'v', 'coherence', 'support_fraction']:
        np.testing.assert_allclose(first[key], second[key], equal_nan=True, atol=1e-12, rtol=1e-12)


class StreamlineFieldTests(unittest.TestCase):
    def test_uniform_direction_and_nan_mask_contract(self):
        xy = square_points()
        vectors = np.tile([3., 4.], (len(xy), 1))
        result = build_stream_grid(xy, vectors, grid_size=41)
        valid = result['valid']
        self.assertEqual(valid.shape, (len(result['y']), len(result['x'])))
        self.assertGreater(int(valid.sum()), 300)
        np.testing.assert_array_equal(np.isfinite(result['u']), valid)
        np.testing.assert_array_equal(np.isfinite(result['v']), valid)
        np.testing.assert_allclose(result['u'][valid], .6, atol=1e-12)
        np.testing.assert_allclose(result['v'][valid], .8, atol=1e-12)
        np.testing.assert_allclose(result['coherence'][valid], 1., atol=1e-12)
        np.testing.assert_allclose(result['support_fraction'][valid], 1., atol=1e-12)
        self.assertTrue(np.all(result['eligible_count'][valid] >= 5))
        self.assertTrue(np.all(result['effective_n'][valid] >= 3))

    def test_positive_per_peak_rescaling_and_global_reversal(self):
        xy = square_points()
        angle = .3 * xy[:, 0] + .2 * xy[:, 1]
        vectors = np.column_stack((np.cos(angle), np.sin(angle)))
        first = build_stream_grid(xy, vectors, grid_size=35)
        scales = np.random.default_rng(13).lognormal(0, 2, len(xy))
        scaled = build_stream_grid(xy, vectors * scales[:, None], grid_size=35)
        assert_same_field(first, scaled)
        reversed_field = build_stream_grid(xy, -vectors, grid_size=35)
        np.testing.assert_array_equal(reversed_field['valid'], first['valid'])
        np.testing.assert_allclose(reversed_field['u'], -first['u'], equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(reversed_field['v'], -first['v'], equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(reversed_field['coherence'], first['coherence'], atol=1e-12)

    def test_opposing_colocated_directions_cancel_without_a_stream(self):
        xy = np.repeat(square_points(31), 2, axis=0)
        vectors = np.tile([[1., 0.], [-1., 0.]], (len(xy)//2, 1))
        result = build_stream_grid(xy, vectors, grid_size=31, n_neighbors=50)
        self.assertFalse(result['valid'].any())
        self.assertTrue(np.isnan(result['u']).all())
        well_supported = (result['eligible_count'] >= 5) & (result['effective_n'] >= 3)
        self.assertTrue(well_supported.any())
        self.assertLess(float(result['coherence'][well_supported].max()), .35)

    def test_unsupported_majority_remains_in_density_denominator(self):
        # Each coordinate has one eligible vector and nine ineligible peaks.
        # Removing the latter before neighbor lookup would create a false stream.
        xy = np.repeat(square_points(31), 10, axis=0)
        vectors = np.tile([1., 0.], (len(xy), 1))
        valid = np.zeros(len(xy), dtype=bool)
        valid[::10] = True
        result = build_stream_grid(xy, vectors, valid=valid, grid_size=31, n_neighbors=100)
        self.assertFalse(result['valid'].any())
        self.assertLess(float(np.nanmax(result['support_fraction'])), .2)
        # A NaN direction means unsupported; it must not erase that peak's
        # contribution to the local density denominator.
        vectors[~valid] = np.nan
        missing = build_stream_grid(xy, vectors, grid_size=31, n_neighbors=100)
        assert_same_field(result, missing)

    def test_visible_empty_gap_between_islands_is_masked(self):
        offsets = square_points(41) * .5
        xy = np.vstack((offsets + [-2., 0.], offsets + [2., 0.]))
        vectors = np.tile([1., 0.], (len(xy), 1))
        result = build_stream_grid(xy, vectors, grid_size=101)
        gap = np.abs(result['x']) < 1.
        self.assertTrue(gap.any())
        self.assertFalse(result['valid'][:, gap].any())
        self.assertTrue(np.isnan(result['u'][:, gap]).all())
        self.assertTrue(result['valid'][:, result['x'] < -1.5].any())
        self.assertTrue(result['valid'][:, result['x'] > 1.5].any())
        # Check the actual plotting backend too: integration must terminate at
        # the masked region rather than merely hiding field samples there.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots()
        try:
            stream = axis.streamplot(result['x'], result['y'], result['u'], result['v'],
                                     density=1.5, minlength=.02)
            segments = stream.lines.get_segments()
            self.assertGreater(len(segments), 0)
            coordinates = np.concatenate(segments)
            self.assertTrue(np.all(np.abs(coordinates[:, 0]) >= 1.))
            self.assertTrue(np.any(coordinates[:, 0] < -1.5))
            self.assertTrue(np.any(coordinates[:, 0] > 1.5))
        finally:
            plt.close(figure)

    def test_no_eligible_vectors_yields_no_field(self):
        xy = square_points(21)
        for vectors, valid in [(np.zeros_like(xy), None),
                               (np.ones_like(xy), np.zeros(len(xy), dtype=bool))]:
            result = build_stream_grid(xy, vectors, valid=valid, grid_size=25)
            self.assertFalse(result['valid'].any())
            self.assertTrue(np.isnan(result['u']).all())
            self.assertTrue(np.isnan(result['v']).all())

    def test_isolated_valid_peak_cannot_create_supported_stream(self):
        xy = square_points(21)
        vectors = np.ones_like(xy)
        valid = np.zeros(len(xy), dtype=bool)
        valid[len(xy)//2] = True
        result = build_stream_grid(xy, vectors, valid=valid, grid_size=25)
        self.assertFalse(result['valid'].any())

    def test_permutation_and_missing_coordinates_do_not_change_field(self):
        xy = square_points(21)
        vectors = np.column_stack((np.ones(len(xy)), .2 * xy[:, 0]))
        # Using all observations avoids arbitrary ties at the KNN cutoff while
        # preserving the finite-radius interpolation and support tests.
        options = dict(grid_size=25, n_neighbors=len(xy))
        first = build_stream_grid(xy, vectors, **options)
        order = np.random.default_rng(7).permutation(len(xy))
        permuted = build_stream_grid(xy[order], vectors[order], **options)
        assert_same_field(first, permuted)
        missing = build_stream_grid(np.vstack((xy, [np.nan, 0.])),
                                    np.vstack((vectors, [1e9, -1e9])), **options)
        assert_same_field(first, missing)


if __name__ == '__main__':
    unittest.main()
