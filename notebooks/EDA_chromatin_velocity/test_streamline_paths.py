"""Independently test rendered streamline support and ordered local direction."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/chrom_velo'))
from streamline_paths import draw_supported_streams


def field(x, y, u, v, valid):
    return dict(x=x, y=y, u=np.where(valid, u, np.nan),
                v=np.where(valid, v, np.nan), valid=valid.copy())


def independently_check_paths(test, grid, result, min_cosine=.099):
    """Sample every returned segment more finely than the renderer, using SciPy.

    A point is supported only when interpolation has finite values; NaNs at any
    interpolation corner propagate. Sampling also includes all drawn vertices.
    No private path-cleaner or production audit function is used here.
    """
    interpolate = [RegularGridInterpolator((grid['y'], grid['x']), grid[key],
                    bounds_error=False, fill_value=np.nan) for key in ['u', 'v']]
    step = min(np.diff(grid['x']).min(), np.diff(grid['y']).min()) * .037
    count = 0
    for path in result['paths']:
        test.assertEqual(path.shape[1], 2)
        test.assertGreaterEqual(len(path), 2)
        test.assertTrue(np.isfinite(path).all())
        for start, end in zip(path[:-1], path[1:]):
            delta = end-start
            length = np.linalg.norm(delta)
            test.assertGreater(length, 0)
            positions = start + np.linspace(0, 1, max(2, int(np.ceil(length/step))+1))[:, None] * delta
            velocity = np.column_stack([f(positions[:, ::-1]) for f in interpolate])
            test.assertTrue(np.isfinite(velocity).all(), f'Unsupported returned segment: {start} -> {end}')
            norm = np.linalg.norm(velocity, axis=1)
            test.assertTrue((norm > 0).all())
            cosine = (velocity @ delta) / (norm * length)
            test.assertGreaterEqual(float(cosine.min()), min_cosine)
            count += len(positions)
    # The visible line collection must contain exactly the audited polylines.
    drawn = result['lines'].get_segments()
    test.assertEqual(len(drawn), len(result['paths']))
    for actual, returned in zip(drawn, result['paths']):
        np.testing.assert_array_equal(actual, returned)
    # Arrow polygon boundaries are displayed too, so independently sample them.
    for arrow in result['arrows']:
        boundary = arrow.get_xy()
        heading = boundary[0] - (boundary[1] + boundary[2]) / 2
        tip_velocity = np.array([f(boundary[0][::-1][None])[0] for f in interpolate])
        denominator = np.linalg.norm(heading) * np.linalg.norm(tip_velocity)
        test.assertGreater(denominator, 0)
        test.assertGreaterEqual(float(heading @ tip_velocity / denominator), min_cosine)
        for start, end in zip(boundary[:-1], boundary[1:]):
            positions = start + np.linspace(0, 1, max(2, int(np.ceil(np.linalg.norm(end-start)/step))+1))[:, None] * (end-start)
            velocity = np.column_stack([f(positions[:, ::-1]) for f in interpolate])
            test.assertTrue(np.isfinite(velocity).all(), 'Arrowhead crosses unsupported grid support')
    return count


class StreamlinePathTests(unittest.TestCase):
    def test_masked_barrier_and_domain_boundaries(self):
        x = np.linspace(-2., 2., 51); y = np.linspace(-1., 1., 31)
        xx, yy = np.meshgrid(x, y)
        valid = (np.abs(xx) > .35) & (np.abs(yy) < .85)
        grid = field(x, y, np.ones_like(xx), np.zeros_like(yy), valid)
        original = {key: value.copy() for key, value in grid.items()}
        figure, axis = plt.subplots()
        try:
            result = draw_supported_streams(axis, grid, density=1.2, seed_limit=180)
            self.assertGreater(len(result['paths']), 0)
            self.assertGreater(independently_check_paths(self, grid, result), 100)
            points = np.concatenate(result['paths'])
            self.assertTrue(np.any(points[:, 0] < -.5))
            self.assertTrue(np.any(points[:, 0] > .5))
            self.assertTrue(np.all(np.abs(points[:, 0]) > .35))
            for key in grid:
                np.testing.assert_array_equal(grid[key], original[key])
        finally:
            plt.close(figure)

    def test_curved_annulus_tangents_and_arrowhead_support(self):
        x = np.linspace(-2., 2., 51); y = np.linspace(-2., 2., 51)
        xx, yy = np.meshgrid(x, y)
        radius = np.hypot(xx, yy)
        grid = field(x, y, -yy, xx, (radius > .55) & (radius < 1.7))
        figure, axis = plt.subplots()
        try:
            result = draw_supported_streams(axis, grid, density=1.2, seed_limit=150)
            self.assertGreater(len(result['paths']), 0)
            self.assertGreater(len(result['arrows']), 0)
            self.assertGreater(independently_check_paths(self, grid, result), 100)
        finally:
            plt.close(figure)

    def test_injected_native_overshoot_and_reversal_are_removed(self):
        x = np.linspace(0., 3., 31); y = np.linspace(0., 2., 21)
        xx, yy = np.meshgrid(x, y)
        grid = field(x, y, np.ones_like(xx), np.zeros_like(yy), (xx < 1.3) | (xx > 1.6))
        # Deliberately supply one out-of-bounds path with a backward section and
        # a crossing through a masked barrier, independent of Matplotlib version.
        candidate = np.array([[-.2, .55], [.95, .55], [.4, .55], [2.9, .55], [3.3, .55]])
        figure, axis = plt.subplots()
        background = axis.scatter([.1], [.1])
        def fake_streamplot(*args, **kwargs):
            lines = LineCollection([candidate]); axis.add_collection(lines)
            return SimpleNamespace(lines=lines)
        try:
            with patch.object(axis, 'streamplot', side_effect=fake_streamplot):
                result = draw_supported_streams(axis, grid, seed_limit=20)
            self.assertGreater(result['audit']['rejected_support_segments'], 0)
            self.assertGreater(result['audit']['rejected_direction_segments'], 0)
            self.assertGreaterEqual(len(result['paths']), 3)
            independently_check_paths(self, grid, result)
            self.assertIn(background, axis.collections)
            self.assertEqual(len(axis.collections), 2)
            for path in result['paths']:
                self.assertTrue((np.diff(path[:, 0]) > 0).all())
        finally:
            plt.close(figure)

    def test_empty_support_returns_empty_artists_and_paths(self):
        x = np.linspace(0., 1., 11); y = np.linspace(0., 1., 11)
        shape = (len(y), len(x))
        grid = field(x, y, np.ones(shape), np.zeros(shape), np.zeros(shape, dtype=bool))
        figure, axis = plt.subplots()
        try:
            with patch.object(axis, 'streamplot', side_effect=AssertionError('Empty support should not integrate')):
                result = draw_supported_streams(axis, grid)
            self.assertEqual(result['paths'], [])
            self.assertEqual(result['arrows'], [])
            self.assertEqual(len(result['lines'].get_segments()), 0)
            self.assertEqual(result['audit']['candidate_seed_cells'], 0)
            self.assertEqual(result['audit']['clean_paths'], 0)
        finally:
            plt.close(figure)

    def test_fixed_seed_rendering_is_deterministic(self):
        x = np.linspace(-1., 1., 25); y = np.linspace(-1., 1., 25)
        xx, yy = np.meshgrid(x, y)
        grid = field(x, y, np.ones_like(xx), .4*xx, np.ones_like(xx, dtype=bool))
        figures = [plt.subplots(), plt.subplots()]
        try:
            outputs = [draw_supported_streams(axis, grid, density=1., seed_limit=80, seed=19)
                       for _, axis in figures]
            self.assertEqual(outputs[0]['audit'], outputs[1]['audit'])
            self.assertGreater(len(outputs[0]['paths']), 0)
            self.assertEqual(len(outputs[0]['paths']), len(outputs[1]['paths']))
            for first, second in zip(outputs[0]['paths'], outputs[1]['paths']):
                np.testing.assert_array_equal(first, second)
            self.assertEqual(len(outputs[0]['arrows']), len(outputs[1]['arrows']))
            for first, second in zip(outputs[0]['arrows'], outputs[1]['arrows']):
                np.testing.assert_array_equal(first.get_xy(), second.get_xy())
            independently_check_paths(self, grid, outputs[0])
        finally:
            for figure, _ in figures:
                plt.close(figure)


if __name__ == '__main__':
    unittest.main()
