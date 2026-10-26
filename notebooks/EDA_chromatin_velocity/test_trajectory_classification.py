"""Mode claims must respect both fitted reversals and support limitations."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/chrom_velo'))
from trajectory_classification import classify_supported_modes


class ClassificationTests(unittest.TestCase):
    def test_partial_reversals_are_uncertain(self):
        rates = np.array([[1., 1., -.4, -.4], [-1., -1., .4, .4]])
        supported = np.array([[True, True, False, False]] * 2)
        np.testing.assert_array_equal(classify_supported_modes(rates, supported), [6, 6])
        np.testing.assert_array_equal(classify_supported_modes(rates, np.ones_like(supported)), [4, 5])

    def test_monotone_no_supported_change_and_floor(self):
        rates = np.array([[1., 2., 0., -.01], [-1., -2., 0., .01], [1., -1., 0., 0.]])
        supported = np.array([[True, True, False, False], [True, True, False, False], [False] * 4])
        np.testing.assert_array_equal(classify_supported_modes(rates, supported), [2, 3, 1])
        self.assertEqual(classify_supported_modes(np.array([.05, -.05]), np.array([True, True])), 1)

    def test_quality_complexity_and_missingness(self):
        rates = np.array([[1., -1., 1., np.nan], [1., 1., 1., 1.], [0., 0., 0., 0.], [np.nan] * 4])
        np.testing.assert_array_equal(classify_supported_modes(
            rates, np.isfinite(rates), quality_valid=[True, False, False, False]), [6, 6, 6, 0])
        # Missing stages never count as a zero or as a supported sign.
        self.assertEqual(classify_supported_modes(
            np.array([1., np.nan, -.5]), np.array([True, True, False])), 6)

    def test_shape_and_validation(self):
        rates = np.ones((2, 3, 5))
        out = classify_supported_modes(rates, np.ones_like(rates, bool), quality_valid=[[True], [False]])
        self.assertEqual(out.dtype, np.int8)
        np.testing.assert_array_equal(out, [[2] * 3, [6] * 3])
        with self.assertRaises(ValueError):
            classify_supported_modes(rates, np.ones(5, bool))
        with self.assertRaises(ValueError):
            classify_supported_modes([1., np.inf], [True, True])
        with self.assertRaises(ValueError):
            classify_supported_modes([1., 2.], [True, True], rate_floor=-1.)


if __name__ == '__main__':
    unittest.main()
