"""Scientific controls for temporal change versus celltype specialization."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "chrom_velo"))
from specialization_synthetic import make_specialization_dataset, evaluate_specialization
from temporal_velocity import STAGES_HPF


class SpecializationSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kwargs = dict(n_constitutive=8, n_per_family_trunk=4,
                          n_per_celltype_branch=4, n_per_control_family=3,
                          n_replicates=3)
        cls.a = make_specialization_dataset(**cls.kwargs)
        cls.truth = cls.a.layers["truth_state"].reshape(-1, 20, 6)
        cls.rate = cls.a.layers["truth_instantaneous_rate"].reshape(-1, 20, 6)

    def mask(self, kind):
        return self.a.obs.true_program.to_numpy() == kind

    def test_schema_and_bounds(self):
        self.assertEqual(self.a.shape, (164, 120))
        self.assertTrue(self.a.obs_names.is_unique)
        self.assertTrue(self.a.var_names.is_unique)
        self.assertTrue(np.isfinite(self.a.X).all())
        self.assertTrue(((self.truth > 0) & (self.truth < 1)).all())
        self.assertEqual(int(self.a.obs.is_hierarchy.sum()), 104)
        self.assertTrue(self.a.obs.loc[~self.a.obs.is_hierarchy, "true_specialization"].isna().all())
        self.assertEqual(self.a.var.time_hpf.tolist()[:6], STAGES_HPF.tolist())

    def test_stable_controls_have_zero_truth_velocity(self):
        mask = self.mask("constitutive_core") | self.mask("stable_celltype_specific") | self.mask("low_signal")
        np.testing.assert_array_equal(self.rate[mask], 0)
        np.testing.assert_array_equal(np.diff(self.truth[mask], axis=-1), 0)
        np.testing.assert_array_equal(self.a.layers["truth_velocity"][mask], 0)
        specific = self.truth[self.mask("stable_celltype_specific")]
        self.assertTrue((specific.max(axis=1) > 0.65).all())
        self.assertTrue((specific.min(axis=1) < 0.07).all())

    def test_uniform_opening_preserves_zero_specificity(self):
        x = self.truth[self.mask("uniform_opening")].astype(float)
        self.assertTrue((np.diff(x, axis=-1) > 0).all())
        proportions = x / x.sum(axis=1, keepdims=True)
        specificity = 1 + np.sum(proportions * np.log(proportions), axis=1) / np.log(20)
        np.testing.assert_allclose(specificity, 0, atol=1e-12)

    def test_target_lineage_opens_and_offtarget_is_constant(self):
        indices = np.flatnonzero(self.mask("lineage_opening"))
        for i in indices:
            target = int(self.a.obs.iloc[i].true_celltype_index)
            x = self.truth[i]
            self.assertLess(x[target, 0], 0.07)
            self.assertGreater(x[target, -1], 0.65)
            self.assertTrue((np.diff(x[target]) > 0).all())
            np.testing.assert_array_equal(np.diff(x[np.arange(20) != target], axis=-1), 0)

    def test_transient_returns_and_reverses(self):
        for i in np.flatnonzero(self.mask("transient_lineage")):
            target = int(self.a.obs.iloc[i].true_celltype_index)
            x = self.truth[i, target]
            self.assertGreater(x[3] - x[0], 0.5)
            self.assertAlmostEqual(float(x[0]), float(x[-1]), places=7)
            self.assertGreater(self.rate[i, target, 2], 0)
            self.assertLess(self.rate[i, target, 4], 0)

    def test_weak_signal_is_below_floor(self):
        x = self.truth[self.mask("low_signal")]
        self.assertTrue((x < 0.02).all())
        np.testing.assert_array_equal(np.diff(x, axis=1), 0)

    def test_analytic_rates_match_numerical_derivative(self):
        times = np.array([11.0, 14.0, 17.0, 19.0, 23.0])
        _, exact = evaluate_specialization(self.a.obs, times)
        h = 1e-4
        plus, _ = evaluate_specialization(self.a.obs, times + h)
        minus, _ = evaluate_specialization(self.a.obs, times - h)
        np.testing.assert_allclose((plus - minus) / (2 * h), exact, atol=2e-9, rtol=2e-7)
        with self.assertRaises(ValueError):
            evaluate_specialization(self.a.obs, [-1, 10])

    def test_replicates_are_independent_and_reproducible(self):
        a2 = make_specialization_dataset(**self.kwargs)
        np.testing.assert_array_equal(a2.X, self.a.X)
        self.assertFalse(np.array_equal(self.a.layers["sampling_rep1"], self.a.layers["sampling_rep2"]))
        expected = np.mean([self.a.layers[f"sampling_rep{r}"] for r in range(1, 4)], axis=0)
        np.testing.assert_array_equal(expected, self.a.X)
        a3 = make_specialization_dataset(seed=43, **self.kwargs)
        self.assertFalse(np.array_equal(a3.X, self.a.X))
        self.assertEqual(a3.shape, self.a.shape)

    def test_cis_graph_and_temporal_mask(self):
        graph = self.a.obsp["coaccessibility"]
        rows, cols = graph.nonzero()
        np.testing.assert_array_equal(self.a.obs.chrom.to_numpy()[rows], self.a.obs.chrom.to_numpy()[cols])
        self.assertEqual((graph - graph.T).nnz, 0)
        self.assertTrue((graph.data > 0).all())
        mask = self.a.layers["truth_velocity_valid"].reshape(-1, 20, 6)
        self.assertTrue(mask[:, :, :-1].all())
        self.assertFalse(mask[:, :, -1].any())
        expected = np.diff(self.truth, axis=-1) / np.diff(STAGES_HPF)
        actual = self.a.layers["truth_velocity"].reshape(-1, 20, 6)
        np.testing.assert_allclose(actual[:, :, :-1], expected, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
