"""Ground-truth and sampling invariants for the 20-celltype benchmark."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'chrom_velo'))
from synthetic_accessibility import (SyntheticConfig, make_dataset, latent_parameters,
                                    evaluate_truth, CELLTYPES, MISSING_STRESS)
from temporal_velocity import temporal_rates


class SyntheticAccessibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SyntheticConfig(n_modules=24, peaks_per_module=6)
        cls.clean = make_dataset(cls.config, 'clean')
        cls.noisy = make_dataset(cls.config, 'noisy')

    def test_matrix_axes_and_bounds(self):
        a = self.noisy
        self.assertEqual(a.shape, (144, 120))
        self.assertEqual(a.var.groupby('celltype').size().tolist(), [6]*20)
        self.assertTrue(a.obs_names.is_unique and a.var_names.is_unique)
        self.assertTrue(np.isfinite(a.X).all())
        self.assertTrue(((a.X > 0) & (a.X < 1)).all())
        np.testing.assert_array_equal(self.clean.X, self.clean.layers['truth_state'])

    def test_analytic_derivative(self):
        args = latent_parameters(self.config)
        value, rate = evaluate_truth(*args, [16.])
        plus, _ = evaluate_truth(*args, [16.0001])
        minus, _ = evaluate_truth(*args, [15.9999])
        np.testing.assert_allclose(rate, (plus-minus)/0.0002, atol=1e-8)

    def test_planted_signs_and_controls(self):
        a = self.clean
        for i, row in a.obs.iterrows():
            idx = a.obs_names.get_loc(i)
            cols = (a.var.celltype == row.primary_celltype).to_numpy()
            diff = np.diff(a.X[idx, cols])
            if row.true_program in ('opening', 'late_opening'):
                self.assertTrue((diff > 0).all())
            elif row.true_program == 'closing':
                self.assertTrue((diff < 0).all())
            elif row.true_program.startswith('stable_'):
                np.testing.assert_array_equal(np.diff(a.X[idx].reshape(20,6), axis=1), 0.)
            elif row.true_program.startswith('transient_'):
                self.assertTrue((diff > 0).any() and (diff < 0).any())

    def test_opposed_rates_cancel_only_when_averaged(self):
        a = self.clean
        for i in np.flatnonzero(a.obs.true_program.to_numpy() == 'opposite_celltypes'):
            row = a.obs.iloc[i]
            primary = (a.var.celltype == row.primary_celltype).to_numpy()
            partner = (a.var.celltype == row.partner_celltype).to_numpy()
            v1, v2 = a.layers['truth_velocity'][i, primary], a.layers['truth_velocity'][i, partner]
            self.assertTrue((v1[:-1] > 0).all() and (v2[:-1] < 0).all())
            np.testing.assert_allclose(v1+v2, 0., atol=3e-8)

    def test_noise_does_not_change_latent_programs(self):
        np.testing.assert_array_equal(self.clean.layers['truth_state'], self.noisy.layers['truth_state'])
        changed = make_dataset(replace(self.config, noise_precision=0.5), 'noisy')
        np.testing.assert_array_equal(changed.layers['truth_state'], self.clean.layers['truth_state'])
        self.assertGreater(np.mean((changed.X-changed.layers['truth_state'])**2),
                           np.mean((self.noisy.X-self.noisy.layers['truth_state'])**2))

    def test_sampling_variance_matches_model(self):
        a = self.noisy
        mu = a.layers['truth_state'].astype(float)
        variance = np.zeros(a.shape)
        for r in range(1, self.config.n_replicates+1):
            precision = a.var[f'n_cells_rep{r}'].to_numpy()*a.var[f'depth_rep{r}'].to_numpy()*self.config.noise_precision
            variance += mu*(1-mu)/(precision+1)/self.config.n_replicates**2
        mse = np.mean((a.X-mu)**2)
        self.assertLess(abs(mse/variance.mean()-1), 0.12)
        self.assertLess(abs(np.mean(a.X-mu)), 5*np.sqrt(variance.sum())/mu.size)

    def test_cis_graph_has_wrong_edges_and_isolates(self):
        a = self.noisy
        g = a.obsp['coaccessibility'].tocoo()
        chrom = a.obs.chrom.to_numpy()
        self.assertTrue((chrom[g.row] == chrom[g.col]).all())
        self.assertEqual((g.tocsr()-g.T.tocsr()).nnz, 0)
        self.assertGreater(a.uns['synthetic']['graph']['wrong_undirected_edges'], 0)
        self.assertTrue((np.asarray(g.tocsr().sum(1)).ravel() == 0).any())

    def test_perfect_and_empty_graph_controls(self):
        perfect = make_dataset(replace(self.config, graph_retention=1.,
                                      graph_false_edge_fraction=0., graph_isolated_fraction=0.))
        self.assertEqual((perfect.obsp['coaccessibility']-perfect.obsp['true_cis_candidates']).nnz, 0)
        empty = make_dataset(replace(self.config, graph_isolated_fraction=1.))
        self.assertEqual(empty.obsp['coaccessibility'].nnz, 0)

    def test_stress_missingness_invalidates_preceding_intervals(self):
        a = make_dataset(self.config, 'stress')
        self.assertEqual(a.n_vars, 116)
        self.assertTrue(all(f'{c}_{t:g}hpf' not in a.var_names for c,t in MISSING_STRESS))
        _, mask, _ = temporal_rates(a.X, a.var, min_cells=20)
        # CT08's missing 14 hpf invalidates 12->14, not 16->19.
        self.assertFalse(mask[:, a.var_names.get_loc('CT08_12hpf')].any())
        self.assertTrue(mask[:, a.var_names.get_loc('CT08_16hpf')].all())
        # Low-cell CT13/12 hpf invalidates both adjoining intervals.
        self.assertFalse(mask[:, a.var_names.get_loc('CT13_10hpf')].any())
        self.assertFalse(mask[:, a.var_names.get_loc('CT13_12hpf')].any())


if __name__ == '__main__':
    unittest.main()
