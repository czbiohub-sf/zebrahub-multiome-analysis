"""Scientific invariants: missingness, units, graph alignment, and null behavior."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'chrom_velo'))
from temporal_velocity import temporal_rates, cis_weights, couple_rates, rate_summary, project_scvelo


class TemporalVelocityTests(unittest.TestCase):
    def test_irregular_time_and_terminal_mask(self):
        t = np.array([10., 12., 14., 16., 19., 24.])
        c = pd.DataFrame({'celltype': 'a', 'time_hpf': t})
        x = np.array([2*t + 1, -3*t + 100, t*0+5])
        v, m, dt = temporal_rates(x, c)
        np.testing.assert_allclose(v[:, :-1], np.tile([[2.], [-3.], [0.]], (1, 5)))
        self.assertFalse(m[:, -1].any())
        np.testing.assert_allclose(dt[:-1], [2, 2, 2, 3, 5])
        change, _, _ = temporal_rates(x, c, divide_by_dt=False)
        np.testing.assert_allclose(change[:, :-1], v[:, :-1]*dt[:-1])
        slope, _, _ = temporal_rates(x, c, form='slope')
        np.testing.assert_allclose(slope, np.tile([[2.], [-3.], [0.]], (1, 6)), atol=1e-12)

    def test_missing_internal_stage_not_bridged(self):
        c = pd.DataFrame({'celltype': ['a']*3, 'time_hpf': [10., 14., 16.]})
        v, m, _ = temporal_rates(np.array([[1., 3., 4.]]), c)
        self.assertFalse(m[0, 0])
        self.assertEqual(v[0, 1], 0.5)
        self.assertFalse(m[0, 2])

    def test_nan_and_low_count_endpoints_remain_unknown(self):
        c = pd.DataFrame({'celltype': ['a']*3, 'time_hpf': [10., 12., 14.], 'n_cells': [50, 5, 50]})
        v, m, _ = temporal_rates(np.array([[1., 2., 3.]]), c, min_cells=20)
        self.assertFalse(m.any())
        _, m, _ = temporal_rates(np.array([[1., np.nan, 3.]]), c)
        self.assertFalse(m.any())

    def test_order_and_celltype_names_are_explicit(self):
        c = pd.DataFrame({'celltype': ['neural_crest']*3, 'time_hpf': [10., 12., 14.]})
        x = np.array([[2., 4., 6.]])
        v, m, _ = temporal_rates(x, c)
        p = [2, 0, 1]
        vp, mp, _ = temporal_rates(x[:, p], c.iloc[p])
        np.testing.assert_array_equal(vp, v[:, p])
        np.testing.assert_array_equal(mp, m[:, p])
        with self.assertRaises(ValueError):
            temporal_rates(np.ones((1, 2)), c.iloc[[0, 0]])

    def test_sparse_cis_graph_and_isolates(self):
        g = sparse.csr_matrix([[0., 0.8, 0.], [0.8, 0., 0.], [0., 0., 0.]])
        w = cis_weights(g, ['chr1', 'chr1', 'chr2'])
        self.assertTrue(sparse.issparse(w))
        np.testing.assert_allclose(np.asarray(w.sum(1)), 1.)
        self.assertEqual(w[2, 2], 1.)
        with self.assertRaises(ValueError):
            cis_weights(g, ['chr1', 'chr2', 'chr2'])
        v = np.array([[1., 0.], [3., 0.], [5., 0.]])
        mask = np.array([[True, False]]*3)
        final, _ = couple_rates(v, mask, w, alpha=1.)
        np.testing.assert_array_equal(final, v)
        final, _ = couple_rates(v, mask, w, alpha=0.)
        self.assertEqual(final[2, 0], 5.)
        self.assertTrue((final[:, 1] == 0).all())

    def test_neighbor_missingness_does_not_bias_or_fill_focal(self):
        w = cis_weights(sparse.csr_matrix([[0., 0.8], [0.8, 0.]]), ['1', '1'])
        v = np.array([[2., 0.], [0., 7.]])
        mask = np.array([[True, False], [False, True]])
        final, _ = couple_rates(v, mask, w, alpha=0.)
        np.testing.assert_allclose(final, v)

    def test_zero_or_uniform_field_has_no_projected_arrows(self):
        import anndata as ad
        x = np.arange(60, dtype='float32').reshape(20, 3)
        a = ad.AnnData(x)
        graph = sparse.eye(20, format='csr')
        a.obsp['distances'] = graph.copy()
        a.uns['neighbors'] = {'distances_key': 'distances'}
        a.obsm['X_umap'] = x[:, :2].copy()
        before = a.obsm['X_umap'].copy()
        for value in [0., 0.1]:
            a.layers['velocity'] = np.full_like(x, value)
            project_scvelo(a, np.ones_like(x, dtype=bool))
            np.testing.assert_array_equal(a.obsm['velocity_umap'], 0.)
            np.testing.assert_array_equal(a.obsm['X_umap'], before)
            self.assertEqual((a.obsp['distances'] != graph).nnz, 0)

    def test_pulse_does_not_disappear_from_magnitude(self):
        v = np.array([[1., -1., 0.], [0., 0., 0.]])
        m = np.array([[True, True, False], [False]*3])
        s = rate_summary(v, m)
        self.assertEqual(s.signed_rate.iloc[0], 0.)
        self.assertEqual(s.magnitude_rms.iloc[0], 1.)
        self.assertTrue(np.isnan(s.signed_rate.iloc[1]))


if __name__ == '__main__':
    unittest.main()
