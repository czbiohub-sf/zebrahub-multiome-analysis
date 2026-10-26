"""Scientific invariants of root-free accessibility-change projection."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/chrom_velo'))
from trajectory_projection import project_interval


def star_fixture(direction=None):
    direction = np.ones(6) if direction is None else np.asarray(direction, dtype=float)
    state = np.full((5, 6), 3.)
    for index, multiple in enumerate([1., 2., -1., -2.], start=1):
        state[index] += multiple*direction
    velocity = np.zeros_like(state); velocity[0] = direction
    graph = sparse.csr_matrix((np.ones(4), (np.zeros(4, dtype=int), np.arange(1, 5))), shape=(5, 5))
    xy = np.array([[0., 0.], [1., 0.], [2., 0.], [-1., 0.], [-2., 0.]])
    xyz = np.column_stack((xy, np.zeros(5)))
    return graph, state, velocity, dict(umap2d=xy, umap3d=xyz)


class TrajectoryProjectionTests(unittest.TestCase):
    def test_uniform_opening_is_preserved_without_feature_centering(self):
        graph, state, velocity, embedding = star_fixture()
        result = project_interval(graph, state, velocity, embeddings=embedding)
        np.testing.assert_allclose(result['probabilities'].toarray()[0], [0, .5, .5, 0, 0])
        np.testing.assert_allclose(result['vectors']['umap2d'][0], [1., 0.])
        np.testing.assert_allclose(result['vectors']['umap3d'][0], [1., 0., 0.])
        self.assertTrue(result['projection_valid']['umap2d'][0])
        self.assertAlmostEqual(result['velocity_rms'][0], 1.)
        self.assertAlmostEqual(result['net_rate'][0], 1.)
        self.assertFalse(result['metadata']['geometric_baseline_subtracted'])

    def test_stationary_broad_state_never_emits_flow_despite_graph_bias(self):
        graph, state, velocity, embedding = star_fixture()
        graph = graph.copy(); graph.data[2:] = 0
        velocity[:] = 0
        result = project_interval(graph, state, velocity, embeddings=embedding)
        np.testing.assert_allclose(result['baseline_vectors']['umap2d'][0], [1., 0.])
        self.assertTrue(result['stationary'].all())
        self.assertEqual(result['probabilities'].nnz, 0)
        self.assertFalse(result['changing'].any())
        self.assertFalse(result['unprojectable'].any())
        np.testing.assert_array_equal(result['vectors']['umap2d'], 0.)

    def test_velocity_sign_flip_selects_actual_opposite_neighbors(self):
        graph, state, velocity, embedding = star_fixture()
        forward = project_interval(graph, state, velocity, embeddings=embedding)
        reverse = project_interval(graph, state, -velocity, embeddings=embedding)
        np.testing.assert_allclose(reverse['probabilities'].toarray()[0], [0, 0, 0, .5, .5])
        np.testing.assert_allclose(reverse['vectors']['umap2d'], -forward['vectors']['umap2d'])
        self.assertEqual(reverse['net_rate'][0], -1.)

    def test_celltype_switch_has_zero_net_change_but_nonzero_dynamics(self):
        graph, state, velocity, embedding = star_fixture([1., -1., 0., 0., 0., 0.])
        result = project_interval(graph, state, velocity, embeddings=embedding)
        self.assertEqual(result['net_rate'][0], 0.)
        self.assertGreater(result['velocity_norm'][0], 1.)
        self.assertTrue(result['changing'][0])
        np.testing.assert_allclose(result['vectors']['umap2d'][0], [1., 0.])

    def test_unchanged_celltypes_remain_in_state_alignment(self):
        graph, state, velocity, embedding = star_fixture()
        velocity[0] = [1., 0., 0., 0., 0., 0.]
        state[1:] = state[0] + np.array([1., 10., 10., 10., 10., 10.])
        result = project_interval(graph, state, velocity, embeddings=embedding)
        self.assertEqual(result['source_feature_count'][0], 6)
        self.assertEqual(result['comparable_count'][0], 4)
        self.assertEqual(result['support_count'][0], 0)
        self.assertTrue(result['unprojectable'][0])
        self.assertFalse(result['stationary'][0])

    def test_shared_mask_fraction_uses_full_panel_and_keeps_observed_zero(self):
        graph, state, velocity, embedding = star_fixture()
        valid = np.ones(state.shape, dtype=bool)
        valid[0, -1] = False
        valid[1, 0] = False  # Only four shared features; reject.
        state[3:] = state[0] + 1
        state[0, 2] = 0.; velocity[0, 2] = 0.  # Measured zeros are valid.
        result = project_interval(graph, state, velocity, valid, embedding)
        self.assertEqual(result['source_feature_count'][0], 5)
        self.assertEqual(result['metadata']['required_shared_features'], 5)
        self.assertEqual(result['probabilities'][0, 1], 0.)
        self.assertTrue(result['active'][0])
        invalid_nan = state.copy(); invalid_nan[~valid] = np.nan
        same = project_interval(graph, invalid_nan, velocity, embeddings=embedding)
        np.testing.assert_allclose(result['probabilities'].toarray(), same['probabilities'].toarray())

    def test_target_can_be_used_when_its_own_velocity_is_unknown(self):
        graph, state, velocity, embedding = star_fixture()
        velocity[1:] = np.nan
        result = project_interval(graph, state, velocity, embeddings=embedding)
        self.assertTrue(result['active'][0])
        self.assertFalse(result['source_supported'][1:].any())
        np.testing.assert_allclose(result['vectors']['umap2d'][0], [1., 0.])

    def test_external_source_gate_does_not_remove_neighbor_states(self):
        graph, state, velocity, embedding = star_fixture()
        enabled = np.array([True, False, False, False, False])
        result = project_interval(graph, state, velocity, embeddings=embedding, source_active=enabled)
        self.assertTrue(result['active'][0])
        disabled = project_interval(graph, state, velocity, embeddings=embedding, source_active=np.zeros(5, bool))
        self.assertTrue(disabled['source_filtered'][0])
        self.assertEqual(disabled['probabilities'].nnz, 0)

    def test_missing_support_stable_and_strong_unprojectable_are_distinct(self):
        graph, state, velocity, embedding = star_fixture()
        state[1:] = state[0]  # No distinct currently accessible neighbor state.
        valid = np.ones(state.shape, bool); valid[1, :3] = False
        result = project_interval(graph, state, velocity, valid, embedding)
        self.assertTrue(result['changing'][0])
        self.assertTrue(result['unprojectable'][0])
        self.assertFalse(result['stationary'][0])
        self.assertFalse(result['source_supported'][1])
        self.assertTrue(result['stationary'][2])
        self.assertEqual(result['probabilities'].nnz, 0)

    def test_low_embedding_coherence_abstains_without_erasing_change(self):
        graph, state, velocity, embedding = star_fixture()
        embedding['umap2d'][2] = [-1., 0.]
        result = project_interval(graph, state, velocity, embeddings={'umap2d': embedding['umap2d']})
        self.assertTrue(result['active'][0])
        self.assertTrue(result['unprojectable'][0])
        self.assertFalse(result['projection_valid']['umap2d'][0])
        self.assertAlmostEqual(result['coherence']['umap2d'][0], 0.)
        self.assertGreater(result['velocity_rms'][0], 0)

    def test_at_least_two_distinct_projected_neighbors_are_required(self):
        graph, state, velocity, embedding = star_fixture()
        embedding['umap2d'][2] = embedding['umap2d'][0]
        result = project_interval(graph, state, velocity, embeddings={'umap2d': embedding['umap2d']})
        self.assertTrue(result['active'][0])
        self.assertEqual(result['projection_neighbor_count']['umap2d'][0], 1)
        self.assertFalse(result['projection_valid']['umap2d'][0])
        np.testing.assert_array_equal(result['display_vectors']['umap2d'][0], 0.)

    def test_positive_velocity_rescaling_preserves_direction_and_reports_amplitude(self):
        graph, state, velocity, embedding = star_fixture()
        first = project_interval(graph, state, velocity, embeddings=embedding)
        second = project_interval(graph, state, velocity*7, embeddings=embedding)
        np.testing.assert_allclose(first['probabilities'].toarray(), second['probabilities'].toarray())
        np.testing.assert_allclose(first['vectors']['umap2d'], second['vectors']['umap2d'])
        np.testing.assert_allclose(second['velocity_rms'], first['velocity_rms']*7)

    def test_probabilities_match_independent_weighted_cosine_loop(self):
        rng = np.random.default_rng(28)
        n, c = 14, 6
        state = rng.uniform(1., 5., (n, c)); velocity = rng.normal(size=(n, c))
        valid = rng.uniform(size=(n, c)) > .08
        dense = rng.uniform(.2, 1., (n, n)); dense[rng.uniform(size=(n, n)) < .3] = 0
        graph = sparse.csr_matrix(dense)
        weights = np.arange(1., c+1)
        embedding = {'umap2d': rng.normal(size=(n, 2)), 'umap3d': rng.normal(size=(n, 3))}
        result = project_interval(graph, state, velocity, valid, embedding, feature_weights=weights, block_size=3)
        expected = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                shared = valid[i] & valid[j]
                if i == j or dense[i, j] <= 0 or shared.sum() < 5:
                    continue
                v = velocity[i, shared]; delta = state[j, shared]-state[i, shared]; w = weights[shared]
                denominator = np.sqrt(np.sum(v*v*w)*np.sum(delta*delta*w))
                cosine = np.sum(v*delta*w)/denominator if denominator else 0.
                if cosine > 0 and cosine >= .2:
                    expected[i, j] = dense[i, j]*cosine
            if np.count_nonzero(expected[i]) < 2:
                expected[i] = 0
            elif expected[i].sum() > 0:
                expected[i] /= expected[i].sum()
        np.testing.assert_allclose(result['probabilities'].toarray(), expected, atol=1e-14)
        for name, xy in embedding.items():
            projected = np.zeros_like(xy)
            for i in range(n):
                for j in range(n):
                    delta = xy[j]-xy[i]
                    if np.linalg.norm(delta) > 0:
                        projected[i] += expected[i, j]*delta/np.linalg.norm(delta)
            np.testing.assert_allclose(result['vectors'][name], projected, atol=1e-14)

    def test_peak_permutation_csr_edge_order_and_block_size_invariance(self):
        graph, state, velocity, embedding = star_fixture()
        original_state, original_velocity = state.copy(), velocity.copy()
        original_embedding = {key:value.copy() for key, value in embedding.items()}
        # Unsorted duplicate entries and a self edge must neither change the
        # summed neighbor weights nor become directed transitions.
        duplicate = sparse.csr_matrix((np.array([.4, 1., .6, 1., 1., 100.]),
            np.array([2, 4, 2, 3, 1, 0]), np.array([0, 6, 6, 6, 6, 6])), shape=(5, 5))
        data_before, indices_before = duplicate.data.copy(), duplicate.indices.copy()
        first = project_interval(duplicate, state, velocity, embeddings=embedding, block_size=1)
        standard = project_interval(graph, state, velocity, embeddings=embedding, block_size=100)
        np.testing.assert_allclose(first['probabilities'].toarray(), standard['probabilities'].toarray())
        order = np.array([2, 4, 0, 3, 1]); inverse = np.argsort(order)
        second = project_interval(graph[order][:, order], state[order], velocity[order],
                                  embeddings={k:v[order] for k,v in embedding.items()}, block_size=2)
        np.testing.assert_allclose(standard['probabilities'].toarray(), second['probabilities'][inverse][:, inverse].toarray())
        for name in embedding:
            np.testing.assert_allclose(standard['vectors'][name], second['vectors'][name][inverse])
            np.testing.assert_array_equal(embedding[name], original_embedding[name])
        np.testing.assert_array_equal(state, original_state)
        np.testing.assert_array_equal(velocity, original_velocity)
        np.testing.assert_array_equal(duplicate.data, data_before)
        np.testing.assert_array_equal(duplicate.indices, indices_before)

    def test_empty_input_no_embedding_and_probability_free_mode(self):
        graph = sparse.csr_matrix((0, 0))
        result = project_interval(graph, np.empty((0, 6)), np.empty((0, 6)),
                                  embeddings={'umap2d': np.empty((0, 2))})
        self.assertEqual(result['probabilities'].shape, (0, 0))
        self.assertEqual(result['vectors']['umap2d'].shape, (0, 2))
        self.assertEqual(len(result['unprojectable']), 0)
        graph, state, velocity, embedding = star_fixture()
        without = project_interval(graph, state, velocity, return_probabilities=False)
        self.assertNotIn('probabilities', without)
        self.assertTrue(without['active'][0])
        self.assertFalse(without['unprojectable'][0])

    def test_no_new_edges_even_if_better_state_neighbors_overlap_in_umap(self):
        graph, state, velocity, embedding = star_fixture()
        graph = sparse.csr_matrix((np.ones(3), ([0, 0, 1], [3, 4, 2])), shape=(5, 5))
        embedding['umap2d'][1:3] = [[1e-5, 0.], [2e-5, 0.]]
        result = project_interval(graph, state, velocity, embeddings=embedding)
        self.assertTrue(result['changing'][0])
        self.assertTrue(result['unprojectable'][0])
        self.assertEqual(result['probabilities'].nnz, 0)
        np.testing.assert_array_equal(result['display_vectors']['umap2d'][0], 0.)

    def test_malformed_inputs_rejected(self):
        graph, state, velocity, embedding = star_fixture()
        for kwargs in [dict(feature_valid=np.ones((5, 5), bool)), dict(min_shared_fraction=1.2),
                       dict(feature_weights=np.zeros(6)), dict(block_size=0),
                       dict(source_active=np.ones(4)), dict(velocity_floor=-1)]:
            with self.assertRaises(ValueError):
                project_interval(graph, state, velocity, **kwargs)
        bad = graph.copy(); bad.data[0] = -1
        with self.assertRaises(ValueError):
            project_interval(bad, state, velocity)


if __name__ == '__main__':
    unittest.main()
