"""Scientific invariants of entropy and graph-supported specialization arrows."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'chrom_velo'))
from specialization_flow import (
    specificity_by_stage, temporal_specificity, aggregate_specificity,
    build_state_graph, structural_flow, rate_guided_vectors, bidirectional_rate_guidance,
)


class SpecificityTests(unittest.TestCase):
    def test_broad_and_one_celltype_limits(self):
        a = np.array([[[.5, .8], [.5, .8], [.5, .8]],
                      [[.5, .8], [0., 0.], [0., 0.]]])
        result = specificity_by_stage(a)
        np.testing.assert_allclose(result['score'][0], 0., atol=1e-15)
        np.testing.assert_allclose(result['score'][1], 1., atol=1e-15)
        self.assertTrue(result['valid'].all())

    def test_multiplicative_scale_invariance_without_pseudocount(self):
        rng = np.random.default_rng(4)
        a = rng.uniform(.2, 1., (5, 4, 3))
        rescaled = a * rng.uniform(.5, 8., (5, 1, 3))
        for shrinkage in (0., .1, .5):
            np.testing.assert_allclose(
                specificity_by_stage(a, shrinkage=shrinkage)['score'],
                specificity_by_stage(rescaled, shrinkage=shrinkage)['score'], atol=1e-14)

    def test_uniform_shrinkage_and_pseudocount_reduce_specificity(self):
        a = np.array([[[.8], [.1], [.05], [.05]]])
        score = specificity_by_stage(a)['score']
        self.assertTrue((specificity_by_stage(a, pseudocount=.1)['score'] < score).all())
        self.assertTrue((specificity_by_stage(a, shrinkage=.1)['score'] < score).all())
        np.testing.assert_allclose(specificity_by_stage(a, shrinkage=1)['score'], 0.)

    def test_stable_specific_and_always_open_have_zero_rates(self):
        a = np.repeat(np.array([[.8, .05, .05], [.8, .8, .8]])[:, :, None], 4, axis=2)
        result = temporal_specificity(a, [10, 12, 19, 24])
        self.assertGreater(result['score'][0, 0], .5)
        np.testing.assert_array_equal(result['interval_rate'], 0.)

    def test_global_opening_is_not_specialization(self):
        a = np.array([[[.2, .5, .8], [.2, .5, .8], [.2, .5, .8]]])
        np.testing.assert_allclose(temporal_specificity(a, [10, 14, 24])['interval_rate'], 0., atol=1e-15)

    def test_closing_outside_target_increases_specificity(self):
        a = np.array([[[.8, .8, .8], [.8, .4, .05], [.8, .4, .05]]])
        result = temporal_specificity(a, [10, 12, 24])
        self.assertTrue((result['interval_rate'] > 0).all())
        expected = np.diff(result['score'], axis=1) / [2., 12.]
        np.testing.assert_allclose(result['interval_rate'], expected)

    def test_time_reversal_reverses_interval_changes(self):
        a = np.array([[[.8, .8, .8], [.8, .3, .05], [.8, .3, .05]]])
        forward = temporal_specificity(a, [10, 12, 24])['interval_rate']
        # Reversing physical intervals also reverses their durations.
        reverse = temporal_specificity(a[:, :, ::-1], [10, 22, 24])['interval_rate']
        np.testing.assert_allclose(reverse, -forward[:, ::-1], atol=1e-15)

    def test_low_signal_stays_invalid_even_with_large_pseudocount(self):
        a = np.array([[[.001, .002], [0., 0.]], [[0., 0.], [0., 0.]]])
        result = temporal_specificity(a, [10, 12], pseudocount=1.)
        self.assertFalse(result['valid'].any())
        self.assertFalse(result['interval_valid'].any())
        self.assertTrue(np.isnan(result['interval_rate']).all())
        self.assertFalse(specificity_by_stage(np.zeros((1, 2, 2)), signal_floor=0.)['valid'].any())

    def test_missing_celltypes_are_compared_on_common_support(self):
        a = np.array([[[.8, .8, .8], [.2, .2, .2], [np.nan, .8, .8]]])
        result = temporal_specificity(a, [10, 12, 14])
        # Descriptive per-stage entropies differ because support differs.
        self.assertGreater(abs(np.diff(result['score'])[0, 0]), .01)
        # Matched endpoint entropies correctly find no temporal change.
        np.testing.assert_array_equal(result['interval_rate'], 0.)
        np.testing.assert_array_equal(result['paired_celltypes'], [[2, 3]])
        masked = a.copy(); masked[np.isnan(masked)] = 0.
        observed = np.ones((3, 3), dtype=bool); observed[2, 0] = False
        explicit = temporal_specificity(masked, [10, 12, 14], observed=observed)
        np.testing.assert_array_equal(explicit['interval_rate'], result['interval_rate'])

    def test_missing_stage_never_bridged_and_single_celltype_invalid(self):
        a = np.ones((2, 3, 3)) * .5
        a[0, :, 1] = np.nan
        a[1, 1:, :] = np.nan
        result = temporal_specificity(a, [10, 12, 14])
        self.assertFalse(result['interval_valid'].any())
        self.assertFalse(result['valid'][1].any())

    def test_nonuniform_time_weights_are_explicit(self):
        scores = np.array([[0., .2, 1.], [np.nan, .2, 1.]])
        equal = aggregate_specificity(scores, [10, 12, 24])
        duration = aggregate_specificity(scores, [10, 12, 24], weighting='duration')
        late = aggregate_specificity(scores, [10, 12, 24], weighting='late')
        np.testing.assert_allclose(equal['stage_weights'], [1/3, 1/3, 1/3])
        np.testing.assert_allclose(duration['stage_weights'], np.array([1, 7, 6]) / 14)
        self.assertAlmostEqual(equal['aggregate'][1], .6)
        self.assertAlmostEqual(equal['effective_weight'][1], 2/3)
        self.assertAlmostEqual(late['stage_weights'][-1] / late['stage_weights'][0], 4.)

    def test_terminal_weighting_never_falls_back_when_final_stage_missing(self):
        scores = np.array([[0., .3, .8], [.2, .4, np.nan], [np.nan, np.nan, .6]])
        result = aggregate_specificity(scores, [10, 12, 24], weighting='terminal')
        np.testing.assert_array_equal(result['stage_weights'], [0., 0., 1.])
        np.testing.assert_allclose(result['aggregate'], [.8, np.nan, .6], equal_nan=True)
        np.testing.assert_array_equal(result['valid'], [True, False, True])
        np.testing.assert_array_equal(result['effective_weight'], [1., 0., 1.])
        valid = np.ones(scores.shape, dtype=bool); valid[0, -1] = False
        self.assertFalse(aggregate_specificity(scores, [10, 12, 24], valid=valid,
                                              weighting='terminal')['valid'][0])

    def test_invalid_accessibility_and_times_rejected(self):
        with self.assertRaises(ValueError):
            specificity_by_stage(np.array([[[-.1, .5], [.2, .5]]]))
        with self.assertRaises(ValueError):
            temporal_specificity(np.ones((1, 2, 2)), [12, 10])


class TauSpecificityTests(unittest.TestCase):
    def test_tau_flat_and_single_active_limits(self):
        a = np.array([[[.5, .8], [.5, .8], [.5, .8]],
                      [[.5, .8], [0., 0.], [0., 0.]]])
        result = specificity_by_stage(a, method='tau')
        np.testing.assert_allclose(result['score'], [[0., 0.], [1., 1.]], atol=1e-15)

    def test_tau_scale_invariance_and_explicit_regularization(self):
        a = np.array([[[.8, .6], [.2, .3], [.1, .2]]])
        score = specificity_by_stage(a, method='tau')['score']
        # Direct definition without regularization.
        np.testing.assert_allclose(score, [[(.75 + .875)/2, (.5 + 2/3)/2]])
        np.testing.assert_allclose(score, specificity_by_stage(a * 4, method='tau')['score'])
        for options in ({'pseudocount': .1}, {'shrinkage': .1}):
            self.assertTrue((specificity_by_stage(a, method='tau', **options)['score'] < score).all())
        np.testing.assert_allclose(specificity_by_stage(a, method='tau', shrinkage=1.)['score'], 0.)
        np.testing.assert_allclose(
            specificity_by_stage(a, method='tau', shrinkage=.1)['score'],
            specificity_by_stage(a * 4, method='tau', shrinkage=.1)['score'])

    def test_tau_missing_support_uses_only_matched_celltypes(self):
        a = np.array([[[.8, .8, .8], [.2, .2, .2], [np.nan, .8, .8]]])
        result = temporal_specificity(a, [10, 12, 24], method='tau')
        np.testing.assert_array_equal(result['interval_rate'], 0.)
        np.testing.assert_array_equal(result['paired_celltypes'], [[2, 3]])
        explicit = np.nan_to_num(a)
        observed = np.isfinite(a)
        np.testing.assert_allclose(
            temporal_specificity(explicit, [10, 12, 24], method='tau', observed=observed)['score'],
            result['score'])

    def test_tau_stable_specificity_and_low_signal(self):
        a = np.array([[[.8, .8, .8], [.05, .05, .05]],
                      [[.001, .001, .001], [.002, .002, .002]],
                      [[0., 0., 0.], [0., 0., 0.]]])
        result = temporal_specificity(a, [10, 12, 24], method='tau', pseudocount=.1)
        np.testing.assert_array_equal(result['interval_rate'][0], 0.)
        self.assertFalse(result['valid'][1:].any())
        self.assertTrue(np.isnan(result['interval_rate'][1:]).all())
        self.assertFalse(specificity_by_stage(a[2:], method='tau', signal_floor=0.)['valid'].any())

    def test_method_must_be_explicitly_supported(self):
        with self.assertRaises(ValueError):
            specificity_by_stage(np.ones((1, 2, 2)), method='invalid')


class StructuralFlowTests(unittest.TestCase):
    def setUp(self):
        self.graph = sparse.csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 1, 0],
                                                 [0, 1, 0, 0], [0, 0, 0, 0]], dtype=float))
        self.score = np.array([0., .2, .5, .9])
        self.xy = np.array([[0., 0.], [1., 0.], [2., 0.], [0., 2.]])

    def test_only_supported_uphill_edges_and_isolated_zero(self):
        result = structural_flow(self.graph, self.score, {'umap': self.xy})
        expected = np.zeros((4, 4)); expected[0, 1] = expected[1, 2] = 1.
        np.testing.assert_array_equal(result['probabilities'].toarray(), expected)
        np.testing.assert_array_equal(result['vectors']['umap'], [[1., 0.], [1., 0.], [0., 0.], [0., 0.]])
        np.testing.assert_array_equal(result['coherence']['umap'], [1., 1., 0., 0.])
        self.assertEqual(result['support_count'][3], 0)
        np.testing.assert_allclose(result['expected_specificity_gain'], [.2, .3, 0., 0.])

    def test_plateau_and_invalid_scores_never_force_direction(self):
        plateau = structural_flow(self.graph, np.full(4, .3), self.xy)
        self.assertEqual(plateau['probabilities'].nnz, 0)
        np.testing.assert_array_equal(plateau['vectors']['embedding'], 0.)
        missing = self.score.copy(); missing[1] = np.nan
        self.assertEqual(structural_flow(self.graph, missing)['probabilities'].nnz, 0)
        near_flat = structural_flow(self.graph, np.array([0., .0001, .0002, 1.]), edge_tolerance=.001)
        self.assertEqual(near_flat['probabilities'].nnz, 0)

    def test_edge_uncertainty_removes_small_noisy_contrast_keeps_large_gradient(self):
        score = np.array([0., .01, .5, .9])
        result = structural_flow(self.graph, score, score_uncertainty=np.full(4, .02),
                                 edge_tolerance=0., edge_confidence_z=1.)
        expected = np.zeros((4, 4)); expected[1, 2] = 1.
        np.testing.assert_allclose(result['probabilities'].toarray(), expected, atol=1e-15)
        np.testing.assert_array_equal(result['support_count'], [0, 1, 0, 0])
        # Large measurement errors may suppress an existing edge but cannot
        # create a shortcut from vertex 0 to the distant vertex 2.
        self.assertEqual(result['probabilities'][0, 2], 0.)

    def test_zero_uncertainty_multiplier_is_exactly_backward_compatible(self):
        original = structural_flow(self.graph, self.score, {'umap': self.xy})
        result = structural_flow(self.graph, self.score, {'umap': self.xy},
                                 score_uncertainty=[.2, .1, 0., 1.], edge_confidence_z=0.)
        np.testing.assert_array_equal(result['probabilities'].toarray(), original['probabilities'].toarray())
        np.testing.assert_array_equal(result['vectors']['umap'], original['vectors']['umap'])
        zero_se = structural_flow(self.graph, self.score, score_uncertainty=np.zeros(4),
                                  edge_confidence_z=2.)
        np.testing.assert_array_equal(zero_se['probabilities'].toarray(), original['probabilities'].toarray())

    def test_invalid_uncertainty_vertices_are_unsupported(self):
        for missing in (np.nan, np.inf, -.1):
            se = np.array([.01, missing, .01, .01])
            result = structural_flow(self.graph, self.score, score_uncertainty=se,
                                     edge_confidence_z=1.)
            self.assertEqual(result['probabilities'].nnz, 0)
        with self.assertRaises(ValueError):
            structural_flow(self.graph, self.score, edge_confidence_z=1.)
        with self.assertRaises(ValueError):
            structural_flow(self.graph, self.score, score_uncertainty=[.1, .1])
        with self.assertRaises(ValueError):
            structural_flow(self.graph, self.score, edge_confidence_z=-1.)

    def test_graph_order_permutation_preserves_output(self):
        original = structural_flow(self.graph, self.score, {'umap': self.xy})
        order = np.array([2, 0, 3, 1])
        permuted = structural_flow(self.graph[order][:, order], self.score[order], {'umap': self.xy[order]})
        np.testing.assert_allclose(permuted['probabilities'].toarray(), original['probabilities'].toarray()[order][:, order])
        np.testing.assert_allclose(permuted['vectors']['umap'], original['vectors']['umap'][order])

    def test_embeddings_do_not_change_transitions(self):
        xyz = np.c_[self.xy, [1., -1., 2., 0.]]
        original = structural_flow(self.graph, self.score, {'umap': self.xy, 'umap3d': xyz})
        warped = structural_flow(self.graph, self.score, {'umap': self.xy[::-1]})
        np.testing.assert_allclose(original['probabilities'].toarray(), warped['probabilities'].toarray())
        self.assertEqual(original['vectors']['umap3d'].shape, (4, 3))

    def test_opposed_embedding_directions_have_zero_coherence(self):
        graph = sparse.csr_matrix([[0., 1., 1.], [1., 0., 0.], [1., 0., 0.]])
        result = structural_flow(graph, [0., .5, .5], np.array([[0., 0.], [-1., 0.], [1., 0.]]))
        self.assertAlmostEqual(result['coherence']['embedding'][0], 0.)
        self.assertEqual(result['support_count'][0], 2)
        self.assertAlmostEqual(result['probabilities'][0].sum(), 1.)

    def test_minimum_neighbor_support_does_not_invent_edges(self):
        result = structural_flow(self.graph, self.score, min_support=2)
        self.assertEqual(result['probabilities'].nnz, 0)
        np.testing.assert_array_equal(result['support_count'], [1, 1, 0, 0])

    def test_scalar_rate_display_preserves_sign_and_gates_unsupported(self):
        directions = np.array([[.5, 0.], [.5, 0.], [0., 0.], [.1, 0.]])
        result = rate_guided_vectors(directions, [.02, -.01, .5, .03], min_coherence=.2)
        np.testing.assert_allclose(result, [[.02, 0.], [-.01, 0.], [0., 0.], [0., 0.]])
        np.testing.assert_array_equal(rate_guided_vectors(directions, np.zeros(4)), 0.)

    def test_negative_rate_at_uphill_tip_uses_actual_downhill_neighbors(self):
        up = structural_flow(self.graph, self.score, {'umap': self.xy})
        down = structural_flow(self.graph, -self.score, {'umap': self.xy})
        self.assertEqual(np.linalg.norm(up['vectors']['umap'][2]), 0.)
        result = bidirectional_rate_guidance(
            up['vectors']['umap'], down['vectors']['umap'], [.02, 0., -.03, -.1])
        np.testing.assert_allclose(result, [[.02, 0.], [0., 0.], [-.03, 0.], [0., 0.]])
        # Reversing an uphill field would lose the tip's supported inward arrow.
        old = rate_guided_vectors(up['vectors']['umap'], [.02, 0., -.03, -.1])
        np.testing.assert_array_equal(old[2], [0., 0.])

    def test_bidirectional_guidance_checks_directional_masks_and_coherence(self):
        up = np.array([[1., 0.], [1., 0.], [1., 0.]])
        down = -up
        rates = np.array([.02, -.03, -.04])
        result = bidirectional_rate_guidance(
            up, down, rates, upward_valid=[False, True, True],
            downward_coherence=[1., .1, .9], min_coherence=.5)
        np.testing.assert_allclose(result, [[0., 0.], [0., 0.], [-.04, 0.]])
        result = bidirectional_rate_guidance(up, down, rates, valid=[True, False, False])
        np.testing.assert_allclose(result, [[.02, 0.], [0., 0.], [0., 0.]])

    def test_bidirectional_guidance_never_invents_plateau_or_invalid_direction(self):
        flat = structural_flow(self.graph, np.full(4, .5), {'umap': self.xy})
        vectors = flat['vectors']['umap']
        np.testing.assert_array_equal(
            bidirectional_rate_guidance(vectors, vectors, [.02, -.03, np.nan, 0.]), 0.)
        with self.assertRaises(ValueError):
            bidirectional_rate_guidance(np.zeros((2, 2)), np.zeros((2, 3)), [0., 0.])

    def test_sparse_state_graph_has_no_self_edges_and_handles_singleton(self):
        states = np.array([[0., 0.], [1., 0.], [2., 0.], [10., 1.]])
        graph = build_state_graph(states, n_neighbors=1)
        self.assertTrue(sparse.isspmatrix_csr(graph))
        np.testing.assert_array_equal(graph.diagonal(), 0.)
        self.assertEqual((graph - graph.T).nnz, 0)
        self.assertEqual(build_state_graph(np.ones((1, 3))).nnz, 0)
        self.assertTrue((graph.data > 0).all())


if __name__ == '__main__':
    unittest.main()
