"""Exact scientific equivalence of bounded-CSR and original atlas projection."""
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/chrom_velo'))
from specialization_flow import structural_flow
from real_sparse_projection import block_structural_flow


class BlockProjectionTests(unittest.TestCase):
    def compare(self, graph, score, embeddings, **kwargs):
        original = structural_flow(graph, score, embeddings, **kwargs)
        for block_size in (1, 7, 1000):
            result = block_structural_flow(graph, score, embeddings, block_size=block_size, **kwargs)
            np.testing.assert_allclose(result['probabilities'].toarray(), original['probabilities'].toarray(), atol=1e-14)
            for key in ('support_count', 'support_fraction', 'active', 'expected_specificity_gain'):
                np.testing.assert_allclose(result[key], original[key], atol=1e-14)
            for key in ('vectors', 'displacement_vectors', 'coherence', 'projection_support'):
                for name in original[key]:
                    np.testing.assert_allclose(result[key][name], original[key][name], atol=1e-14)

    def test_random_graphs_orientation_powers_masks_and_neighbor_gates(self):
        rng = np.random.default_rng(2)
        dense = rng.uniform(0, 1, (31, 31)); dense[dense < .8] = 0.
        dense[30] = 0.; dense[:, 30] = 0.
        graph = sparse.csr_matrix(dense.astype('float32'))
        score = rng.uniform(0, 1, 31); score[3] = np.nan
        embeddings = {'2d': rng.normal(size=(31, 2)), '3d': rng.normal(size=(31, 3))}
        embeddings['2d'][7] = embeddings['2d'][8]
        before = graph.copy()
        for sign in (-1, 1):
            for power in (.5, 1., 2.):
                for minimum in (1, 3):
                    self.compare(graph, sign * score, embeddings, edge_tolerance=.05,
                                 gradient_power=power, min_support=minimum)
        np.testing.assert_array_equal(graph.data, before.data)
        np.testing.assert_array_equal(graph.indices, before.indices)

    def test_plateaus_isolates_and_zero_weight_edges(self):
        graph = sparse.csr_matrix(([1., 0., 1., 2.], ([0, 0, 1, 2], [1, 2, 0, 2])), shape=(4, 4))
        self.compare(graph, [1., 1., 1., np.nan], np.arange(8).reshape(4, 2))
        self.compare(sparse.csr_matrix((4, 4)), [0., .2, .3, .4], np.arange(8).reshape(4, 2))

    def test_uncertainty_margin_and_duplicate_entries(self):
        graph = sparse.csr_matrix((np.array([.2, .3, .5, .1]), np.array([1, 1, 2, 0]),
                                   np.array([0, 3, 4, 4, 4])), shape=(4, 4))
        self.assertFalse(graph.has_canonical_format)
        self.compare(graph, [0., .5, .8, .4], np.arange(8).reshape(4, 2),
                     score_uncertainty=[.01, .05, np.nan, -.1], edge_confidence_z=1.5, edge_tolerance=0)

    def test_empty_graph(self):
        self.compare(sparse.csr_matrix((0, 0)), np.array([]), {})


if __name__ == '__main__':
    unittest.main()
