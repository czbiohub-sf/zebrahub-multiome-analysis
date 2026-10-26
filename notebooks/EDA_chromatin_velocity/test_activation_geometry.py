"""Small graph controls independent of the real atlas topology."""
import sys
from pathlib import Path
import numpy as np
import unittest
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/chrom_velo'))
from activation_geometry import reference_distances,region_graph_totals,select_reference_core


def make_graph(n, edges):
    r=[];c=[];v=[]
    for a,b,w in edges:r.extend([a,b]);c.extend([b,a]);v.extend([w,w])
    return sparse.csr_matrix((v,(r,c)),shape=(n,n))


def test_multiroot_hops_and_unreachable_isolate():
    g=make_graph(8,[(0,1,1),(1,2,1),(2,3,1),(3,4,1),(5,6,1)])
    b=np.zeros(8,bool);b[[0,4,7]]=True
    r=reference_distances(g,b)
    np.testing.assert_array_equal(r['distance_hops'],[0,1,2,1,0,np.inf,np.inf,0])
    np.testing.assert_allclose(r['distance_weighted'],r['distance_hops'])
    assert r['n_components']==3


def test_weak_affinity_can_favor_longer_path_and_input_unchanged():
    g=make_graph(3,[(0,1,.001),(0,2,1),(2,1,1)])
    original=g.copy();r=reference_distances(g,[True,False,False])
    assert r['distance_hops'][1]==1
    assert r['distance_weighted'][1]==2
    assert (g!=original).nnz==0


def test_empty_reference_and_bad_graph():
    g=make_graph(2,[(0,1,.5)])
    assert np.isinf(reference_distances(g,[False,False])['distance_hops']).all()
    with unittest.TestCase().assertRaisesRegex(ValueError,'symmetric'):
        reference_distances(sparse.csr_matrix([[0,.5],[0,0]]),[True,False])


def test_region_aggregation_preserves_original_edge_totals():
    g=make_graph(4,[(0,1,1),(1,2,.5),(2,3,.25)])
    count,weight=region_graph_totals(g,[0,0,1,1],block_size=1)
    np.testing.assert_array_equal(count,[[2,1],[1,2]])
    np.testing.assert_allclose(weight,[[2,.5],[.5,.5]])
    assert count.sum()==g.nnz
    assert np.isclose(weight.sum(),g.data.sum())


def test_core_reference_minimal_and_deterministic():
    reference=np.array([True]*9+[False]);codes=np.array([2]*5+[0]*2+[1]*2+[3])
    core,chosen=select_reference_core(reference,codes,coverage=.7)
    np.testing.assert_array_equal(chosen,[2,0])
    assert core.sum()==7
    assert np.all(core<=reference)


if __name__=='__main__':
    tests=[test_multiroot_hops_and_unreachable_isolate,test_weak_affinity_can_favor_longer_path_and_input_unchanged,test_empty_reference_and_bad_graph,test_region_aggregation_preserves_original_edge_totals,test_core_reference_minimal_and_deterministic]
    for test in tests:test();print('PASS',test.__name__)
    print(f'{len(tests)} numerical geometry controls passed')
