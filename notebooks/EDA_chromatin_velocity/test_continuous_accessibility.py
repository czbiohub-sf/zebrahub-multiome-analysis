"""Scientific invariants of the continuous peak-profile hierarchy."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'/'chrom_velo'))
from continuous_accessibility import ContinuousConfig,make_continuous,evaluate_hierarchy


class ContinuousAccessibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=make_continuous(ContinuousConfig(n_constitutive=8,n_per_family_trunk=12,n_per_celltype_branch=6))

    def test_matrix_and_probabilities(self):
        a=self.a
        self.assertEqual(a.shape,(176,120))
        self.assertTrue(a.obs_names.is_unique)
        self.assertTrue(((a.layers['truth_state']>0)&(a.layers['truth_state']<1)).all())
        self.assertTrue(np.isfinite(a.X).all())

    def test_constitutive_core_is_high_and_has_zero_true_rate(self):
        a=self.a
        root=a.obs.true_program.to_numpy()=='constitutive_core'
        x=a.layers['truth_state'][root]
        self.assertTrue((x>0.64).all())
        np.testing.assert_array_equal(x,np.repeat(x[:,:1],120,axis=1))
        np.testing.assert_array_equal(a.layers['truth_velocity'][root],0.)
        np.testing.assert_array_equal(a.layers['truth_instantaneous_rate'][root],0.)

    def test_early_accessibility_is_shared_across_celltypes(self):
        x=self.a.layers['truth_state'].reshape(len(self.a),20,6)[:,:,0]
        np.testing.assert_allclose(x,np.repeat(x[:,:1],20,axis=1),atol=1e-7)

    def test_exact_analytic_derivative(self):
        _,rate=evaluate_hierarchy(self.a.obs,[16.])
        plus,_=evaluate_hierarchy(self.a.obs,[16.0001])
        minus,_=evaluate_hierarchy(self.a.obs,[15.9999])
        np.testing.assert_allclose(rate,(plus-minus)/0.0002,atol=1e-8)

    def test_junctions_have_no_profile_gap(self):
        p=self.a.obs.iloc[[8]].copy()
        p['true_family_weight']=0.;p['true_celltype_weight']=0.
        root,_=evaluate_hierarchy(p,[10,16,24])
        np.testing.assert_allclose(root,p.true_baseline.iloc[0])
        # F endpoint and C branch origin have identical mixing weights.
        p['true_family_weight']=1.
        family,_=evaluate_hierarchy(p,[10,16,24])
        q=p.copy();q['true_program']='celltype_branch'
        branch,_=evaluate_hierarchy(q,[10,16,24])
        np.testing.assert_array_equal(family,branch)
        # The one-sided limit is continuous too, not just the exact endpoint.
        q['true_family_weight']=1-1e-6;q['true_celltype_weight']=1e-6
        near,_=evaluate_hierarchy(q,[10,16,24])
        self.assertLess(abs(near-family).max(),1e-6)

    def test_target_opens_while_off_family_closes(self):
        a=self.a
        for i,row in a.obs.iterrows():
            if row.true_program=='constitutive_core': continue
            idx=a.obs_names.get_loc(i)
            target=int(row.true_celltype_index)
            outside=(int(row.true_family)+1)%4*5
            rates=a.layers['truth_velocity'][idx].reshape(20,6)
            self.assertTrue((rates[target,:5]>0).all())
            self.assertTrue((rates[outside,:5]<0).all())

    def test_cis_graph_and_terminal_masks(self):
        a=self.a;g=a.obsp['coaccessibility'].tocoo()
        chrom=a.obs.chrom.to_numpy()
        self.assertTrue((chrom[g.row]==chrom[g.col]).all())
        self.assertEqual((g.tocsr()-g.T.tocsr()).nnz,0)
        self.assertFalse(a.layers['truth_velocity_valid'][:,5::6].any())


if __name__=='__main__':
    unittest.main()
