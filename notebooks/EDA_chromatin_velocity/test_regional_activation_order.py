import unittest,itertools
import numpy as np
from regional_activation_order import phase_window_center,row_spearman,exact_tail


class RegionalOrderTests(unittest.TestCase):
    def test_interval_amounts_and_scale_invariance(self):
        t=np.array([10,12,14,16,19,24]);x=np.array([[0,2,2,2,4,4],[4,4,2,2,2,0]],float)
        c=phase_window_center(x,t)
        self.assertAlmostEqual(c[0,0],(11+17.5)/2)
        self.assertAlmostEqual(c[1,1],(13+21.5)/2)
        np.testing.assert_allclose(phase_window_center(x*np.array([[3],[.2]]),t),c)
        self.assertTrue(np.isnan(c[0,1]));self.assertTrue(np.isnan(c[1,0]))

    def test_stable_and_missing_are_undefined(self):
        t=[10,12,14,16,19,24];x=np.array([[2]*6,[0,1,np.nan,2,3,4]],float)
        self.assertTrue(np.isnan(phase_window_center(x,t)).all())
        rho,count=row_spearman(np.arange(7),np.arange(7,dtype=float),np.array([1,1,1,1,1,0,0],bool))
        self.assertTrue(np.isnan(rho));self.assertEqual(count,5)

    def test_true_outward_schedule_exceeds_shuffled_timing(self):
        t=np.array([10,12,14,16,19,24]);distance=np.arange(15)
        onset=np.repeat(np.arange(1,6),3)
        x=(np.arange(6)[None,:]>=onset[:,None]).astype(float)
        permutations=np.array(list(itertools.permutations(range(6))))
        windows=phase_window_center(x[...,permutations],t)[:,:,0].T
        rho,count=row_spearman(distance,windows,np.ones(15,bool))
        self.assertGreater(rho[0],.95)
        self.assertLess(exact_tail(rho,rho[0]),.10)
        self.assertLess(np.nanmedian(rho),rho[0])
        self.assertEqual(count[0],15)
        # Exact enumeration includes identity, so a finite observed statistic
        # cannot receive a zero tail fraction.
        self.assertGreater(exact_tail(rho,rho[0]),0)

    def test_time_order_invariant_region_selection(self):
        x=np.array([[0,1,2,1,0,1],[4,4,4,4,4,4]],float)
        selection=np.ptp(x,axis=-1)/x.mean(axis=-1)>=.1
        p=[4,1,5,0,2,3]
        np.testing.assert_array_equal(selection,np.ptp(x[:,p],axis=-1)/x[:,p].mean(axis=-1)>=.1)


if __name__=='__main__':unittest.main()
