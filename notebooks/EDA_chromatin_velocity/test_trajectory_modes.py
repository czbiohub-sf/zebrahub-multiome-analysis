"""Focused numerical checks for six-stage trajectory fitting."""
from pathlib import Path
import sys, unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/chrom_velo'))
from trajectory_modes import fit_trajectories
T=np.array([10,12,14,16,19,24.])

class TrajectoryTests(unittest.TestCase):
    def fit(self,y,**kwargs):
        return fit_trajectories(np.asarray(y),T,measurement_variance=np.full(np.shape(y),.0001),**kwargs)
    def test_stationary_broad(self):
        r=self.fit(np.full((8,20,6),10.))
        np.testing.assert_allclose(r['rates'],0,atol=1e-12)
        self.assertTrue((r['mode']==1).all());self.assertFalse(r['rate_supported'].any())
    def test_actual_time_spacing_and_common_mode(self):
        y=np.broadcast_to(2+.8*(T-10),(4,20,6))
        r=self.fit(y)
        np.testing.assert_allclose(r['rates'],.8,atol=1e-10)
        self.assertTrue((r['mode']==2).all())
    def test_peak_and_trough(self):
        y=np.array([12-np.abs(T-14),2+np.abs(T-16)])
        r=self.fit(y)
        np.testing.assert_allclose(r['fitted'],y,atol=1e-10)
        np.testing.assert_array_equal(r['mode'],[4,5])
    def test_missing_stage_not_interpolated_or_bridged(self):
        y=2+.8*(T-10);obs=np.ones(6,bool);obs[2]=False
        r=self.fit(y,observed=obs)
        self.assertTrue(np.isnan(r['fitted'][2]));self.assertTrue(np.isnan(r['rates'][1:3]).all())
        np.testing.assert_allclose(r['rates'][[0,3,4]],.8,atol=1e-10)
    def test_insufficient(self):
        r=self.fit(np.arange(6.),observed=[1,1,1,0,0,0])
        self.assertEqual(r['mode'],0);self.assertFalse(r['fit_valid']);self.assertTrue(np.isnan(r['rates']).all())
    def test_amplitude_not_erased(self):
        y=np.array([12-np.abs(T-14),2+.8*(T-10)])
        r=self.fit(y)
        s=fit_trajectories(y*7,T,measurement_variance=np.full(y.shape,.0001)*49,rate_floor=.35)
        np.testing.assert_allclose(s['rates'],r['rates']*7,atol=1e-10)
        np.testing.assert_array_equal(r['mode'],s['mode'])
    def test_complex_high_signal_abstains(self):
        r=self.fit([0,10,0,10,0,10])
        self.assertEqual(r['mode'],6);self.assertFalse(r['rate_supported'].any())
    def test_weight_scale_invariance(self):
        y=[1,2,3,5,4,2]
        r=self.fit(y,weights=[3,4,5,2,1,.5])
        s=self.fit(y,weights=np.array([3,4,5,2,1,.5])*100)
        np.testing.assert_allclose(r['fitted'],s['fitted'],atol=1e-10)

if __name__=='__main__':unittest.main()
