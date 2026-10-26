"""Scientific invariants of the independently generated trajectory controls."""
import unittest

import numpy as np

from trajectory_benchmark import FAMILIES, TIMES, make_benchmark, rate_mode
from trajectory_modes import fit_trajectories


class TrajectoryBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_benchmark(seed=42,n_per_family=8,n_celltypes=4)

    def rows(self,family):
        return self.data["metadata"].family.to_numpy()==family

    def test_stationary_controls_have_exact_zero_truth_rates(self):
        for family in FAMILIES[:3]+("isolated_technical_spike",):
            np.testing.assert_array_equal(self.data["true_rates"][self.rows(family)],0.)

    def test_uniform_change_and_celltype_switch_have_different_mean_behavior(self):
        for family,sign in [("uniform_opening",1),("uniform_closing",-1)]:
            values=self.data["truth"][self.rows(family)]
            np.testing.assert_allclose(values[:,1:],np.broadcast_to(values[:,:1],values[:,1:].shape))
            self.assertTrue((sign*np.diff(values,axis=-1)>0).all())
        switches=self.data["truth"][self.rows("celltype_switch")].mean(axis=1)
        np.testing.assert_allclose(switches,np.broadcast_to(switches[:,:1],switches.shape),atol=1e-14)

    def test_broad_pulses_and_troughs_reverse_on_actual_nonuniform_time_grid(self):
        for family,mode in [("broad_pulse",3),("broad_trough",4)]:
            values=self.data["truth"][self.rows(family)]
            np.testing.assert_allclose(values[:,:,0],values[:,:,-1],atol=1e-14)
            rates=np.diff(values,axis=-1)/np.diff(TIMES)
            np.testing.assert_array_equal(rate_mode(rates,floor=0.001),mode)
        opening=self.data["true_rates"][self.rows("uniform_opening")]
        np.testing.assert_allclose(opening,np.broadcast_to(opening[:,:,:1],opening.shape),atol=1e-14)

    def test_true_narrow_pulse_and_technical_spike_are_not_identifiable(self):
        a,b=self.rows("narrow_true_pulse"),self.rows("isolated_technical_spike")
        np.testing.assert_array_equal(self.data["counts"][a],self.data["counts"][b])
        np.testing.assert_array_equal(self.data["values"][a],self.data["values"][b])
        self.assertTrue((self.data["truth"][a]!=self.data["truth"][b]).any())
        result=fit_trajectories(self.data["values"],TIMES,observed=self.data["observed"],
                                weights=self.data["exposure"],
                                measurement_variance=self.data["measurement_variance"])
        for key in ["mode","model","rate_supported","fitted","rates"]:
            np.testing.assert_allclose(result[key][a],result[key][b],equal_nan=True)

    def test_support_uses_total_counts_and_does_not_impute_low_signal(self):
        expected=self.data["counts"].sum(axis=-1)>=20
        np.testing.assert_array_equal(self.data["count_supported"],expected)
        self.assertTrue(self.data["observed"].all())
        self.assertTrue((~expected).any())
        self.assertTrue(((self.data["counts"]==0)&self.data["observed"]).any())
        np.testing.assert_allclose(self.data["values"],self.data["counts"]*1e6/self.data["exposure"])

    def test_independent_seeds_and_nonnegative_unscaled_truth(self):
        duplicate=make_benchmark(seed=42,n_per_family=8,n_celltypes=4)
        independent=make_benchmark(seed=43,n_per_family=8,n_celltypes=4)
        np.testing.assert_array_equal(duplicate["values"],self.data["values"])
        self.assertFalse(np.array_equal(independent["values"],self.data["values"]))
        self.assertTrue((self.data["truth"]>=0).all())
        self.assertGreater(self.data["truth"].max(),1.)
        self.assertGreaterEqual(self.data["exposure"].min(),.5e6)
        self.assertLessEqual(self.data["exposure"].max(),30e6)
        with self.assertRaises(ValueError):make_benchmark(n_per_family=200)


if __name__=="__main__":
    unittest.main()
