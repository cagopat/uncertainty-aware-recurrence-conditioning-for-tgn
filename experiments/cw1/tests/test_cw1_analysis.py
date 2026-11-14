"""Only handmade posteriors/probabilities; never saved scientific rows."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cw1_analysis import (posterior_disagreement,disagreement_strata,metrics,auc,
    grouped_metrics,contrast_definitions,block_summary,aggregate,METHODS)
from cw1_data import labels_for_clock


def handmade():
    posterior=np.array([[.6,0,.4,0,0,0],[0,0,0,1,0,0]],float)
    d=posterior_disagreement(posterior)
    data={'x':np.zeros((2,16),bool),'y':np.zeros((2,16),bool),
          'clock_index':np.array([0,3]),'labels':labels_for_clock(np.array([0,3]),np.arange(1,9))}
    p={m:np.full((2,8),.5) for m in METHODS}
    p.update(fitted_clock_posterior=posterior,fitted_period_probability=d['period_masses'],
             fitted_context_soft=d['posterior_soft_probability'],fitted_context_hard=d['posterior_MAP_probability'],
             context_entropy=d['period_entropy_bits'],context_map_period=np.array([2,4]))
    return data,p


class AnalysisTests(unittest.TestCase):
    def test_disagreement_is_not_MAP_regret(self):
        d=posterior_disagreement(np.array([[.6,0,.4,0,0,0]]))
        self.assertAlmostEqual(d['disagreement'][0,2],.24)
        self.assertAlmostEqual(d['posterior_MAP_regret'][0,2],.16)
        self.assertEqual(d['disagreement'][0,0],0)
        np.testing.assert_allclose(d['posterior_soft_probability'][0],[1,0,.6,0,1,0,.6,0])

    def test_zero_mass_not_missing_probability(self):
        d=posterior_disagreement(np.array([[1,0,0,0,0,0]],float))
        np.testing.assert_array_equal(d['disagreement'],np.zeros((1,8)))
        np.testing.assert_array_equal(d['posterior_MAP_regret'],np.zeros((1,8)))
        self.assertEqual(d['conditional_defined'].tolist(),[[True,False]])

    def test_fixed_boundary_partition(self):
        values=np.array([0,.001,.01,.02,.05,.051,.25])
        masks=disagreement_strata(values)
        np.testing.assert_array_equal(sum(m.astype(int) for m in masks.values()),np.ones(7))
        self.assertEqual(masks['positive_to_0.01'].tolist(),[False,True,True,False,False,False,False])

    def test_metrics_exact_ties_and_unclipped_Brier(self):
        self.assertEqual(auc([0,1],[.5,.5]),.5)
        self.assertIsNone(auc([1,1],[0,1]))
        self.assertEqual(metrics(np.array([0,1]),np.array([1.,0.]))['brier'],1)
        self.assertTrue(np.isfinite(metrics(np.array([0,1]),np.array([1.,0.]))['logloss']))
        self.assertEqual(grouped_metrics(np.array([[1,0],[0,1]]),np.array([[.9,.1],[.8,.2]]))['auc'],1)

    def test_comparator_and_empty_strata_no_row_replication(self):
        defs=contrast_definitions()
        self.assertEqual(defs['soft_fixed_minus_equal_fixed']['coefficients'],{'neural_soft_fixed':1.,'neural_equal_fixed':-1.})
        data,p=handmade()
        block,_=block_summary(data,p)
        spec={'blocks':[0,1], 'bootstrap_seed':3,'bootstrap_replicates':10}
        value=aggregate([block,block],spec)
        self.assertEqual(value['contrasts_h5_h8']['soft_fixed_minus_equal_fixed']['blocks'],2)
        self.assertIsNone(value['strata']['own_at_least_three']['methods']['neural_soft_fixed']['h5_h8']['brier']['mean'])
        self.assertEqual(sum(r['rows'] for r in block['disagreement']['D_strata_h5_h8'].values()),8)

    def test_fitted_MAP_error_directions_are_separate(self):
        from cw1_analysis import strata
        data,p=handmade()
        p['context_map_period']=np.array([4,2])
        groups=strata(data,p)
        self.assertEqual(groups['fitted_MAP_true2_selected4'].tolist(),[True,False])
        self.assertEqual(groups['fitted_MAP_true4_selected2'].tolist(),[False,True])


if __name__=='__main__':unittest.main()
