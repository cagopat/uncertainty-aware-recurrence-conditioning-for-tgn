"""Only handmade posteriors/probabilities; never saved scientific rows."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sw1_analysis import (posterior_disagreement,disagreement_strata,metrics,auc,
    grouped_metrics,contrast_definitions,block_summary,aggregate,METHODS,
    assignment_diagnostics, conditional_block_bootstrap, support_diagnostics, PRIMARY)
from sw1_data import labels_for_clock


def handmade():
    posterior=np.array([[.6,0,.4,0,0,0],[0,.1,0,.9,0,0]],float)
    d=posterior_disagreement(posterior)
    data={'x':np.zeros((2,16),bool),'y':np.zeros((2,16),bool),
          'clock_index':np.array([0,3]),'labels':labels_for_clock(np.array([0,3]),np.arange(1,9))}
    p={m:np.full((2,8),.5) for m in METHODS}
    p.update(fitted_clock_posterior=posterior,fitted_period_probability=d['period_masses'],
             fitted_context_soft=d['posterior_soft_probability'],fitted_context_hard=d['posterior_MAP_probability'],
             context_entropy=d['period_entropy_bits'],context_map_period=np.array([2,4]),
             reassignment_donor_index=np.array([1,0]),reassigned_period_probability=d['period_masses'][[1,0]])
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
        self.assertEqual(PRIMARY, 'soft_fixed_minus_support_reassigned_fixed')
        self.assertEqual(defs[PRIMARY]['role'], 'primary')
        self.assertEqual(defs[PRIMARY]['coefficients'],
                         {'neural_soft_fixed':1.,'neural_support_reassigned_fixed':-1.})
        self.assertEqual(defs['soft_fixed_minus_equal_fixed']['coefficients'],{'neural_soft_fixed':1.,'neural_equal_fixed':-1.})
        self.assertEqual(defs['soft_fixed_minus_equal_fixed']['role'], 'secondary')
        self.assertEqual(defs['support_reassigned_fixed_minus_equal_fixed']['coefficients'],
                         {'neural_support_reassigned_fixed':1.,'neural_equal_fixed':-1.})
        data,p=handmade()
        block,_=block_summary(data,p)
        spec={'blocks':[0,1], 'bootstrap_seed':3,'bootstrap_replicates':10}
        value=aggregate([block,block],spec)
        self.assertEqual(value['contrasts_h5_h8']['soft_fixed_minus_equal_fixed']['blocks'],2)
        self.assertIsNone(value['strata']['own_at_least_three']['methods']['neural_soft_fixed']['h5_h8']['brier']['mean'])
        self.assertEqual(sum(r['rows'] for r in block['disagreement']['D_strata_h5_h8'].values()),8)
        conditional=value['contrasts_h5_h8'][PRIMARY]['support_subgroups']['support_eligible']
        self.assertEqual(conditional['total_blocks'],2)
        self.assertEqual(conditional['defined_resamples'],10)
        self.assertEqual(conditional['all_empty_resamples'],0)
        self.assertEqual(conditional['mean'],value['contrasts_h5_h8'][PRIMARY]['mean'])

    def test_assignment_verified_not_selected_and_uncertainty_stays_original(self):
        data,p=handmade()
        diagnostic=assignment_diagnostics(data,p)
        self.assertTrue(diagnostic['motif_weight_distribution_preserved_exactly'])
        self.assertEqual(diagnostic['changed_normalized_weight_vectors'],2)
        block, details=block_summary(data,p)
        np.testing.assert_array_equal(details['period_masses'],p['fitted_period_probability'])
        self.assertEqual(block['strata']['period_entropy_gt_half_bit']['pairs'],1)
        self.assertEqual(block['strata']['reassigned_weights_changed']['pairs'],2)
        self.assertEqual(block['strata']['reassigned_MAP_changed']['pairs'],2)

    def test_equal_value_donors_remain_legitimate(self):
        data={'x':np.array([[1]+[0]*15,[1]*16,[0]*16,[0]*16],bool),
              'y':np.zeros((4,16),bool)}
        original=np.array([[1.,0.],[1.,0.],[.2,.8],[.2,.8]])
        donors=np.array([0,1,3,2])
        p={'fitted_period_probability':original,'reassigned_period_probability':original[donors],
           'reassignment_donor_index':donors}
        d=assignment_diagnostics(data,p)
        self.assertEqual(d['unchanged_normalized_weight_vectors_despite_different_donor'],2)
        self.assertEqual(d['unchanged_normalized_weight_vectors_including_anchors'],4)
        self.assertEqual(d['anchored_motifs'],2)
        self.assertEqual(d['changed_normalized_weight_vectors'],0)
        self.assertEqual(d['normalized_weight_L1_displacement_mean'],0.)

    def test_permutation_does_not_claim_event_slot_distribution_match(self):
        data,p=handmade()
        data['x'][0,:]=True
        d=assignment_diagnostics(data,p)
        self.assertFalse(d['event_count_weighted_distribution_preservation_required'])
        self.assertNotEqual(d['event_count_weighted_original_mean_weight'],
                            d['event_count_weighted_reassigned_mean_weight'])

    def test_invalid_assignment_rejected(self):
        data,p=handmade()
        bad=dict(p,reassignment_donor_index=np.array([0,1]))
        with self.assertRaises(RuntimeError):assignment_diagnostics(data,bad)
        bad=dict(p,reassignment_donor_index=np.array([1,1]))
        with self.assertRaises(RuntimeError):assignment_diagnostics(data,bad)
        bad=dict(p,reassigned_period_probability=p['fitted_period_probability'].copy())
        with self.assertRaises(RuntimeError):assignment_diagnostics(data,bad)
        bad=dict(p,reassignment_donor_index=np.array([1.,0.]))
        with self.assertRaises(RuntimeError):assignment_diagnostics(data,bad)

    def test_no_silent_incomplete_block_deletion(self):
        data,p=handmade()
        block,_=block_summary(data,p)
        with self.assertRaises(RuntimeError):
            aggregate([block],{'blocks':[0,1],'bootstrap_seed':3,'bootstrap_replicates':10})

    def test_original_MAP_metadata_cannot_silently_use_donor(self):
        data,p=handmade()
        p['context_map_period']=p['context_map_period'][p['reassignment_donor_index']]
        with self.assertRaises(RuntimeError):block_summary(data,p)

    def test_analysis_saved_fixture_layout(self):
        """Handwritten two-row arrays only; no generated cohorts or fitting."""
        from sw1_analysis import analyze
        from sw1_io import read_json, write_json, sha
        data,p=handmade()
        data['pair_index']=np.array([0,1],dtype=np.int64)
        original=p['fitted_period_probability']
        weights=original/np.linalg.norm(original,axis=1,keepdims=True)
        assignment={'donor_index':p['reassignment_donor_index'],
                    'original_period_probability':original,
                    'reassigned_period_probability':p['reassigned_period_probability'],
                    'original_normalized_weights':weights,
                    'reassigned_normalized_weights':weights[p['reassignment_donor_index']]}
        selected={'selected_qualified':True,'selected_endpoint':0,'fit_sha256':'a'*64}
        spec={'blocks':[0,1],'bootstrap_seed':3,'bootstrap_replicates':10}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for block in spec['blocks']:
                unit=f'SW05_B{block:02d}'
                evaluation=root/'results/evaluate'/unit
                training=root/'results/assets'/unit
                evaluation.mkdir(parents=True);training.mkdir(parents=True)
                np.savez_compressed(evaluation/'DATA.npz',**data)
                np.savez_compressed(evaluation/'PREDICTIONS.npz',**p)
                np.savez_compressed(training/'DATA.npz',x=data['x'],y=data['y'],
                                    fitted_period_probability=original)
                for destination in (evaluation,training):
                    np.savez_compressed(destination/'ASSIGNMENT.npz',**assignment)
                    write_json(destination/'ASSIGNMENT_REPORT.json',
                               {'assignment_sha256':sha(destination/'ASSIGNMENT.npz')})
                    report={'block':block,'unit':unit,'data_sha256':sha(destination/'DATA.npz'),
                            'source_fit_record_sha256':'b'*64,
                            'assignment_sha256':sha(destination/'ASSIGNMENT.npz'),
                            'assignment_report_sha256':sha(destination/'ASSIGNMENT_REPORT.json'),
                            'pairing':{'fixture':True}}
                    if destination==evaluation:report['predictions_sha256']=sha(destination/'PREDICTIONS.npz')
                    write_json(destination/'REPORT.json',report)
            # Selection and seed regeneration belong to their separate tests;
            # here mocks prevent source lookup and any study seed draw.
            with mock.patch('sw1_data.selected_fit',return_value=(None,selected,'b'*64)), \
                 mock.patch('sw1_data.verify_assignment',return_value=True) as verification:
                analyze(root,root/'analysis',spec,cohorts=['SW05_B00','SW05_B01'])
            self.assertEqual(verification.call_count,4)
            summary=read_json(root/'analysis/SUMMARY.json')
            self.assertEqual(summary['conditions']['SW05']['primary'],PRIMARY)
            self.assertEqual(len(summary['training_assignment_diagnostics']),2)
            self.assertFalse(summary['population_parameters_refitted'])
            self.assertTrue((root/'analysis/BOOTSTRAP_INDICES.npz').is_file())
            self.assertTrue((root/'analysis/SUPPORT_DIAGNOSTICS.json').is_file())
            self.assertTrue((root/'analysis/SW05_B00_SUPPORT_DIAGNOSTICS.npz').is_file())

    def test_fitted_MAP_error_directions_are_separate(self):
        from sw1_analysis import strata
        data,p=handmade()
        p['context_map_period']=np.array([4,2])
        groups=strata(data,p)
        self.assertEqual(groups['fitted_MAP_true2_selected4'].tolist(),[True,False])
        self.assertEqual(groups['fitted_MAP_true4_selected2'].tolist(),[False,True])

    def test_conditional_bootstrap_keeps_full_blocks_and_reports_empty_resamples(self):
        indices=np.array([[0,0,2,2],[1,1,3,3],[0,1,2,2]],dtype=np.int64)
        result=conditional_block_bootstrap([None,.2,None,.4],indices)
        self.assertEqual(result['total_blocks'],4)
        self.assertEqual(result['nonempty_blocks'],2)
        self.assertEqual(result['all_empty_resamples'],1)
        self.assertEqual(result['defined_resamples'],2)
        self.assertAlmostEqual(result['mean'],.3)
        np.testing.assert_allclose(result['interval_percentile95'],[.2025,.2975])
        empty=conditional_block_bootstrap([None]*4,indices)
        self.assertIsNone(empty['mean'])
        self.assertIsNone(empty['interval_percentile95'])
        self.assertEqual(empty['all_empty_resamples'],3)

    def test_noneligible_singletons_are_anchors_and_may_have_same_predictions(self):
        data={'x':np.zeros((2,16),bool),'y':np.zeros((2,16),bool)}
        original=np.array([[.6,.4],[1.,0.]])
        p={'fitted_period_probability':original,'reassigned_period_probability':original.copy(),
           'reassignment_donor_index':np.arange(2)}
        report=assignment_diagnostics(data,p)
        self.assertEqual(report['eligible_motifs'],1)
        self.assertEqual(report['eligible_singleton_motifs'],1)
        self.assertEqual(report['movable_motifs'],0)
        self.assertEqual(report['self_donors'],2)
        self.assertEqual(report['changed_mass_vectors'],0)

    def test_support_deletion_is_rejected_even_if_global_multiset_preserved(self):
        data={'x':np.zeros((2,16),bool),'y':np.zeros((2,16),bool)}
        original=np.array([[.6,.4],[1.,0.]])
        p={'fitted_period_probability':original,'reassigned_period_probability':original[::-1].copy(),
           'reassignment_donor_index':np.array([1,0])}
        with self.assertRaisesRegex(RuntimeError,'retain their own weights'):
            assignment_diagnostics(data,p)

    def test_float32_underflow_is_anchored_not_treated_as_dual_positive(self):
        data={'x':np.zeros((3,16),bool),'y':np.zeros((3,16),bool)}
        original=np.array([[1e-50,1.],[.6,.4],[.2,.8]])
        donors=np.array([0,2,1])
        p={'fitted_period_probability':original,'reassigned_period_probability':original[donors],
           'reassignment_donor_index':donors}
        report=assignment_diagnostics(data,p)
        self.assertEqual(report['eligible_motifs'],2)
        self.assertEqual(report['anchored_motifs'],1)
        self.assertTrue(report['normalized_float32_support_preserved_per_recipient'])

    def test_support_diagnostic_does_not_require_network_use_of_positive_channel(self):
        data,p=handmade()
        report, arrays=support_diagnostics(data,p)
        self.assertEqual(report['new_raw_P4_support_loss'],0)
        self.assertEqual(report['new_effective_P4_support_loss'],0)
        self.assertTrue(arrays['eligible'].all())
        # All handmade neural forecasts are .5: this must remain a valid
        # descriptive result even with two nonzero phase channels.
        neural=report['groups']['support_eligible']['methods']['neural_support_reassigned_fixed']
        self.assertEqual(neural['maximum_h_plus_two_probability_difference'],0)
        self.assertEqual(neural['exact_two_step_symmetric_motifs'],2)


if __name__=='__main__':unittest.main()
