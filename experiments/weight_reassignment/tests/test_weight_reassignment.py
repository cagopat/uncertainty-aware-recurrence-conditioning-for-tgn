"""Handwritten-array tests for weight reassignment; no fitting or model runs."""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import weight_reassignment_data as data
import weight_reassignment_encoding as encoding
from weight_reassignment_worker import evaluation_group_barrier


class ReassignmentTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((ROOT/'SPEC.json').read_text())
        self.mass = np.asarray([[.9, .1], [.3, .7], [.5, .5], [1., 0.]], dtype=np.float64)
        self.donor = np.asarray([1, 2, 0, 3], dtype=np.int64)

    def test_fixed_spec_and_identifiers(self):
        data.validate_spec(self.spec)
        self.assertEqual(data.block_number('SW05_B15'), 15)
        self.assertEqual(data.split_fit('SW05_B00__support_reassigned_fixed'), ('SW05_B00', 'support_reassigned_fixed'))
        for bad in ('R05_B00', 'SW05_B16', 'SW05_B0', 'F05_B00'):
            with self.assertRaises(RuntimeError):
                data.block_number(bad)
        changed = copy.deepcopy(self.spec)
        changed['reassignment']['pool'] = 'minibatch'
        with self.assertRaises(RuntimeError):
            data.validate_spec(changed)

    def test_permutation_uniform_rejection_not_rotation(self):
        donor, draws = data.stable_derangement(5, 8419)
        again, again_draws = data.stable_derangement(5, 8419)
        np.testing.assert_array_equal(donor, again)
        np.testing.assert_array_equal(np.sort(donor), np.arange(5))
        self.assertTrue(np.all(donor != np.arange(5)))
        self.assertEqual(draws, again_draws)
        # Inspect the actual primitive; this is not a numerical uniformity experiment.
        source = inspect.getsource(data.stable_derangement)
        self.assertIn('generator.permutation(size)', source)
        self.assertNotIn('roll(', source)

    def test_rejection_bound_fails_without_fallback(self):
        class IdentityOnly:
            def permutation(self, size):
                return np.arange(size)
        with patch.object(data.np.random, 'default_rng', return_value=IdentityOnly()):
            with self.assertRaisesRegex(RuntimeError, 'no fallback'):
                data.stable_derangement(4, 11, maximum_draws=3)

    def test_exact_multiset_original_preserved_and_anchor_self(self):
        original = self.mass.copy()
        a = data.reassignment_arrays(self.mass, self.donor)
        np.testing.assert_array_equal(self.mass, original)
        np.testing.assert_array_equal(a['original_period_probability'], original)
        np.testing.assert_array_equal(a['reassigned_period_probability'], original[self.donor])
        np.testing.assert_array_equal(a['reassigned_normalized_weights'],
                                      a['original_normalized_weights'][self.donor])
        np.testing.assert_allclose(np.linalg.norm(a['reassigned_normalized_weights'], axis=1), 1., rtol=0, atol=1e-14)
        np.testing.assert_array_equal(a['eligible'], [True, True, True, False])
        np.testing.assert_array_equal(a['movable'], a['eligible'])
        np.testing.assert_array_equal(a['anchored'], [False, False, False, True])
        self.assertEqual(a['donor_index'][-1], 3)
        np.testing.assert_array_equal(a['raw_support'], encoding.support_profile(original)['raw_support'])
        for invalid in (np.arange(4), np.asarray([1, 1, 3, 2]), np.asarray([4, 0, 1, 2])):
            with self.assertRaises(RuntimeError):
                data.reassignment_arrays(self.mass, invalid)

    def test_equal_weight_ties_are_not_rejected(self):
        tied = np.tile([[.5, .5]], (4, 1))
        a = data.reassignment_arrays(tied, np.asarray([1, 0, 3, 2]))
        np.testing.assert_array_equal(a['original_period_probability'], a['reassigned_period_probability'])
        self.assertTrue(np.all(a['donor_index'] != np.arange(4)))

    def test_zero_and_float32_underflow_support_are_anchored(self):
        masses = np.asarray([[.9, .1], [.3, .7], [1., 0.], [0., 1.],
                             [1., 1e-50], [1e-50, 1.]], dtype=np.float64)
        profile = encoding.support_profile(masses)
        np.testing.assert_array_equal(profile['eligible'], [True, True, False, False, False, False])
        self.assertTrue(profile['raw_support'][4].all())
        np.testing.assert_array_equal(profile['effective_support'][4], [True, False])
        donors = np.asarray([1, 0, 2, 3, 4, 5])
        a = data.reassignment_arrays(masses, donors)
        np.testing.assert_array_equal(a['reassigned_period_probability'][2:], masses[2:])
        bad = np.asarray([1, 2, 0, 3, 4, 5])
        with self.assertRaises(RuntimeError):
            data.reassignment_arrays(masses, bad)

    def test_singleton_or_no_eligible_pool_remains_unchanged(self):
        for masses in (np.asarray([[.5, .5], [1., 0.], [1., 1e-50]]),
                       np.asarray([[1., 0.], [0., 1.]]), np.asarray([[.5, .5]])):
            a = data.reassignment_arrays(masses, np.arange(len(masses)))
            self.assertFalse(a['movable'].any())
            self.assertTrue(a['anchored'].all())
            np.testing.assert_array_equal(a['original_period_probability'], a['reassigned_period_probability'])
        fixture = copy.deepcopy(self.spec)
        fixture['train_pairs'] = fixture['evaluation_pairs'] = 3
        masses = np.asarray([[.5, .5], [1., 0.], [1., 1e-50]])
        with patch.object(data, 'validate_spec', return_value=None), \
             patch.object(data, 'stable_derangement', side_effect=AssertionError('No draw allowed for singleton')):
            a, r = data.assignment(fixture, 'SW05_B00', 'train', masses)
        self.assertEqual(r['rejection_draws'], 0)
        self.assertTrue(r['singleton_eligible_anchored'])
        self.assertEqual(r['self_donor_count'], 3)

    def test_support_guard_uses_saved_cohort_mask_not_batch_eligibility(self):
        # A one-row minibatch is movable because the full cohort has peers.
        original = np.asarray([[.9, .1]])
        assigned = np.asarray([[.3, .7]])
        report = encoding.support_assignment_guard(original, assigned, np.asarray([True]))
        self.assertTrue(report['raw_support_preserved'])
        with self.assertRaisesRegex(RuntimeError, 'Anchored'):
            encoding.support_assignment_guard(original, assigned, np.asarray([False]))
        with self.assertRaisesRegex(RuntimeError, 'mathematical'):
            encoding.support_assignment_guard(original, np.asarray([[1., 0.]]), np.asarray([True]))
        with self.assertRaisesRegex(RuntimeError, 'float32'):
            encoding.support_assignment_guard(original, np.asarray([[1., 1e-50]]), np.asarray([True]))

    def test_pool_permutation_uses_eligible_local_indices_only(self):
        fixture = copy.deepcopy(self.spec)
        fixture['train_pairs'] = fixture['evaluation_pairs'] = 4
        with patch.object(data, 'validate_spec', return_value=None), \
             patch.object(data, 'stable_derangement', return_value=(np.asarray([2, 0, 1]), 3)) as draw:
            a, report = data.assignment(fixture, 'SW05_B00', 'train', self.mass)
        self.assertEqual(draw.call_args.args[0], 3)
        np.testing.assert_array_equal(a['donor_index'], [2, 0, 1, 3])
        self.assertEqual(report['anchored_count'], 1)
        self.assertEqual(report['movable_count'], 3)
        self.assertFalse(report['zero_self_donors'])
        self.assertTrue(report['zero_movable_self_donors'])

    def test_assignment_determinism_and_saved_array_tampering(self):
        # Four hand-written rows; disable only the scientific population-size
        # validator for this small fixture. No scientific data are generated.
        fixture = copy.deepcopy(self.spec)
        fixture['train_pairs'] = fixture['evaluation_pairs'] = 4
        with patch.object(data, 'validate_spec', return_value=None):
            arrays, report = data.assignment(fixture, 'SW05_B00', 'train', self.mass)
            self.assertTrue(data.verify_assignment(fixture, 'SW05_B00', 'train', self.mass, arrays, report))
            changed = {k: v.copy() for k, v in arrays.items()}
            changed['reassigned_period_probability'][0] = [.2, .8]
            with self.assertRaises(RuntimeError):
                data.verify_assignment(fixture, 'SW05_B00', 'train', self.mass, changed, report)
        self.assertFalse(report['future_labels_or_truth_used_for_assignment'])
        self.assertFalse(report['selection_on_weight_change_or_effect'])

    def test_stream_separation_and_assignment_input_boundary(self):
        keys = [data.stream_seed(self.spec, 'SW05_B00', split, component)
                for split, component in [('train', 'weight_assignment'),
                                         ('evaluation', 'weight_assignment'),
                                         ('train', 'candidate_clock'),
                                         ('model', 'paired_initialization')]]
        self.assertEqual(len(set(keys)), len(keys))
        self.assertNotEqual(keys[0], data.stream_seed(self.spec, 'SW05_B01', 'train', 'weight_assignment'))
        self.assertEqual(list(inspect.signature(data.assignment).parameters), ['spec', 'unit', 'split', 'masses'])


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.mass = np.asarray([[.9, .1], [.3, .7], [.5, .5]], dtype=np.float64)
        self.valid = np.asarray([[True, True, False]]*3)
        self.delta = np.asarray([[1., 2., 3.], [5., 6., 7.], [9., 10., 11.]])

    def test_all_arms_same_width_norm_padding_and_cycle(self):
        channels = encoding.bank_coordinates(self.delta, self.valid)
        next_cycle = encoding.bank_coordinates(self.delta+4, self.valid)
        np.testing.assert_array_equal(channels, next_cycle)
        for arm in encoding.ARMS:
            bank = encoding.phase_bank(channels, self.mass, arm)
            self.assertEqual(bank.shape, (3, 3, 16))
            self.assertEqual(bank.dtype, np.float32)
            np.testing.assert_allclose(np.linalg.norm(bank, axis=-1)[self.valid], 2., atol=2e-7, rtol=0)
            np.testing.assert_array_equal(bank[~self.valid], 0.)

    def test_actual_bank_support_matches_normalized_float32_coefficients(self):
        masses = np.asarray([[1., 0.], [0., 1.], [1., 1e-50],
                             [1e-50, 1.], [.5, .5], [1., 1e-44]])
        delta = np.tile(np.arange(1, 9, dtype=float), (len(masses), 1))
        valid = np.ones(delta.shape, dtype=bool)
        channels = encoding.bank_coordinates(delta, valid)
        bank = encoding.phase_bank(channels, masses, 'support_reassigned_fixed')
        actual = np.any(bank[:, :, :4].reshape(len(masses), 8, 2, 2) != 0, axis=-1)
        support = encoding.support_profile(masses)['effective_support']
        np.testing.assert_array_equal(actual, np.broadcast_to(support[:, None, :], actual.shape))
        # No epsilon deletes a subnormal but representable positive channel.
        self.assertTrue(support[-1].all())

    def test_reassigned_means_only_externally_mapped_masses(self):
        donor = np.asarray([1, 2, 0])
        channels = encoding.bank_coordinates(self.delta, self.valid)
        reassigned = encoding.phase_bank(channels, self.mass[donor], 'support_reassigned_fixed')
        soft_same_masses = encoding.phase_bank(channels, self.mass[donor], 'soft_fixed')
        np.testing.assert_array_equal(reassigned, soft_same_masses)
        # Batching cannot change donor identity; singleton uses same supplied row.
        singleton = np.concatenate([encoding.phase_bank(channels[i:i+1], self.mass[donor][i:i+1],
                                                        'support_reassigned_fixed') for i in range(3)])
        np.testing.assert_array_equal(reassigned, singleton)
        equal_a = encoding.phase_bank(channels, self.mass, 'equal_fixed')
        equal_b = encoding.phase_bank(channels, self.mass[donor], 'equal_fixed')
        np.testing.assert_array_equal(equal_a, equal_b)

    def test_metadata_hides_horizon_relevance_but_not_own_history(self):
        metadata = np.arange(3*2*6, dtype=np.float32).reshape(3, 2, 6)
        original = metadata.copy()
        mapped = encoding.decoder_metadata(metadata)
        np.testing.assert_array_equal(metadata, original)
        np.testing.assert_array_equal(mapped[:, :, :2], 0.)
        np.testing.assert_array_equal(mapped[:, :, 2], 1.)
        np.testing.assert_array_equal(mapped[:, :, 3:], original[:, :, 3:])


class BarrierTests(unittest.TestCase):
    def test_all_48_final_reports_required_before_test_generation(self):
        spec = json.loads((ROOT/'SPEC.json').read_text())
        with tempfile.TemporaryDirectory(prefix='sw1_barrier_fixture_') as temporary:
            work = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, 'All 48'):
                evaluation_group_barrier(work, 'SW05_B00', spec)
            for block in range(16):
                for arm in encoding.ARMS:
                    fit = f'SW05_B{block:02d}__{arm}'
                    path = work/'results/train'/fit/'REPORT.json'
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps({'fit': fit, 'arm': arm, 'epoch': 16,
                                                'selection': 'final_epoch_only'}))
            result = evaluation_group_barrier(work, 'SW05_B00', spec)
            self.assertEqual(result['completed_fit_reports'], 48)
            last = work/'results/train/SW05_B15__support_reassigned_fixed/REPORT.json'
            changed = json.loads(last.read_text())
            changed['epoch'] = 15
            last.write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, 'completed final-epoch'):
                evaluation_group_barrier(work, 'SW05_B00', spec)


if __name__ == '__main__':
    unittest.main()
