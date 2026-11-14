"""Handwritten array/algebra tests only; no generated cohorts or inference."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cw1_encoding as e


class ClockWeightEncoding(unittest.TestCase):
    def setUp(self):
        self.mass = np.asarray([[.5, .5], [.8, .2], [.25, .75], [1., 0.], [0., 1.]])
        self.lag = np.tile(np.asarray([1., 2., 3., 4., 8.]), (5, 1))
        self.valid = np.ones((5, 5), bool)
        self.valid[:, -1] = False
        self.channels = e.bank_coordinates(self.lag, self.valid)

    def test_exact_four_weight_maps_and_norms(self):
        self.assertEqual(e.ARMS, ('hard_fixed', 'equal_fixed', 'soft_fixed', 'period4_fixed'))
        radius = np.linalg.norm(self.mass, axis=1)
        expected = {'hard_fixed': [[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]],
                    'equal_fixed': np.full((5, 2), 1./np.sqrt(2.)),
                    'soft_fixed': self.mass/radius[:, None],
                    'period4_fixed': np.tile([0., 1.], (5, 1))}
        for arm in e.ARMS:
            with self.subTest(arm=arm):
                result = e.bank_weights(self.mass, arm)
                np.testing.assert_array_equal(result['effective_period_weights'], expected[arm])
                np.testing.assert_array_equal(result['applied_radial_factor'], np.ones(5))
                np.testing.assert_allclose(np.sum(result['effective_period_weights']**2, axis=1), 1., atol=1e-15)
                bank = e.phase_bank(self.channels, self.mass, arm)
                self.assertEqual(bank.shape, (5, 5, 16))
                np.testing.assert_allclose(np.linalg.norm(bank.astype(float), axis=-1)[:, :4], 2., atol=2e-7, rtol=0)
                np.testing.assert_array_equal(bank[:, -1], np.zeros((5, 16)))

    def test_equal_and_period4_do_not_use_mass_values(self):
        for arm in ('equal_fixed', 'period4_fixed'):
            np.testing.assert_array_equal(e.phase_bank(self.channels, self.mass, arm),
                                          e.phase_bank(self.channels, self.mass[::-1], arm))
        equal = e.bank_weights(self.mass, 'equal_fixed')['effective_period_weights']
        self.assertGreater(equal[0].sum(), 1.)  # These are not probabilities.

    def test_correct_coincidences_not_false_all_arm_degeneracy(self):
        hard = e.phase_bank(self.channels, self.mass, 'hard_fixed')
        soft = e.phase_bank(self.channels, self.mass, 'soft_fixed')
        equal = e.phase_bank(self.channels, self.mass, 'equal_fixed')
        p4 = e.phase_bank(self.channels, self.mass, 'period4_fixed')
        np.testing.assert_array_equal(hard[3:], soft[3:])
        np.testing.assert_array_equal(equal[0], soft[0])
        np.testing.assert_array_equal(p4[4], hard[4])
        self.assertFalse(np.array_equal(p4[3], hard[3]))
        self.assertFalse(np.array_equal(equal[3], hard[3]))

    def test_genuine_period4_distinguishes_two_steps_but_all_repeat_four(self):
        later = e.bank_coordinates(self.lag+4, self.valid)
        for arm in e.ARMS:
            np.testing.assert_array_equal(e.phase_bank(self.channels, self.mass, arm),
                                          e.phase_bank(later, self.mass, arm))
        bank = e.phase_bank(self.channels, self.mass, 'period4_fixed')
        self.assertFalse(np.array_equal(bank[:, 0], bank[:, 2]))
        hard2 = e.phase_bank(self.channels, np.tile([[1., 0.]], (5, 1)), 'hard_fixed')
        np.testing.assert_array_equal(hard2[:, 0], hard2[:, 2])

    def test_invalid_masses_and_unknown_arm_rejected(self):
        for bad in ([[.1, .8]], [[np.nan, .5]], [[-.1, 1.1]], [[0., 0.]]):
            with self.assertRaises(RuntimeError):
                e.bank_weights(bad, 'equal_fixed')
        with self.assertRaises(RuntimeError):
            e.bank_weights(self.mass, 'soft_adaptive')

    def test_empty_own_and_all_empty_remain_valid(self):
        own = np.zeros((2, 16), bool)
        neighbor = np.zeros_like(own)
        neighbor[0, [1, 5]] = True
        a = e.encoding_records(own, neighbor, [[.2, .8], [.5, .5]], [1, 5])
        self.assertFalse(a['valid'][0].any())
        self.assertTrue(a['valid'][1].any())
        self.assertFalse(a['valid'][6:].any())
        for arm in e.ARMS:
            bank = e.phase_bank(a['channels'], a['period_probability'], arm)
            self.assertTrue(np.isfinite(bank).all())
            np.testing.assert_array_equal(bank[6:], np.zeros_like(bank[6:]))
        self.assertNotIn('shared_indicator', a)

    def test_tied_event_order_and_float_metadata(self):
        own = np.zeros((1, 16), bool); neighbor = np.zeros_like(own)
        own[0, [0, 4]] = True; neighbor[0, [0, 2]] = True
        a = e.encoding_records(own, neighbor, [[.5, .5]], [1, 5])
        np.testing.assert_array_equal(a['event_stream'][1, -4:], [0, 1, 1, 0])
        np.testing.assert_array_equal(a['event_times'][1, -4:], [80, 80, 82, 84])
        original = np.array([[[.125, 0., 1., .0625, .9375, 1.]]], np.float32)
        copied = original.copy()
        changed = e.decoder_metadata(original)
        np.testing.assert_array_equal(original, copied)
        np.testing.assert_array_equal(changed[:, :, 1:], original[:, :, 1:])
        self.assertEqual(changed[0, 0, 0], 0.)
        self.assertEqual(changed.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
