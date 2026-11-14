"""SW1 data and fitted-clock contracts. No work executes on import.

The generation entry point is scientific work, reserved for
explicitly authorized Colab stages. Authoring tests use handwritten arrays.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from weight_reassignment_io import canonical_hash, read_json, require
from weight_reassignment_encoding import period_probabilities, support_profile, support_assignment_guard

ARMS = ('soft_fixed', 'equal_fixed', 'support_reassigned_fixed')
CONDITIONS = ('SW05',)
RHOS = {'SW05': .5}
CLOCKS = np.asarray([(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)], dtype=np.int64)
TIMES = np.arange(80, 96, dtype=np.int64)


def validate_spec(spec):
    fixed = {'study': 'SW1', 'version': '1.0.0', 'namespace': 'SW1/1.0.0',
             'blocks': list(range(16)), 'arms': list(ARMS), 'base_seed': 2026111303,
             'bootstrap_seed': 2026111304, 'bootstrap_replicates': 10000,
             'train_pairs': 2048, 'evaluation_pairs': 4096,
             'epochs': 16, 'batch_size': 64, 'training_horizons': [1, 2, 3, 4],
             'evaluation_horizons': list(range(1, 9)), 'primary_horizons': [5, 6, 7, 8],
             'cutoff': 95, 'monitored_times': list(range(80, 96)),
             'learning_rate': .001, 'optimizer': 'Adam', 'optimizer_betas': [.9, .999],
             'optimizer_epsilon': 1e-8, 'weight_decay': 0, 'gradient_clip_norm': 1,
             'dropout': 0, 'expected_trainable_parameters': 50434,
             'updates_per_epoch': 32, 'total_updates_per_fit': 512,
             'total_target_presentations_per_fit': 131072, 'expected_fit_count': 48,
             'expected_cohort_count': 16, 'expected_calibration_fit_count': 0,
             'checkpoint_selection': 'final_epoch_16_only', 'scientific_pass_fail_rule': None,
             'primary_contrast': 'soft_fixed_minus_support_reassigned_fixed',
             'memory_message_transform_changed': False, 'decoder_raw_horizon': False,
             'decoder_hidden_shared_indicator': False, 'edge_features_scaled': False,
             'new_population_parameter_fitting': False,
             'replay_old_neural_checkpoints': False}
    for key, value in fixed.items():
        require(spec.get(key) == value, 'SW1 specification contract differs: '+key)
    conditions = spec.get('conditions', [])
    require([x['id'] for x in conditions] == list(CONDITIONS) and
            all(x['rho'] == RHOS[x['id']] and x['group'] == 'retained'
                for x in conditions), 'Condition population changed')
    expected = {'period_two_probability': .5, 'own_weights': [1/3]*3,
                'own_rates': [.875, .1875, .875], 'neighbor_rate': .75,
                'sharing_probability': .5, 'phase_given_period': 'uniform',
                'alternating_parity': 'fair_hidden_opportunity_index_parity',
                'independent_neighbor_can_match_by_chance': True, 'false_positive_records': False}
    require(spec['generator'] == expected, 'Observation generator changed')
    require(spec['disagreement']['bins'] == [0, .01, .05, .25], 'Fixed disagreement bins changed')
    reassignment = spec.get('reassignment', {})
    require(reassignment.get('algorithm') == 'support_preserving_uniform_derangement' and
            reassignment.get('pool') == 'dual_positive_support_within_block_split' and
            reassignment.get('eligibility') == 'raw_and_normalized_float32_both_positive' and
            reassignment.get('anchor_policy') == 'noneligible_and_singleton' and
            reassignment.get('max_attempts') == 10000,
            'Stable support-preserving reassignment contract changed')


def block_number(unit):
    require(isinstance(unit, str) and len(unit) == 8 and unit.startswith('SW05_B') and
            unit[6:].isdigit(), 'Expected SW05_B00 through SW05_B15')
    block = int(unit[6:])
    require(0 <= block < 16 and unit == f'SW05_B{block:02d}', 'Undeclared block')
    return block


def split_fit(fit):
    pieces = fit.split('__')
    require(len(pieces) == 2 and pieces[1] in ARMS, 'Invalid SW1 fit identifier')
    block_number(pieces[0])
    return pieces[0], pieces[1]


def stream_seed(spec, unit, split, component):
    """New named streams; arm absent, training/evaluation/assignment distinct."""
    block = block_number(unit)
    key = [spec['namespace'], int(spec['base_seed']), 'new_SW1', block, split, component]
    return int.from_bytes(hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).digest()[:16], 'big')


def model_seed(spec, unit):
    return stream_seed(spec, unit, 'model', 'paired_initialization') % (2**31-1)


def labels_for_clock(clock_index, horizons):
    clocks, h = np.asarray(clock_index), np.asarray(horizons)
    require(clocks.ndim == 1 and clocks.dtype.kind in 'iu' and np.all((clocks>=0)&(clocks<6)), 'Invalid clock indices')
    require(h.ndim == 1 and h.dtype.kind in 'iu' and np.all((h>=1)&(h<=8)), 'Invalid horizons')
    p, phase = CLOCKS[clocks].T
    return ((95+h[None, :]) % p[:, None] == phase[:, None]).astype(np.float32)


def recording_transform(own_clock, neighbor_clock, law, parity, own_uniform, neighbor_uniform):
    p, phase = CLOCKS[np.asarray(own_clock)].T
    pn, phasen = CLOCKS[np.asarray(neighbor_clock)].T
    own_opportunity = TIMES[None, :] % p[:, None] == phase[:, None]
    neighbor_opportunity = TIMES[None, :] % pn[:, None] == phasen[:, None]
    opportunity_index = np.cumsum(own_opportunity, axis=1)-1
    law, parity = np.asarray(law), np.asarray(parity)
    eligible = own_opportunity & ((law[:, None] != 2) | (opportunity_index % 2 == parity[:, None]))
    rates = np.asarray([.875, .1875, .875])[law][:, None]
    return eligible & (np.asarray(own_uniform)<rates), neighbor_opportunity & (np.asarray(neighbor_uniform)<.75)


def generate(spec, unit, split, count):
    """Scientific cohort generation: only called by approved Colab stages."""
    validate_spec(spec)
    block_number(unit)
    sizes = {'train': 'train_pairs', 'evaluation': 'evaluation_pairs'}
    require(split in sizes and count == spec[sizes[split]], 'Undeclared population size/split')
    def rng(component):
        return np.random.default_rng(stream_seed(spec, unit, split, component))
    prior = np.asarray([.25, .25, .125, .125, .125, .125])
    own = np.searchsorted(prior.cumsum(), rng('candidate_clock').random(count), side='right')
    independent = np.searchsorted(prior.cumsum(), rng('independent_neighbor_clock').random(count), side='right')
    shared = rng('sharing').random(count) < .5
    neighbor = np.where(shared, own, independent)
    law = np.minimum((3*rng('own_recording_law').random(count)).astype(np.int64), 2)
    parity = rng('alternating_parity').integers(0, 2, size=count)
    x, y = recording_transform(own, neighbor, law, parity,
                              rng('own_records').random((count, 16)), rng('neighbor_records').random((count, 16)))
    return dict(x=x, y=y, clock_index=own, neighbor_clock_index=neighbor, shared_indicator=shared,
                realized_law=law, alternating_parity=parity, pair_index=np.arange(count, dtype=np.int64))


def selected_fit(work, unit):
    from clock_inference import validate_theta
    block = block_number(unit)
    work = Path(work)
    record = read_json(work/'source/SELECTED_FITS.json')['blocks'][f'B{block:02d}']
    require(record['block'] == block and record['source_unit'] == f'F05_B{block:02d}',
            'Retained CW1 F05 fit mapping changed')
    require(isinstance(record['fit_sha256'], str) and len(record['fit_sha256']) == 64,
            'Missing authenticated source fit hash')
    theta = np.asarray(record['selected_theta'], dtype=np.float64)
    validate_theta(theta)
    return theta, record, canonical_hash(record)


def fitted_inputs(own, neighbor, theta):
    from clock_inference import predict
    prediction = predict(own, neighbor, theta)
    mass = prediction['context_period_masses']
    require(mass.shape == (len(own), 2) and np.isfinite(mass).all() and
            np.all(mass>=0) and np.allclose(mass.sum(axis=1), 1, rtol=0, atol=1e-10), 'Invalid fitted masses')
    return mass, prediction


def true_theta(unit):
    block_number(unit)
    return np.asarray([.5, 1/3, 1/3, .875, .1875, .875, .75, .5])


def stable_derangement(size, seed, maximum_draws=10000):
    """Uniform permutation conditioned on no fixed points, not cyclic rotation.

    No cohort, labels, weights or strata are consulted. Rejecting full uniform
    permutations until no self donor yields a uniform derangement. A bound
    exhaustion is a technical failure; never select on changed weight values.
    """
    require(type(size) is int and size >= 2 and type(seed) is int and seed >= 0,
            'Derangement needs >=2 rows and a nonnegative integer seed')
    require(type(maximum_draws) is int and maximum_draws >= 1,
            'Invalid derangement rejection bound')
    generator = np.random.default_rng(seed)
    recipients = np.arange(size, dtype=np.int64)
    for draw in range(1, maximum_draws+1):
        donor = generator.permutation(size).astype(np.int64)
        if np.all(donor != recipients):
            return donor, draw
    raise RuntimeError('Derangement rejection bound exhausted; no fallback or weight-selected retry')


def reassignment_arrays(masses, donor):
    """Pure support-preserving mapping contract; anchors keep their own row."""
    masses = period_probabilities(masses)
    donor = np.asarray(donor)
    size = len(masses)
    require(donor.dtype.kind in 'iu' and donor.shape == (size,) and
            np.array_equal(np.sort(donor), np.arange(size)), 'Donors must be a full permutation')
    profile = support_profile(masses)
    eligible = profile['eligible']
    movable = eligible.copy() if int(eligible.sum()) >= 2 else np.zeros(size, dtype=bool)
    recipients = np.arange(size)
    require(np.all(donor[~movable] == recipients[~movable]),
            'Noneligible and singleton recipients must retain self assignment')
    require(np.all(donor[movable] != recipients[movable]) and
            np.all(movable[donor[movable]]),
            'Movable recipients require nonself donors within eligible support pool')
    assigned = masses[donor].copy()
    normalized = masses/np.linalg.norm(masses, axis=1)[:, None]
    assigned_normalized = assigned/np.linalg.norm(assigned, axis=1)[:, None]
    require(np.array_equal(assigned_normalized, normalized[donor]),
            'Normalized donor weights differ from original normalization')
    support_assignment_guard(masses, assigned, movable)
    return {'donor_index': donor.astype(np.int64),
            'original_period_probability': masses.copy(),
            'reassigned_period_probability': assigned,
            'original_normalized_weights': normalized,
            'reassigned_normalized_weights': assigned_normalized,
            'raw_support': profile['raw_support'],
            'effective_support': profile['effective_support'],
            'eligible': eligible, 'movable': movable, 'anchored': ~movable}


def assignment(spec, unit, split, masses):
    """Materialize once per cohort, before any minibatch/horizon expansion."""
    validate_spec(spec)
    require(split in ('train', 'evaluation'), 'No cross-split or calibration donors')
    size = spec['train_pairs' if split == 'train' else 'evaluation_pairs']
    masses = period_probabilities(masses, size)
    seed = stream_seed(spec, unit, split, 'weight_assignment')
    profile = support_profile(masses)
    eligible_indices = np.flatnonzero(profile['eligible'])
    donor, draws = np.arange(size, dtype=np.int64), 0
    if len(eligible_indices) >= 2:
        local_donor, draws = stable_derangement(int(len(eligible_indices)), seed,
                                               spec['reassignment']['max_attempts'])
        donor[eligible_indices] = eligible_indices[local_donor]
    arrays = reassignment_arrays(masses, donor)
    displacement = np.abs(arrays['original_normalized_weights']-
                          arrays['reassigned_normalized_weights']).sum(axis=1)
    changed = np.any(arrays['original_period_probability'] !=
                     arrays['reassigned_period_probability'], axis=1)
    normalized_changed = np.any(arrays['original_normalized_weights'] !=
                                arrays['reassigned_normalized_weights'], axis=1)
    float32_changed = np.any(arrays['original_normalized_weights'].astype(np.float32) !=
                             arrays['reassigned_normalized_weights'].astype(np.float32), axis=1)
    report = {'unit': unit, 'split': split, 'motif_count': size, 'seed': seed,
              'algorithm': 'support_preserving_uniform_derangement', 'rejection_draws': draws,
              'maximum_draws': spec['reassignment']['max_attempts'],
              'pool': 'dual_positive_support_within_block_split',
              'eligible_count': int(arrays['eligible'].sum()),
              'movable_count': int(arrays['movable'].sum()),
              'anchored_count': int(arrays['anchored'].sum()),
              'singleton_eligible_anchored': int(arrays['eligible'].sum()) == 1,
              'self_donor_count': int((donor == np.arange(size)).sum()),
              'zero_self_donors': bool(np.all(donor != np.arange(size))),
              'zero_movable_self_donors': True,
              'mathematical_support_preserved_per_recipient': True,
              'float32_support_preserved_per_recipient': True,
              'anchors_preserved_exactly': True,
              'raw_P2_only_count': int((arrays['raw_support'][:, 0] & ~arrays['raw_support'][:, 1]).sum()),
              'raw_P4_only_count': int((~arrays['raw_support'][:, 0] & arrays['raw_support'][:, 1]).sum()),
              'raw_both_positive_count': int(arrays['raw_support'].all(axis=1).sum()),
              'float32_both_positive_count': int(arrays['effective_support'].all(axis=1).sum()),
              'raw_both_but_float32_single_count': int((arrays['raw_support'].all(axis=1) &
                                                       ~arrays['effective_support'].all(axis=1)).sum()),
              'changed_weight_vectors': int(changed.sum()),
              'unchanged_weight_vectors': int((~changed).sum()),
              'unchanged_movable_weight_vectors_from_ties': int((arrays['movable'] & ~changed).sum()),
              'changed_normalized_float64_vectors': int(normalized_changed.sum()),
              'changed_float32_coefficient_vectors': int(float32_changed.sum()),
              'mean_normalized_weight_L1_displacement': float(displacement.mean()),
              'maximum_normalized_weight_L1_displacement': float(displacement.max()),
              'donor_indices_sha256': hashlib.sha256(donor.astype('<i8').tobytes()).hexdigest(),
              'motif_level_empirical_weight_distribution_exact': True,
              'event_slot_weight_distribution_preserved': False,
              'stable_across_epochs_horizons_batches': True,
              'future_labels_or_truth_used_for_assignment': False,
              'transductive_test_pool_control': split == 'evaluation',
              'selection_on_weight_change_or_effect': False}
    return arrays, report


def verify_assignment(spec, unit, split, masses, arrays, report):
    """Deterministically reproduce only the already declared map and arithmetic."""
    expected, expected_report = assignment(spec, unit, split, masses)
    require(set(arrays) == set(expected), 'Assignment archive field set changed')
    for key in expected:
        require(np.array_equal(arrays[key], expected[key]), 'Assignment array differs: '+key)
    for key, value in expected_report.items():
        require(report.get(key) == value, 'Assignment report differs: '+key)
    return True


def schedule(spec, unit, epoch):
    require(type(epoch) is int and 1 <= epoch <= spec['epochs'], 'Invalid epoch')
    order = np.random.default_rng(stream_seed(spec, unit, 'training_order', str(epoch))).permutation(spec['train_pairs'])
    return [order[i:i+spec['batch_size']] for i in range(0, len(order), spec['batch_size'])]


def schedule_hash(batches):
    h = hashlib.sha256()
    for batch in batches:
        a = np.asarray(batch, dtype='<i8')
        h.update(len(a).to_bytes(8, 'little')); h.update(a.tobytes())
    return h.hexdigest()


def budget(spec):
    require(spec['train_pairs'] % spec['batch_size'] == 0, 'Incomplete minibatch')
    updates = spec['train_pairs']//spec['batch_size']
    return {'updates_per_epoch': updates, 'updates_per_fit': updates*spec['epochs'],
            'motif_presentations_per_fit': spec['train_pairs']*spec['epochs'],
            'label_presentations_per_fit': spec['train_pairs']*spec['epochs']*len(spec['training_horizons'])}


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}
