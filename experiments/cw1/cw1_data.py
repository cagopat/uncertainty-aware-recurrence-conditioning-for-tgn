"""CW1 data and fitted-clock contracts. No work executes on import.

The generation and calibration entry points are scientific work, reserved for
explicitly authorized Colab stages. Authoring tests use handwritten arrays.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from cw1_io import canonical_hash, read_json, require, sha, write_json

ARMS = ('hard_fixed', 'equal_fixed', 'soft_fixed', 'period4_fixed')
CONDITIONS = ('R05', 'F00', 'F05', 'F10')
RHOS = dict(R05=.5, F00=0., F05=.5, F10=1.)
CLOCKS = np.asarray([(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)], dtype=np.int64)
TIMES = np.arange(80, 96, dtype=np.int64)
FC1_OPTIMIZER_SHA256 = '8974f5f9c9c647ab4a7d397ace28738986e5ac7e9debc42643610a7ba237939f'


def validate_spec(spec):
    fixed = {'study': 'CW1', 'version': '1.0.0', 'namespace': 'CW1/1.0.0',
             'blocks': list(range(16)), 'arms': list(ARMS), 'base_seed': 2026111201,
             'bootstrap_seed': 2026111202, 'bootstrap_replicates': 10000,
             'train_pairs': 2048, 'evaluation_pairs': 4096, 'calibration_pairs': 7680,
             'epochs': 16, 'batch_size': 64, 'training_horizons': [1, 2, 3, 4],
             'evaluation_horizons': list(range(1, 9)), 'primary_horizons': [5, 6, 7, 8],
             'cutoff': 95, 'monitored_times': list(range(80, 96)),
             'learning_rate': .001, 'optimizer': 'Adam', 'optimizer_betas': [.9, .999],
             'optimizer_epsilon': 1e-8, 'weight_decay': 0, 'gradient_clip_norm': 1,
             'dropout': 0, 'expected_trainable_parameters': 50434,
             'updates_per_epoch': 32, 'total_updates_per_fit': 512,
             'total_target_presentations_per_fit': 131072, 'expected_fit_count': 256,
             'expected_cohort_count': 64, 'expected_calibration_fit_count': 48,
             'checkpoint_selection': 'final_epoch_16_only', 'scientific_pass_fail_rule': None,
             'primary_contrast': 'soft_fixed_minus_equal_fixed',
             'memory_message_transform_changed': False, 'decoder_raw_horizon': False,
             'decoder_hidden_shared_indicator': False, 'edge_features_scaled': False,
             'new_population_parameter_fitting': 'fresh_group_only',
             'replay_old_neural_checkpoints': False}
    for key, value in fixed.items():
        require(spec.get(key) == value, 'CW1 specification contract differs: '+key)
    conditions = spec.get('conditions', [])
    require([x['id'] for x in conditions] == list(CONDITIONS) and
            all(x['rho'] == RHOS[x['id']] and x['group'] == ('retained' if x['id']=='R05' else 'fresh')
                for x in conditions), 'Condition population changed')
    expected = {'period_two_probability': .5, 'own_weights': [1/3]*3,
                'own_rates': [.875, .1875, .875], 'neighbor_rate': .75,
                'sharing_probability': .5, 'phase_given_period': 'uniform',
                'alternating_parity': 'fair_hidden_opportunity_index_parity',
                'independent_neighbor_can_match_by_chance': True, 'false_positive_records': False}
    require(spec['generator'] == expected, 'Observation generator changed')
    require(spec['disagreement']['bins'] == [0, .01, .05, .25], 'Fixed disagreement bins changed')
    # Verify against the authenticated inherited specification, never executed.
    original = read_json(Path(__file__).with_name('FC1_SOURCE_PINS.json'))
    require(isinstance(original, dict), 'Missing original FC1 source pins')
    optimizer = spec['calibration_fit']
    require(canonical_hash(optimizer) == FC1_OPTIMIZER_SHA256, 'Exact FC1 optimizer/start configuration changed')
    require(len(optimizer['starts']) == 6 and optimizer['em_maxiter'] == 300 and
            optimizer['slsqp_maxiter'] == 500 and optimizer['continuation_maxiter'] == 1000 and
            optimizer['stationarity_tolerance'] == 1e-6 and optimizer['ftol'] == 1e-12,
            'FC1 fixed fitting procedure changed')


def block_number(unit):
    require(isinstance(unit, str) and len(unit) == 7 and unit[:3] in CONDITIONS and
            unit[3:5] == '_B' and unit[5:].isdigit(), 'Expected R05/F00/F05/F10_B00 through B15')
    block = int(unit[5:])
    require(0 <= block < 16 and unit == f'{unit[:3]}_B{block:02d}', 'Undeclared block')
    return block


def split_fit(fit):
    pieces = fit.split('__')
    require(len(pieces) == 2 and pieces[1] in ARMS, 'Invalid CW1 fit identifier')
    block_number(pieces[0])
    return pieces[0], pieces[1]


def stream_seed(spec, unit, split, component):
    """Fresh rho cohorts share all draws; only the sharing threshold differs.

    Arm is absent. R05 is an independent stream, not a fresh-fit comparator
    on identical histories. Calibration/train/evaluation streams are distinct.
    """
    block = block_number(unit)
    group = 'retained' if unit[:3] == 'R05' else 'fresh_CRN'
    key = [spec['namespace'], int(spec['base_seed']), group, block, split, component]
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
    sizes = {'calibration': 'calibration_pairs', 'train': 'train_pairs', 'evaluation': 'evaluation_pairs'}
    require(split in sizes and count == spec[sizes[split]], 'Undeclared population size/split')
    require(split != 'calibration' or unit.startswith('F'), 'Retained calibration must not be regenerated')
    def rng(component):
        return np.random.default_rng(stream_seed(spec, unit, split, component))
    prior = np.asarray([.25, .25, .125, .125, .125, .125])
    own = np.searchsorted(prior.cumsum(), rng('candidate_clock').random(count), side='right')
    independent = np.searchsorted(prior.cumsum(), rng('independent_neighbor_clock').random(count), side='right')
    shared = rng('sharing').random(count) < RHOS[unit[:3]]
    neighbor = np.where(shared, own, independent)
    law = np.minimum((3*rng('own_recording_law').random(count)).astype(np.int64), 2)
    parity = rng('alternating_parity').integers(0, 2, size=count)
    x, y = recording_transform(own, neighbor, law, parity,
                              rng('own_records').random((count, 16)), rng('neighbor_records').random((count, 16)))
    return dict(x=x, y=y, clock_index=own, neighbor_clock_index=neighbor, shared_indicator=shared,
                realized_law=law, alternating_parity=parity, pair_index=np.arange(count, dtype=np.int64))


def selected_fit(work, unit):
    from fc1_core import validate_theta
    block = block_number(unit)
    work = Path(work)
    if unit.startswith('R05'):
        record = read_json(work/'source/SELECTED_FITS.json')['blocks'][f'B{block:02d}']
        require(record['block'] == block and record['source_unit'] == f'U{3*block+1:03d}', 'Retained fit mapping changed')
    else:
        record = read_json(work/'results/calibrate'/unit/'FIT.json')
        require(record['unit'] == unit and record['calibration_size'] == 7680, 'Fresh fit mapping changed')
        endpoints = record['endpoints']
        eligible = [i for i,e in enumerate(endpoints) if e['feasible'] and e['mean_nll'] is not None and np.isfinite(e['mean_nll'])]
        chosen = min(eligible, key=lambda i: (endpoints[i]['mean_nll'], i))
        require(chosen == record['selected_endpoint'] and record['selected_theta'] == endpoints[chosen]['theta'],
                'Selected fit is not the lowest finite feasible endpoint')
        record = dict(record, fit_sha256=sha(work/'results/calibrate'/unit/'FIT.json'))
    theta = np.asarray(record['selected_theta'], dtype=np.float64)
    validate_theta(theta)
    return theta, record, canonical_hash(record)


def fitted_inputs(own, neighbor, theta):
    from fc1_core import predict
    prediction = predict(own, neighbor, theta)
    mass = prediction['context_period_masses']
    require(mass.shape == (len(own), 2) and np.isfinite(mass).all() and
            np.all(mass>=0) and np.allclose(mass.sum(axis=1), 1, rtol=0, atol=1e-10), 'Invalid fitted masses')
    return mass, prediction


def true_theta(unit):
    block_number(unit)
    return np.asarray([.5, 1/3, 1/3, .875, .1875, .875, .75, RHOS[unit[:3]]])


def calibration(work, output, unit, spec):
    """Scientific fit entry point. Only the x/y histories enter fit_counts."""
    from fc1_core import fit_counts
    validate_spec(spec)
    require(unit.startswith('F'), 'Retained parameters may not be refitted')
    # Original fitting settings are independently bound to authenticated source.
    require(spec['calibration_fit'] == read_json(Path(work)/'source/FC1_SPEC.json')['optimizer'],
            'Fresh calibration settings differ from inherited FC1')
    data = generate(spec, unit, 'calibration', spec['calibration_pairs'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'CALIBRATION.npz', x=data['x'], y=data['y'], pair_index=data['pair_index'])
    fitted = fit_counts(data['x'], data['y'], spec['calibration_fit'])
    fitted.update(unit=unit, block=block_number(unit), calibration_size=spec['calibration_pairs'],
                  calibration_sha256=sha(output/'CALIBRATION.npz'))
    write_json(output/'FIT.json', fitted)
    write_json(output/'REPORT.json', {'stage': 'calibrate', 'unit': unit,
               'fit_sha256': sha(output/'FIT.json'), 'calibration_sha256': sha(output/'CALIBRATION.npz'),
               'selected_qualified': fitted['selected_qualified'], 'selected_residual': fitted['selected_residual'],
               'selected_endpoint': fitted['selected_endpoint'], 'unqualified_dropped': False,
               'fitting_inputs': ['candidate recorded history', 'neighbor recorded history'],
               'hidden_clocks_laws_sharing_or_future_labels_supplied': False,
               'optimizer_contract_sha256': canonical_hash(spec['calibration_fit'])})


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
