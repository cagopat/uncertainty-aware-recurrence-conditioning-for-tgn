"""SW1 observed-history packing and three constant-norm clock-weight controls.

No Torch, model construction, generated cohorts, fitting or disk access occurs
on import. Period masses are supplied by the authenticated fixed FC1 fit.
Neither hidden sharing status nor generator truth is an argument here.
"""
from __future__ import annotations

import numpy as np

ARMS = ('soft_fixed', 'equal_fixed', 'support_reassigned_fixed')
TIME_DIM = 16
NEIGHBORS = 32
CUTOFF = 95
TIMES = np.arange(80, 96, dtype=np.int64)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def records(value, label):
    value = np.asarray(value)
    require(value.dtype.kind == 'b' and value.ndim == 2 and value.shape[1] == 16
            and len(value) > 0, f'{label} must be nonempty Boolean Bx16 records')
    return value.copy()


def period_probabilities(value, rows=None):
    value = np.asarray(value, dtype=np.float64)
    require(value.ndim == 2 and value.shape[1] == 2 and
            (rows is None or len(value) == rows), 'Expected Bx2 fitted period masses')
    require(np.isfinite(value).all() and np.all((value >= 0) & (value <= 1)) and
            np.allclose(value.sum(axis=1), 1., atol=1e-12, rtol=0),
            'Period masses must be nonnegative, finite and sum to one')
    return value.copy()


def support_profile(masses):
    """Observed-only mathematical and representable phase-channel support.

    No tolerance, probability floor or hidden clock is used. A mathematically
    positive mass may round to zero after normalization and float32 casting;
    such a row is anchored rather than mixed with dual-channel recipients.
    Integer P2/P4 phases always have one coordinate of magnitude one, so this
    coefficient support is also checked against the actual valid-slot bank.
    """
    masses = period_probabilities(masses)
    normalized = masses / np.linalg.norm(masses, axis=1, keepdims=True)
    coefficients = normalized.astype(np.float32)
    raw = masses > 0
    effective = coefficients > 0
    return {'raw_support': raw, 'effective_support': effective,
            'normalized_weights': normalized,
            'float32_coefficients': coefficients,
            'eligible': raw.all(axis=1) & effective.all(axis=1)}


def support_assignment_guard(original, supplied, movable):
    """Pure batch guard; the full-cohort map is verified separately.

    movable is sliced from the saved full-cohort eligibility decision, never
    recalculated using the batch size. Anchored inputs remain byte-identical.
    """
    original = period_probabilities(original)
    supplied = period_probabilities(supplied, len(original))
    movable = np.asarray(movable)
    require(movable.dtype == bool and movable.shape == (len(original),),
            'Expected saved cohort-level movable mask for support guard')
    before, after = support_profile(original), support_profile(supplied)
    require(np.array_equal(before['raw_support'], after['raw_support']),
            'Support-preserving assignment changed mathematical period support')
    require(np.array_equal(before['effective_support'], after['effective_support']),
            'Support-preserving assignment changed float32 effective period support')
    require(np.all(~movable | before['eligible']), 'An ineligible row was declared movable')
    require(np.array_equal(original[~movable], supplied[~movable]),
            'Anchored recipient weights changed')
    return {'rows': len(original), 'movable_rows': int(movable.sum()),
            'anchored_rows': int((~movable).sum()), 'raw_support_preserved': True,
            'float32_support_preserved': True, 'anchored_weights_unchanged': True,
            'eligibility_uses_truth_or_labels': False}


def hard_period_probabilities(value):
    """Marginal MAP period; ties select period two, not a single MAP clock."""
    value = period_probabilities(value)
    result = np.zeros_like(value)
    result[np.arange(len(value)), np.argmax(value, axis=1)] = 1.
    return result


def bank_weights(masses, arm):
    """Return constant-unit-radius feature weights, not forecast probabilities.

    Equal retains both channels at 1/sqrt(2). Soft normalizes own fitted
    masses, while reassigned uses the same normalization on masses assigned
    once outside this module by a persisted cohort-level donor permutation.
    This function never permutes rows: batch size, order, epoch and horizon
    cannot change a recipient's assigned weights. Original posterior radius
    is saved for audit only, not applied. Four copies give bank norm two.
    """
    require(arm in ARMS, 'Unregistered SW1 arm')
    masses = period_probabilities(masses)
    original_radius = np.sqrt(np.sum(masses*masses, axis=1))
    require(np.all(original_radius >= np.sqrt(.5)-1e-12) and
            np.all(original_radius <= 1.+1e-12), 'Invalid posterior radial factor')
    if arm == 'equal_fixed':
        direction = np.full_like(masses, 1./np.sqrt(2.))
    else:
        direction = masses/original_radius[:, None]
    radius = np.ones(len(masses), dtype=np.float64)
    effective = direction.copy()
    require(np.allclose(np.sum(effective**2, axis=1), 1., atol=1e-12, rtol=0),
            'All three clock-weight controls require unit L2 weights')
    return {'original_period_radial_factor': original_radius,
            'period_direction_weights': direction,
            'applied_radial_factor': radius,
            'effective_period_weights': effective}


def decoder_metadata(metadata):
    """Zero direct horizon and the obsolete shared-status slot in all arms.

    The visible slot is always one. Own count, last-record age and median
    availability remain observed-only metadata. No true relevance flag is used.
    """
    value = np.asarray(metadata)
    require(value.ndim == 3 and value.shape[-1] == 6 and value.dtype == np.float32
            and np.isfinite(value).all(), 'Expected finite BxHx6 float32 metadata')
    result = value.copy()
    result[:, :, 0:2] = 0.
    result[:, :, 2] = 1.
    return result


def bank_coordinates(delta, valid):
    """Canonical modulo then float64 trigonometry, finally float32 channels."""
    delta, valid = np.asarray(delta, dtype=np.float64), np.asarray(valid)
    require(delta.ndim == 2 and valid.shape == delta.shape and valid.dtype.kind == 'b'
            and np.isfinite(delta).all() and np.all(delta[valid] > 0),
            'Invalid lag/validity arrays')
    channels = []
    for period in (2, 4):
        angle = 2. * np.pi * (np.remainder(delta, period) / period)
        channels.extend((np.where(valid, np.cos(angle), 0.),
                         np.where(valid, np.sin(angle), 0.)))
    return np.stack(channels, axis=-1).astype(np.float32)


def phase_bank(channels, masses, arm):
    require(arm in ARMS, 'Unregistered SW1 arm')
    channels = np.asarray(channels)
    require(channels.ndim == 3 and channels.shape[-1] == 4 and
            channels.dtype == np.float32 and np.isfinite(channels).all(),
            'Expected finite Qxslotsx4 float32 channels')
    masses = period_probabilities(masses, len(channels))
    effective = bank_weights(masses, arm)['effective_period_weights']
    weights = np.repeat(effective, 2, axis=1)[:, None, :]
    weighted = (channels.astype(np.float64) * weights).astype(np.float32)
    # Four copies give norm2 for every arm at every valid slot.
    return np.tile(weighted, (1, 1, TIME_DIM // 4))


def encoding_records(own, neighbor, masses, horizons):
    """Exact recent-32 sampler packing without generation or inference.

    Order is motif, horizon, node role. All observed candidate and visible
    neighbor events are retained. The same candidate-conditioned period mass
    is used at every node role/horizon; no per-edge true clock is available.
    """
    own, neighbor = records(own, 'own'), records(neighbor, 'neighbor')
    require(own.shape == neighbor.shape, 'Own/neighbor shapes differ')
    masses = period_probabilities(masses, len(own))
    horizons = np.asarray(horizons)
    require(horizons.ndim == 1 and horizons.dtype.kind in 'iu' and len(horizons)
            and len(np.unique(horizons)) == len(horizons)
            and np.all((horizons >= 1) & (horizons <= 8)),
            'Horizons must be unique integers from one through eight')
    nrows = len(own) * len(horizons) * 3
    a = {
        'motif_index': np.empty(nrows, np.int64),
        'node_role': np.empty(nrows, np.int8),
        'horizon': np.empty(nrows, np.int8),
        'query_node_ids': np.empty(nrows, np.int64),
        'query_times': np.empty(nrows, np.float64),
        'neighbor_node_ids': np.zeros((nrows, NEIGHBORS), np.int64),
        'event_times': np.zeros((nrows, NEIGHBORS), np.int16),
        'event_stream': np.full((nrows, NEIGHBORS), -1, np.int8),
        'valid': np.zeros((nrows, NEIGHBORS), bool),
    }
    row = 0
    for motif in range(len(own)):
        roles = []
        for role in range(3):
            events = []
            if role in (0, 1):
                events.extend((int(t), 0, 1 if role == 0 else 0) for t in TIMES[own[motif]])
            if role in (1, 2):
                events.extend((int(t), 1, 2 if role == 1 else 1) for t in TIMES[neighbor[motif]])
            events.sort(key=lambda x: (x[0], x[1]))
            require(len(events) <= NEIGHBORS, 'Neighbor truncation is outside this design')
            roles.append(events)
        for h in horizons:
            for role, events in enumerate(roles):
                a['motif_index'][row], a['node_role'][row] = motif, role
                a['horizon'][row], a['query_times'][row] = h, CUTOFF + h
                a['query_node_ids'][row] = 3 * motif + role + 1
                for slot, (timestamp, stream, other) in enumerate(events, start=NEIGHBORS-len(events)):
                    a['valid'][row, slot] = True
                    a['event_times'][row, slot] = timestamp
                    a['event_stream'][row, slot] = stream
                    a['neighbor_node_ids'][row, slot] = 3 * motif + other + 1
                row += 1
    a['period_probability'] = masses[a['motif_index']]
    a['map_period'] = np.asarray((2, 4), np.int64)[np.argmax(a['period_probability'], axis=1)]
    a['cutoff_age'] = np.where(a['valid'], CUTOFF-a['event_times'], 0).astype(np.float32)
    a['channels'] = bank_coordinates(a['query_times'][:, None]-a['event_times'], a['valid'])
    return a

