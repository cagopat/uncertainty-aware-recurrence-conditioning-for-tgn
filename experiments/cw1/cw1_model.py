"""CW1 query-only fitted-period representation, with unchanged functional TGN.

Import is non-neural. make_model and preflight_fixture are reserved for explicit
user-authorized Colab execution. No source FC1 fit is optimized by this module.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import MethodType

import numpy as np

from cw1_encoding import (ARMS, TIME_DIM, bank_weights, decoder_metadata, encoding_records,
                          phase_bank, records, require)

EXPECTED_TRAINABLE_PARAMETERS = 50434
FUSION_TRAINABLE_PARAMETERS = 528
ATOL = 2e-5
RTOL = 2e-5


def inherited_module():
    root = Path(__file__).resolve().parent / 'inherited'
    require((root / 'gj1_model.py').is_file(), 'Authenticated inherited model is missing')
    if str(root) not in sys.path:
        sys.path.append(str(root))
    module = importlib.import_module('gj1_model')
    require(Path(module.__file__).resolve() == root / 'gj1_model.py',
            'A different gj1_model was imported')
    return module


def learned_weight_fingerprint(model):
    return inherited_module().learned_weight_fingerprint(model)


class EncodingContext:
    def __init__(self, model, own, neighbor, masses, horizons):
        self.model = model
        self.arrays = encoding_records(own, neighbor, masses, horizons)
        self.calls = 0
        self.report = None

    def apply(self, neighbor_delta_times, neighbor_time_features, neighbor_edge_features,
              neighbor_node_ids, query_node_ids=None, query_times=None,
              neighbor_edge_ids=None, layer=0):
        import torch
        self.calls += 1
        require(self.calls == 1 and layer == 1, 'Exactly one layer-one query hook is required')
        a = self.arrays
        require(np.array_equal(query_node_ids, a['query_node_ids']), 'Query row identity changed')
        require(np.array_equal(query_times, a['query_times']), 'Query dates changed')
        require(np.array_equal(neighbor_node_ids, a['neighbor_node_ids']), 'Sampler members changed')
        event_times = np.asarray(query_times)[:, None]-np.asarray(neighbor_delta_times)
        require(np.array_equal(event_times, a['event_times']), 'Sampled occurrence/padding times changed')
        require(neighbor_edge_ids is not None and
                np.array_equal(np.asarray(neighbor_edge_ids) > 0, a['valid']),
                'Sampled event validity changed')
        t, e = neighbor_time_features, neighbor_edge_features
        require(t.ndim == e.ndim == 3 and t.shape[-1] == TIME_DIM and
                t.shape[:2] == e.shape[:2] == a['valid'].shape and
                t.dtype == e.dtype == torch.float32 and t.device == e.device,
                'Unexpected query feature layout/dtype/device')
        valid = torch.as_tensor(a['valid'], device=t.device)
        query_np = phase_bank(a['channels'], a['period_probability'], self.model.cw1_arm)
        query = torch.as_tensor(query_np, device=t.device)
        cutoff = self.model.host.time_encoder(torch.as_tensor(a['cutoff_age'], device=t.device))
        require(cutoff.shape == t.shape, 'Cutoff-age feature width changed')
        mixed = self.model.cw1_query_fusion(torch.cat((query, cutoff), dim=-1))
        out_t = torch.where(valid[..., None], mixed, t)
        require(torch.equal(out_t[~valid], t[~valid]), 'Padding features changed')
        require(bool(torch.isfinite(out_t).all()), 'Nonfinite query features')
        a['query_input_norm'] = query.detach().norm(dim=-1).cpu().numpy()
        a.update(bank_weights(a['period_probability'], self.model.cw1_arm))
        a['expected_query_input_norm'] = np.where(
            a['valid'], 2.*a['applied_radial_factor'][:, None], 0.)
        require(np.array_equal(query_np[~a['valid']], np.zeros_like(query_np[~a['valid']])),
                'Invalid neighbor bank slots must remain zero')
        norm_difference = np.abs(a['query_input_norm'][a['valid']] -
                                 a['expected_query_input_norm'][a['valid']])
        require(np.allclose(a['query_input_norm'][a['valid']], a['expected_query_input_norm'][a['valid']],
                            atol=1e-6, rtol=1e-6), 'Constant-norm clock-weight bank contract failed')
        a['cutoff_feature_norm'] = cutoff.detach().norm(dim=-1).cpu().numpy()
        a['output_time_norm'] = out_t.detach().norm(dim=-1).cpu().numpy()
        a['output_edge_norm'] = e.detach().norm(dim=-1).cpu().numpy()
        self.report = {
            'arm': self.model.cw1_arm, 'hook_calls': self.calls,
            'query_rows': len(valid), 'valid_slots': int(a['valid'].sum()),
            'empty_query_rows': int(np.sum(~a['valid'].any(axis=1))),
            'padding_unchanged': True, 'edge_features_unscaled': True,
            'cutoff': 95, 'period_inference': 'fixed_authenticated_retained_or_fresh_fitted_context_period_marginals',
            'true_sharing_indicator_exposed': False,
            'decoder_raw_horizon_exposed': False,
            'historical_message_transform_changed': False,
            'shared_parameters_may_coadapt': True,
            'clock_weight_rule': self.model.cw1_arm,
            'bank_squared_norm': '4',
            'norm_matched_across_all_four_arms': True,
            'norm_canary_absolute_tolerance': 1e-6,
            'norm_canary_relative_tolerance': 1e-6,
            'norm_canary_maximum_difference': float(np.max(norm_difference)) if norm_difference.size else 0.,
            'norm_canary_scope': 'Valid query-bank slots before learned cutoff-age fusion; not the fused/output feature norms',
            'adaptive_magnitude_applied': False,
            'bank_weights_are_probability_forecasts': False,
            'six_clock_phase_posterior_consumed_directly': False,
            'direct_fitted_forecast_consumed': False,
        }
        return out_t, e


def make_model(arm, seed, device):
    """Construct only during approved Colab preflight/training/evaluation."""
    require(arm in ARMS, 'Unregistered CW1 neural arm')
    source = inherited_module()
    model = source.make_model('native', seed, device)
    import torch
    model.cw1_query_fusion = torch.nn.Linear(2*TIME_DIM, TIME_DIM, bias=True).to(device)
    with torch.no_grad():
        model.cw1_query_fusion.weight.copy_(torch.cat((.5*torch.eye(TIME_DIM),
                                                     .5*torch.eye(TIME_DIM)), dim=1).to(device))
        model.cw1_query_fusion.bias.zero_()
    require(model.parameter_counts()['trainable'] == EXPECTED_TRAINABLE_PARAMETERS,
            'CW1 trainable parameter count changed')
    require(sum(p.numel() for p in model.cw1_query_fusion.parameters()) == FUSION_TRAINABLE_PARAMETERS,
            'CW1 fusion capacity changed')
    host_module = importlib.import_module('train.models.MemoryModel')
    expected = Path(source.__file__).resolve().parent/'runtime/train/models/MemoryModel.py'
    require(Path(host_module.__file__).resolve() == expected, 'Another temporal host was imported')
    original_hook = host_module._keat_apply
    original_metadata = source.recurrence_inputs
    original_forward = model.forward_batch
    model.cw1_arm = arm
    model.cw1_last_report = None
    model.cw1_last_arrays = None
    model.cw1_last_metadata = None

    def forward_batch(self, own, neighbor, period_probability, horizons):
        require(host_module._keat_apply is original_hook and source.recurrence_inputs is original_metadata,
                'Another query hook is active; isolated sequential workers are required')
        own, neighbor = records(own, 'own'), records(neighbor, 'neighbor')
        context = EncodingContext(self, own, neighbor, period_probability, horizons)
        self.cw1_last_report = self.cw1_last_arrays = self.cw1_last_metadata = None

        def scoped_metadata(*args, **kwargs):
            metadata, phi = original_metadata(*args, **kwargs)
            metadata = decoder_metadata(metadata)
            self.cw1_last_metadata = metadata.copy()
            return metadata, np.zeros_like(phi)

        host_module._keat_apply, source.recurrence_inputs = context.apply, scoped_metadata
        try:
            # False is a constant unused legacy metadata placeholder, NOT the
            # candidate's hidden sharing state. Neighbor records always visible.
            result = original_forward(own, neighbor, False, True, horizons, 'none')
            require(context.calls == 1 and context.report is not None, 'Query transform was not exercised')
            require(self.arm == 'native', 'Unexpected inherited decoder arm')
            self.cw1_last_report, self.cw1_last_arrays = context.report, context.arrays
            return result
        finally:
            host_module._keat_apply, source.recurrence_inputs = original_hook, original_metadata

    def forward(self, own, neighbor, period_probability, horizons):
        return self.forward_batch(own, neighbor, period_probability, horizons)

    model.forward_batch = MethodType(forward_batch, model)
    model.forward = MethodType(forward, model)
    return model


def preflight_fixture(device):
    """User-run tiny handwritten fixtures; zero optimizer steps or study cases.

    Tests neural routing, matched initialization, phase identity, padding,
    absent own observations, batch tolerance and full-history gradient flow.
    Never called by author-side tests.
    """
    import torch
    own = np.zeros((3, 16), bool)
    neighbor = np.zeros_like(own)
    own[0, [0, 4, 8, 12]] = True
    neighbor[0, [2, 6, 10, 14]] = True
    neighbor[1, [1, 5, 9, 13]] = True
    # Row one has no own records, row two has neither stream recorded.
    mass = np.asarray([[.5, .5], [.25, .75], [.4, .6]], dtype=np.float64)
    models = {arm: make_model(arm, 43119, device) for arm in ARMS}
    inherited = inherited_module()
    host_module = importlib.import_module('train.models.MemoryModel')
    untouched_hook, untouched_metadata = host_module._keat_apply, inherited.recurrence_inputs
    initial = {arm: learned_weight_fingerprint(model) for arm, model in models.items()}
    require(len(set(initial.values())) == 1, 'Four-arm matched initialization differs')
    result = {'optimizer_steps': 0, 'generated_study_cases': 0,
              'arms': list(ARMS), 'matched_initial_parameters': True,
              'initialization_fingerprints': initial,
              'trainable_parameters': EXPECTED_TRAINABLE_PARAMETERS,
              'batch_logit_absolute_tolerance': ATOL,
              'batch_logit_relative_tolerance': RTOL,
              'four_step_probability_absolute_tolerance': 1e-5,
              'gradient_checks': {}, 'forward_checks': {}}
    one_hot_logits, restored_checks, history_states = {}, {}, {}
    one_hot_mass = np.asarray([[1., 0.], [0., 1.], [1., 0.]])
    for model in models.values():
        model.eval()
        before = learned_weight_fingerprint(model)
        with torch.no_grad():
            logits, _ = model.forward_batch(own, neighbor, mass, list(range(1, 9)))
            values = logits.detach().cpu().numpy()
            require(model.last_state_report['functional_state'] and
                    model.last_state_report['query_read_only'] and
                    model.last_state_report['pending_messages'] == 0,
                    'Functional-memory or pending-message contract changed')
            state_hash = model.last_state_report['state_sha256']
            history_states[model.cw1_arm] = state_hash
            require(np.allclose(values[:, :4], values[:, 4:], atol=ATOL, rtol=RTOL),
                    'Four-step input symmetry failed to preserve logits')
            probability = torch.sigmoid(logits.double()).detach().cpu().numpy()
            probability_cycle_error = float(np.max(np.abs(probability[:, :4]-probability[:, 4:])))
            require(probability_cycle_error <= 1e-5,
                    'Four-step probability symmetry exceeds the scientific 1e-5 canary')
            require(np.all(model.cw1_last_metadata[:, :, :2] == 0) and
                    np.all(model.cw1_last_metadata[:, :, 2] == 1),
                    'Raw horizon or hidden relevance reached decoder')
            require(model.cw1_last_report['padding_unchanged'], 'Padding check absent')
            per_row = []
            for index in range(len(own)):
                one, _ = model.forward_batch(own[index:index+1], neighbor[index:index+1],
                                             mass[index:index+1], list(range(1, 9)))
                per_row.append(one.detach().cpu().numpy()[0])
            independent = np.asarray(per_row)
            require(np.allclose(values, independent, atol=ATOL, rtol=RTOL),
                    'Canonical versus singleton batching exceeded fixed tolerance')
            one_hot, _ = model.forward_batch(own, neighbor, one_hot_mass, list(range(1, 9)))
            one_hot_logits[model.cw1_arm] = one_hot.detach().cpu().numpy()
            # An in-memory, zero-update checkpoint roundtrip checks restoration
            # without a cohort, optimizer, training update or disk checkpoint.
            import io
            buffer = io.BytesIO()
            torch.save({'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        'rng': torch.get_rng_state().cpu(),
                        'cuda_rng': [s.cpu() for s in torch.cuda.get_rng_state_all()]}, buffer)
            buffer.seek(0)
            restored_state = torch.load(buffer, map_location='cpu', weights_only=True)
            restored = make_model(model.cw1_arm, 43119, device)
            restored.load_state_dict(restored_state['model'], strict=True)
            torch.set_rng_state(restored_state['rng'])
            torch.cuda.set_rng_state_all(restored_state['cuda_rng'])
            restored.eval()
            resumed, _ = restored.forward_batch(own, neighbor, mass, list(range(1, 9)))
            resumed_values = resumed.detach().cpu().numpy()
            require(np.allclose(resumed_values, values, atol=ATOL, rtol=RTOL),
                    'Zero-update checkpoint restoration changed fixture logits')
            require(restored.last_state_report['state_sha256'] == state_hash and
                    learned_weight_fingerprint(restored) == before,
                    'Zero-update restoration changed memory or learned weights')
            restored_checks[model.cw1_arm] = {
                'maximum_logit_difference': float(np.max(np.abs(resumed_values-values))),
                'functional_history_state_identical': True, 'optimizer_steps': 0,
                'scope': 'Handwritten fixture state_dict and RNG roundtrip, not trained checkpoint replay'}
            del restored, restored_state
        require(learned_weight_fingerprint(model) == before, 'Read-only fixture changed weights')
        result['forward_checks'][model.cw1_arm] = {
            'four_step_max_abs_logit_difference': float(np.max(np.abs(values[:, :4]-values[:, 4:]))),
            'four_step_max_abs_probability_difference': probability_cycle_error,
            'batch_max_abs_logit_difference': float(np.max(np.abs(values-independent))),
            'no_own_and_no_any_record_rows_finite': True,
            'padding_and_metadata_checked': True,
        }
        model.train()
        model.capture_gradient = True
        model.zero_grad(set_to_none=True)
        try:
            logits, _ = model.forward_batch(own, neighbor, mass, [1, 2, 3, 4])
            (logits.square().mean()+logits.mean()).backward()
            early = model.earliest_memory
            require(early is not None and early.grad is not None, 'No earliest-memory gradient')
            groups = {'earliest_memory': float(early.grad.detach().abs().sum().cpu())}
            for label, module in (('time_encoder', model.host.time_encoder),
                                  ('memory_gru', model.host.memory_updater.memory_updater),
                                  ('attention', model.host.embedding_module.temporal_conv_layers),
                                  ('query_fusion', model.cw1_query_fusion),
                                  ('decoder', model.decoder)):
                gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
                require(gradients and all(bool(torch.isfinite(g).all()) for g in gradients),
                        f'Missing/nonfinite {label} gradient')
                groups[label] = sum(float(g.detach().abs().sum().cpu()) for g in gradients)
            require(all(v > 0 for v in groups.values()), 'A required historical/query gradient is zero')
            require(learned_weight_fingerprint(model) == before, 'Gradient fixture changed weights')
            result['gradient_checks'][model.cw1_arm] = groups
        finally:
            model.zero_grad(set_to_none=True)
            model.capture_gradient = False
            model.earliest_memory = None
            model.eval()
    require(host_module._keat_apply is untouched_hook and
            inherited.recurrence_inputs is untouched_metadata,
            'A temporary query hook leaked outside its forward call')
    result['global_query_hooks_restored'] = True
    require(len(set(history_states.values())) == 1,
            'Matched initialization must reconstruct identical historical memory across arms')
    result['historical_memory_identical_at_matched_initialization'] = True
    require(np.array_equal(one_hot_logits['hard_fixed'], one_hot_logits['soft_fixed']),
            'One-hot hard and normalized-soft inputs must give identical logits')
    require(np.array_equal(one_hot_logits['period4_fixed'][1], one_hot_logits['hard_fixed'][1]),
            'Always-P4 must agree with hard on the one-hot-P4 handwritten row')
    weights = {arm: bank_weights(mass, arm)['effective_period_weights'] for arm in ARMS}
    require(np.array_equal(weights['period4_fixed'], np.tile([[0., 1.]], (3, 1))),
            'The period-four control must ignore candidate posterior weights')
    require(np.array_equal(weights['equal_fixed'], np.full((3, 2), 1./np.sqrt(2.))),
            'The equal control must use both clock banks at 1/sqrt(2)')
    require(np.array_equal(weights['equal_fixed'][0], weights['soft_fixed'][0]),
            'Equal and normalized-soft weights must coincide at posterior tie')
    result['one_hot_hard_soft_bitwise_equal'] = True
    result['one_hot_period4_row_bitwise_equal'] = True
    result['equal_and_period4_weights_verified'] = True
    result['all_arms_valid_pre_fusion_bank_norm_two'] = True
    result['checkpoint_roundtrip'] = restored_checks
    result['cross_arm_scope'] = 'Matched initialization; equal-both and always-P4 need not match hard/soft away from their exact defining coincidences'
    return result

