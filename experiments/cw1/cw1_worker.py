"""CW1 explicitly authorized Colab scientific stages.

Importing this file does not generate data, construct a model, fit parameters,
or load a checkpoint. The author must not invoke these stage functions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np

from cw1_io import (canonical_hash, copy_exact, read_json, require, restore_unit,
                    seal_unit, sha, write_json)
from cw1_data import (block_number, budget, fitted_inputs, generate, labels_for_clock,
                      load_npz, model_seed, schedule, schedule_hash, selected_fit,
                      split_fit, true_theta, validate_spec)
from cw1_encoding import ARMS


def checkpoint_identity(identity, fit, epoch):
    return {'study_identity': canonical_hash(identity), 'phase': 'checkpoint', 'unit': fit, 'epoch': epoch}


def assets(work, output, unit, spec):
    block = block_number(unit)
    theta, selected, source_hash = selected_fit(work, unit)
    data = generate(spec, unit, 'train', spec['train_pairs'])
    mass, _ = fitted_inputs(data['x'], data['y'], theta)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'DATA.npz', x=data['x'], y=data['y'],
                        labels=labels_for_clock(data['clock_index'], spec['training_horizons']),
                        fitted_period_probability=mass,
                        pair_index=data['pair_index'])
    write_json(output/'REPORT.json', {'stage': 'assets', 'block': block, 'unit': unit,
               'pairs': spec['train_pairs'], 'label_horizons': spec['training_horizons'],
               'data_sha256': sha(output/'DATA.npz'), 'source_fit_record_sha256': source_hash,
               'source_fit_sha256': selected['fit_sha256'],
               'source_selected_qualified': selected.get('selected_qualified', selected.get('qualified')),
               'namespace': spec['namespace'], 'scientific_generation_authorized': True,
               'latent_variables_saved': False, 'future_test_labels_generated': False,
               'recording_law_parameters_refitted_in_assets': False,
               'parameter_source': 'retained_fit' if unit.startswith('R05_') else 'fresh_unlabeled_calibration',
               'model_input_fields': ['x', 'y', 'fitted_period_probability'],
               'counts': {'own_zero': int((data['x'].sum(axis=1) == 0).sum()),
                          'own_below_three': int((data['x'].sum(axis=1) < 3).sum())}})


def load_training(work, unit, spec):
    root = work/'results/assets'/unit
    report, data = read_json(root/'REPORT.json'), load_npz(root/'DATA.npz')
    theta, _, source_hash = selected_fit(work, unit)
    require(report['data_sha256'] == sha(root/'DATA.npz') and
            report['source_fit_record_sha256'] == source_hash, 'Training data/source identity differs')
    require(set(data) == {'x', 'y', 'labels', 'fitted_period_probability', 'pair_index'},
            'Undeclared training fields, including latent diagnostic fields, are forbidden')
    require(data['x'].dtype == data['y'].dtype == bool and
            data['x'].shape == data['y'].shape == (spec['train_pairs'], 16) and
            data['labels'].shape == (spec['train_pairs'], 4) and
            data['labels'].dtype == np.float32 and np.isin(data['labels'], [0, 1]).all() and
            data['fitted_period_probability'].dtype == np.float64 and
            data['pair_index'].dtype == np.int64 and
            np.array_equal(data['pair_index'], np.arange(spec['train_pairs'], dtype=np.int64)),
            'Training shape/type/labels differ')
    mass, _ = fitted_inputs(data['x'], data['y'], theta)
    require(np.array_equal(mass, data['fitted_period_probability']), 'Saved fitted posterior differs')
    return data, report, source_hash


def cpu_tree(value):
    """Torch is imported only during approved neural execution."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    require(value is None or isinstance(value, (str, int, float, bool)), 'Unsupported checkpoint value')
    return value


def train(work, output, fit, spec, resume_root):
    import torch
    from cw1_model import make_model, learned_weight_fingerprint
    unit, arm = split_fit(fit)
    block = block_number(unit)
    identity = read_json(work/'IDENTITY.json')
    data, assets_report, source_hash = load_training(work, unit, spec)
    seed = model_seed(spec, unit)
    model = make_model(arm, seed, 'cuda')
    require(model.parameter_counts()['trainable'] == spec['expected_trainable_parameters'],
            'Neural parameter capacity differs')
    init_hash = learned_weight_fingerprint(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=spec['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0.)
    inputs = {'assets_data_sha256': assets_report['data_sha256'],
              'assets_report_sha256': sha(work/'results/assets'/unit/'REPORT.json'),
              'source_fit_record_sha256': source_hash, 'init_sha256': init_hash,
              'model_seed': seed, 'spec_sha256': sha(work/'SPEC.json')}
    resumed, history, last_payload = 0, [], None
    restore_root = work/'checkpoint_restores'/fit
    for epoch in range(1, spec['epochs']+1):
        checkpoint_root = resume_root/f'epoch_{epoch:03d}'
        destination = restore_root/f'epoch_{epoch:03d}'
        receipt = restore_unit(checkpoint_root, destination, checkpoint_identity(identity, fit, epoch), inputs)
        if not receipt:
            require(not any((resume_root/f'epoch_{later:03d}'/'COMPLETE.json').exists()
                            for later in range(epoch+1, spec['epochs']+1)), 'Noncontiguous completed epochs')
            break
        record = read_json(destination/'REPORT.json')
        require(record['fit'] == fit and record['epoch'] == epoch and record['inputs'] == inputs,
                'Checkpoint receipt/report differs')
        history = read_json(destination/'TRAINING.json')['epochs']
        require(len(history) == epoch and [h['epoch'] for h in history] == list(range(1, epoch+1)),
                'Checkpoint epoch history is not contiguous')
        resumed, last_payload = epoch, destination
    if resumed:
        saved = torch.load(last_payload/'CHECKPOINT.pt', map_location='cpu', weights_only=True)
        require(saved['identity'] == identity and saved['inputs'] == inputs and saved['fit'] == fit and
                saved['epoch'] == resumed, 'Checkpoint identity differs')
        model.load_state_dict(saved['model'], strict=True)
        checkpoint_report = read_json(last_payload/'REPORT.json')
        require(learned_weight_fingerprint(model) == checkpoint_report['trained_parameter_sha256'],
                'Restored checkpoint learned parameters differ from its saved report')
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print(json.dumps({'fit': fit, 'restored_completed_epoch': resumed}), flush=True)
    counts = budget(spec)
    for epoch in range(resumed+1, spec['epochs']+1):
        started, losses, norms, query_reports = time.monotonic(), [], [], []
        batches = schedule(spec, unit, epoch)
        require(len(batches) == counts['updates_per_epoch'], 'Epoch update count differs')
        model.train()
        for indices in batches:
            optimizer.zero_grad(set_to_none=True)
            logits, gate = model.forward_batch(data['x'][indices], data['y'][indices],
                                               data['fitted_period_probability'][indices],
                                               spec['training_horizons'])
            require(gate is None and tuple(logits.shape) == (spec['batch_size'], 4), 'Unexpected decoder output')
            target = torch.as_tensor(data['labels'][indices], device='cuda', dtype=torch.float32)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            require(bool(torch.isfinite(loss)), 'Nonfinite training loss; no automatic numerical fallback')
            loss.backward()
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                    'Nonfinite training gradient')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip_norm'])
            require(bool(torch.isfinite(norm)), 'Nonfinite gradient norm')
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),
                    'Nonfinite trained parameter; stop without a completed epoch')
            losses.append(float(loss.detach().cpu()))
            norms.append(float(norm.detach().cpu()))
            query_reports.append(model.cw1_last_report)
        record = {'epoch': epoch, 'updates': len(losses),
                  'motif_presentations': spec['train_pairs'],
                  'label_presentations': spec['train_pairs']*4,
                  'mean_BCE': float(np.mean(losses)), 'maximum_preclip_gradient_norm': float(max(norms)),
                  'batch_losses': losses, 'schedule_sha256': schedule_hash(batches),
                  'seconds': time.monotonic()-started}
        history.append(record)
        local = Path(tempfile.mkdtemp(prefix=f'{fit}_epoch{epoch:03d}_', dir=work))
        torch.save({'identity': identity, 'inputs': inputs, 'fit': fit, 'epoch': epoch,
                    'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
                    'torch_rng': torch.get_rng_state().cpu(),
                    'cuda_rng': [state.cpu() for state in torch.cuda.get_rng_state_all()],
                    'randomness': 'Only Torch RNG plus pure per-cohort-unit/epoch named order stream. No stochastic model dropout.'},
                   local/'CHECKPOINT.pt')
        write_json(local/'TRAINING.json', {'epochs': history, 'query_reports_current_epoch': query_reports})
        write_json(local/'REPORT.json', {'fit': fit, 'epoch': epoch, 'inputs': inputs,
                   'trained_parameter_sha256': learned_weight_fingerprint(model),
                   'checkpoint_sha256': sha(local/'CHECKPOINT.pt'),
                   'selection': 'Every epoch archived; final epoch16 exclusively evaluated',
                   'state_contract': 'Functional per-batch memory reconstructed from observed records; pending bank unused. No persistent training memory is required for replay.'})
        seal_unit(local, resume_root/f'epoch_{epoch:03d}', checkpoint_identity(identity, fit, epoch),
                  inputs, uuid.uuid4().hex)
        last_payload = local
        print(json.dumps({'fit': fit, **{k: v for k, v in record.items() if k != 'batch_losses'}}), flush=True)
    require(len(history) == spec['epochs'] and sum(row['updates'] for row in history) == counts['updates_per_fit'],
            'Completed training budget differs')
    output.mkdir(parents=True, exist_ok=True)
    copy_exact(last_payload/'CHECKPOINT.pt', output/'CHECKPOINT.pt')
    write_json(output/'TRAINING.json', {'epochs': history})
    write_json(output/'REPORT.json', {'stage': 'train', 'fit': fit, 'block': block, 'arm': arm,
               'epoch': spec['epochs'], 'model_seed': seed, 'init_sha256': init_hash,
               'source_fit_record_sha256': source_hash, 'asset_data_sha256': assets_report['data_sha256'],
               'checkpoint_sha256': sha(output/'CHECKPOINT.pt'), 'checkpoint_inputs': inputs,
               'parameter_counts': model.parameter_counts(), 'budget': counts,
               'trained_parameter_sha256': learned_weight_fingerprint(model),
               'selection': 'final_epoch_only', 'resumed_completed_epochs': resumed,
               'training_labels': 'Underlying candidate events at h1-h4 only; all positives and absences, no negative sampling',
               'test_labels_used': False})


def pairing_checks(work, unit, spec):
    require(tuple(spec['arms']) == ARMS, 'The declared four fixed-norm arms changed')
    records = {arm: read_json(work/'results/train'/f'{unit}__{arm}'/'REPORT.json') for arm in spec['arms']}
    histories = {arm: read_json(work/'results/train'/f'{unit}__{arm}'/'TRAINING.json')['epochs'] for arm in spec['arms']}
    reference = ARMS[0]
    for key in ('init_sha256', 'model_seed', 'source_fit_record_sha256', 'asset_data_sha256', 'budget', 'parameter_counts'):
        require(all(records[arm][key] == records[reference][key] for arm in ARMS),
                f'Four-arm pairing differs: {key}')
    orders = {arm: [epoch['schedule_sha256'] for epoch in histories[arm]] for arm in spec['arms']}
    require(all(orders[arm] == orders[reference] and len(orders[arm]) == spec['epochs'] for arm in ARMS),
            'Four-arm paired training order differs')
    require(all(records[arm]['arm'] == arm and records[arm]['fit'] == f'{unit}__{arm}'
                for arm in ARMS), 'Saved fit identifiers do not match the four arms')
    return records, {'arms': list(ARMS), 'reference_arm': reference,
                     'same_initialization': True, 'same_order_all_16_epochs': True,
                     'same_source_fit': True, 'same_training_assets': True, 'same_budget': True,
                     'same_parameter_count': True, 'initialization_sha256': records[reference]['init_sha256'],
                     'model_seed': records[reference]['model_seed'], 'epoch_schedule_sha256': orders[reference]}


def evaluation_group_barrier(work, unit, spec):
    """Require all final-epoch reports in this declared group before test data.

    Receipt/hash authentication remains the session's responsibility. Keeping
    this check in the worker prevents a direct invocation from silently
    skipping the retained-64 or fresh-192 training-completion barrier.
    """
    require(tuple(spec['arms']) == ARMS and spec['blocks'] == list(range(16)),
            'Evaluation population differs from the locked four-arm design')
    condition_id = unit.rsplit('_B', 1)[0]
    matches = [row for row in spec['conditions'] if row['id'] == condition_id]
    require(len(matches) == 1, 'Unknown evaluation cohort')
    group = matches[0]['group']
    conditions = [row['id'] for row in spec['conditions'] if row['group'] == group]
    expected = 64 if group == 'retained' else 192
    require(group in ('retained', 'fresh') and len(conditions)*16*len(ARMS) == expected,
            'Group fit population changed')
    for condition in conditions:
        for block in spec['blocks']:
            for arm in ARMS:
                fit = f'{condition}_B{block:02d}__{arm}'
                path = Path(work)/'results/train'/fit/'REPORT.json'
                require(path.is_file(), f'All{expected} {group} fits must be sealed/restored before test generation: {fit}')
                report = read_json(path)
                require(report['fit'] == fit and report['arm'] == arm and
                        report['epoch'] == spec['epochs'] and report['selection'] == 'final_epoch_only',
                        'An evaluation group member is not a completed final-epoch fit')
    return {'group': group, 'completed_fit_reports': expected}


def save_neural_evidence(output, arm, checkpoint_hash, logits, probability,
                         first_arrays, pair_index, horizons, batch_reports, tolerance):
    """Save each arm before checking its symmetry, so a failure is auditable.

    This is a pure storage/arithmetic helper. Failed evidence is not a unit
    completion and never permits analysis to proceed past the canary.
    """
    require(arm in ARMS, 'Undeclared neural evidence arm')
    output = Path(output)
    raw = output/f'NEURAL_{arm}_RAW.npz'
    audit = output/f'INPUT_AUDIT_FIRST_BATCH_{arm}.npz'
    np.savez_compressed(raw, logits=logits, probability=probability,
                        pair_index=pair_index, horizons=np.asarray(horizons))
    np.savez_compressed(audit, **first_arrays)
    error = float(np.max(np.abs(probability[:, :4] - probability[:, 4:])))
    record = {'arm': arm, 'checkpoint_sha256': checkpoint_hash,
              'maximum_four_step_probability_difference': error,
              'absolute_tolerance': tolerance, 'passed': bool(error <= tolerance),
              'raw_predictions_sha256': sha(raw), 'first_batch_audit_sha256': sha(audit),
              'all_batch_reports': batch_reports,
              'scope': 'Incremental evidence only; not a completed scientific unit.'}
    write_json(output/f'NEURAL_{arm}_EVIDENCE.json', record)
    if not record['passed']:
        failure = output/f'DIAGNOSTIC_{arm}_CYCLE_FAILURE.npz'
        np.savez_compressed(failure, logits=logits, probability=probability,
                            pair_index=pair_index, horizons=np.asarray(horizons),
                            absolute_cycle_difference=np.abs(probability[:, :4]-probability[:, 4:]))
        write_json(output/f'DIAGNOSTIC_{arm}_CYCLE_FAILURE.json',
                   {**record, 'diagnostic_sha256': sha(failure),
                    'completion_authorized': False,
                    'policy': 'Stop; preserve the original probabilities and tensors. No tolerance change or automatic retry.'})
    require(record['passed'], 'Imposed four-step probability symmetry canary failed; raw outputs and diagnostic were preserved')
    return error


def evaluate(work, output, unit, spec):
    import torch
    import fc1_core
    from cw1_model import make_model, learned_weight_fingerprint
    # Session authenticates/restores all sealed fits in the requested group.
    # This defense-in-depth check precedes any evaluation-cohort generation.
    evaluation_group_barrier(work, unit, spec)
    block = block_number(unit)
    theta, selected, source_hash = selected_fit(work, unit)
    identity = read_json(work/'IDENTITY.json')
    fit_records, paired = pairing_checks(work, unit, spec)
    data = generate(spec, unit, 'evaluation', spec['evaluation_pairs'])
    data['labels'] = labels_for_clock(data['clock_index'], spec['evaluation_horizons'])
    mass, fitted = fitted_inputs(data['x'], data['y'], theta)
    known = fc1_core.predict(data['x'], data['y'], true_theta(unit))
    predictions = {f'fitted_{name}': fitted[name] for name in
                   ('own_hard', 'own_soft', 'context_hard', 'context_soft')}
    predictions.update(known_law_context_soft=known['context_soft'],
                       fitted_clock_posterior=fitted['posterior_context'],
                       fitted_period_probability=mass,
                       context_entropy=fitted['context_entropy'],
                       context_map_period=fitted['context_map_period'])
    output.mkdir(parents=True, exist_ok=True)
    # Preserve the common evaluated cases and fixed baselines before any
    # neural arm, including when a later device error or canary stops the unit.
    np.savez_compressed(output/'DATA.npz', **data)
    np.savez_compressed(output/'BASELINE_PREDICTIONS.npz', **predictions)
    neural_reports = {}
    for arm in spec['arms']:
        fit = f'{unit}__{arm}'
        checkpoint = work/'results/train'/fit/'CHECKPOINT.pt'
        require(sha(checkpoint) == fit_records[arm]['checkpoint_sha256'], 'Final checkpoint hash differs')
        model = make_model(arm, model_seed(spec, unit), 'cuda')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(saved['identity'] == identity and saved['fit'] == fit and saved['epoch'] == 16 and
                saved['inputs'] == fit_records[arm]['checkpoint_inputs'], 'Evaluated checkpoint identity differs')
        model.load_state_dict(saved['model'], strict=True)
        model.eval()
        before = learned_weight_fingerprint(model)
        require(before == fit_records[arm]['trained_parameter_sha256'], 'Loaded learned weights differ')
        chunks, batch_reports, first_arrays = [], [], None
        with torch.no_grad():
            for start in range(0, spec['evaluation_pairs'], spec['batch_size']):
                stop = start+spec['batch_size']
                logits, gate = model.forward_batch(data['x'][start:stop], data['y'][start:stop],
                                                   mass[start:stop], spec['evaluation_horizons'])
                require(gate is None and tuple(logits.shape) == (spec['batch_size'], 8), 'Evaluation layout differs')
                require(bool(torch.isfinite(logits).all()), 'Nonfinite evaluated logits')
                chunks.append(logits.cpu().numpy())
                batch_reports.append({'start': start, 'stop': stop, 'query': model.cw1_last_report,
                                      'state': model.last_state_report})
                if first_arrays is None:
                    first_arrays = {k: np.asarray(v) for k, v in model.cw1_last_arrays.items()}
                    first_arrays['decoder_metadata'] = np.asarray(model.cw1_last_metadata)
        logits = np.concatenate(chunks)
        # Stable float64 logistic scoring preserves the raw float32 logits separately.
        probability = np.empty(logits.shape, dtype=np.float64)
        positive = logits >= 0
        probability[positive] = 1 / (1 + np.exp(-logits[positive].astype(np.float64)))
        exponential = np.exp(logits[~positive].astype(np.float64))
        probability[~positive] = exponential / (1 + exponential)
        cycle_error = save_neural_evidence(
            output, arm, sha(checkpoint), logits, probability, first_arrays,
            data['pair_index'], spec['evaluation_horizons'], batch_reports,
            spec['cycle_probability_absolute_tolerance'])
        require(learned_weight_fingerprint(model) == before, 'Evaluation changed learned weights')
        predictions[f'neural_{arm}'] = probability
        predictions[f'neural_{arm}_logit'] = logits
        neural_reports[arm] = {'checkpoint_sha256': sha(checkpoint),
                              'learned_parameters_unchanged': True,
                              'maximum_four_step_probability_difference': cycle_error,
                              'all_batch_reports': batch_reports,
                              'tensor_audit_scope': 'Prespecified first64 evaluation pairs only; all batch summaries and all posterior inputs saved.'}
        del model, saved
        torch.cuda.empty_cache()
    for name in ('fitted_own_hard', 'fitted_own_soft', 'fitted_context_hard', 'fitted_context_soft', 'known_law_context_soft'):
        require(np.array_equal(predictions[name][:, :4], predictions[name][:, 4:]), 'Clock baseline cycle identity changed')
    np.savez_compressed(output/'PREDICTIONS.npz', **predictions)
    write_json(output/'REPORT.json', {'stage': 'evaluate', 'block': block, 'unit': unit,
               'pairs': spec['evaluation_pairs'], 'horizons': spec['evaluation_horizons'],
               'data_sha256': sha(output/'DATA.npz'), 'predictions_sha256': sha(output/'PREDICTIONS.npz'),
               'source_fit_sha256': selected['fit_sha256'], 'source_fit_record_sha256': source_hash,
               'pairing': paired, 'neural': neural_reports,
               'evaluated_neural_arms': list(ARMS),
               'neural_forecast_rows': spec['evaluation_pairs']*len(spec['evaluation_horizons'])*len(ARMS),
               'training_during_evaluation': False, 'recording_parameters_refitted': False,
               'hidden_fields_passed_to_neural_models': False,
               'interpretation': 'The four arms compare hard, equal-both, normalized fitted-soft and always-period-four banks, all pre-fusion norm2. Retraining includes parameter coadaptation; contrasts do not isolate an acute attention mechanism. h5-h8 repeats h1-h4 by supplied symmetry, not learned extrapolation.'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['assets', 'train', 'evaluate', 'analyze'])
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--unit', default='all')
    parser.add_argument('--resume-root', type=Path)
    args = parser.parse_args()
    spec = read_json(args.work/'SPEC.json')
    validate_spec(spec)
    if args.phase == 'assets':
        assets(args.work, args.output, args.unit, spec)
    elif args.phase == 'train':
        require(args.resume_root is not None, 'Explicit immutable checkpoint root required')
        train(args.work, args.output, args.unit, spec, args.resume_root)
    elif args.phase == 'evaluate':
        evaluate(args.work, args.output, args.unit, spec)
    else:
        from cw1_analysis import analyze
        require(args.unit == 'all', 'Analysis requires all blocks')
        analyze(args.work, args.output, spec)


if __name__ == '__main__':
    main()

