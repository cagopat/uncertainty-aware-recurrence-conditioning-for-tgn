"""SW1 saved-prediction arithmetic; no scientific work on import."""
from pathlib import Path
import platform
import numpy as np
from sw1_io import read_json, require, sha, write_json
from sw1_data import CLOCKS, TIMES, load_npz, ARMS, CONDITIONS, block_number, support_profile
LOGLOSS_CLIP=1e-7
PRIMARY="soft_fixed_minus_support_reassigned_fixed"
METHODS=tuple("neural_"+a for a in ARMS)+(
    "fitted_own_hard","fitted_own_soft","fitted_context_hard",
    "fitted_context_soft","known_law_context_soft")
SUPPORT_GROUPS = ('support_eligible', 'support_movable', 'support_anchored',
                  'support_eligible_singleton', 'support_P2_only',
                  'support_P4_only', 'support_float32_underflow_anchor')

def auc(labels, scores):
    labels, scores = np.asarray(labels).reshape(-1), np.asarray(scores).reshape(-1)
    positive = labels == 1
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if not npos or not nneg:
        return None
    order = np.argsort(scores, kind='stable')
    ranked = scores[order]
    starts = np.r_[0, np.flatnonzero(ranked[1:] != ranked[:-1]) + 1]
    stops = np.r_[starts[1:], len(ranked)]
    ranks = np.empty(len(ranked), float)
    for start, stop in zip(starts, stops):
        ranks[order[start:stop]] = (start + 1 + stop) / 2
    return float((ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(label, prediction):
    label, prediction = np.asarray(label), np.asarray(prediction, dtype=np.float64)
    require(label.shape == prediction.shape and np.isin(label, [0, 1]).all(), 'Metric alignment/labels differ')
    require(np.isfinite(prediction).all() and np.all((prediction >= 0) & (prediction <= 1)),
            'Invalid forecast probabilities')
    if not label.size:
        return {'rows': 0, 'brier': None, 'logloss': None, 'auc': None,
                'brier_sum': 0., 'logloss_sum': 0.}
    clipped = np.clip(prediction, LOGLOSS_CLIP, 1-LOGLOSS_CLIP)
    return {'rows': int(label.size), 'brier': float(np.mean((prediction-label)**2)),
            'logloss': float(-np.mean(label*np.log(clipped)+(1-label)*np.log1p(-clipped))),
            'auc': auc(label, prediction), 'brier_sum': float(np.sum((prediction-label)**2)),
            'logloss_sum': float(-np.sum(label*np.log(clipped)+(1-label)*np.log1p(-clipped)))}


def grouped_metrics(labels, probabilities):
    """Equal-horizon summaries, never cross-time pooled AUC."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    require(labels.ndim == 2 and labels.shape == probabilities.shape, 'Expected pair-by-horizon metrics')
    answer = metrics(labels, probabilities)
    values = [auc(labels[:, h], probabilities[:, h]) if len(labels) else None for h in range(labels.shape[1])]
    defined = [value for value in values if value is not None]
    answer['auc'] = float(np.mean(defined)) if defined else None
    answer['auc_defined_horizons'] = len(defined)
    answer['auc_horizon_count'] = labels.shape[1]
    answer['auc_convention'] = 'Equal mean of defined within-horizon AUCs; no pooled AUC'
    return answer


def strata(data, predictions):
    count = data['x'].sum(axis=1)
    median = np.full(len(count), np.nan)
    available = count >= 3
    for index in np.flatnonzero(available):
        median[index] = np.median(np.diff(TIMES[data['x'][index]]))
    true_period = CLOCKS[data['clock_index'], 0]
    masks = {'all': np.ones(len(count), bool), 'own_zero': count == 0,
             'both_histories_empty': (count == 0) & (data['y'].sum(axis=1) == 0),
             'own_one_or_two': (count > 0) & (count < 3), 'own_below_three': count < 3,
             'own_at_least_three': count >= 3,
             'period_entropy_gt_half_bit': predictions['context_entropy'] > .5,
             'period_entropy_le_half_bit': predictions['context_entropy'] <= .5,
             'median_unavailable': ~available, 'median_correct': available & (median == true_period),
             'median_wrong': available & (median != true_period),
             'median_true2_est4': available & (true_period == 2) & (median == 4),
             'median_true4_est2': available & (true_period == 4) & (median == 2),
             'fitted_MAP_correct': predictions['context_map_period'] == true_period,
             'fitted_MAP_wrong': predictions['context_map_period'] != true_period,
             'fitted_MAP_true2_selected4': (true_period == 2) & (predictions['context_map_period'] == 4),
             'fitted_MAP_true4_selected2': (true_period == 4) & (predictions['context_map_period'] == 2),
             'true_period2': true_period == 2, 'true_period4': true_period == 4}
    original = predictions['fitted_period_probability']
    reassigned = predictions['reassigned_period_probability']
    changed = np.any(original != reassigned, axis=1)
    map_changed = (original[:, 0] >= original[:, 1]) != (reassigned[:, 0] >= reassigned[:, 1])
    masks.update(reassigned_weights_changed=changed, reassigned_weights_unchanged=~changed,
                 reassigned_MAP_changed=map_changed, reassigned_MAP_unchanged=~map_changed)
    profile = support_profile(original)
    eligible = profile['eligible']
    movable = eligible & (int(eligible.sum()) >= 2)
    raw, effective = profile['raw_support'], profile['effective_support']
    masks.update(support_eligible=eligible, support_movable=movable,
                 support_anchored=~movable,
                 support_eligible_singleton=eligible & ~movable,
                 support_P2_only=raw[:, 0] & ~raw[:, 1],
                 support_P4_only=~raw[:, 0] & raw[:, 1],
                 support_float32_underflow_anchor=raw.all(1) & ~effective.all(1))
    return masks



def bootstrap_mean(values, repetitions, seed, indices=None):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and len(values) and np.isfinite(values).all(), 'Invalid block values')
    index = (np.random.default_rng(seed).integers(0, len(values), size=(repetitions, len(values)))
             if indices is None else np.asarray(indices))
    require(index.shape == (repetitions, len(values)) and index.dtype.kind in 'iu' and
            np.all((index >= 0) & (index < len(values))), 'Invalid whole-block bootstrap indices')
    sample = values[index].mean(axis=1)
    return {'mean': float(values.mean()), 'interval_percentile95': np.quantile(sample, [.025, .975]).tolist(),
            'blocks': int(len(values)), 'favorable_negative_blocks': int((values < 0).sum()),
            'block_values': values.tolist(), 'resamples': int(repetitions)}


def conditional_block_bootstrap(values, indices):
    """Secondary conditional mean with the original full-block resampling.

    A missing subgroup in a block remains None, never zero. Each resample
    averages its defined subgroup block values. Entirely empty resamples
    are omitted only from the percentile calculation and counted explicitly.
    This is not a bootstrap of rows or a redraw until coverage is obtained.
    """
    values = list(values)
    require(all(v is None or np.isfinite(v) for v in values), 'Invalid conditional block values')
    index = np.asarray(indices)
    require(index.ndim == 2 and index.shape[1] == len(values) and index.dtype.kind in 'iu'
            and np.all((index >= 0) & (index < len(values))), 'Conditional bootstrap indices differ')
    array = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    valid = np.isfinite(array)
    sampled = array[index]
    present = np.isfinite(sampled)
    count = present.sum(1)
    defined = count > 0
    sums = np.where(present, sampled, 0.).sum(1)
    means = sums[defined] / count[defined]
    return {'mean': float(array[valid].mean()) if valid.any() else None,
            'interval_percentile95': np.quantile(means, [.025, .975]).tolist() if len(means) else None,
            'nonempty_blocks': int(valid.sum()), 'total_blocks': len(values),
            'negative_nonempty_blocks': int((array[valid] < 0).sum()),
            'block_values': values, 'resamples': len(index),
            'defined_resamples': int(defined.sum()), 'all_empty_resamples': int((~defined).sum()),
            'scope': 'Descriptive secondary conditional equal-nonempty-block mean; same full-block bootstrap indices as primary. Empty groups are not zero and all-empty resamples are reported, not redrawn.'}


def posterior_disagreement(posterior):
    """Conditional-period forecasts, disagreement and fitted-model MAP regret.

    D is posterior variance between conditional forecast probabilities; it is
    not itself MAP regret. The latter is (p_MAP-p_soft)^2 under this fitted
    posterior, and need not equal realized empirical Brier differences.
    Zero-mass conditionals are stored as zero only with explicit flags.
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    require(posterior.ndim == 2 and posterior.shape[1] == 6 and np.isfinite(posterior).all()
            and np.all(posterior>=0) and np.allclose(posterior.sum(1),1,rtol=0,atol=1e-10),
            'Invalid saved clock posterior')
    targets = ((95+np.arange(1,9)[None,:]) % CLOCKS[:,0,None] == CLOCKS[:,1,None]).astype(float)
    masses = np.column_stack([posterior[:,:2].sum(1),posterior[:,2:].sum(1)])
    components = np.zeros((len(posterior),2,8),float)
    for k, sl in enumerate((slice(0,2),slice(2,6))):
        numerator = posterior[:,sl]@targets[sl]
        np.divide(numerator,masses[:,k,None],out=components[:,k,:],where=masses[:,k,None]>0)
    soft = posterior@targets
    hard = components[np.arange(len(posterior)),np.where(masses[:,0]>=masses[:,1],0,1)]
    disagreement = masses[:,0,None]*masses[:,1,None]*(components[:,0]-components[:,1])**2
    logarithm=np.zeros_like(masses)
    np.log2(masses,out=logarithm,where=masses>0)
    require(np.all(disagreement>=0) and np.all(disagreement<=.25+1e-12), 'D outside theoretical bounds')
    return {'period_masses':masses,'conditional_period_probability':components,
            'conditional_defined':masses>0,'disagreement':disagreement,
            'component_probability_difference':components[:,0]-components[:,1],
            'period_entropy_bits':-np.sum(masses*logarithm,axis=1),
            'posterior_soft_probability':soft,'posterior_MAP_probability':hard,
            'posterior_MAP_regret':(hard-soft)**2}


def disagreement_strata(values):
    values = np.asarray(values)
    require(np.isfinite(values).all() and np.all(values>=0) and np.all(values<=.25+1e-12), 'Invalid D values')
    return {'zero':values==0,'positive_to_0.01':(values>0)&(values<=.01),
            'over_0.01_to_0.05':(values>.01)&(values<=.05),'over_0.05':values>.05}


def contrast_definitions():
    pairs = [('soft_fixed','support_reassigned_fixed'),('soft_fixed','equal_fixed'),
             ('support_reassigned_fixed','equal_fixed')]
    result = {a+'_minus_'+b:{'coefficients':{'neural_'+a:1.,'neural_'+b:-1.},
              'role':'primary' if (a,b)==('soft_fixed','support_reassigned_fixed') else 'secondary',
              'interpretation':'Negative favors '+a+'; no equivalence conclusion from an interval spanning zero.'}
              for a,b in pairs}
    for arm in ARMS:
        result[arm+'_minus_fitted_context_soft'] = {'coefficients':{'neural_'+arm:1.,'fitted_context_soft':-1.},
          'role':'mandatory_baseline','interpretation':'Negative favors neural over the direct fitted probability forecast.'}
    for a,b in [('fitted_context_soft','fitted_context_hard'),('fitted_context_soft','fitted_own_soft'),
                ('fitted_context_soft','known_law_context_soft')]:
        result[a+'_minus_'+b] = {'coefficients':{a:1.,b:-1.},'role':'reference','interpretation':'Negative favors first method.'}
    return result


def assignment_diagnostics(data, predictions):
    """Verify the saved map; never generate or select a scientific reassignment.

    Preservation is over motifs, not valid event slots. Donors with identical
    weights are legitimate. Only eligible rows are deranged; noneligible or
    singleton rows are exact self anchors. No effect-selected redraw occurs.
    """
    original = np.asarray(predictions['fitted_period_probability'], dtype=np.float64)
    assigned = np.asarray(predictions['reassigned_period_probability'], dtype=np.float64)
    donors = np.asarray(predictions['reassignment_donor_index'])
    n = len(data['x'])
    require(original.shape == assigned.shape == (n, 2) and n >= 1,
            'Reassignment requires aligned motif-by-two period masses')
    require(donors.shape == (n,) and donors.dtype.kind in 'iu' and
            np.array_equal(np.sort(donors), np.arange(n)), 'Donor indices are not a permutation')
    require(np.isfinite(original).all() and np.all(original >= 0) and
            np.allclose(original.sum(1), 1, rtol=0, atol=1e-10), 'Invalid original period masses')
    require(np.array_equal(assigned, original[donors]), 'Reassigned masses differ from exact donor mapping')
    own_support, donor_support = support_profile(original), support_profile(assigned)
    eligible = own_support['eligible']
    movable = eligible & (int(eligible.sum()) >= 2)
    require(np.all(donors[~movable] == np.arange(n)[~movable]),
            'Noneligible and singleton motifs must retain their own weights')
    require(np.all(donors[movable] != np.arange(n)[movable]) and
            np.all(movable[donors[movable]]), 'Movable donors must be different eligible motifs')
    require(np.array_equal(own_support['raw_support'], donor_support['raw_support']) and
            np.array_equal(own_support['effective_support'], donor_support['effective_support']),
            'Per-recipient raw or float32 support changed')
    weights = own_support['normalized_weights']
    assigned_weights = donor_support['normalized_weights']
    require(np.array_equal(assigned_weights, weights[donors]), 'Normalized donor mapping changed')
    order = np.lexsort((weights[:, 1], weights[:, 0]))
    assigned_order = np.lexsort((assigned_weights[:, 1], assigned_weights[:, 0]))
    require(np.array_equal(weights[order], assigned_weights[assigned_order]),
            'Motif-level empirical normalized-weight distribution changed')
    changed_mass = np.any(original != assigned, axis=1)
    changed = np.any(weights != assigned_weights, axis=1)
    changed_float32 = np.any(own_support['float32_coefficients'] != donor_support['float32_coefficients'], axis=1)
    distance = np.abs(weights-assigned_weights).sum(1)
    # This detects the expected difference in event-slot weighting but does
    # not impose equality at that different, undeclared sampling unit.
    event_counts = data['x'].sum(1) + data['y'].sum(1)
    total_events = int(event_counts.sum())
    weighted_original = ((weights*event_counts[:, None]).sum(0)/total_events).tolist() if total_events else None
    weighted_assigned = ((assigned_weights*event_counts[:, None]).sum(0)/total_events).tolist() if total_events else None
    return {'motifs': n, 'eligible_motifs': int(eligible.sum()), 'movable_motifs': int(movable.sum()),
            'anchored_motifs': int((~movable).sum()),
            'eligible_singleton_motifs': int((eligible & ~movable).sum()),
            'self_donors': int((donors == np.arange(n)).sum()),
            'zero_self_donors_among_movable': True, 'anchors_exactly_preserved': True,
            'raw_support_preserved_per_recipient': True,
            'normalized_float32_support_preserved_per_recipient': True,
            'raw_support_changed_recipients': 0, 'effective_support_changed_recipients': 0,
            'permutation_verified': True,
            'exact_original_to_donor_mapping_verified': True,
            'motif_weight_distribution_preserved_exactly': True,
            'changed_normalized_weight_vectors': int(changed.sum()),
            'changed_mass_vectors': int(changed_mass.sum()),
            'changed_float32_coefficient_vectors': int(changed_float32.sum()),
            'unchanged_normalized_weight_vectors_despite_different_donor': int((~changed & movable).sum()),
            'unchanged_normalized_weight_vectors_including_anchors': int((~changed).sum()),
            'changed_weight_fraction': float(changed.mean()),
            'normalized_weight_L1_displacement_mean': float(distance.mean()),
            'normalized_weight_L1_displacement_maximum': float(distance.max()),
            'maximum_normalized_weight_norm_error': float(np.max(np.abs(np.linalg.norm(weights, axis=1)-1))),
            'event_count_weighted_original_mean_weight': weighted_original,
            'event_count_weighted_reassigned_mean_weight': weighted_assigned,
            'event_count_weighted_distribution_preservation_required': False,
            'scope': 'Stable label-free support-eligible same-split donor pool, exact self anchors elsewhere; transductive diagnostic, not a deployment predictor. One assignment per block/split; no permutation selection. Support is raw positivity plus normalized float32 coefficient positivity, not a confidence threshold.'}


def support_diagnostics(data, predictions):
    """Prespecified saved-score support and two-step consistency diagnostics.

    Hidden clocks and labels are evaluation diagnostics only. An unchanged
    positive channel does not force the learned network to use that channel,
    so h+2 equality/inequality is reported rather than used as a scientific gate.
    """
    own = support_profile(predictions['fitted_period_probability'])
    assigned = support_profile(predictions['reassigned_period_probability'])
    require(np.array_equal(own['raw_support'], assigned['raw_support']) and
            np.array_equal(own['effective_support'], assigned['effective_support']),
            'Support changed in saved diagnostic input')
    true_period = CLOCKS[data['clock_index'], 0]
    true_four = true_period == 4
    groups = strata(data, predictions)
    rows = {}
    for name in ('all', *SUPPORT_GROUPS, 'true_period2', 'true_period4'):
        mask = groups[name]
        counts = int(mask.sum())
        rows[name] = {'motifs': counts,
            'labels_different_h_vs_h_plus_two': int(np.sum(data['labels'][mask, :6] != data['labels'][mask, 2:])),
            'methods': {}}
        for method in METHODS:
            p = predictions[method][mask]
            diff = np.abs(p[:, :6] - p[:, 2:])
            rows[name]['methods'][method] = {
                'maximum_h_plus_two_probability_difference': float(diff.max()) if counts else None,
                'mean_h_plus_two_probability_difference': float(diff.mean()) if counts else None,
                'exact_two_step_symmetric_motifs': int(np.all(diff == 0, axis=1).sum()),
                'brier_h5_h8': float(np.mean((p[:, 4:] - data['labels'][mask, 4:])**2)) if counts else None}
    raw_loss = own['raw_support'][:, 1] & ~assigned['raw_support'][:, 1]
    effective_loss = own['effective_support'][:, 1] & ~assigned['effective_support'][:, 1]
    require(not raw_loss.any() and not effective_loss.any(), 'Reassignment eliminated P4 support')
    arrays = {'original_raw_support': own['raw_support'],
              'assigned_raw_support': assigned['raw_support'],
              'original_effective_support': own['effective_support'],
              'assigned_effective_support': assigned['effective_support'],
              'eligible': own['eligible'],
              'movable': groups['support_movable'], 'anchored': groups['support_anchored'],
              'original_float32_coefficients': own['float32_coefficients'],
              'assigned_float32_coefficients': assigned['float32_coefficients']}
    report = {'groups': rows,
              'new_raw_P4_support_loss': int(raw_loss.sum()),
              'new_effective_P4_support_loss': int(effective_loss.sum()),
              'true_P4_new_raw_P4_support_loss': int((true_four & raw_loss).sum()),
              'true_P4_new_effective_P4_support_loss': int((true_four & effective_loss).sum()),
              'true_P4_original_P2_only_raw_support': int((true_four & ~own['raw_support'][:, 1]).sum()),
              'true_P4_assigned_P2_only_raw_support': int((true_four & ~assigned['raw_support'][:, 1]).sum()),
              'scope': 'Prespecified saved-score diagnostic. No new model calls. Support preservation is a technical requirement; favorable h+2 behavior is not. Conditional strata compare differently trained networks.'}
    return report, arrays


def direct_predictions(theta, data, spec, unit=None):
    """Approved-stage forecast helper, not called on import or by author tests."""
    from fc1_core import predict
    from sw1_data import true_theta
    require(unit is not None, 'Known-law reference requires explicit condition')
    fitted = predict(data['x'],data['y'],theta)
    known = predict(data['x'],data['y'],true_theta(unit))
    probabilities = {'fitted_'+name:fitted[name] for name in ('own_hard','own_soft','context_hard','context_soft')}
    probabilities['known_law_context_soft'] = known['context_soft']
    metadata = {'fitted_clock_posterior':fitted['posterior_context'],
                'fitted_period_probability':fitted['context_period_masses'],
                'context_entropy':fitted['context_entropy'],'context_map_period':fitted['context_map_period']}
    return probabilities,metadata


def _validate_saved(data,predictions,methods):
    n = len(data['x'])
    require(data['x'].shape == data['y'].shape == (n,16) and np.isin(data['x'],[0,1]).all()
            and np.isin(data['y'],[0,1]).all(), 'Saved history alignment differs')
    require(data['labels'].shape == (n,8), 'Eight horizon labels required')
    from sw1_data import labels_for_clock
    require(np.array_equal(data['labels'],labels_for_clock(data['clock_index'],np.arange(1,9))),
            'Saved labels disagree with saved latent evaluation clocks')
    for method in methods:
        require(predictions[method].shape == (n,8), 'Saved probability shape differs: '+method)
    details = posterior_disagreement(predictions['fitted_clock_posterior'])
    require(np.allclose(details['posterior_soft_probability'],predictions['fitted_context_soft'],rtol=0,atol=1e-12)
            and np.allclose(details['posterior_MAP_probability'],predictions['fitted_context_hard'],rtol=0,atol=1e-12),
            'Saved direct forecasts disagree with saved posterior')
    require(np.allclose(details['period_masses'],predictions['fitted_period_probability'],rtol=0,atol=1e-12),
            'Saved period masses disagree with clock posterior')
    require(np.allclose(details['period_entropy_bits'],predictions['context_entropy'],rtol=0,atol=1e-12),
            'Saved period entropy disagrees with clock posterior')
    expected_map=np.where(details['period_masses'][:,0] >= details['period_masses'][:,1],2,4)
    require(np.array_equal(predictions['context_map_period'],expected_map),
            'Saved marginal MAP period disagrees with original recipient posterior')
    return details


def masked_query_metrics(labels,predictions,mask):
    """Query-subset Brier/logloss; AUC still averaged within each horizon."""
    labels,predictions,mask=np.asarray(labels),np.asarray(predictions),np.asarray(mask)
    require(mask.dtype == bool and mask.shape == labels.shape == predictions.shape,'Invalid query mask')
    value=metrics(labels[mask],predictions[mask])
    aucs=[auc(labels[mask[:,h],h],predictions[mask[:,h],h]) for h in range(labels.shape[1])]
    defined=[a for a in aucs if a is not None]
    value['auc']=float(np.mean(defined)) if defined else None
    value['auc_defined_horizons']=len(defined)
    value['auc_convention']='Equal mean of defined within-horizon selected-query AUCs; never pooled across time'
    return value


def block_summary(data,predictions,methods=METHODS):
    labels=data['labels']
    details=_validate_saved(data,predictions,methods)
    groups=strata(data,predictions)
    result={}
    for group,mask in groups.items():
        result[group]={'pairs':int(mask.sum()),'methods':{}}
        for method in methods:
            p=predictions[method]
            result[group]['methods'][method]={'h1_h4':grouped_metrics(labels[mask,:4],p[mask,:4]),
                'h5_h8':grouped_metrics(labels[mask,4:],p[mask,4:]),
                'per_horizon':{str(h+1):metrics(labels[mask,h],p[mask,h]) for h in range(8)}}
    # D(q) is stratified at query level, never silently converted to a pair mean.
    drows={}
    for name,mask in disagreement_strata(details['disagreement'][:,4:]).items():
        drows[name]={'rows':int(mask.sum()),'pairs_with_at_least_one_selected_query':int(mask.any(1).sum()),
            'methods':{m:masked_query_metrics(labels[:,4:],predictions[m][:,4:],mask) for m in methods},
            'mean_D':float(details['disagreement'][:,4:][mask].mean()) if mask.any() else None,
            'mean_posterior_MAP_regret':float(details['posterior_MAP_regret'][:,4:][mask].mean()) if mask.any() else None}
    diagnostic={'mean_D_h5_h8':float(details['disagreement'][:,4:].mean()),
        'mean_posterior_MAP_regret_h5_h8':float(details['posterior_MAP_regret'][:,4:].mean()),
        'zero_mass_period_components':int((~details['conditional_defined']).sum()),
        'D_strata_h5_h8':drows,
        'D_per_horizon':{str(h+1):float(details['disagreement'][:,h].mean()) for h in range(8)},
        'interpretation':'D and posterior MAP regret are fitted-model quantities, not empirical neural treatment effects.'}
    support_report, _ = support_diagnostics(data, predictions)
    return {'strata':result,'disagreement':diagnostic,
            'assignment':assignment_diagnostics(data,predictions),
            'support': support_report},details


def _summarize_values(values):
    nonempty=[v for v in values if v is not None]
    return {'mean':float(np.mean(nonempty)) if nonempty else None,'nonempty_blocks':len(nonempty),'block_values':values}


def aggregate(blocks,spec,methods=METHODS,definitions=None):
    require(len(blocks)==len(spec['blocks']), 'Incomplete-case block deletion is prohibited')
    definitions=contrast_definitions() if definitions is None else definitions
    summary={}
    for group in blocks[0]['strata']:
        row={'pair_counts_by_block':[b['strata'][group]['pairs'] for b in blocks],'methods':{}}
        for method in methods:
            row['methods'][method]={window:{metric:_summarize_values([b['strata'][group]['methods'][method][window][metric] for b in blocks])
                for metric in ('brier','logloss','auc')} for window in ('h1_h4','h5_h8')}
            row['methods'][method]['per_horizon']={str(h):{metric:_summarize_values(
                [b['strata'][group]['methods'][method]['per_horizon'][str(h)][metric] for b in blocks])
                for metric in ('brier','logloss','auc')} for h in range(1,9)}
        summary[group]=row
    indices=np.random.default_rng(spec['bootstrap_seed']).integers(0,len(blocks),size=(spec['bootstrap_replicates'],len(blocks)))
    contrasts={}
    for name,definition in definitions.items():
        coefficients=definition['coefficients']
        require(set(coefficients)<=set(methods) and sum(coefficients.values())==0,'Invalid contrast')
        values=[sum(c*b['strata']['all']['methods'][m]['h5_h8']['brier'] for m,c in coefficients.items()) for b in blocks]
        value=bootstrap_mean(values,spec['bootstrap_replicates'],spec['bootstrap_seed'],indices)
        value['negative_blocks']=value.pop('favorable_negative_blocks')
        value.update(definition)
        # Descriptive subgroup paired differences, never treating rows as replications.
        value['subgroups']={group:_summarize_values([
            sum(c*b['strata'][group]['methods'][m]['h5_h8']['brier'] for m,c in coefficients.items())
            if b['strata'][group]['pairs'] else None for b in blocks]) for group in blocks[0]['strata']}
        value['support_subgroups'] = {
            group: conditional_block_bootstrap(value['subgroups'][group]['block_values'], indices)
            for group in SUPPORT_GROUPS}
        value['D_strata']={nameD:_summarize_values([
            sum(c*b['disagreement']['D_strata_h5_h8'][nameD]['methods'][m]['brier'] for m,c in coefficients.items())
            if b['disagreement']['D_strata_h5_h8'][nameD]['rows'] else None for b in blocks])
            for nameD in blocks[0]['disagreement']['D_strata_h5_h8']}
        contrasts[name]=value
    dsummary={}
    for key in blocks[0]['disagreement']['D_strata_h5_h8']:
        dsummary[key]={'rows_by_block':[b['disagreement']['D_strata_h5_h8'][key]['rows'] for b in blocks],
                      'methods':{m:_summarize_values([b['disagreement']['D_strata_h5_h8'][key]['methods'][m]['brier'] for b in blocks]) for m in methods}}
    return {'strata':summary,'contrasts_h5_h8':contrasts,'D_strata_h5_h8':dsummary,
        'mean_D_h5_h8':_summarize_values([b['disagreement']['mean_D_h5_h8'] for b in blocks]),
        'mean_posterior_MAP_regret_h5_h8':_summarize_values([b['disagreement']['mean_posterior_MAP_regret_h5_h8'] for b in blocks]),
        'primary':PRIMARY,'methods':list(methods),'scientific_pass_fail_rule':None,'equivalence_margin':None,
        'assignment_diagnostics_by_block':[b['assignment'] for b in blocks],
        'support_diagnostics_by_block':[b['support'] for b in blocks],
        'support_conditional_interval_policy': 'The original full-block bootstrap indices are reused. A subgroup contributes its conditional mean only in blocks where present; all-empty resamples are counted and have no defined statistic. These intervals are secondary, descriptive and not multiplicity-adjusted.',
        'interval_scope':'Pointwise whole paired-block percentile bootstrap. Conditional on retained CW1 F05 population fits, generator and declared training/permutation procedure. Blocks combine new cohorts, initialization and one permutation; no independent horizons, events or permutations.',
        'multiplicity':'One named primary contrast; secondary comparisons and subgroups descriptive/exploratory. No simultaneous interval or discovery claim.',
        'empty_stratum_policy':'null, never zero; nonempty-block count retained',
        'D_stratum_unit':'h5-h8 query; no outcome-driven threshold selection',
        'logloss_clip_only':LOGLOSS_CLIP}


def analyze(work,output,spec,cohorts=None):
    from sw1_data import selected_fit, verify_assignment
    work,output=Path(work),Path(output)
    chosen=list(CONDITIONS) if cohorts is None else list(cohorts)
    # Accept either condition identifiers or the complete unit list used by session.
    if chosen and '_B' in chosen[0]:
        identifiers=chosen
        chosen=list(dict.fromkeys(u.split('_B')[0] for u in identifiers))
        require(set(identifiers)=={f'{c}_B{b:02d}' for c in chosen for b in spec['blocks']},'Incomplete condition cohort')
    require(chosen and len(set(chosen))==len(chosen) and set(chosen)<=set(CONDITIONS),'Unknown/duplicate analysis condition')
    blocks,summary,inputs={}, {}, {}
    qualification={}
    training_assignment={}
    output.mkdir(parents=True,exist_ok=True)
    for condition in chosen:
        selected=[]
        for block in spec['blocks']:
            unit=f'{condition}_B{block:02d}';root=work/'results/evaluate'/unit
            report=read_json(root/'REPORT.json')
            require(report['block']==block and report['data_sha256']==sha(root/'DATA.npz') and
                    report['predictions_sha256']==sha(root/'PREDICTIONS.npz'),'Evaluation hash/identity mismatch')
            data,predictions=load_npz(root/'DATA.npz'),load_npz(root/'PREDICTIONS.npz')
            _,fit_record,fit_hash=selected_fit(work,unit)
            require(report['source_fit_record_sha256']==fit_hash,'Evaluation selected fit changed')
            qualification[unit]={'selected_qualified':fit_record.get('selected_qualified',fit_record.get('qualified')),
                                 'selected_endpoint':fit_record['selected_endpoint'],
                                 'fit_sha256':fit_record['fit_sha256'],
                                 'excluded_from_analysis':False}
            assignment=load_npz(root/'ASSIGNMENT.npz')
            assignment_report=read_json(root/'ASSIGNMENT_REPORT.json')
            require(report['assignment_sha256'] == sha(root/'ASSIGNMENT.npz') ==
                    assignment_report['assignment_sha256'] and
                    report['assignment_report_sha256'] == sha(root/'ASSIGNMENT_REPORT.json'),
                    'Evaluation assignment hash differs')
            require(np.array_equal(assignment['donor_index'],predictions['reassignment_donor_index']) and
                    np.array_equal(assignment['original_period_probability'],predictions['fitted_period_probability']) and
                    np.array_equal(assignment['reassigned_period_probability'],predictions['reassigned_period_probability']),
                    'Standalone assignment and prediction-side assignment differ')
            original=predictions['fitted_period_probability']
            weights=original/np.linalg.norm(original,axis=1,keepdims=True)
            require(np.array_equal(assignment['original_normalized_weights'],weights) and
                    np.array_equal(assignment['reassigned_normalized_weights'],weights[assignment['donor_index']]),
                    'Standalone normalized assignment differs')
            verify_assignment(spec,unit,'evaluation',original,assignment,assignment_report)
            training_root=work/'results/assets'/unit
            training_data=load_npz(training_root/'DATA.npz')
            training_arrays=load_npz(training_root/'ASSIGNMENT.npz')
            training_assignment_report=read_json(training_root/'ASSIGNMENT_REPORT.json')
            training_report=read_json(training_root/'REPORT.json')
            require(training_report['source_fit_record_sha256'] == fit_hash and
                    training_report['data_sha256'] == sha(training_root/'DATA.npz') and
                    training_report['assignment_sha256'] == sha(training_root/'ASSIGNMENT.npz') ==
                    training_assignment_report['assignment_sha256'] and
                    training_report['assignment_report_sha256'] == sha(training_root/'ASSIGNMENT_REPORT.json'),
                    'Training assignment/source hash differs')
            verify_assignment(spec,unit,'train',training_data['fitted_period_probability'],
                              training_arrays,training_assignment_report)
            training_prediction_view={
                'fitted_period_probability':training_data['fitted_period_probability'],
                'reassigned_period_probability':training_arrays['reassigned_period_probability'],
                'reassignment_donor_index':training_arrays['donor_index']}
            training_assignment[unit]=assignment_diagnostics(training_data,training_prediction_view)
            blocks[unit],details=block_summary(data,predictions)
            np.savez_compressed(output/f'{unit}_POSTERIOR_DIAGNOSTICS.npz',pair_index=data['pair_index'],**details)
            _, support_arrays = support_diagnostics(data, predictions)
            np.savez_compressed(output/f'{unit}_SUPPORT_DIAGNOSTICS.npz',
                                pair_index=data['pair_index'], **support_arrays)
            selected.append(blocks[unit]);inputs[unit]={'report_sha256':sha(root/'REPORT.json'),
                'data_sha256':sha(root/'DATA.npz'),'predictions_sha256':sha(root/'PREDICTIONS.npz'),
                'assignment_sha256':sha(root/'ASSIGNMENT.npz'),
                'assignment_report_sha256':sha(root/'ASSIGNMENT_REPORT.json'),
                'training_assignment_sha256':sha(training_root/'ASSIGNMENT.npz'),
                'training_assignment_report_sha256':sha(training_root/'ASSIGNMENT_REPORT.json'),
                'pairing':report['pairing']}
        summary[condition]=aggregate(selected,spec)
    indices=np.random.default_rng(spec['bootstrap_seed']).integers(0,len(spec['blocks']),size=(spec['bootstrap_replicates'],len(spec['blocks'])))
    np.savez_compressed(output/'BOOTSTRAP_INDICES.npz',indices=indices,block_order=np.asarray(spec['blocks']))
    write_json(output/'BLOCK_METRICS.json',blocks)
    write_json(output/'SUPPORT_DIAGNOSTICS.json',
               {'blocks': {unit: row['support'] for unit, row in blocks.items()},
                'prespecified': True, 'new_model_calls': False,
                'interpretation': 'Support loss is forbidden; h+2 equality is descriptive. Positive phase-channel availability does not establish its effective use by the trained model.'})
    write_json(output/'SUMMARY.json',{'conditions':summary,
        'all_selected_fits_retained':True,'population_parameters_refitted':False,
        'calibration_qualification':qualification,
        'training_assignment_diagnostics':training_assignment,
        'assignment_control_limitations':[
            'Motif-level normalized-weight distribution is preserved, not event-slot-weighted allocation.',
            'Only raw-and-float32 dual-positive motifs form the same-split donor pool; noneligible and singleton motifs are exact self anchors.',
            'Support-eligible same-split reassignment is a transductive diagnostic; donors supply no future labels.',
            'Raw and normalized float32 support is preserved per recipient, but confidence, counts, phase, relative amplitudes and training gradients can still change.',
            'The contrast changes correspondence with history-derived features, not calibrated probability correctness alone.',
            'One permutation per block/split is fixed across epochs/horizons; permutations are not independent replications.',
            'Distinct donor indices may still carry numerically identical weights; no redraw to force different values.',
            'Anchored motifs have unchanged supplied input weights, not necessarily unchanged predictions from separately trained networks.',
            'D, entropy and period-error groups are computed from the recipient original posterior, never reassigned weights.'],
        'scientific_pass_fail_rule':None,'equivalence_margin':None})
    write_json(output/'REPORT.json',{'stage':'analyze','conditions':chosen,'complete_cohorts':len(blocks),
        'arithmetic_runtime':{'python':platform.python_version(),'numpy':np.__version__,'device':'CPU'},
        'methods':list(METHODS),'inputs':inputs,'new_training_or_fitting':False,
        'summary_sha256':sha(output/'SUMMARY.json'),'block_metrics_sha256':sha(output/'BLOCK_METRICS.json'),
        'support_diagnostics_sha256':sha(output/'SUPPORT_DIAGNOSTICS.json'),
        'bootstrap_indices_sha256':sha(output/'BOOTSTRAP_INDICES.npz'),
        'interpretation':'Soft versus support-preserving reassignment tests history-to-weight correspondence while preserving raw/float32 support, motif-level weight distribution and pre-fusion norm; trainable networks may coadapt. It does not isolate probability calibration, guarantee use of both positive channels, prove a general GNN advantage, or demonstrate learned extrapolation.'})


