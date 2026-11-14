"""CW1 saved-prediction arithmetic; no scientific work on import."""
from pathlib import Path
import platform
import numpy as np
from clock_weights_io import read_json, require, sha, write_json
from clock_weights_data import CLOCKS, TIMES, load_npz, ARMS, CONDITIONS, block_number
LOGLOSS_CLIP=1e-7
PRIMARY="soft_fixed_minus_equal_fixed"
METHODS=tuple("neural_"+a for a in ARMS)+(
    "fitted_own_hard","fitted_own_soft","fitted_context_hard",
    "fitted_context_soft","known_law_context_soft")

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
    pairs = [('soft_fixed','equal_fixed'),('equal_fixed','hard_fixed'),('soft_fixed','hard_fixed'),
             ('period4_fixed','hard_fixed'),('soft_fixed','period4_fixed'),('equal_fixed','period4_fixed')]
    result = {a+'_minus_'+b:{'coefficients':{'neural_'+a:1.,'neural_'+b:-1.},
              'role':'primary' if (a,b)==('soft_fixed','equal_fixed') else 'secondary',
              'interpretation':'Negative favors '+a+'; no equivalence conclusion from an interval spanning zero.'}
              for a,b in pairs}
    for arm in ARMS:
        result[arm+'_minus_fitted_context_soft'] = {'coefficients':{'neural_'+arm:1.,'fitted_context_soft':-1.},
          'role':'mandatory_baseline','interpretation':'Negative favors neural over the direct fitted probability forecast.'}
    for a,b in [('fitted_context_soft','fitted_context_hard'),('fitted_context_soft','fitted_own_soft'),
                ('fitted_context_soft','known_law_context_soft')]:
        result[a+'_minus_'+b] = {'coefficients':{a:1.,b:-1.},'role':'reference','interpretation':'Negative favors first method.'}
    return result


def direct_predictions(theta, data, spec, unit=None):
    """Approved-stage forecast helper, not called on import or by author tests."""
    from clock_inference import predict
    from clock_weights_data import true_theta
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
    from clock_weights_data import labels_for_clock
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
    return {'strata':result,'disagreement':diagnostic},details


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
        'interval_scope':'Pointwise whole paired-block percentile bootstrap. Conditional on generator and declared fitting procedure; R05 additionally conditional on retained fits. Not independent horizons, events or queries.',
        'multiplicity':'One named primary contrast; secondary comparisons, subgroups and historical audit exploratory. No simultaneous interval or discovery claim.',
        'empty_stratum_policy':'null, never zero; nonempty-block count retained',
        'D_stratum_unit':'h5-h8 query; no outcome-driven threshold selection',
        'logloss_clip_only':LOGLOSS_CLIP}


def saved_audit(work,output,spec):
    """Explicitly approved CPU historical arithmetic; no generation or fitting."""
    from clock_weights_source import verify_imported
    work,output=Path(work),Path(output)
    provenance=verify_imported(work/'source')
    methods=('neural_hard_fixed','neural_soft_fixed','neural_hard_adaptive','neural_soft_adaptive',
             'fitted_context_hard','fitted_context_soft')
    definitions={'soft_fixed_minus_hard_fixed':{'coefficients':{'neural_soft_fixed':1.,'neural_hard_fixed':-1.},'role':'historical_exploratory'},
                 'soft_adaptive_minus_hard_adaptive':{'coefficients':{'neural_soft_adaptive':1.,'neural_hard_adaptive':-1.},'role':'historical_exploratory'},
                 'direct_soft_minus_MAP':{'coefficients':{'fitted_context_soft':1.,'fitted_context_hard':-1.},'role':'historical_exploratory'}}
    output.mkdir(parents=True,exist_ok=True)
    blocks={}
    for block in spec['blocks']:
        unit=f'B{block:02d}';root=work/'source/historical_evaluate'/unit
        data,predictions=load_npz(root/'DATA.npz'),load_npz(root/'PREDICTIONS.npz')
        blocks[unit],details=block_summary(data,predictions,methods)
        np.savez_compressed(output/f'{unit}_POSTERIOR_DIAGNOSTICS.npz',pair_index=data['pair_index'],**details)
    summary=aggregate(list(blocks.values()),spec,methods,definitions)
    summary['primary']=None
    summary['scope']='Retrospective EM1 saved evidence, not a new neural comparison; equal/period4 arms do not exist in these old predictions.'
    write_json(output/'BLOCK_METRICS.json',blocks)
    write_json(output/'SUMMARY.json',summary)
    write_json(output/'REPORT.json',{'stage':'saved-audit','historical_only':True,'blocks':16,
        'arithmetic_runtime':{'python':platform.python_version(),'numpy':np.__version__,'device':'CPU'},
        'source_provenance':provenance,'new_data_generated':False,'models_executed':False,'parameters_fitted':False,
        'thresholds_selected_from_results':False,'summary_sha256':sha(output/'SUMMARY.json'),
        'block_metrics_sha256':sha(output/'BLOCK_METRICS.json'),
        'scientific_interpretation':'Observed D/entropy/loss associations do not isolate a causal encoding mechanism.'})


def analyze(work,output,spec,cohorts=None):
    from clock_weights_data import selected_fit
    work,output=Path(work),Path(output)
    chosen=list(CONDITIONS) if cohorts is None else list(cohorts)
    # Accept either condition identifiers or the complete unit list used by session.
    if chosen and '_B' in chosen[0]:
        identifiers=chosen
        chosen=list(dict.fromkeys(u[:3] for u in identifiers))
        require(set(identifiers)=={f'{c}_B{b:02d}' for c in chosen for b in spec['blocks']},'Incomplete condition cohort')
    require(chosen and len(set(chosen))==len(chosen) and set(chosen)<=set(CONDITIONS),'Unknown/duplicate analysis condition')
    blocks,summary,inputs={}, {}, {}
    pairing_data={}
    qualification={}
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
            if condition.startswith('F'):
                pairing_data[unit]={name:data[name] for name in ('x','clock_index','realized_law','alternating_parity','pair_index','shared_indicator')}
            blocks[unit],details=block_summary(data,predictions)
            np.savez_compressed(output/f'{unit}_POSTERIOR_DIAGNOSTICS.npz',pair_index=data['pair_index'],**details)
            selected.append(blocks[unit]);inputs[unit]={'report_sha256':sha(root/'REPORT.json'),
                'data_sha256':sha(root/'DATA.npz'),'predictions_sha256':sha(root/'PREDICTIONS.npz'),
                'pairing':report['pairing']}
        summary[condition]=aggregate(selected,spec)
    interactions={}
    crn_check=None
    if all(c in chosen for c in ('F00','F05','F10')):
        for block in spec['blocks']:
            reference=pairing_data[f'F00_B{block:02d}']
            for condition in ('F05','F10'):
                comparison=pairing_data[f'{condition}_B{block:02d}']
                for name in ('x','clock_index','realized_law','alternating_parity','pair_index'):
                    require(np.array_equal(reference[name],comparison[name]),'Fresh common-random-number pairing differs: '+name)
            require(not reference['shared_indicator'].any() and pairing_data[f'F10_B{block:02d}']['shared_indicator'].all(),
                    'Independent/shared control thresholds differ')
        crn_check={'passed':True,'blocks':len(spec['blocks']),
                   'exact_saved_fields':['x','clock_index','realized_law','alternating_parity','pair_index'],
                   'independent_and_shared_threshold_controls_checked':True}
        for first,second in [('F05','F00'),('F05','F10'),('F10','F00')]:
            a=summary[first]['contrasts_h5_h8'][PRIMARY]['block_values']
            b=summary[second]['contrasts_h5_h8'][PRIMARY]['block_values']
            value=bootstrap_mean(np.asarray(a)-np.asarray(b),spec['bootstrap_replicates'],spec['bootstrap_seed'])
            value['negative_blocks']=value.pop('favorable_negative_blocks')
            value['interpretation']='Difference of soft-minus-equal effects; neither sign is a universal success criterion. Fresh CRN block pairs are jointly resampled.'
            interactions[first+'_minus_'+second]=value
    indices=np.random.default_rng(spec['bootstrap_seed']).integers(0,len(spec['blocks']),size=(spec['bootstrap_replicates'],len(spec['blocks'])))
    np.savez_compressed(output/'BOOTSTRAP_INDICES.npz',indices=indices,block_order=np.asarray(spec['blocks']))
    write_json(output/'BLOCK_METRICS.json',blocks)
    write_json(output/'SUMMARY.json',{'conditions':summary,'fresh_rho_interactions':interactions,
        'retained_vs_fresh_not_a_paired_comparison':True,'all_selected_fits_retained':True,
        'calibration_qualification':qualification,'fresh_common_random_number_check':crn_check,
        'scientific_pass_fail_rule':None,'equivalence_margin':None})
    write_json(output/'REPORT.json',{'stage':'analyze','conditions':chosen,'complete_cohorts':len(blocks),
        'arithmetic_runtime':{'python':platform.python_version(),'numpy':np.__version__,'device':'CPU'},
        'methods':list(METHODS),'inputs':inputs,'new_training_or_fitting':False,
        'summary_sha256':sha(output/'SUMMARY.json'),'block_metrics_sha256':sha(output/'BLOCK_METRICS.json'),
        'bootstrap_indices_sha256':sha(output/'BOOTSTRAP_INDICES.npz'),
        'interpretation':'A neural soft-versus-equal comparison isolates the use of fitted period-channel weights at fixed bank norm, not pure uncertainty, a general GNN advantage or calibrated forecast probabilities.'})


