# Main synthetic protocol and design choices

This document describes the **Clock-weight comparison** and **Support-preserving weight reassignment** workflows. `configs/clock_weights.json` and `configs/weight_reassignment.json` retain the exact original scientific specifications. `SOURCE_PROVENANCE.json` records the original implementation hashes and the naming-only changes used in this release. The portable launcher is new orchestration, not an original Colab session or retrospective claim of preregistration.

## Question

When observed history leaves more than one recurring clock plausible, does assigning fitted weights to candidate period channels improve TGN beyond hard selection or equal weighting? Does the advantage require associating those weights with the appropriate history, rather than merely retaining their distribution?

## Population and observation law

Every independent motif has nodes a,b,c, a candidate edge a→b and a visible neighbor b→c. Candidate periods are 2 or 4, each with probability .5, and phase is uniform within period. The six clock probabilities are therefore (.25,.25,.125,.125,.125,.125).

The neighbor shares the candidate's exact period and phase with probability rho. Otherwise it draws an independent clock and may still match by chance. The clock-weight comparison uses rho=0,.5,1, with fresh calibration in each condition and common random numbers across conditions within block; the threshold determining shared versus independent changes. This controls the amount of useful neighboring evidence while retaining pairing. The generating rho is not supplied to fitted inference.

Candidate recording mechanisms are equally frequent: independent retention .875, independent retention .1875, or alternating opportunity eligibility with hidden fair parity and retention .875. Neighbor retention is .75. There are no false-positive records. All sixteen integer timestamps 80–95 are monitored. Recorded absences at otherwise possible opportunities contribute likelihood information. Future labels are underlying deterministic clock events, not whether those events would be recorded.

The finite period family, recording-mechanism family and uniform conditional phase are supplied assumptions. Their population parameters are fitted. This is correctly specified synthetic inference, not evidence of robustness to unknown period families or observation laws.

## Calibration and information access

Each fresh condition/block fits eight population coordinates from 7,680 unlabeled candidate–neighbor history pairs. Six fixed starts use bounded EM, SLSQP refinement and one declared bounded continuation. Selection is the lowest finite feasible calibration mean negative log likelihood, not best downstream prediction. An unqualified selected endpoint is retained and flagged, not replaced by a worse objective endpoint. The projected-gradient threshold is 1e-6; it is first-order numerical qualification, not proof of global optimum.

There are 16 blocks per condition. The weight-reassignment study retains all sixteen selected fits from the half-shared-neighbor condition, with no favorable subset. Neural training and test histories are separate from the calibration sample. The direct fitted probability baseline uses the full posterior over period and phase; TGN's query-time representation uses its two period marginals together with observed event histories.

No true per-motif period, hidden recording law, parity or sharing indicator is provided to operational forecasts. True quantities are retained only for evaluation and diagnostic subgroup definitions. Empty own histories are retained and receive a posterior; they are not discarded because a median-gap estimate is unavailable.

## TGN representation

For an event time t and query q, let d=q−t and

\[
u_P(d)=[\cos(2\pi(d\bmod P)/P),\sin(2\pi(d\bmod P)/P)].
\]

The query bank is four repetitions of `[w2*u2(d), w4*u4(d)]`. Weight vectors have norm one, so valid 16-coordinate banks have norm two:

| Arm | Weights |
|---|---|
| Hard | One-hot marginal-MAP period; tie to P2 |
| Equal | (1/sqrt(2),1/sqrt(2)) |
| Fitted soft | (m2,m4)/sqrt(m2²+m4²) |
| Fixed P4 | (0,1) |
| Support-preserving reassigned | Another eligible history's normalized marginals, evaluated on the recipient's own event lags |

The query bank is concatenated with cutoff-age features at T−t and passed through the same learned fusion. It replaces the time-feature route entering attention keys and values, not the event features. Historical message time encoding is unchanged. The three-node functional TGN uses memory width 32, time width 16, edge width 4, one attention layer, two heads and at most 32 neighbors; every arm has 50,434 trainable parameters.

Raw h/8 and obsolete engineered recurrence cues are neutralized in these studies. With periods 2 and 4, changing q by 4 leaves the relevant query representation unchanged. Prediction equality is checked; accuracy remains the substantive outcome. This explicit symmetry must not be called learned recurrence extrapolation.

## Reassigned-weight control

The weight-reassignment study trains fitted-soft, equal and support-preserving reassigned arms. Eligibility requires both raw float64 masses and both normalized float32 weights to be positive. Eligible rows form one pool per block/split. A uniformly drawn permutation is rejected until it has no fixed points; there is no cyclic-rotation fallback. Noneligible rows and an eligible singleton are anchored to themselves. Maps are fixed across epochs, batches and horizons; training/evaluation maps have distinct seeds.

This preserves exact per-recipient raw/effective support and the whole-cohort motif-level weight multiset. It does not preserve event-slot-weighted distributions or dependence on sparsity, entropy, phase or true clock. It uses other unlabeled histories in the evaluation cohort and is a transductive negative control, not a deployable forecast rule. No label or hidden truth enters assignment, and no map is redrawn to obtain a stronger disruption.

## Training and test contract

Each condition/block has 2,048 training motifs and 4,096 test motifs. Every retained candidate is scored; there is no negative sampling or future-positive filtering. Neural training uses h1–h4; evaluation uses h1–h8 at a fixed cutoff with no intermediate ingestion. The primary endpoint is equally weighted h5–h8 Brier.

All arms share histories, initial TGN parameters, minibatch schedules and fixed budgets within block: 16 epochs, 32 updates/epoch, batch size 64, 512 updates total, 131,072 target presentations, Adam learning rate .001, gradient clip norm 1, and no dropout or weight decay. Final epoch 16 is mandatory; no validation or test checkpoint selection. The clock-weight comparison requires all 192 fits complete before test generation; the weight-reassignment study requires all 48.

The clock-weight comparison includes all three sharing conditions. A one-condition reduced run is not implemented as a silent replacement for the 192-fit barrier. `--unit` allows completing one unit at a time but never changes the full population required for evaluation.

## Forecasts and metrics

Every evaluation retains neural arms, own-history hard/soft direct forecasts, context hard/soft direct forecasts and a known-law context reference. The primary contrast is fitted-soft minus equal in the clock-weight comparison, and fitted-soft minus support-preserving reassignment in the weight-reassignment study. The direct fitted-soft forecast is a mandatory comparator, including when it beats every TGN arm.

Brier is computed without clipping. Log loss alone clips at 1e-7. AUC is within horizon with half credit for exact ties; undefined single-class results remain missing. Motifs are averaged within horizon, horizons equally, then blocks equally. Paired pointwise 95% intervals resample all 16 blocks 10,000 times, preserving comparisons within each sampled block. Horizons, motifs and model arms are not independent fitted-model repetitions. There is no equivalence margin, multiplicity-adjusted discovery claim or scientific pass/fail threshold.

Report all blocks, horizons and declared contrasts; preserve conditional-support and uncertainty subgroups with counts. Do not use true-clock diagnostic subgroups as deployable routers. No outcome-dependent extra epochs, new seeds or discarded blocks are allowed within the recorded design.
