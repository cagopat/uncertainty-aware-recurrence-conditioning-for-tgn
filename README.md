# Uncertainty-Aware Recurrence Conditioning for TGN

Reproduction code for two core synthetic comparisons: **fresh clock calibration** (CW1) and **support-preserving reassignment of clock weights** (SW1).

The intervention keeps several candidate periods available in TGN's attention-side time representation. Hard selection, equal weights and fitted weights are compared at the same input norm. A direct fitted probability forecast remains a mandatory baseline: improving a TGN representation does **not** establish an advantage over direct forecasting or prove that graph learning is necessary.

This repository contains the generator, population clock fitting, functional TGN host, training, evaluation, exact configurations and table generation. **No Google Drive folder or earlier experiment archive is required.** The sixteen original CW1 F05 calibration fits used by SW1 are bundled; they can also be regenerated.

## Setup

Use Python 3.13 and a CUDA GPU for neural training/evaluation. The recorded runs used Python 3.13.16, NumPy 2.4.6, pandas 2.2.3, SciPy 1.16.3, Torch 2.14.1+cu130, CUDA 13.0 and an NVIDIA L4. The original protocols also permit an A100-SXM4-40GB or Tesla T4 when selected before the run. See [environment and verification limits](docs/REPRODUCIBILITY.md).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-recorded.txt
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

The dependency file records the original stack; wheel availability on every platform is not guaranteed. Do not silently substitute a different stack and describe the result as exact replay. A deliberately changed environment can be recorded using the launcher's explicit environment-difference option.

## Rerun the main comparisons

Run these commands from the repository root. `plan` is read-only and does not generate data or train models. `--execute` explicitly authorizes computation.

### Fresh calibration and sharing conditions

```bash
python scripts/run_experiments.py fresh plan --run-dir runs/fresh
python scripts/run_experiments.py fresh all --run-dir runs/fresh --execute
```

This fits 48 population clock models, then 192 neural models: 16 blocks × three neighbor-sharing conditions × four arms. It generates held-out test cases only after all 192 neural fits are complete. The four arms are hard, equal, fitted-soft and fixed-period-four phase encoding. All direct probability baselines are evaluated too.

### Support-preserving shuffle with the original sixteen fits

```bash
python scripts/run_experiments.py shuffle plan --run-dir runs/shuffle
python scripts/run_experiments.py shuffle all --run-dir runs/shuffle --execute
```

This uses the bundled calibration estimates and trains 48 new neural models: 16 blocks × fitted-soft, equal and support-preserving reassigned weights. It reproduces the original **conditional-on-calibration** comparison without requiring the old CW1 archive.

To use the sixteen F05 fits from your new calibration run instead:

```bash
python scripts/run_experiments.py shuffle all --run-dir runs/shuffle-refitted \
  --calibration-run runs/fresh --execute
```

This is a new end-to-end rerun of the procedure, not byte-identical recovery of the original fitted records. Use a different output directory; the launcher binds the chosen inputs into the run identity.

### Status and tables

```bash
python scripts/run_experiments.py fresh status --run-dir runs/fresh
python scripts/run_experiments.py fresh tables --run-dir runs/fresh --execute
python scripts/run_experiments.py shuffle tables --run-dir runs/shuffle --execute
```

The `all` workflow includes table generation. Tables retain every arm and direct baseline, per-horizon and paired-block comparisons, and unfavorable results. `tables` uses saved evaluation arrays; it does not refit or retrain models. See [execution details](docs/REPRODUCIBILITY.md) for stage-by-stage use, verification and recovery.

## What is included

| Path | Contents |
|---|---|
| `experiments/cw1/`, `experiments/sw1/` | Unchanged scientific modules and modified TGL/DyGLib-derived functional host |
| `configs/` | Exact original scientific specifications, seeds and budgets |
| `inputs/cw1_f05/` | Sixteen authenticated original calibration fit records, selection mapping and provenance |
| `inputs/FC1_SPEC.json` | Original clock-fitting configuration required by fresh calibration |
| `scripts/` | Portable execution, verification and table commands; no Drive dependency |
| `tests/` | Handwritten technical fixtures, not scientific experiment runs |
| `docs/` | Protocol, scope and reproducibility boundaries |
| `SOURCE_PROVENANCE.json`, `FILE_MANIFEST.json` | Original source hashes and public-package integrity |

Only required synthetic fitted inputs are bundled. Neural checkpoints, raw prediction arrays and historical audit archives are not included; the workflow produces new ones. The repository is small enough for ordinary Git and does not need Git LFS for its initial contents.

## Scope and interpretation

This initial release covers the main fresh-calibration and support-preserving shuffle comparisons, **not every experiment in the project's history or every manuscript appendix table**. Earlier matched-magnitude comparisons (EM1), retained-calibration CW1 R05, flight/email studies, and the later dynamic-context DC1 study are outside this runnable release. See [scope](docs/SCOPE.md).

Training covers h1–h4 and evaluation covers h1–h8. With periods 2 and 4 and the raw horizon input neutralized, h and h+4 phase representations repeat by construction. Longer-horizon accuracy here is not evidence that TGN independently discovered a recurrence law. All motifs share the same three-node topology.

The public wrapper has software tests, but no new scientific experiment is run during repository preparation. It reuses the original scientific source unchanged. Original experiments and their saved-output audits are distinct from an independently completed run of this new portable wrapper.

## License and attribution

Project-authored contributions are MIT licensed. Inherited components retain their original notices; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This is a modified TGL/DyGLib-derived TGN host, not released KEAT-TGN and not an official TGB leaderboard implementation.
