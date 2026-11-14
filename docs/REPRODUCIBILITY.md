# Execution and reproducibility

## Two kinds of reproduction

1. **Rerun the scientific procedure:** regenerate synthetic observations, fit the clock model, train TGN and evaluate. The clock-weight comparison does this. Using `--calibration-run` also makes the weight-reassignment study depend on newly fitted clocks.
2. **Repeat the original conditional comparison:** train/evaluate the weight-reassignment study with its sixteen original clock fits. This is its default workflow. It does not independently reproduce the calibration fits themselves.

Neither is a promise of byte-identical neural scores on different hardware. No previous model checkpoint or prediction is downloaded by these commands. The bundled fits are scientific inputs, not hidden neural predictions.

## Environment

| Component | Original recorded value |
|---|---|
| Python | 3.13.16 |
| NumPy | 2.4.6 |
| pandas | 2.2.3 |
| SciPy | 1.16.3 |
| Torch | 2.14.1+cu130 |
| CUDA | 13.0 |
| GPU | NVIDIA L4 |

`requirements-recorded.txt` describes that software stack. Installation and CUDA driver compatibility depend on the user's machine; package availability was not established by installing the scientific environment during repository preparation. Do not silently substitute a version when installation fails. Use an appropriate documented environment or explicitly label a portability run.

The launcher compares software against the scientific configuration. `--allow-environment-differences` records departures for a **new** portability run; it does not allow changing software or GPU midway through an existing run. The original protocols permit L4, A100-SXM4-40GB and T4 selected before neural work. Exact GPU identity is bound when the first neural fixture/training stage begins. The full Python version is recorded separately from the original major/minor compatibility rule.

Calibration uses CPU calculations. Neural preflight, training and evaluation require CUDA because the original worker logic explicitly uses it. Table analysis needs the scientific numerical environment but no GPU; standalone rendering of an existing `SUMMARY.json` uses only the Python standard library:

```bash
python scripts/render_tables.py runs/clock-weights/work/results/tables/all/SUMMARY.json \
  --output formatted-tables
```

## Stage-by-stage workflow

The examples below are alternatives to `all`; they do not change the study sizes.

```bash
python scripts/run_experiments.py clock-weights calibrate --run-dir runs/clock-weights --execute
python scripts/run_experiments.py clock-weights assets --run-dir runs/clock-weights --execute
python scripts/run_experiments.py clock-weights preflight --run-dir runs/clock-weights --execute
python scripts/run_experiments.py clock-weights train --run-dir runs/clock-weights --execute
python scripts/run_experiments.py clock-weights evaluate --run-dir runs/clock-weights --execute
python scripts/run_experiments.py clock-weights tables --run-dir runs/clock-weights --execute
```

Weight reassignment follows the same sequence without `calibrate`. Its default calibration inputs are bundled. A new `--calibration-run` must provide all sixteen completed half-shared-neighbor units, not a selected subset. The original fitting function and objective are used; the public wrapper does not fabricate old Colab receipts.

The clock-weight comparison's `all` command performs 48 clock fits and 192 neural fits, not a small smoke test. Weight reassignment performs 48 neural fits. Review `plan` and storage before authorizing either. Training retains all sixteen epoch checkpoints, final models, optimizer/RNG state written by the original worker, input arrays, predictions, logs and completion hashes. There is no automatic checkpoint deletion. Allow tens of GB of output space. Runtime depends on GPU, numerical fitting and storage speed; no new speed benchmark was run for this release.

The original launcher names `fresh` and `shuffle` remain accepted aliases for `clock-weights` and `weight-reassignment`. New documentation uses the descriptive names. Scientific condition and unit keys inside saved records retain their original values so they remain traceable to the experiment evidence; the mapping is in [scope](SCOPE.md#historical-identifier-map).

**Existing runs are bound to their source version.** The aliases do not make an output directory created by the first public commit compatible with the renamed code. Complete those runs using commit `b4c836b`, or start the renamed release in a new run directory. Do not rewrite receipts or change saved hashes to force a resume. The naming update preserves scientific algorithms, seed namespaces and checkpoint attributes; it still changes the public source identity.

## Interruption and retry

Calling a completed stage again verifies its completion receipt and file hashes, then skips it. It does not silently retrain completed fits. Status counts recorded completion pointers and explicitly does not rehash every payload.

An incomplete attempt stops automatic progress. Inspect its `worker.log` and `FAILURE.json` before an explicit retry. For example:

```bash
python scripts/run_experiments.py clock-weights train --run-dir runs/clock-weights \
  --unit F05_B00__soft_fixed --execute --retry-failed
```

The original worker restores verified contiguous completed epochs with optimizer and RNG states; an incomplete epoch is not considered complete. The public wrapper's integration has not been established by an independent GPU interruption/replay experiment. Do not delete locks or alter checkpoint receipts to bypass checks.

After a dead runtime, first confirm that no other worker remains active. A stale lock can then be preserved and released explicitly:

```bash
python scripts/run_experiments.py clock-weights unlock --run-dir runs/clock-weights --old-runtime-stopped
```

The lock is moved into the run's history, not discarded. `--unit` never reduces the all-fits-before-test requirement. Changing configuration, code or inputs requires a new run directory; do not mix results across identities.

## Output layout

| Location under a run directory | Meaning |
|---|---|
| `RUN.json`, `GPU.json` | Reproduction identity, software/input/code hashes and selected GPU |
| `units/<phase>/<unit>/attempts/` | Preserved attempts, command logs and immutable payloads |
| `units/<phase>/<unit>/COMPLETE.json` | Successful unit identity, input dependencies and file hashes |
| `checkpoints/<fit>/epoch_*/` | Original worker's checkpoint publications |
| `work/results/` | Verified working copies for the preserved scientific functions |
| `work/results/tables/all/` | Summary, block metrics, bootstrap indices and diagnostics |
| `work/results/tables/all/tables/` | Readable Markdown and CSV tables |

## Technical validation without experiments

```bash
python -m pip install -r requirements-test.txt
python scripts/check_repository.py
python -m unittest discover -s tests -p 'test_*.py'
```

These are CPU-only software checks using handwritten arrays and fabricated workflow records. They block Torch import and scientific generator/fitter/training entry points. They test phase norms, metric ties, reassignment support, barriers, immutable inputs and table formatting. They are **not** a new scientific replication, model-quality test or assurance of neural convergence.

CI runs this technical suite, not GPU training. Repository preparation leaves all historical evidence unchanged. `SOURCE_PROVENANCE.json` records original scientific source hashes and the name-only module/import transformations; `FILE_MANIFEST.json` covers the distributable package. After an intentional repository edit, maintainers can update the latter with `python scripts/update_manifest.py --write`, rerun the checks and commit the change. Updating a manifest does not make a changed scientific protocol equivalent to the original.
