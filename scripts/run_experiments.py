#!/usr/bin/env python3
"""Run the clock-weight comparison and support-preserving weight reassignment.

Help, plan and status use only the standard library. Scientific work requires
--execute. This wrapper does not impersonate the original Colab run's identity.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {"fresh": ("F00", "F05", "F10"), "shuffle": ("SW05",)}
PREFIX = {"fresh": "clock_weights", "shuffle": "weight_reassignment"}
PUBLIC_NAMES = {"fresh": "clock-weights", "shuffle": "weight-reassignment"}
STUDY_NAMES = {"fresh": "Clock-weight comparison",
               "shuffle": "Support-preserving weight reassignment"}
STUDY_ALIASES = {**{name: key for key, name in PUBLIC_NAMES.items()},
                 "fresh": "fresh", "shuffle": "shuffle"}
PHASES = ("plan", "status", "preflight", "calibrate", "assets", "train", "evaluate", "tables", "all", "unlock")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def atomic_json(path, value):
    """Create immutable metadata; an identical existing record is permitted."""
    path = Path(path)
    if path.exists():
        require(read(path) == value, f"Existing metadata differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    # Locking serializes writers. Never overwrite an existing differing record.
    require(not path.exists(), f"Concurrent metadata creation: {path}")
    temp.rename(path)


def inventory(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"Symlink is not a portable evidence file: {path}")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            result[path.relative_to(root).as_posix()] = digest(path)
    return result


def copy_exact(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    require(source.is_file() and not source.is_symlink(), f"Missing/unsafe source: {source}")
    actual = digest(source)
    require(expected is None or actual == expected, f"Source hash changed: {source}")
    if destination.exists():
        require(destination.is_file() and not destination.is_symlink() and digest(destination) == actual,
                f"Existing output differs; preserve and inspect: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    require(digest(destination) == actual, f"Copy verification failed: {destination}")


def units(study, phase, spec):
    cohorts = [f"{condition}_B{block:02d}" for condition in CONDITIONS[study] for block in spec["blocks"]]
    if phase == "calibrate":
        return cohorts if study == "fresh" else []
    if phase in ("assets", "evaluate"):
        return cohorts
    if phase == "train":
        return [f"{unit}__{arm}" for unit in cohorts for arm in spec["arms"]]
    return ["all"]


def source_inventory(study):
    code = ROOT / "experiments" / PREFIX[study]
    require(code.is_dir(), f"Missing vendored source: {code}")
    result = {"experiments/" + PREFIX[study] + "/" + k: v for k, v in inventory(code).items()}
    for path in (ROOT / "configs" / (PREFIX[study] + ".json"), Path(__file__).resolve(),
                 ROOT / "scripts/render_tables.py"):
        result[path.relative_to(ROOT).as_posix()] = digest(path)
    if study == "fresh":
        result["inputs/clock_fitting_spec.json"] = digest(ROOT / "inputs/clock_fitting_spec.json")
    return result


def verify_fit(record, expected_unit):
    require(record["unit"] == expected_unit and record["calibration_size"] == 7680, "Calibration identity differs")
    endpoints = record["endpoints"]
    valid = [i for i, row in enumerate(endpoints) if row["feasible"] and
             row["mean_nll"] is not None and math.isfinite(row["mean_nll"])]
    require(valid, "No finite feasible calibration endpoint")
    selected = min(valid, key=lambda i: (endpoints[i]["mean_nll"], i))
    require(record["selected_endpoint"] == selected and record["selected_theta"] == endpoints[selected]["theta"],
            "Calibration selection is not the lowest finite feasible objective")
    require(len(record["selected_theta"]) == 8 and all(math.isfinite(x) for x in record["selected_theta"]),
            "Invalid calibration parameter vector")


def verify_selection(root):
    """Verify the bundled or newly built sixteen-fit handoff without inference."""
    root = Path(root)
    selected = read(root / "SELECTED_FITS.json")
    require(set(selected["blocks"]) == {f"B{b:02d}" for b in range(16)}, "Exactly sixteen clock fits are required")
    for block in range(16):
        key, unit = f"B{block:02d}", f"F05_B{block:02d}"
        row = selected["blocks"][key]
        fit_path = root / unit / "FIT.json"
        record = read(fit_path)
        verify_fit(record, unit)
        require(row["block"] == block and row["source_unit"] == unit and
                row["fit_sha256"] == digest(fit_path) and row["selected_theta"] == record["selected_theta"] and
                row["selected_endpoint"] == record["selected_endpoint"] and
                row["selected_qualified"] == record["selected_qualified"], "Selected-fit handoff differs: " + unit)
    return selected


def verify_unit(run, phase, unit, identity=None, restore=True):
    run = Path(run)
    root = run / "units" / phase / unit
    receipt_path = root / "COMPLETE.json"
    require(receipt_path.is_file(), f"Missing completed unit: {phase}/{unit}")
    receipt = read(receipt_path)
    expected_identity = read(run / "RUN.json")["identity"] if identity is None else identity
    require(receipt["identity"] == {"run": canonical_hash(expected_identity), "phase": phase, "unit": unit},
            f"Unit identity changed: {phase}/{unit}")
    relative = Path(receipt["payload"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe payload path")
    payload = root / relative
    require(inventory(payload) == receipt["files"] and (payload / "REPORT.json").is_file(),
            f"Completed payload changed: {phase}/{unit}")
    if restore:
        target = run / "work/results" / phase / unit
        for name, expected in receipt["files"].items():
            copy_exact(payload / name, target / name, expected)
        require(inventory(target) == receipt["files"], f"Unexpected restored files: {target}")
    return receipt, payload


def environment(spec, neural, allow_differences):
    current = {"python_major_minor": ".".join(platform.python_version_tuple()[:2]),
               "python_version": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "torch"):
        current[package] = importlib.metadata.version(package)
    expected = spec["environment"]
    differences = {key: {"current": value, "recorded": expected[key]}
                   for key, value in current.items() if key in expected and expected[key] != value}
    gpu = None
    if neural:
        import torch
        require(torch.cuda.is_available(), "A CUDA GPU is required for the unchanged neural workers")
        gpu = {"cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)}
        if gpu["cuda"] != expected["cuda"]:
            differences["cuda"] = {"current": gpu["cuda"], "recorded": expected["cuda"]}
        if gpu["device"] not in expected["allowed_devices"]:
            differences["device"] = {"current": gpu["device"], "recorded_allowed": expected["allowed_devices"]}
    require(not differences or allow_differences,
            "Runtime differs from the recorded configuration: " + json.dumps(differences) +
            ". Use --allow-environment-differences only for an explicitly labeled portability run.")
    return current, gpu, differences


@contextmanager
def run_lock(run):
    run.mkdir(parents=True, exist_ok=True)
    require(not run.is_symlink(), "Run directory cannot be a symlink")
    path = run / ".RUNNING.json"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError(f"Run lock exists: {path}. Stop other workers; use unlock --old-runtime-stopped only after checking.")
    token = uuid.uuid4().hex
    with handle:
        json.dump({"token": token, "pid": os.getpid(), "started_at": stamp()}, handle)
    try:
        yield
    finally:
        if path.is_file() and read(path).get("token") == token:
            target = run / "lock_history" / (stamp() + "_" + token + ".json")
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)


def prepare_run(args, spec, neural):
    run, work = args.run_dir, args.run_dir / "work"
    existing = read(run / "RUN.json") if (run / "RUN.json").is_file() else None
    require(not existing or not args.allow_environment_differences or existing["allow_environment_differences"],
            "Portability policy cannot change after run initialization; use a new run directory")
    allow = args.allow_environment_differences or bool(existing and existing["allow_environment_differences"])
    software, gpu, differences = environment(spec, neural, allow)
    source = source_inventory(args.study)
    if existing:
        require(existing["study"] == args.study and existing["identity"]["source_files"] == source,
                "Repository source/configuration differs from this run; use a new run directory")
        require(existing["identity"]["software"] == software, "Software changed within a run; restore the bound runtime")
        if args.calibration_run:
            require(existing.get("calibration_run") == str(args.calibration_run), "Calibration source changed within run")
        require(read(work / "SPEC.json") == spec, "Run specification changed")
        require(read(work / "IDENTITY.json") == existing["identity"], "Run identity changed")
        for name, expected in existing["identity"]["input_files"].items():
            require(digest(work / "source" / name) == expected, "Bound input changed: " + name)
        metadata = existing
    else:
        work.mkdir(parents=True, exist_ok=True)
        require(not (work / "IDENTITY.json").exists(), "Unrecognized partial setup; preserve it and use a new run directory")
        if args.study == "fresh":
            copy_exact(ROOT / "inputs/clock_fitting_spec.json", work / "source/CLOCK_FITTING_SPEC.json")
            require(read(work / "source/CLOCK_FITTING_SPEC.json")["optimizer"] == spec["calibration_fit"], "Clock-fitting optimizer mismatch")
        else:
            if args.calibration_run:
                selected = {"source_study": "CW1", "source_condition": "F05", "calibration_size": 7680,
                            "source_kind": "new_portable_calibration_run", "blocks": {}}
                parent = read(args.calibration_run / "RUN.json")
                require(parent["study"] == "fresh", "--calibration-run must point to a fresh-calibration run")
                for block in range(16):
                    unit = f"F05_B{block:02d}"
                    receipt, payload = verify_unit(args.calibration_run, "calibrate", unit, restore=False)
                    fit = read(payload / "FIT.json")
                    verify_fit(fit, unit)
                    copy_exact(payload / "FIT.json", work / "source" / unit / "FIT.json")
                    selected["blocks"][f"B{block:02d}"] = {
                        "block": block, "source_unit": unit, "fit_sha256": digest(payload / "FIT.json"),
                        "selected_theta": fit["selected_theta"], "selected_endpoint": fit["selected_endpoint"],
                        "selected_qualified": fit["selected_qualified"], "selected_residual": fit["selected_residual"],
                        "portable_receipt_sha256": canonical_hash(receipt)}
                atomic_json(work / "source/SELECTED_FITS.json", selected)
            else:
                bundled = ROOT / "inputs/half_shared_clock_fits"
                verify_selection(bundled)
                for path in (bundled / "SELECTED_FITS.json", bundled / "provenance.json"):
                    copy_exact(path, work / "source" / path.name)
                for block in range(16):
                    unit = f"F05_B{block:02d}"
                    copy_exact(bundled / unit / "FIT.json", work / "source" / unit / "FIT.json")
            verify_selection(work / "source")
        identity = {"kind": "portable_reproduction_v1", "study": args.study,
                    "scientific_study": spec["study"], "source_files": source, "software": software,
                    "input_files": inventory(work / "source"), "not_original_colab_identity": True,
                    "fresh_scope": list(CONDITIONS[args.study])}
        metadata = {"study": args.study, "identity": identity, "created_at": stamp(),
                    "allow_environment_differences": allow, "initial_environment_differences": differences,
                    "calibration_run": str(args.calibration_run) if args.calibration_run else None}
        copy_exact(ROOT / "configs" / (PREFIX[args.study] + ".json"), work / "SPEC.json")
        atomic_json(work / "IDENTITY.json", identity)
        atomic_json(run / "RUN.json", metadata)
    if gpu is not None:
        atomic_json(run / "GPU.json", gpu)
    return metadata


def dependencies(study, phase):
    if phase in ("calibrate", "preflight"):
        return []
    result = ["calibrate"] if study == "fresh" else []
    if phase in ("train", "evaluate", "tables"):
        result += ["assets", "preflight"]
    if phase in ("evaluate", "tables"):
        result += ["train"]
    if phase == "tables":
        result += ["evaluate"]
    return result


def execute_unit(args, spec, metadata, phase, unit, upstream):
    run, work = args.run_dir, args.run_dir / "work"
    root = run / "units" / phase / unit
    if (root / "COMPLETE.json").exists():
        receipt, _ = verify_unit(run, phase, unit, metadata["identity"])
        require(receipt["inputs"] == upstream, "Completed unit dependencies changed")
        print(json.dumps({"phase": phase, "unit": unit, "status": "verified_skip"}), flush=True)
        return
    attempts = root / "attempts"
    require(args.retry_failed or not attempts.exists() or not any(attempts.iterdir()),
            f"Incomplete attempt exists at {attempts}; preserve and inspect before --retry-failed")
    attempt = stamp() + "_" + uuid.uuid4().hex[:10]
    directory = attempts / attempt
    payload = directory / "payload"
    payload.mkdir(parents=True)
    job = {"study": args.study, "phase": phase, "unit": unit, "work": str(work),
           "output": str(payload), "resume_root": str(run / "checkpoints" / unit)}
    atomic_json(directory / "JOB.json", job)
    atomic_json(directory / "RESERVATION.json", {"identity": metadata["identity"], "inputs": upstream,
                                                  "started_at": stamp(), "attempt": attempt})
    command = [sys.executable, str(Path(__file__).resolve()), "_worker", str(directory / "JOB.json"), "--execute"]
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    try:
        with (directory / "worker.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, env=env, cwd=ROOT)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        require(code == 0 and (payload / "REPORT.json").is_file(), f"Worker failed; inspect {directory / 'worker.log'}")
        receipt = {"identity": {"run": canonical_hash(metadata["identity"]), "phase": phase, "unit": unit},
                   "inputs": upstream, "payload": payload.relative_to(root).as_posix(),
                   "files": inventory(payload), "completed_at": stamp(), "attempt": attempt}
        atomic_json(root / "COMPLETE.json", receipt)
        verify_unit(run, phase, unit, metadata["identity"])
        print(json.dumps({"phase": phase, "unit": unit, "status": "complete"}), flush=True)
    except BaseException as error:
        atomic_json(directory / "FAILURE.json", {"error": repr(error), "at": stamp(), "evidence_preserved": True})
        raise


def worker(job):
    """Called only by an explicitly authorized child process."""
    from check_repository import check
    check(ROOT)
    study, phase = job["study"], job["phase"]
    prefix = PREFIX[study]
    sys.path.insert(0, str(ROOT / "experiments" / prefix))
    work, output = Path(job["work"]), Path(job["output"])
    spec = read(work / "SPEC.json")
    data = importlib.import_module(prefix + "_data")
    data.validate_spec(spec)
    if phase == "calibrate":
        require(study == "fresh", "Shuffle cannot refit clocks")
        data.calibration(work, output, job["unit"], spec)
    elif phase == "preflight":
        model = importlib.import_module(prefix + "_model")
        result = model.preflight_fixture("cuda")
        atomic_json(output / "REPORT.json", {"passed": True, "fixture": result, "optimizer_steps": 0,
                                              "scientific_cohorts_generated": False})
    elif phase == "tables":
        analysis = importlib.import_module(prefix + "_analysis")
        analysis.analyze(work, output, spec, cohorts=list(CONDITIONS[study]))
        from render_tables import render
        render(output / "SUMMARY.json", output / "tables")
    else:
        module = importlib.import_module(prefix + "_worker")
        if phase == "train":
            module.train(work, output, job["unit"], spec, Path(job["resume_root"]))
        else:
            getattr(module, phase)(work, output, job["unit"], spec)


def plan(study, spec):
    return {"study": PUBLIC_NAMES[study], "study_name": STUDY_NAMES[study],
            "original_protocol_id": spec["study"], "conditions": list(CONDITIONS[study]),
            "calibration_fits": len(units(study, "calibrate", spec)),
            "neural_fits": len(units(study, "train", spec)), "epochs_per_fit": spec["epochs"],
            "train_horizons": spec["training_horizons"], "evaluation_horizons": spec["evaluation_horizons"],
            "arms": spec["arms"], "training_pairs_per_cohort": spec["train_pairs"],
            "test_pairs_per_cohort": spec["evaluation_pairs"], "config": "configs/" + PREFIX[study] + ".json",
            "execution": False, "excludes_earlier_retained_calibration_and_magnitude_studies": study == "fresh",
            "calibration_source_default": "sixteen original half-shared-neighbor clock fits" if study == "shuffle" else None,
            "warning": "h5-h8 repeats the supplied four-step symmetry, not learned extrapolation"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        require(len(argv) == 3 and argv[2] == "--execute", "Internal workers require explicit execution authorization")
        worker(read(argv[1]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=tuple(STUDY_ALIASES),
                        help="clock-weights or weight-reassignment; fresh/shuffle remain compatibility aliases")
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--run-dir", type=Path, help="New or resumed output folder; defaults to runs/STUDY")
    parser.add_argument("--execute", action="store_true", help="Explicitly enable requested scientific/fixture stages")
    parser.add_argument("--unit", help="One declared unit; never reduces the all-fit-before-test barrier")
    parser.add_argument("--calibration-run", type=Path,
                        help="Weight reassignment only: use all sixteen half-shared-neighbor fits from a clock-weight run")
    parser.add_argument("--allow-environment-differences", action="store_true",
                        help="Record a portability run; never silently claim the original software/hardware")
    parser.add_argument("--retry-failed", action="store_true", help="Preserve prior attempts and explicitly retry inspected failures")
    parser.add_argument("--old-runtime-stopped", action="store_true", help="Required only to preserve and release a stale lock")
    args = parser.parse_args(argv)
    args.study = STUDY_ALIASES[args.study]
    args.run_dir = (args.run_dir or ROOT / "runs" / PUBLIC_NAMES[args.study]).expanduser().resolve()
    if args.calibration_run:
        args.calibration_run = args.calibration_run.expanduser().resolve()
        require(args.study == "shuffle", "--calibration-run applies only to shuffle")
    require(args.run_dir != ROOT and ROOT not in args.run_dir.parents or args.run_dir.is_relative_to(ROOT / "runs"),
            "Inside the repository, write results only under runs/")
    require(not ROOT.is_relative_to(args.run_dir), "Run directory cannot be a parent of the repository")
    spec = read(ROOT / "configs" / (PREFIX[args.study] + ".json"))
    if args.phase == "plan":
        print(json.dumps(plan(args.study, spec), indent=2))
        return
    if args.phase == "status":
        counts = {}
        for phase in ("calibrate", "assets", "preflight", "train", "evaluate", "tables"):
            expected = units(args.study, phase, spec)
            counts[phase] = {"expected": len(expected), "recorded_complete": sum(
                (args.run_dir / "units" / phase / unit / "COMPLETE.json").is_file() for unit in expected)}
        print(json.dumps({"run_dir": str(args.run_dir), "counts": counts,
                          "lock_present": (args.run_dir / ".RUNNING.json").exists(),
                          "status_only": True, "payload_hashes_rechecked": False}, indent=2))
        return
    if args.phase == "unlock":
        require(args.old_runtime_stopped, "First stop all workers; then explicitly supply --old-runtime-stopped")
        path = args.run_dir / ".RUNNING.json"
        require(path.is_file() and not path.is_symlink(), "No regular run lock found")
        preserved = args.run_dir / "lock_history" / (stamp() + "_explicit_stale_recovery.json")
        preserved.parent.mkdir(parents=True, exist_ok=True)
        path.rename(preserved)
        print("Preserved stale lock:", preserved)
        return
    require(args.execute, "No experiment was run. Supply --execute after reviewing the protocol and printed plan.")
    from check_repository import check
    check(ROOT)
    require(not (args.phase == "calibrate" and args.study == "shuffle"), "Shuffle retains its clock fits; there is no calibration stage")
    require(not (args.phase == "all" and args.unit), "--unit is a phase-level recovery control, not an all-study subset")
    stages = ["preflight"] + (["calibrate"] if args.study == "fresh" else []) + ["assets", "train", "evaluate", "tables"]
    stages = stages if args.phase == "all" else [args.phase]
    with run_lock(args.run_dir):
        for phase in stages:
            metadata = prepare_run(args, spec, neural=phase in ("preflight", "train", "evaluate"))
            upstream = {}
            for dependency in dependencies(args.study, phase):
                for unit in units(args.study, dependency, spec):
                    receipt, _ = verify_unit(args.run_dir, dependency, unit, metadata["identity"])
                    upstream[dependency + "/" + unit] = canonical_hash(receipt)
            selected = units(args.study, phase, spec)
            if args.unit:
                require(args.unit in selected, "Unit not in frozen workload: " + args.unit)
                selected = [args.unit]
            for unit in selected:
                execute_unit(args, spec, metadata, phase, unit, upstream)


if __name__ == "__main__":
    main()
