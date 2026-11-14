#!/usr/bin/env python3
"""Read-only release checks. No scientific implementation is imported or run."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", "runs", "results", "logs", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".cff"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def repository_files(root):
    root = Path(root)
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise ValueError(f"Repository symlink is not allowed: {relative}")
        if path.is_file() and relative.as_posix() != "FILE_MANIFEST.json":
            yield relative.as_posix(), path


def validate_manifest(root):
    root = Path(root)
    manifest = json.loads((root / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Expected schema_version 1 and a relative-path-to-SHA256 files object")
    declared = manifest["files"]
    for name, expected in declared.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or "\\" in name
                or path.as_posix() != name or not re.fullmatch(r"[0-9a-f]{64}", str(expected))):
            raise ValueError(f"Invalid manifest entry: {name}")
    actual = dict(repository_files(root))
    if set(actual) != set(declared):
        raise ValueError(f"Manifest coverage differs; missing={sorted(set(actual)-set(declared))}; "
                         f"unexpected={sorted(set(declared)-set(actual))}")
    for name, path in actual.items():
        if digest(path) != declared[name]:
            raise ValueError(f"File hash changed: {name}")
    return len(actual)


def scan_text(name, text):
    """Reject private machine paths and recognisable credential forms, not author names."""
    checks = {
        "private home path": r"/(?:Users|home)/[A-Za-z0-9_.-]+/",
        "GitHub credential": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
        "AWS access key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "credential-bearing URL": r"https?://[^\s/@:]+:[^\s/@]+@",
    }
    for label, pattern in checks.items():
        if re.search(pattern, text):
            raise ValueError(f"Potential {label} in {name}; inspect before publication")


def check_inputs(root):
    root = Path(root)
    fits = root / "inputs" / "cw1_f05"
    expected = {f"F05_B{block:02d}" for block in range(16)}
    actual = {p.name for p in fits.iterdir() if p.is_dir()}
    if actual != expected:
        raise ValueError("The pinned input bundle must contain exactly F05_B00 through F05_B15")
    selected = json.loads((fits / "SELECTED_FITS.json").read_text(encoding="utf-8"))
    provenance = json.loads((fits / "provenance.json").read_text(encoding="utf-8"))
    if set(selected.get("blocks", {})) != {f"B{block:02d}" for block in range(16)}:
        raise ValueError("Selected-fit map must contain all sixteen blocks")
    if set(provenance.get("fits", {})) != expected:
        raise ValueError("Fit provenance must identify all sixteen source units")
    for unit in sorted(expected):
        path = fits / unit / "FIT.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("unit") != unit:
            raise ValueError(f"Fit identity mismatch: {unit}")
        if record.get("calibration_size") != 7680:
            raise ValueError(f"Unexpected calibration sample size: {unit}")
        theta = record.get("selected_theta")
        if not isinstance(theta, list) or len(theta) != 8:
            raise ValueError(f"Missing eight-coordinate selected fit: {unit}")
        mapped = selected["blocks"][unit.removeprefix("F05_")]
        origin = provenance["fits"][unit]
        if (mapped.get("source_unit") != unit or mapped.get("selected_theta") != theta
                or mapped.get("selected_endpoint") != record.get("selected_endpoint")
                or mapped.get("selected_qualified") != record.get("selected_qualified")
                or mapped.get("fit_sha256") != digest(path)
                or origin.get("sha256") != digest(path)
                or origin.get("path") != path.relative_to(root).as_posix()):
            raise ValueError(f"Selected-fit/provenance mapping differs: {unit}")
    for name in ("SELECTED_FITS.json", "provenance.json"):
        if not (fits / name).is_file():
            raise ValueError(f"Missing fit-bundle provenance: {name}")
    if not (root / "inputs" / "FC1_SPEC.json").is_file():
        raise ValueError("Missing original clock-fitting specification")
    return len(expected)


def check_source_provenance(root):
    root = Path(root)
    provenance = json.loads((root / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
    records = provenance.get("files", {})
    if provenance.get("scientific_modules_modified") is not False or not records:
        raise ValueError("Scientific source must retain its original provenance")
    for name, record in records.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or "\\" in name
                or path.as_posix() != name or not name.startswith("experiments/")):
            raise ValueError(f"Unsafe source-provenance path: {name}")
        if record.get("byte_identical") is not True or digest(root / name) != record.get("sha256"):
            raise ValueError(f"Original scientific source changed: {name}")
    required = {path.relative_to(root).as_posix()
                for path in (root / "experiments").rglob("*.py")
                if "tests" not in path.relative_to(root).parts and "__pycache__" not in path.parts}
    if not required <= set(records):
        raise ValueError(f"Scientific source missing provenance: {sorted(required-set(records))}")
    for study in ("cw1", "sw1"):
        if digest(root / "configs" / f"{study}.json") != digest(root / "experiments" / study / "SPEC.json"):
            raise ValueError(f"Public configuration differs from original specification: {study}")
    return len(records)


def check(root):
    root = Path(root).resolve()
    count = validate_manifest(root)
    parsed = 0
    scanned = 0
    for name, path in repository_files(root):
        if path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE", "NOTICE", ".gitignore"}:
            text = path.read_text(encoding="utf-8")
            scan_text(name, text)
            scanned += 1
            if path.suffix == ".py":
                ast.parse(text, filename=name)
                parsed += 1
    fit_count = check_inputs(root)
    source_count = check_source_provenance(root)
    for study in ("cw1", "sw1"):
        spec = json.loads((root / "configs" / f"{study}.json").read_text(encoding="utf-8"))
        if spec.get("blocks") != list(range(16)) or spec.get("epochs") != 16:
            raise ValueError(f"Unexpected declared full-study budget: {study}")
        if spec.get("training_horizons") != [1, 2, 3, 4] or spec.get("primary_horizons") != [5, 6, 7, 8]:
            raise ValueError(f"Unexpected horizon contract: {study}")
    return {"passed": True, "manifest_files": count, "python_files_parsed": parsed,
            "text_files_scanned": scanned, "bundled_fits": fit_count,
            "original_source_files_verified": source_count,
            "scientific_code_imported": False, "experiments_executed": False,
            "scope": "Source/fit-bundle integrity, syntax and basic publication hygiene; not model replay or scientific validation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        report = check(args.repository)
    except (OSError, ValueError, SyntaxError, json.JSONDecodeError) as error:
        print(json.dumps({"passed": False, "error": str(error), "experiments_executed": False}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
