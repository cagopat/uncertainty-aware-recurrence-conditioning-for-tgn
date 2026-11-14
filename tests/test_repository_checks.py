"""Handwritten and fabricated-file tests; no scientific data or models."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repo_checks", ROOT / "scripts" / "check_repository.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class RepositoryChecks(unittest.TestCase):
    def write_fixture(self, directory):
        root = Path(directory)
        (root / "example.txt").write_text("handwritten test fixture\n", encoding="utf-8")
        (root / "FILE_MANIFEST.json").write_text(json.dumps({
            "schema_version": 1, "files": {"example.txt": checks.digest(root / "example.txt")}
        }), encoding="utf-8")
        return root

    def test_manifest_checks_hash_and_exact_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.write_fixture(temporary)
            self.assertEqual(checks.validate_manifest(root), 1)
            (root / "example.txt").write_text("altered fixture\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                checks.validate_manifest(root)

    def test_unmanifested_file_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.write_fixture(temporary)
            (root / "unexpected.txt").write_text("not covered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage differs"):
                checks.validate_manifest(root)

    def test_output_and_cache_files_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.write_fixture(temporary)
            (root / "results").mkdir()
            (root / "results" / "new_run.txt").write_text("fabricated result", encoding="utf-8")
            self.assertEqual(checks.validate_manifest(root), 1)

    def test_traversal_and_noncanonical_paths_rejected(self):
        for name in ("../outside", "/absolute", "a/../b", "a\\b", "a//b"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = self.write_fixture(temporary)
                (root / "FILE_MANIFEST.json").write_text(json.dumps({
                    "schema_version": 1, "files": {name: "0" * 64}
                }), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Invalid manifest"):
                    checks.validate_manifest(root)

    def test_publication_hygiene_detects_examples_without_printing_them(self):
        bad = ["/" + "Users/example/private/file.py",
               "gh" + "p_" + "a" * 36,
               "https:" + "//example:password@example.invalid/path"]
        for value in bad:
            with self.assertRaises(ValueError):
                checks.scan_text("fixture", value)
        checks.scan_text("fixture", "https://example.invalid/public and relative/file.py")

    def test_original_source_provenance_is_independent_of_public_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiments/cw1").mkdir(parents=True)
            (root / "experiments/sw1").mkdir(parents=True)
            (root / "configs").mkdir()
            science = root / "experiments/cw1/fixture.py"
            science.write_text("# handwritten source fixture\n", encoding="utf-8")
            for study in ("cw1", "sw1"):
                (root / "experiments" / study / "SPEC.json").write_text("{}\n", encoding="utf-8")
                (root / "configs" / f"{study}.json").write_text("{}\n", encoding="utf-8")
            (root / "SOURCE_PROVENANCE.json").write_text(json.dumps({
                "scientific_modules_modified": False,
                "files": {"experiments/cw1/fixture.py": {"byte_identical": True, "sha256": checks.digest(science)}}
            }), encoding="utf-8")
            self.assertEqual(checks.check_source_provenance(root), 1)
            science.write_text("# changed fixture\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Original scientific source changed"):
                checks.check_source_provenance(root)


if __name__ == "__main__":
    unittest.main()
