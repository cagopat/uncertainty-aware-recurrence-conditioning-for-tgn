"""Portable wrapper contracts using fake metadata and handwritten summaries."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("portable_runner_fixture", "run_experiments.py")
renderer = load("table_renderer_fixture", "render_tables.py")


class RunnerContracts(unittest.TestCase):
    def test_descriptive_commands_match_compatibility_aliases(self):
        for canonical, alias, expected in (
                ("clock-weights", "fresh", "Clock-weight comparison"),
                ("weight-reassignment", "shuffle", "Support-preserving weight reassignment")):
            results = []
            for name in (canonical, alias):
                with contextlib.redirect_stdout(io.StringIO()) as captured, \
                     patch.object(runner, "prepare_run", side_effect=AssertionError("No run preparation")), \
                     patch.object(runner.importlib, "import_module", side_effect=AssertionError("No scientific imports")):
                    runner.main([name, "plan"])
                    results.append(json.loads(captured.getvalue()))
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0]["study"], canonical)
            self.assertEqual(results[0]["study_name"], expected)

    def test_plan_and_status_do_not_import_science_or_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-created"
            for phase in ("plan", "status"):
                with self.subTest(phase=phase), contextlib.redirect_stdout(io.StringIO()) as captured, \
                     patch.object(runner, "prepare_run", side_effect=AssertionError("No run preparation")), \
                     patch.object(runner.importlib, "import_module", side_effect=AssertionError("No scientific imports")):
                    runner.main(["fresh", phase, "--run-dir", str(output)])
                    self.assertIsInstance(json.loads(captured.getvalue()), dict)
                    self.assertFalse(output.exists())

    def test_scientific_stages_require_explicit_execution(self):
        for study in ("fresh", "shuffle"):
            for phase in ("calibrate", "assets", "preflight", "train", "evaluate", "tables", "all"):
                with self.subTest(study=study, phase=phase), \
                     patch.object(runner, "prepare_run", side_effect=AssertionError("No run preparation")), \
                     patch.object(runner, "run_lock", side_effect=AssertionError("No output creation")):
                    with self.assertRaisesRegex(RuntimeError, "No experiment was run"):
                        runner.main([study, phase])

    def test_full_workloads_not_silently_reduced(self):
        fresh = json.loads((ROOT / "configs/clock_weights.json").read_text())
        shuffle = json.loads((ROOT / "configs/weight_reassignment.json").read_text())
        self.assertEqual(len(runner.units("fresh", "calibrate", fresh)), 48)
        self.assertEqual(len(runner.units("fresh", "train", fresh)), 192)
        self.assertEqual(len(runner.units("shuffle", "train", shuffle)), 48)
        self.assertEqual(runner.units("shuffle", "calibrate", shuffle), [])

    def test_handwritten_fit_selection_keeps_lower_unqualified_endpoint(self):
        theta = [.5, .3, .3, .8, .2, .8, .75, .5]
        record = {"unit": "F05_B00", "calibration_size": 7680,
                  "selected_endpoint": 0, "selected_theta": theta,
                  "endpoints": [{"theta": theta, "mean_nll": 1., "feasible": True, "qualified": False},
                                {"theta": theta, "mean_nll": 2., "feasible": True, "qualified": True}]}
        runner.verify_fit(record, "F05_B00")
        record["selected_endpoint"] = 1
        with self.assertRaisesRegex(RuntimeError, "lowest finite feasible"):
            runner.verify_fit(record, "F05_B00")


class TableContracts(unittest.TestCase):
    def fixture(self):
        def metrics(value):
            return {name: {"mean": value} for name in ("brier", "logloss", "auc")}
        methods = {
            "neural_soft_fixed": {"h5_h8": metrics(.04), "per_horizon": {"5": metrics(.04)}},
            "fitted_context_soft": {"h5_h8": metrics(.03), "per_horizon": {"5": metrics(.03)}}}
        return {"conditions": {"F05": {"strata": {"all": {"methods": methods}},
                 "contrasts_h5_h8": {"soft_minus_equal": {
                     "mean": -.002, "interval_percentile95": [-.003, -.001], "negative_blocks": 16}}}}}

    def test_renderer_retains_interval_sign_and_direct_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "SUMMARY.json"
            source.write_text(json.dumps(self.fixture()), encoding="utf-8")
            output = root / "tables"
            renderer.render(source, output)
            text = (output / "TABLES.md").read_text()
            self.assertIn("fitted_context_soft", text)
            self.assertIn("-0.002000000", text)
            self.assertIn("[-0.003000000, -0.001000000]", text)
            self.assertIn("not learned extrapolation", text)
            before = (output / "TABLES.md").read_bytes()
            renderer.render(source, output)
            self.assertEqual((output / "TABLES.md").read_bytes(), before)

    def test_different_tables_do_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "SUMMARY.json"
            source.write_text(json.dumps(self.fixture()), encoding="utf-8")
            output = root / "tables"
            renderer.render(source, output)
            (output / "TABLES.md").write_text("preserved manually reviewed fixture", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Existing table differs"):
                renderer.render(source, output)
            self.assertEqual((output / "TABLES.md").read_text(), "preserved manually reviewed fixture")


if __name__ == "__main__":
    unittest.main()
