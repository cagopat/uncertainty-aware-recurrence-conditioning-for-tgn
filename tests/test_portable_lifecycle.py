"""Storage/authorization lifecycle tests with a fake child, never a model worker."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("portable_lifecycle_runner", ROOT / "scripts/run_experiments.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeChild:
    """Handwritten REPORT writer that cannot import any scientific implementation."""
    returncode = 0

    def __init__(self, command, **kwargs):
        assert command[-1] == "--execute"
        assert command[-3] == "_worker"
        self.job = json.loads(Path(command[-2]).read_text())
        output = Path(self.job["output"])
        (output / "REPORT.json").write_text(json.dumps({"fixture_only": True, "unit": self.job["unit"]}))
        self.stdout = io.StringIO("handwritten storage fixture; no model executed\n")

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        pass


class FailedChild(FakeChild):
    returncode = 3


class PortableLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="portable_fixture_")
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "run"
        (self.run / "work").mkdir(parents=True)
        self.args = SimpleNamespace(run_dir=self.run, study="fresh", retry_failed=False)
        self.metadata = {"identity": {"kind": "handwritten_test_only", "software": {}}}
        self.phase, self.unit = "assets", "F05_B00"
        self.upstream = {"calibrate/F05_B00": "fixture_dependency_hash"}

    @property
    def unit_root(self):
        return self.run / "units" / self.phase / self.unit

    def execute(self, child=FakeChild):
        with patch.object(runner.subprocess, "Popen", child), redirect_stdout(io.StringIO()):
            runner.execute_unit(self.args, {}, self.metadata, self.phase, self.unit, self.upstream)

    def test_complete_then_skip_is_immutable_and_does_not_start_child(self):
        self.execute()
        complete = self.unit_root / "COMPLETE.json"
        original_bytes = complete.read_bytes()
        report = self.run / "work/results" / self.phase / self.unit / "REPORT.json"
        self.assertTrue(json.loads(report.read_text())["fixture_only"])
        with patch.object(runner.subprocess, "Popen", side_effect=AssertionError("Completed unit restarted")), redirect_stdout(io.StringIO()):
            runner.execute_unit(self.args, {}, self.metadata, self.phase, self.unit, self.upstream)
        self.assertEqual(complete.read_bytes(), original_bytes)
        self.assertEqual(len(list((self.unit_root / "attempts").iterdir())), 1)

    def test_tampered_completed_payload_fails_without_reexecution(self):
        self.execute()
        receipt = json.loads((self.unit_root / "COMPLETE.json").read_text())
        (self.unit_root / receipt["payload"] / "REPORT.json").write_text('{"tampered": true}')
        with patch.object(runner.subprocess, "Popen", side_effect=AssertionError("Tamper must not trigger rerun")), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Completed payload changed"):
                runner.execute_unit(self.args, {}, self.metadata, self.phase, self.unit, self.upstream)

    def test_tampered_restored_copy_is_not_overwritten(self):
        self.execute()
        report = self.run / "work/results" / self.phase / self.unit / "REPORT.json"
        changed = '{"changed_restored_copy": true}'
        report.write_text(changed)
        with patch.object(runner.subprocess, "Popen", side_effect=AssertionError("Tamper must not trigger rerun")), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Existing output differs"):
                runner.execute_unit(self.args, {}, self.metadata, self.phase, self.unit, self.upstream)
        self.assertEqual(report.read_text(), changed)

    def test_failure_is_preserved_and_retry_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError, "Worker failed"):
            self.execute(FailedChild)
        attempts = self.unit_root / "attempts"
        first = next(attempts.iterdir())
        failure_bytes = (first / "FAILURE.json").read_bytes()
        self.assertFalse((self.unit_root / "COMPLETE.json").exists())
        with self.assertRaisesRegex(RuntimeError, "Incomplete attempt exists"):
            self.execute()
        self.assertEqual(len(list(attempts.iterdir())), 1)
        self.args.retry_failed = True
        self.execute()
        self.assertEqual(len(list(attempts.iterdir())), 2)
        self.assertEqual((first / "FAILURE.json").read_bytes(), failure_bytes)
        self.assertTrue((self.unit_root / "COMPLETE.json").exists())

    def test_dependency_change_cannot_silently_skip(self):
        self.execute()
        changed = {**self.upstream, "calibrate/F05_B01": "new_dependency"}
        with patch.object(runner.subprocess, "Popen", side_effect=AssertionError("Dependency change must not rerun")), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "dependencies changed"):
                runner.execute_unit(self.args, {}, self.metadata, self.phase, self.unit, changed)


if __name__ == "__main__":
    unittest.main()
