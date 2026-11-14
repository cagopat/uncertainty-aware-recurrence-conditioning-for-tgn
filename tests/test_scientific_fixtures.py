"""Run preserved handwritten fixtures with neural/scientific execution blocked.

These are software tests, not a small scientific replication. In particular,
the GPU preflight is not part of this suite.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NoNeuralFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("Neural-framework import is forbidden in the CPU fixture suite")
        return None


GUARD = NoNeuralFramework()
if "torch" in sys.modules:
    raise AssertionError("Start fixture tests in a fresh Python process without Torch")
sys.meta_path.insert(0, GUARD)


def forbidden(*args, **kwargs):
    raise AssertionError("Scientific generation, fitting, inference or training is forbidden in these tests")


BLOCKED_FUNCTIONS = {
    "cw1_data": {"generate", "calibration", "fitted_inputs"},
    "sw1_data": {"generate", "fitted_inputs"},
    "cw1_worker": {"assets", "load_training", "train", "evaluate", "main"},
    "sw1_worker": {"assets", "load_training", "train", "evaluate", "main"},
    "cw1_model": {"inherited_module", "make_model", "preflight_fixture"},
    "sw1_model": {"inherited_module", "make_model", "preflight_fixture"},
    "cw1_analysis": {"saved_audit", "direct_predictions"},
    "sw1_analysis": {"direct_predictions"},
    "fc1_core": {"fit_counts", "predict", "local_sensitivity", "objective", "stationarity", "em_update"},
}


def prevent_scientific_call(frame, event, arg):
    # Call-time protection also covers functions imported later or retained under
    # an alias before module attributes are replaced below.
    if event == "call":
        module = frame.f_globals.get("__name__", "")
        if frame.f_code.co_name in BLOCKED_FUNCTIONS.get(module, ()):
            forbidden()


sys.setprofile(prevent_scientific_call)


def block_loaded_entry_points():
    # Replace aliases too, not just attributes on the defining module. The
    # profile guard above is the final boundary for later/dynamic imports.
    for module in tuple(sys.modules.values()):
        if module is None or not hasattr(module, "__dict__"):
            continue
        for name, value in tuple(vars(module).items()):
            if isinstance(value, types.FunctionType) and value.__name__ in BLOCKED_FUNCTIONS.get(value.__module__, ()):
                setattr(module, name, forbidden)


def import_file(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite([standard_tests])
    selections = {
        "cw1": ["test_cw1_encoding", "test_cw1_analysis", "test_cw1_data", "test_cw1_worker"],
        "sw1": ["test_sw1_reassignment", "test_sw1_analysis"],
    }
    for study, names in selections.items():
        package = ROOT / "experiments" / study
        sys.path.insert(0, str(package))
        try:
            for name in names:
                fixture = package / "tests" / (name + ".py")
                if not fixture.is_file():
                    raise AssertionError(f"Missing preserved fixture file: {fixture.relative_to(ROOT)}")
                module = import_file("preserved_" + name, fixture)
                # Imports define APIs only. Disable scientific entry points before
                # unittest invokes any fixture, leaving pure array helpers intact.
                block_loaded_entry_points()
                suite.addTests(loader.loadTestsFromModule(module))
        finally:
            sys.path.remove(str(package))
    return suite
