"""Mocked storage/barrier tests and static training contracts; no neural work."""
import ast
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import clock_weights_worker as w


def spec_fixture():
    return {'arms': list(w.ARMS), 'blocks': list(range(16)), 'epochs': 16,
            'conditions': [{'id': c, 'group': 'retained' if c == 'R05' else 'fresh'}
                           for c in ('R05', 'F00', 'F05', 'F10')]}


def report_fixture(path):
    fit = path.parent.name
    return {'fit': fit, 'arm': fit.split('__')[1], 'epoch': 16, 'selection': 'final_epoch_only'}


class WorkerContracts(unittest.TestCase):
    def test_retained_barrier_does_not_require_fresh_fits(self):
        reads = []
        def read(path):
            reads.append(path)
            return report_fixture(path)
        with mock.patch.object(Path, 'is_file', return_value=True), mock.patch.object(w, 'read_json', side_effect=read):
            got = w.evaluation_group_barrier(Path('/mock/work'), 'R05_B00', spec_fixture())
        self.assertEqual(got, {'group': 'retained', 'completed_fit_reports': 64})
        self.assertEqual(len(reads), 64)
        self.assertTrue(all(p.parent.name.startswith('R05_') for p in reads))

    def test_fresh_barrier_requires_every_rho_fit(self):
        reads = []
        def read(path):
            reads.append(path)
            return report_fixture(path)
        with mock.patch.object(Path, 'is_file', return_value=True), mock.patch.object(w, 'read_json', side_effect=read):
            got = w.evaluation_group_barrier(Path('/mock/work'), 'F05_B03', spec_fixture())
        self.assertEqual(got['completed_fit_reports'], 192)
        self.assertEqual({p.parent.name[:3] for p in reads}, {'F00', 'F05', 'F10'})
        missing = '/mock/work/results/train/F10_B15__period4_fixed/REPORT.json'
        with mock.patch.object(Path, 'is_file', lambda p: str(p) != missing), mock.patch.object(w, 'read_json', side_effect=report_fixture):
            with self.assertRaisesRegex(RuntimeError, 'All192'):
                w.evaluation_group_barrier(Path('/mock/work'), 'F00_B00', spec_fixture())

    def test_barrier_rejects_incomplete_epoch(self):
        def read(path):
            value = report_fixture(path); value['epoch'] = 15
            return value
        with mock.patch.object(Path, 'is_file', return_value=True), mock.patch.object(w, 'read_json', side_effect=read):
            with self.assertRaisesRegex(RuntimeError, 'final-epoch'):
                w.evaluation_group_barrier(Path('/mock/work'), 'R05_B00', spec_fixture())

    def test_canary_failure_preserves_raw_predictions_before_stopping(self):
        p = np.zeros((2, 8), np.float64); p[1, 7] = .1
        saves, jsons = [], []
        def savez(path, **values):
            saves.append((path.name, values))
        def write(path, value):
            jsons.append((path.name, value))
        with mock.patch.object(w.np, 'savez_compressed', side_effect=savez), \
             mock.patch.object(w, 'sha', return_value='a'*64), \
             mock.patch.object(w, 'write_json', side_effect=write):
            with self.assertRaisesRegex(RuntimeError, 'symmetry canary failed'):
                w.save_neural_evidence(Path('/mock/out'), 'equal_fixed', 'b'*64,
                                      p.astype(np.float32), p, {'valid': np.ones((1, 1), bool)},
                                      np.arange(2), list(range(1, 9)), [], 1e-5)
        self.assertEqual(saves[0][0], 'NEURAL_equal_fixed_RAW.npz')
        np.testing.assert_array_equal(saves[0][1]['probability'], p)
        self.assertTrue(any('CYCLE_FAILURE' in name for name, _ in saves))
        self.assertFalse(jsons[-1][1]['completion_authorized'])

    def test_static_training_budget_resume_and_no_top_level_science(self):
        text = (ROOT/'clock_weights_worker.py').read_text()
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(n.name.startswith('torch') for n in node.names))
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self.fail('Top-level worker execution')
        for contract in ("model_seed(spec, unit)", "schedule(spec, unit, epoch)",
                         "'torch_rng': torch.get_rng_state().cpu()", "'cuda_rng':",
                         "optimizer.load_state_dict(saved['optimizer'])", "weights_only=True",
                         "saved['identity'] == identity", "'selection': 'final_epoch_only'"):
            self.assertIn(contract, text)
        evaluator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
        calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(evaluator) if isinstance(n, ast.Call)]
        barrier = min(line for line, name in calls if name == 'evaluation_group_barrier')
        generation = min(line for line, name in calls if name == 'generate')
        self.assertLess(barrier, generation)
        self.assertIn("true_theta(unit)", text)
        self.assertIn("fitted_clock_posterior=fitted['posterior_context']", text)


if __name__ == '__main__':
    unittest.main()
