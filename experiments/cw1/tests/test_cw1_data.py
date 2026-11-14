"""Handmade arithmetic and namespace fixtures; no generation/fitting."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cw1_data import (ARMS,block_number,split_fit,stream_seed,model_seed,
    labels_for_clock,recording_transform,budget,selected_fit,true_theta,validate_spec)
from cw1_io import write_json


class DataTests(unittest.TestCase):
    def test_spec_and_unit_contract(self):
        spec=json.loads((ROOT/'SPEC.json').read_text())
        validate_spec(spec)
        for arm in ARMS:
            self.assertEqual(split_fit('F10_B15__'+arm),('F10_B15',arm))
        for bad in ['B00','F20_B00','F05_B16','F05_B1','../F05_B01']:
            with self.assertRaises(RuntimeError):block_number(bad)
        spec['base_seed']+=1
        with self.assertRaises(RuntimeError):validate_spec(spec)
        spec=json.loads((ROOT/'SPEC.json').read_text())
        spec['calibration_fit']['starts'][0][0]=.51
        with self.assertRaises(RuntimeError):validate_spec(spec)

    def test_fresh_common_random_namespace_retained_separate(self):
        spec={'namespace':'CW1/1.0.0','base_seed':2026111201}
        for component in ['candidate_clock','independent_neighbor_clock','sharing','own_records','neighbor_records']:
            values=[stream_seed(spec,c+'_B02','calibration',component) for c in ['F00','F05','F10']]
            self.assertEqual(len(set(values)),1)
            self.assertNotEqual(values[0],stream_seed(spec,'R05_B02','calibration',component))
        self.assertEqual(model_seed(spec,'F00_B02'),model_seed(spec,'F10_B02'))
        self.assertNotEqual(stream_seed(spec,'F05_B02','train','own'),stream_seed(spec,'F05_B02','evaluation','own'))

    def test_handmade_labels_and_recording(self):
        labels=labels_for_clock(np.array([0,2]),np.arange(1,9))
        self.assertEqual(labels.tolist(),[[1,0,1,0,1,0,1,0],[1,0,0,0,1,0,0,0]])
        x,y=recording_transform(np.array([0,0]),np.array([2,2]),np.array([0,2]),np.array([0,1]),np.zeros((2,16)),np.zeros((2,16)))
        self.assertEqual(np.flatnonzero(x[1]).tolist(),[2,6,10,14])
        self.assertEqual(np.flatnonzero(y[0]).tolist(),[0,4,8,12])
        self.assertEqual(true_theta('F00_B00')[-1],0)
        self.assertEqual(true_theta('F10_B00')[-1],1)

    def test_unqualified_lower_likelihood_endpoint_retained(self):
        theta=[.5,.3,.3,.8,.2,.8,.75,.5]
        record={'unit':'F05_B00','calibration_size':7680,'selected_theta':theta,
                'selected_endpoint':0,'selected_qualified':False,'endpoints':[
                {'theta':theta,'mean_nll':1.,'feasible':True,'qualified':False},
                {'theta':theta,'mean_nll':2.,'feasible':True,'qualified':True}]}
        with tempfile.TemporaryDirectory() as folder:
            write_json(Path(folder)/'results/calibrate/F05_B00/FIT.json',record)
            result,metadata,digest=selected_fit(folder,'F05_B00')
            self.assertFalse(metadata['selected_qualified'])
            self.assertEqual(len(metadata['fit_sha256']),64)
            np.testing.assert_array_equal(result,theta)

    def test_no_fitting_on_import_and_unlabeled_fit_interface(self):
        tree=ast.parse((ROOT/'cw1_data.py').read_text())
        calibration=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calibration')
        calls=[n for n in ast.walk(calibration) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='fit_counts']
        self.assertEqual(len(calls),1)
        self.assertEqual([ast.unparse(a) for a in calls[0].args], ["data['x']","data['y']","spec['calibration_fit']"])
        self.assertEqual(budget({'train_pairs':2048,'batch_size':64,'epochs':16,'training_horizons':[1,2,3,4]})['updates_per_fit'],512)


if __name__=='__main__':unittest.main()
