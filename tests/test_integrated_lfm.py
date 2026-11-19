"""Exercise shared CLI training/resume/evaluation with native LFM and all new modes."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]


class IntegratedTests(unittest.TestCase):
    def test_train_resume_reload_evaluate_and_reject_protocol_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for split in ('train', 'val'):
                d = root / split
                d.mkdir()
                for i in range(2):
                    np.savez(d / f'{split}-{i}.npz', image=np.full((16, 160, 160), i*70, np.uint8), target=np.int64(i))
            env = dict(os.environ, TINY_FAMILY='lfm25')
            base = [sys.executable, str(ROOT / 'tests/run_tiny.py'), '--model_id', 'tiny_lfm25', '--dtype', 'fp32',
                    '--num_workers', '0', '--protocol', 'answer_letter', '--max_new_tokens', '2']
            def run(cmd, success=True):
                p = subprocess.run(base + cmd, env=env, text=True, capture_output=True)
                if success:
                    self.assertEqual(p.returncode, 0, (p.stdout+p.stderr)[-4000:])
                else:
                    self.assertNotEqual(p.returncode, 0)
                return p
            for mode, extension in [('lora', 'none'), ('module', 'late'), ('lora_module', 'rca_plugin'), ('module', 'rca_memory')]:
                with self.subTest(mode=mode, extension=extension):
                    out = root / (mode + extension)
                    extra = ['--extension', extension, '--extension_width', '16', '--extension_symbols', '8']
                    train = ['train', '--train_dir', str(root/'train'), '--val_dir', str(root/'val'),
                             '--out_dir', str(out), '--batch_size', '1', '--grad_accum', '1',
                             '--max_steps', '2', '--val_samples', '2', '--val_batch_size', '1', '--val_every', '1',
                             '--save_every', '1', '--train_mode', mode, '--lora_r', '2', '--lora_alpha', '4'] + extra
                    p = run(train + ['--time_limit_min', '0.00001'])
                    self.assertIn('time budget reached', p.stdout)
                    p = run(train)
                    self.assertIn('resumed from step', p.stdout)
                    self.assertTrue((out/'final/protocol.json').exists())
                    if mode == 'module':
                        self.assertFalse((out/'final/model.safetensors').exists())
                        self.assertFalse((out/'final/adapter_config.json').exists())
                    uninterrupted = root / ('uninterrupted-' + mode + extension)
                    ref_args = [str(uninterrupted) if arg == str(out) else arg for arg in train]
                    run(ref_args)
                    for weights in (out/'final').glob('*.safetensors'):
                        reference = load_file(str(uninterrupted/'final'/weights.name))
                        for key, tensor in load_file(str(weights)).items():
                            self.assertTrue(torch.equal(tensor, reference[key]), (mode, extension, key))
                    results = root / ('eval-' + extension)
                    ev = ['eval', '--adapter', str(out/'best'), '--npz_dir', str(root/'val'), '--out_dir', str(results)] + extra
                    run(ev)
                    p = run(ev)
                    self.assertIn('0 to go', p.stdout)
                    run(['eval', '--merge_dir', str(results), '--expected_n', '2'])
                    report = json.loads((results/'summary.json').read_text())
                    self.assertTrue(report['complete'])
                    self.assertEqual(report['protocol'], 'answer_letter')
                    # The stored run identity catches objective changes on resume.
                    run(train + ['--answer_loss_scope', 'answer'], success=False)
                    run(ev + ['--merge_adapter'], success=False)
                    if mode == 'module' and extension == 'late':
                        before = hashlib.sha256((out/'best/extension.safetensors').read_bytes()).hexdigest()
                        joint = root/'staged-joint'
                        staged = [str(joint) if arg == str(out) else arg for arg in train]
                        run(staged + ['--train_mode', 'lora_module', '--init_extension', str(out/'best')])
                        self.assertTrue((joint/'final/adapter_config.json').exists())
                        self.assertTrue((joint/'final/extension.safetensors').exists())
                        self.assertEqual(before, hashlib.sha256((out/'best/extension.safetensors').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
