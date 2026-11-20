"""Fault injection for committed checkpoints; no model downloads or CUDA."""
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from pgmvlm import checkpoints as ck
from pgmvlm.shutdown import StopController


class CheckpointRecovery(unittest.TestCase):
    def writer(self, value):
        return lambda path: save_file({'weight': torch.tensor([value], dtype=torch.float)}, str(path/'adapter_model.safetensors'))

    def test_interrupted_write_preserves_committed_model_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            state=dict(step=1,opt={'state': 1},rng=torch.get_rng_state(),best_checkpoint=None)
            first=ck.commit(root,'last',self.writer(1),state=state)
            with patch.object(ck,'publish',side_effect=OSError('injected crash before pointer')):
                with self.assertRaises(OSError):ck.commit(root,'last',self.writer(2),state=dict(state,step=2))
            folder,path=ck.resume_checkpoint(root)
            self.assertEqual(folder.name,first['directory'])
            restored=torch.load(path,weights_only=False)
            self.assertEqual(restored['step'],1)
            self.assertTrue(torch.equal(restored['rng'],state['rng']))
            self.assertEqual(ck.resolve_checkpoint(root/'last'),folder)

    def test_corruption_falls_back_and_rewinds_logs_and_best(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            best1=ck.commit(root,'best',self.writer(1),selection={'step':1})
            first=ck.commit(root,'last',self.writer(1),state=dict(step=1,best_checkpoint=best1))
            best2=ck.commit(root,'best',self.writer(2),selection={'step':2})
            second=ck.commit(root,'last',self.writer(2),state=dict(step=2,best_checkpoint=best2))
            (root/'metrics.jsonl').write_text('{"step":1}\n{"step":2}\n{"step":')
            (root/'.checkpoints'/second['directory']/'adapter_model.safetensors').write_bytes(b'interrupted')
            with self.assertWarnsRegex(UserWarning,'Recovering'):
                folder,path=ck.resume_checkpoint(root)
            self.assertEqual(folder.name,first['directory'])
            self.assertEqual(json.loads((root/'best.json').read_text())['step'],1)
            self.assertEqual((root/'metrics.jsonl').read_text(),'{"step":1}\n')
            self.assertEqual(torch.load(path,weights_only=False)['step'],1)

    def test_committed_pointer_repairs_missing_links_and_lock_excludes_writer(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            ck.commit(root,'last',self.writer(1),state=dict(step=1,best_checkpoint=None))
            (root/'last').unlink();(root/'state.pt').unlink()
            ck.resume_checkpoint(root)
            self.assertTrue((root/'last/adapter_model.safetensors').exists())
            self.assertTrue((root/'state.pt').exists())
            with ck.RunLock(root):
                with self.assertRaisesRegex(RuntimeError,'Another process'):
                    with ck.RunLock(root):pass
            with ck.RunLock(root):pass

    def test_retention_and_legacy_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'last').mkdir();(root/'last/original.txt').write_text('keep')
            torch.save(dict(step=0),root/'state.pt')
            for step in range(5):
                ck.commit(root,'last',self.writer(step),state=dict(step=step,best_checkpoint=None))
            self.assertEqual(len(list((root/'.checkpoints').glob('last-*'))),2)
            self.assertEqual(len(list((root/'.checkpoints').glob('legacy-last-*/original.txt'))),1)

    def test_reader_prevents_retention_from_deleting_in_use_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            ck.commit(root,'best',self.writer(1),selection={'step':1})
            with ck.checkpoint_reader(root/'best') as reading:
                ck.commit(root,'best',self.writer(2),selection={'step':2})
                self.assertTrue(reading.exists())
            ck.prune(root)
            self.assertFalse(reading.exists())

    def test_signal_and_slurm_deadline_and_sentinel(self):
        with patch.dict(os.environ,{'VLM_DEADLINE_EPOCH':'1000'},clear=True),patch('time.time',return_value=900),patch('time.monotonic',return_value=30):
            self.assertTrue(StopController(reserve_seconds=120).exhausted())
        previous=signal.getsignal(signal.SIGUSR1)
        with patch.dict(os.environ,{},clear=True),StopController() as stop:
            os.kill(os.getpid(),signal.SIGUSR1)
            self.assertTrue(stop.exhausted())
            self.assertEqual(stop.reason,'SIGUSR1')
        self.assertEqual(signal.getsignal(signal.SIGUSR1),previous)
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{'VLM_STOP_FILE':td+'/stop'},clear=True):
            stop=StopController();self.assertFalse(stop.exhausted())
            Path(td+'/stop').touch();self.assertTrue(stop.exhausted())

if __name__=='__main__':unittest.main()
