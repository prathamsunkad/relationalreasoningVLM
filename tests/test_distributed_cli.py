"""Exercise the actual multi-process trainer and its per-rank checkpoint replay."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]

class DistributedCLI(unittest.TestCase):
    def test_two_rank_native_lfm_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for split in ('train','val'):
                (root/split).mkdir()
                for i in range(4):
                    np.savez(root/split/f'{i}.npz',image=np.full((16,160,160),i*40,np.uint8),target=np.int64(i))
            base=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                  str(ROOT/'tests/run_tiny.py'),'--model_id','tiny_lfm25','--dtype','fp32','--num_workers','0',
                  '--protocol','answer_letter','--max_new_tokens','2','train',
                  '--train_dir',str(root/'train'),'--val_dir',str(root/'val'),
                  '--train_mode','lora_module','--extension','rca_plugin','--extension_width','16',
                  '--lora_r','2','--lora_alpha','4','--batch_size','1','--global_batch','4',
                  '--max_steps','2','--val_samples','2','--val_batch_size','1','--val_every','1','--save_every','1']
            def run(output, extra=()):
                result=subprocess.run(base+['--out_dir',str(output)]+list(extra),
                    env=dict(os.environ,TINY_FAMILY='lfm25'),capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,(result.stdout+result.stderr)[-5000:])
                return result.stdout
            resumed,reference=root/'resumed',root/'reference'
            self.assertIn('time budget reached',run(resumed,['--time_limit_min','0.00001']))
            self.assertIn('resumed from step',run(resumed))
            run(reference)
            state=torch.load(resumed/'state.pt',map_location='cpu',weights_only=False)
            self.assertEqual(len(state['rng_by_rank']),2)
            self.assertEqual(state['step'],2)
            for path in (resumed/'final').glob('*.safetensors'):
                expected=load_file(str(reference/'final'/path.name))
                for name,value in load_file(str(path)).items():
                    self.assertTrue(torch.equal(value,expected[name]),(path.name,name))

if __name__=='__main__':unittest.main()
