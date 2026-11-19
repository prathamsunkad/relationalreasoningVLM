"""Send a real SIGUSR1 to the trainer and compare its continuation to a clean run."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import numpy as np
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]

class SignalResume(unittest.TestCase):
    def test_signal_commits_complete_update_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data=root/'data';data.mkdir()
            for i in range(2):
                np.savez(data/f'{i}.npz',image=np.full((16,160,160),40*i,np.uint8),target=np.int64(i))
            command=[sys.executable,str(ROOT/'tests/run_tiny.py'),'--model_id','tiny_lfm25','--dtype','fp32',
                '--num_workers','0','--protocol','answer_letter','--max_new_tokens','2','train',
                '--train_dir',str(data),'--train_mode','lora_module','--extension','rca_plugin',
                '--extension_width','16','--lora_r','2','--lora_alpha','4','--batch_size','1','--grad_accum','1',
                '--save_every','1','--log_every','1','--val_every','1000']
            env=dict(os.environ,TINY_FAMILY='lfm25')
            out=root/'interrupted'
            with (root/'log').open('w') as log:
                process=subprocess.Popen(command+['--out_dir',str(out),'--max_steps','50'],env=env,stdout=log,stderr=log)
                try:
                    deadline=time.monotonic()+60
                    while not (out/'.checkpoints/last.json').exists():
                        if process.poll() is not None or time.monotonic()>deadline:
                            self.fail((root/'log').read_text()[-4000:])
                        time.sleep(.02)
                    process.send_signal(signal.SIGUSR1)
                    self.assertEqual(process.wait(timeout=30),0)
                finally:
                    if process.poll() is None:process.kill();process.wait()
            state=torch.load(out/'state.pt',weights_only=False)
            self.assertGreaterEqual(state['step'],1);self.assertLess(state['step'],50)
            target=state['step']+1
            for destination in (out,root/'reference'):
                result=subprocess.run(command+['--out_dir',str(destination),'--max_steps',str(target)],env=env,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,(result.stdout+result.stderr)[-4000:])
            for path in (out/'final').glob('*.safetensors'):
                reference=load_file(str(root/'reference/final'/path.name))
                for name,tensor in load_file(str(path)).items():self.assertTrue(torch.equal(tensor,reference[name]),name)

if __name__=='__main__':unittest.main()
