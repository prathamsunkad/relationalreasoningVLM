import tempfile
from pathlib import Path
import unittest
import torch
from safetensors.torch import load_file
from tools.init_matched_probes import export

class MatchedInitialization(unittest.TestCase):
    def test_shared_depth_readout_and_location_tensors(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'initial'
            result=export(root,input_dim=32,width=16,symbols=8,locations=(0,2))
            weights={n:load_file(str(root/n/'extension.safetensors')) for n in result['arms']}
            for key,value in weights['llm_d2'].items():
                self.assertTrue(torch.equal(value,weights['llm_d6'][key]))
                self.assertTrue(torch.equal(value,weights['classifier_d2']['probe.'+key]))
                self.assertTrue(torch.equal(value,weights['location_0'][key]))
                self.assertTrue(torch.equal(value,weights['location_2'][key]))
            for key,value in weights['classifier_only'].items():
                if key.startswith('head.'):
                    self.assertTrue(torch.equal(value,weights['classifier_d2'][key]))
                    self.assertTrue(torch.equal(value,weights['classifier_d6'][key]))

if __name__=='__main__':unittest.main()
