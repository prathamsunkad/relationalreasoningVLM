import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_staged_abstractor_native import fixture
from pgmvlm.extensions.pretrained_connectors import install,DESIGNS
from tools.train_pretrained_connectors import validate_pilot,lr_at,save_training,restore_training,Banks
from tools import pretrained_connector_pilot as common

class TrainingTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 def test_gate_rejects_partial_or_failed_checks(self):
  config=dict(max_updates=16,validation_questions=32)
  arm=dict(status='passed',updates=16,frozen_weights_unchanged=True,
           validation_before=dict(complete=True,count=32),validation_after=dict(complete=True,count=32))
  summary=dict(passed=True,complete=True,arms={d:copy.deepcopy(arm) for d in DESIGNS})
  check={k:True for k in ('passed','deep_gradient','reload_exact','next_update_exact','disabled_exact','manual_native_greedy_parity')}
  checks={d:check.copy() for d in DESIGNS};validate_pilot(summary,checks,config)
  for field in ('updates','frozen_weights_unchanged','status'):
   broken=copy.deepcopy(summary);broken['arms']['tokens'][field]=False
   with self.assertRaises(ValueError):validate_pilot(broken,checks,config)
  checks['reader']['next_update_exact']=False
  with self.assertRaises(ValueError):validate_pilot(summary,checks,config)
  self.assertAlmostEqual(lr_at(14999,15000,3e-4,150),3e-5)
  self.assertGreater(lr_at(149,15000,3e-4,150),lr_at(0,15000,3e-4,150))

 def test_committed_resume_replays_next_update_and_rewinds_logs(self):
  model,_,inputs=fixture();ctl=install(model,'reader',layer_index=2,width=16,heads=4)
  bank=torch.randn(1,8,9,32);opt=torch.optim.AdamW(ctl.module.parameters(),lr=.003)
  def step():
   opt.zero_grad(set_to_none=True)
   with ctl.question(bank):logits=model(**inputs).logits[:,-1]
   torch.nn.functional.cross_entropy(logits,torch.tensor([12])).backward();opt.step()
  for _ in range(3):step()
  progress=dict(updates=3,best_loss=None,best_checkpoint=None);plan={'dataset':'fixed'}
  with tempfile.TemporaryDirectory() as folder,patch('tools.train_pretrained_connectors.require_headroom'):
   root=Path(folder);save_training(root,ctl,opt,progress,plan)
   step();expected=copy.deepcopy(ctl.module.state_dict())
   (root/'learning-curve.jsonl').write_text('{"update":3}\n{"update":4}\n')
   got=restore_training(root,ctl,opt,plan);self.assertEqual(got,progress)
   self.assertEqual((root/'learning-curve.jsonl').read_text(),'{"update":3}\n')
   step()
   for key,value in ctl.module.state_dict().items():torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
   with self.assertRaises(ValueError):restore_training(root,ctl,opt,{'dataset':'different'})
  ctl.remove()

if __name__=='__main__':unittest.main()
