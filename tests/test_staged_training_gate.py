import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from pgmvlm.extensions.staged_abstractor import DESIGNS,StagedAbstractor
from tools.train_staged_abstractor import validate_pilot,save,restore
from types import SimpleNamespace


class StagedTrainingGateTests(unittest.TestCase):
    def complete(self):
        config=dict(max_updates=16,validation_questions=32)
        result=dict(status='passed',updates=16,validation=dict(count=32),
            base_bitwise_unchanged=True,checkpoint_reload_exact=True,next_update_replay_exact=True,
            checks=dict(zero_share_native_exact=True,candidate_permutation=True,deep_gradients=True,added_weights_update=True))
        return dict(status='passed',base_bitwise_unchanged=True,
                    results={n:copy.deepcopy(result) for n in DESIGNS}),config

    def test_only_all_seven_complete_passes_release_gate(self):
        summary,config=self.complete();validate_pilot(summary,config)
        for field,value in (('status','failed'),('updates',15),('next_update_replay_exact',False),
                            ('validation',dict(count=31)),('checks',{})):
            bad=copy.deepcopy(summary);bad['results']['late'][field]=value
            with self.assertRaises(ValueError):validate_pilot(bad,config)
        del summary['results']['dual']
        with self.assertRaises(ValueError):validate_pilot(summary,config)

    def test_transactional_resume_rejects_changed_identity_or_corruption(self):
        torch.set_num_threads(2);torch.manual_seed(11)
        module=StagedAbstractor('late',vision_dim=32,text_dim=32,width=16,heads=4,depth=2)
        ctl=SimpleNamespace(module=module);opt=torch.optim.AdamW(module.parameters(),lr=1e-3)
        def step():
            opt.zero_grad();loss=sum(p.square().mean() for p in module.parameters());loss.backward();opt.step()
        step();plan=dict(case='A',source_commit='test')
        with tempfile.TemporaryDirectory() as d,patch('tools.train_staged_abstractor.require_headroom'):
            path=Path(d)/'latest.pt'
            save(path,ctl,opt,dict(updates=7,best_update=4),plan,'stable-ids')
            step();expected=copy.deepcopy(module.state_dict())
            self.assertEqual(restore(path,ctl,opt,plan,'stable-ids')['updates'],7)
            step()
            for name,value in module.state_dict().items():self.assertTrue(torch.equal(expected[name],value),name)
            for newplan,ids in ((dict(case='B'),'stable-ids'),(plan,'changed-ids')):
                with self.assertRaises(ValueError):restore(path,ctl,opt,newplan,ids)
            with path.open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):restore(path,ctl,opt,plan,'stable-ids')


if __name__=='__main__':unittest.main()
