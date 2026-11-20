import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tiny_lfm import tiny_lfm
from pgmvlm.extensions import attach_extension, extension_module, save_model, check_checkpoint_protocol
from pgmvlm.protocols import AnswerLetterProtocol


def options(**kwargs):
    return SimpleNamespace(**dict(dict(extension='late', extension_width=16, extension_depth=2,
        extension_heads=4, extension_symbols=8, correction_cap=None, correction_scale=1.,
        readout='llm', insertion_layer=None, protocol='answer_letter', answer_loss_scope='answer',
        max_new_tokens=2, model_id='tiny_lfm25', revision=None), **kwargs))


class SharedExtensions(unittest.TestCase):
    def test_all_diagnostic_readouts_train_only_new_weights_and_reload(self):
        torch.set_num_threads(1)
        for extra in ({'readout':'classifier'}, {'readout':'classifier_only'}, {'extension_depth':6, 'readout':'classifier'},
                      {'insertion_layer':0}, {'correction_cap':0.1}, {'extension':'rca_memory', 'correction_scale':0.01}):
            with self.subTest(**extra), tempfile.TemporaryDirectory() as td:
                model, proc = tiny_lfm()
                model.requires_grad_(False)
                frozen = {n:p.detach().clone() for n,p in model.named_parameters()}
                args = options(**extra)
                attach_extension(model, proc, args)
                protocol = AnswerLetterProtocol(proc, 'lfm25', args)
                item=dict(file='x', target=0, context=np.zeros((8,160,160),np.uint8), candidates=np.zeros((8,160,160),np.uint8))
                batch=protocol.collator([item])
                opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
                for i in range(2):
                    opt.zero_grad()
                    loss=protocol.training(model,batch,torch.device('cpu'),torch.float32,'causal_ce')['loss']
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward();opt.step()
                for n,p in model.named_parameters():
                    if n in frozen:
                        self.assertFalse(p.requires_grad)
                        self.assertIsNone(p.grad)
                        self.assertTrue(torch.equal(p, frozen[n]))
                save_model(model, td, args)
                check_checkpoint_protocol(td, args)
                clone, cp=tiny_lfm();clone.requires_grad_(False)
                attach_extension(clone, cp, args, td)
                for n,v in extension_module(model).state_dict().items():
                    self.assertTrue(torch.equal(v,extension_module(clone).state_dict()[n]))
                with torch.no_grad():
                    self.assertEqual(len(protocol.records(model,batch,torch.device('cpu'),torch.float32)),1)


if __name__=='__main__':unittest.main()
