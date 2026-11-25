"""All seven connectors against a tiny REAL native LFM, with no pretrained data."""
import copy
from types import SimpleNamespace
import unittest
import torch
from transformers import Lfm2VlConfig,Lfm2VlForConditionalGeneration
from pgmvlm.extensions.staged_abstractor import DESIGNS,install


def fixture():
    torch.manual_seed(71)
    config=Lfm2VlConfig(image_token_id=396,downsample_factor=2,projector_hidden_size=64,
        text_config=dict(model_type='lfm2',hidden_size=32,intermediate_size=64,block_multiple_of=8,
            num_hidden_layers=6,num_attention_heads=4,num_key_value_heads=2,
            layer_types=['conv','conv','full_attention','conv','full_attention','conv'],vocab_size=512,
            pad_token_id=0,bos_token_id=1,eos_token_id=2),
        vision_config=dict(model_type='siglip2_vision_model',hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=4,patch_size=2,num_patches=256,vision_use_head=False))
    base=Lfm2VlForConditionalGeneration(config).eval().requires_grad_(False)
    ids=torch.tensor([[1]+[396]*64+[9,10,11]])
    inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=torch.randn(1,256,12),
                pixel_attention_mask=torch.ones(1,256,dtype=torch.long),spatial_shapes=torch.tensor([[16,16]]),use_cache=False)
    processor=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a,**k:[10,11]))
    return base,processor,inputs


class NativeStagedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)

    def test_seven_designs_frozen_reload_remove_and_next_update(self):
        for design in DESIGNS:
            with self.subTest(design=design):
                base,processor,inp=fixture()
                original={n:p.detach().clone() for n,p in base.named_parameters()}
                with torch.no_grad(): native=base(**inp).logits
                ctl=install(base,processor,design,width=16,heads=4,depth=2)
                ctl.module.fusion.force_share=0.
                with torch.no_grad(): noop=base(**inp).logits
                self.assertTrue(torch.equal(native,noop))
                ctl.module.fusion.force_share=None
                with torch.no_grad(): changed=base(**inp).logits
                self.assertGreater(float((changed-native).abs().max()),0)
                opt=torch.optim.AdamW(ctl.module.parameters(),lr=3e-4)
                def step():
                    opt.zero_grad(set_to_none=True)
                    logits=base(**inp).logits[:,-1]
                    torch.nn.functional.cross_entropy(logits,torch.tensor([12])).backward()
                    opt.step()
                step()
                self.assertGreater(float(ctl.module.content[1].weight.grad.abs().sum()),0)
                self.assertGreater(float(ctl.module.layers[0].relation.q.weight.grad.abs().sum()),0)
                for name,p in base.named_parameters():
                    if name in original:
                        self.assertFalse(p.requires_grad);self.assertIsNone(p.grad)
                        self.assertTrue(torch.equal(original[name],p))
                saved=copy.deepcopy(ctl.module.state_dict());optim=copy.deepcopy(opt.state_dict())
                with torch.no_grad(): expected=base(**inp).logits
                step();updated=copy.deepcopy(ctl.module.state_dict())
                ctl.remove()
                with torch.no_grad(): restored_native=base(**inp).logits
                self.assertTrue(torch.equal(native,restored_native))
                ctl=install(base,processor,design,width=16,heads=4,depth=2)
                ctl.module.load_state_dict(saved)
                opt=torch.optim.AdamW(ctl.module.parameters(),lr=3e-4);opt.load_state_dict(optim)
                with torch.no_grad(): actual=base(**inp).logits
                self.assertTrue(torch.equal(expected,actual))
                step()
                for n,p in ctl.module.state_dict().items(): self.assertTrue(torch.equal(updated[n],p),n)
                ctl.set_enabled(False)
                with torch.no_grad(): bypass=base(**inp).logits
                self.assertTrue(torch.equal(native,bypass))
                ctl.remove()

    def test_rejects_answer_suffix_cache_and_unfrozen_base(self):
        base,processor,inp=fixture()
        base.requires_grad_(True)
        with self.assertRaises(ValueError): install(base,processor,'late',width=16,heads=4,depth=2)
        base.requires_grad_(False)
        ctl=install(base,processor,'late',width=16,heads=4,depth=2)
        for bad in ({**inp,'use_cache':True},{**inp,'labels':inp['input_ids']},
                    {**inp,'input_ids':torch.cat([inp['input_ids'],torch.tensor([[12]])],dim=1)}):
            with self.assertRaises(ValueError): base(**bad)
        ctl.remove()


if __name__=='__main__': unittest.main()
