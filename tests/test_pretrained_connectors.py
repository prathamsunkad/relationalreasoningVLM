"""Native LFM correctness: frozen weights, causal masking, interfaces, replay."""
import copy
import unittest
import torch
from test_staged_abstractor_native import fixture
from pgmvlm.extensions.pretrained_connectors import DESIGNS,install,bound
from pgmvlm.pretrained_relations import pool_relations,question_seed


class ConnectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)

    def test_native_forward_gradients_causality_and_resume(self):
        for design in DESIGNS:
            with self.subTest(design=design):
                base,_,inputs=fixture()
                weights={n:p.clone().detach() for n,p in base.named_parameters()}
                with torch.no_grad(): native=base(**inputs).logits
                ctl=install(base,design,layer_index=2,width=16,heads=4)
                bank=torch.randn(1,8,9,32)
                def forward(inp=inputs):
                    with ctl.question(bank): return base(**inp).logits
                initial=forward()
                self.assertEqual(native.shape,initial.shape)
                if design!='tokens': self.assertTrue(torch.equal(initial,native))
                opt=torch.optim.AdamW(ctl.module.parameters(),lr=.003)
                def step():
                    opt.zero_grad(set_to_none=True)
                    F=torch.nn.functional
                    F.cross_entropy(forward()[:,-1],torch.tensor([12])).backward()
                    opt.step()
                for _ in range(3): step()
                self.assertGreater(float(ctl.module.input.weight.grad.abs().sum()),0)
                self.assertFalse(torch.equal(native,forward()))
                # Changing a future answer cannot influence earlier prompt logits.
                changed={**inputs,'input_ids':inputs['input_ids'].clone()}
                changed['input_ids'][0,-1]=17
                torch.testing.assert_close(forward()[:,:-1],forward(changed)[:,:-1],rtol=0,atol=0)
                # Teacher-forced first answer logits equal prefix-only inference.
                prefix={**inputs,'input_ids':inputs['input_ids'][:,:-1],
                        'attention_mask':inputs['attention_mask'][:,:-1]}
                torch.testing.assert_close(forward()[:,-2],forward(prefix)[:,-1],rtol=1e-5,atol=1e-6)
                saved=copy.deepcopy(ctl.module.state_dict());optim=copy.deepcopy(opt.state_dict())
                expected=forward().detach();step();updated=copy.deepcopy(ctl.module.state_dict())
                ctl.remove()
                ctl=install(base,design,layer_index=2,width=16,heads=4)
                ctl.module.load_state_dict(saved)
                opt=torch.optim.AdamW(ctl.module.parameters(),lr=.003);opt.load_state_dict(optim)
                torch.testing.assert_close(expected,forward(),rtol=0,atol=0)
                step()
                for n,p in ctl.module.state_dict().items(): self.assertTrue(torch.equal(p,updated[n]),n)
                for n,p in base.named_parameters():
                    if n in weights:
                        self.assertFalse(p.requires_grad);self.assertIsNone(p.grad)
                        self.assertTrue(torch.equal(weights[n],p))
                ctl.set_enabled(False)
                torch.testing.assert_close(native,base(**inputs).logits,rtol=0,atol=0)
                ctl.remove()

    def test_bfloat16_frozen_model_has_finite_learning(self):
        for design in DESIGNS:
            with self.subTest(design=design):
                base,_,inputs=fixture();base.to(torch.bfloat16)
                inputs={k:v.to(torch.bfloat16) if isinstance(v,torch.Tensor) and v.is_floating_point() else v for k,v in inputs.items()}
                ctl=install(base,design,layer_index=2,width=16,heads=4)
                bank=torch.randn(1,8,9,32)
                opt=torch.optim.AdamW(ctl.module.parameters(),lr=.003)
                for _ in range(3):
                    opt.zero_grad(set_to_none=True)
                    with ctl.question(bank): logits=base(**inputs).logits[:,-1].float()
                    torch.nn.functional.cross_entropy(logits,torch.tensor([12])).backward()
                    for p in ctl.module.parameters():
                        if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all())
                    opt.step()
                self.assertGreater(float(ctl.module.input.weight.grad.abs().sum()),0)
                ctl.remove()

    def test_memory_and_cache_guards(self):
        base,_,inputs=fixture();ctl=install(base,'reader',layer_index=2,width=16,heads=4)
        with self.assertRaises(ValueError): base(**inputs)
        bank=torch.randn(1,8,9,32)
        with ctl.question(bank):
            with self.assertRaises(ValueError): base(**{**inputs,'use_cache':True})
            with self.assertRaises(ValueError): base(**{**inputs,'labels':inputs['input_ids']})
        self.assertIsNone(ctl.bank)
        ctl.remove()

    def test_pooling_preserves_classifier_mean_and_bound(self):
        x=torch.randn(2,8,144,32)
        torch.testing.assert_close(pool_relations(x).mean(2),x.mean(2))
        native=torch.ones(1,2,8)
        self.assertTrue(torch.equal(bound(torch.zeros_like(native),native,.1),torch.zeros_like(native)))
        tiny=native*1e-5
        torch.testing.assert_close(bound(tiny,native,.1),tiny)
        self.assertLessEqual(float(bound(native*100,native,.1).square().mean().sqrt()),.100001)
        self.assertEqual(question_seed('x',3),question_seed('x',3))
        self.assertNotEqual(question_seed('x',3),question_seed('y',3))


if __name__=='__main__': unittest.main()
