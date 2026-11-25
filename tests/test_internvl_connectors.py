"""Native InternVL/Qwen3 + frozen LoRA, not a mock attention implementation."""
import copy
import unittest
from types import SimpleNamespace

import torch
from transformers import InternVLConfig, InternVLForConditionalGeneration
from pgmvlm.models import add_lora, candidate_scores
from pgmvlm.extensions.internvl_connectors import install
from pgmvlm.extensions.pretrained_connectors import DESIGNS


def fixture():
    torch.manual_seed(41)
    cfg = InternVLConfig(image_token_id=60, image_seq_length=4, downsample_ratio=.5,
        text_config=dict(model_type='qwen3', hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            layer_types=['full_attention'] * 3, vocab_size=64, pad_token_id=0,
            bos_token_id=1, eos_token_id=2, tie_word_embeddings=False),
        vision_config=dict(model_type='internvl_vision', hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, image_size=[8, 8], patch_size=[2, 2],
            use_mean_pooling=True))
    model = add_lora(InternVLForConditionalGeneration(cfg), 'internvl', r=2, alpha=4, dropout=0.)
    model.eval().requires_grad_(False)
    ids = torch.tensor([[1, 60, 60, 60, 60, 5, 6, 7]] * 8)
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        pixel_values=torch.randn(8, 3, 8, 8), targets=torch.tensor([2]))
    return model, batch, torch.randn(8, 9, 32)


class InternVLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_all_interfaces_freezing_permutation_replay_and_causality(self):
        for design in DESIGNS:
            with self.subTest(design=design):
                model, batch, bank = fixture()
                base = model.get_base_model()
                originals = {n: p.detach().clone() for n, p in model.named_parameters()}
                st = SimpleNamespace(yes_id=10, no_id=11)
                def score(b=batch):
                    return candidate_scores(model, b, st, torch.device('cpu'), torch.float32)
                with torch.no_grad(): native = score()
                ctl = install(base, design, layer_index=1, width=16, heads=4)
                def forward(b=batch, memory=bank):
                    with ctl.question(memory):
                        return score(b)
                if design != 'tokens':
                    torch.testing.assert_close(native, forward(), rtol=0, atol=0)
                opt = torch.optim.AdamW(ctl.module.parameters(), lr=.003)
                def step():
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(forward(), batch['targets'])
                    loss.backward()
                    opt.step()
                    return loss
                for _ in range(3): step()
                self.assertGreater(float(ctl.module.input.weight.grad.abs().sum()), 0)
                self.assertGreater(float((forward() - native).abs().max()), 0)
                perm = torch.tensor([6, 2, 0, 7, 3, 5, 1, 4])
                permuted = {k: v[perm] if k != 'targets' else v for k, v in batch.items()}
                torch.testing.assert_close(forward(permuted, bank[perm]), forward()[:, perm], rtol=1e-5, atol=1e-6)
                weights, optim = copy.deepcopy(ctl.module.state_dict()), copy.deepcopy(opt.state_dict())
                expected = forward().detach()
                step()
                updated = copy.deepcopy(ctl.module.state_dict())
                ctl.remove()
                ctl = install(base, design, layer_index=1, width=16, heads=4)
                ctl.module.load_state_dict(weights)
                opt = torch.optim.AdamW(ctl.module.parameters(), lr=.003)
                opt.load_state_dict(optim)
                torch.testing.assert_close(expected, forward(), rtol=0, atol=0)
                step()
                for n, p in ctl.module.state_dict().items():
                    torch.testing.assert_close(updated[n], p, rtol=0, atol=0)
                # Changed future prompt token cannot alter earlier hidden states.
                inputs = {k: v for k, v in batch.items() if k != 'targets'}
                with ctl.question(bank):
                    h1 = model(**inputs, use_cache=False, output_hidden_states=True).hidden_states[-1]
                inputs['input_ids'] = inputs['input_ids'].clone()
                inputs['input_ids'][:, -1] = 15
                with ctl.question(bank):
                    h2 = model(**inputs, use_cache=False, output_hidden_states=True).hidden_states[-1]
                self.assertEqual(h1.shape[1], batch['input_ids'].shape[1])
                torch.testing.assert_close(h1[:, :-1], h2[:, :-1], rtol=0, atol=0)
                for n, p in model.named_parameters():
                    if n in originals:
                        self.assertFalse(p.requires_grad)
                        self.assertIsNone(p.grad)
                        torch.testing.assert_close(originals[n], p, rtol=0, atol=0)
                ctl.set_enabled(False)
                torch.testing.assert_close(native, score(), rtol=0, atol=0)
                ctl.remove()
                torch.testing.assert_close(native, score(), rtol=0, atol=0)

    def test_guards(self):
        model, batch, bank = fixture()
        ctl = install(model.get_base_model(), 'reader', layer_index=1, width=16, heads=4)
        inputs = {k: v for k, v in batch.items() if k != 'targets'}
        with self.assertRaises(ValueError): model(**inputs, use_cache=False)
        for bad in ({'use_cache': True}, {'labels': inputs['input_ids']}):
            with ctl.question(bank), self.assertRaises(ValueError):
                model(**{**inputs, 'use_cache': False, **bad})
        with ctl.question(bank[:1]), self.assertRaises(ValueError):
            model(**inputs, use_cache=False)
        with ctl.question(bank.clone().requires_grad_()), self.assertRaises(ValueError):
            model(**inputs, use_cache=False)
        ctl.remove()


if __name__ == '__main__':
    unittest.main()
