"""Cached single-token interleave on native tiny InternVL/Qwen3 and PEFT LoRA."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from pgmvlm.models import candidate_scores
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "relational"))
from cached_interleave import CachedInterleave
from relvlm import RelationalVLM
from test_internvl_connectors import fixture


def setup():
    model, batch, bank = fixture()
    batch['input_ids'] = batch['input_ids'].clone()
    batch['input_ids'][:, 5] = 59
    mask = batch['input_ids'].eq(59)
    base = model.get_base_model()
    backbone = SimpleNamespace(model=base, d_llm=32, lm=base.model.language_model)
    return model, batch, bank, mask, backbone


def train_lora(model):
    """The shared fixture freezes LoRA; this experiment updates it jointly."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_('lora_' in name)
    for module in model.modules():
        dropouts = getattr(module, 'lora_dropout', None)
        if isinstance(dropouts, nn.ModuleDict):
            for adapter in list(dropouts):
                dropouts[adapter] = nn.Dropout(.05)


def optimizer(model, connector):
    return torch.optim.AdamW([
        {'params': list(connector.parameters()), 'lr': .003},
        {'params': [p for p in model.parameters() if p.requires_grad], 'lr': .002},
    ])


def score(model, batch):
    return candidate_scores(model, batch, SimpleNamespace(yes_id=10, no_id=11),
                            torch.device('cpu'), torch.float32)


def step(model, connector, opt, batch, bank, mask):
    model.train()
    connector.train()
    opt.zero_grad(set_to_none=True)
    with connector.question(bank, mask):
        scores = score(model, batch)
        loss = F.cross_entropy(scores, batch['targets'])
        loss.backward()
    opt.step()
    return loss.detach()


class CachedInterleaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_repository_parity_native_embeddings_and_context_cleanup(self):
        model, batch, bank, mask, backbone = setup()
        with torch.no_grad():
            native = score(model, batch)
        connector = CachedInterleave(backbone, input_dim=32)
        self.addCleanup(connector.remove)
        with torch.no_grad():
            connector.tok_gain.fill_(1.7)
        # Independent reference using the repository method and copied weights.
        embedding = backbone.model.get_input_embeddings().weight.detach()
        reference = SimpleNamespace(
            rel_mode='pool', d=32, tok_proj=copy.deepcopy(connector.tok_proj),
            tok_gain=connector.tok_gain.detach().clone(),
            emb_rms=embedding.float().square().mean(-1).add(1e-6).sqrt().mean())
        expected = RelationalVLM.relation_tokens(reference, {'r': bank.mean(1)})
        torch.testing.assert_close(connector.emb_rms, reference.emb_rms, rtol=0, atol=0)
        seen = []

        def capture(module, args, kwargs):
            seen.append(kwargs['inputs_embeds'].detach().clone())

        # Register after the connector so this observes the actual LM input.
        handle = backbone.lm.register_forward_pre_hook(capture, with_kwargs=True)
        self.addCleanup(handle.remove)
        with torch.no_grad():
            torch.testing.assert_close(score(model, batch), native, rtol=0, atol=0)
            native_embeddings = seen.pop()
            with connector.question(bank, mask):
                injected = score(model, batch)
                embeddings = seen.pop()
            torch.testing.assert_close(embeddings[mask], expected, rtol=0, atol=0)
            torch.testing.assert_close(embeddings[~mask], native_embeddings[~mask], rtol=0, atol=0)
            self.assertEqual(embeddings.shape, native_embeddings.shape)
            self.assertGreater(float((injected - native).abs().max()), 0)
            torch.testing.assert_close(score(model, batch), native, rtol=0, atol=0)
            # Cleanup must also happen if forward/training exits exceptionally.
            with self.assertRaisesRegex(RuntimeError, 'deliberate failure'):
                with connector.question(bank, mask):
                    raise RuntimeError('deliberate failure')
            torch.testing.assert_close(score(model, batch), native, rtol=0, atol=0)
            connector.remove()
            torch.testing.assert_close(score(model, batch), native, rtol=0, atol=0)

    def test_joint_gradients_frozen_weights_permutation_and_dropout_replay(self):
        model, batch, bank, mask, backbone = setup()
        train_lora(model)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        connector = CachedInterleave(backbone, input_dim=32)
        self.addCleanup(connector.remove)
        # External connector parameters must not silently join the backbone state.
        self.assertTrue({id(p) for p in connector.parameters()}.isdisjoint(
            {id(p) for p in model.parameters()}))
        opt = optimizer(model, connector)
        for _ in range(3):
            self.assertTrue(torch.isfinite(step(model, connector, opt, batch, bank, mask)))
        for name, parameter in connector.named_parameters():
            with self.subTest(connector_parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
        lora_changed = False
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIn('lora_', name)
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                lora_changed |= bool(parameter.grad.abs().sum() > 0)
            else:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        self.assertTrue(lora_changed)
        self.assertFalse(bank.requires_grad)
        self.assertIsNone(bank.grad)
        model.eval()
        with torch.no_grad(), connector.question(bank, mask):
            expected_scores = score(model, batch)
        perm = torch.tensor([6, 2, 0, 7, 3, 5, 1, 4])
        permuted = {k: v[perm] if k != 'targets' else v for k, v in batch.items()}
        with torch.no_grad(), connector.question(bank[perm], mask[perm]):
            torch.testing.assert_close(score(model, permuted), expected_scores[:, perm],
                                       rtol=1e-5, atol=1e-6)

        # Save adapters, connector, optimizer and CPU RNG; leave frozen base out.
        checkpoint = dict(
            lora={n: v.detach().clone() for n, v in model.state_dict().items() if 'lora_' in n},
            connector=copy.deepcopy(connector.state_dict()),
            optimizer=copy.deepcopy(opt.state_dict()), rng=torch.get_rng_state())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'resume.pt'
            torch.save(checkpoint, path)
            expected_loss = step(model, connector, opt, batch, bank, mask)
            expected_lora = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            expected_connector = copy.deepcopy(connector.state_dict())
            expected_rng = torch.get_rng_state()
            replay, replay_batch, replay_bank, replay_mask, replay_backbone = setup()
            train_lora(replay)
            restored = CachedInterleave(replay_backbone, input_dim=32)
            self.addCleanup(restored.remove)
            replay_opt = optimizer(replay, restored)
            saved = torch.load(path, map_location='cpu', weights_only=True)
            missing, unexpected = replay.load_state_dict(saved['lora'], strict=False)
            self.assertFalse(unexpected)
            self.assertTrue(all('lora_' not in name for name in missing))
            restored.load_state_dict(saved['connector'])
            replay_opt.load_state_dict(saved['optimizer'])
            replay.eval()
            with torch.no_grad(), restored.question(replay_bank, replay_mask):
                torch.testing.assert_close(score(replay, replay_batch), expected_scores, rtol=0, atol=0)
            torch.set_rng_state(saved['rng'])
            replay_loss = step(replay, restored, replay_opt, replay_batch, replay_bank, replay_mask)
            torch.testing.assert_close(replay_loss, expected_loss, rtol=0, atol=0)
            for name, parameter in replay.named_parameters():
                expected = expected_lora[name] if parameter.requires_grad else frozen[name]
                torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
            for name, tensor in restored.state_dict().items():
                torch.testing.assert_close(tensor, expected_connector[name], rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_invalid_cache_and_placeholder_masks_are_rejected(self):
        model, batch, bank, mask, backbone = setup()
        connector = CachedInterleave(backbone, input_dim=32)
        self.addCleanup(connector.remove)
        doubled = mask.clone()
        doubled[:, 6] = True
        missing = mask.clone()
        missing[0] = False
        # Equal total count is insufficient: each candidate needs one placeholder.
        unequal = mask.clone()
        unequal[0] = False
        unequal[1, 6] = True
        invalid = [
            (bank.clone().requires_grad_(), mask),
            (bank[:, 0], mask),
            (bank[:, :, :-1], mask),
            (bank[:7], mask),
            (bank, mask.float()),
            (bank, doubled),
            (bank, missing),
            (bank, unequal),
        ]
        with torch.no_grad():
            native = score(model, batch)
        for index, (bad_bank, bad_mask) in enumerate(invalid):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    with connector.question(bad_bank, bad_mask):
                        score(model, batch)
                with torch.no_grad():
                    torch.testing.assert_close(score(model, batch), native, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
