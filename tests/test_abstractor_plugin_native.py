"""Bounded native LFM/PEFT integration checks; no downloads or pretrained weights.

Run in the pinned Transformers 5.1.0 / PEFT environment:
    PYTHONPATH=benchmarks python -m unittest discover -s benchmarks/tests \
        -p 'test_abstractor_plugin_native.py' -v

The tiny model has a second full-attention layer after the intervention, so
visual-only corrections can reach answer positions just as in the real model.
"""

import copy
import unittest
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from transformers import Lfm2VlConfig, Lfm2VlForConditionalGeneration

from pgmvlm.extensions.abstractor_plugin import install


IMAGE_ID = 396


def fixture(seed=31):
    torch.manual_seed(seed)
    config = Lfm2VlConfig(
        image_token_id=IMAGE_ID, downsample_factor=2, projector_hidden_size=64,
        text_config=dict(
            model_type="lfm2", hidden_size=32, intermediate_size=64,
            block_multiple_of=8, num_hidden_layers=6, num_attention_heads=4,
            num_key_value_heads=2,
            layer_types=["conv", "conv", "full_attention", "conv", "full_attention", "conv"],
            vocab_size=512, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        ),
        vision_config=dict(
            model_type="siglip2_vision_model", hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, patch_size=2,
            num_patches=16, vision_use_head=False,
        ),
    )
    model = Lfm2VlForConditionalGeneration(config)
    model.requires_grad_(False)
    model.eval()
    inputs = dict(
        input_ids=torch.tensor([[1] + [IMAGE_ID] * 8 + [9, 10, 11, 12]]),
        attention_mask=torch.ones(1, 13, dtype=torch.long),
        pixel_values=torch.randn(2, 16, 12),
        pixel_attention_mask=torch.ones(2, 16, dtype=torch.long),
        spatial_shapes=torch.tensor([[4, 4], [4, 4]]),
    )
    processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [10, 11]))
    return model, inputs, processor


def attach(model, processor):
    return install(model, processor, layer_index=2, width=16, heads=4, depth=2)


def open_output(controller):
    # Exercise a nonzero branch independently of training convergence.
    with torch.no_grad():
        torch.manual_seed(52)
        controller.plugin.output.weight.normal_(mean=0.0, std=0.04)


class NativeAbstractorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_initial_noop_disabled_and_removed_restore_backbone(self):
        model, inputs, processor = fixture()
        with torch.no_grad():
            expected = model(**inputs, use_cache=False).logits.clone()
        controller = attach(model, processor)
        with torch.no_grad():
            initial = model(**inputs, use_cache=False).logits
        self.assertTrue(torch.equal(expected, initial))

        open_output(controller)
        with torch.no_grad():
            active = model(**inputs, use_cache=False).logits
        self.assertGreater(float((active - expected).abs().max()), 0.0)
        controller.set_enabled(False)
        with torch.no_grad():
            bypassed = model(**inputs, use_cache=False).logits
        self.assertTrue(torch.equal(expected, bypassed))

        controller.set_enabled(True)
        controller.remove()
        self.assertFalse(hasattr(model, "abstractor_plugin"))
        with torch.no_grad():
            removed = model(**inputs, use_cache=False).logits
        self.assertTrue(torch.equal(expected, removed))

    def test_intervention_changes_only_visual_positions_locally(self):
        model, inputs, processor = fixture()
        attention = model.model.language_model.layers[2].self_attn
        observed = {}

        def before_plugin(module, args, result):
            observed["native"] = result[0].detach().clone()

        def after_plugin(module, args, result):
            observed["augmented"] = result[0].detach().clone()

        before_handle = attention.register_forward_hook(before_plugin)
        controller = attach(model, processor)
        after_handle = attention.register_forward_hook(after_plugin)
        try:
            open_output(controller)
            with torch.no_grad():
                model(**inputs, use_cache=False)
            image_mask = inputs["input_ids"] == IMAGE_ID
            self.assertTrue(torch.equal(observed["native"][~image_mask],
                                        observed["augmented"][~image_mask]))
            self.assertGreater(float((observed["native"][image_mask] -
                                      observed["augmented"][image_mask]).abs().max()), 0.0)
        finally:
            before_handle.remove()
            after_handle.remove()
            controller.remove()

    def test_no_supervised_suffix_leakage_and_generation_cache_equivalence(self):
        model, inputs, processor = fixture()
        controller = attach(model, processor)
        open_output(controller)
        altered = {**inputs, "input_ids": inputs["input_ids"].clone()}
        altered["input_ids"][0, -1] = 22
        with torch.no_grad():
            first = model(**inputs, use_cache=False).logits[:, :-1]
            second = model(**altered, use_cache=False).logits[:, :-1]
        self.assertTrue(torch.equal(first, second))

        prefix = {key: (value[:, :-1] if key in ("input_ids", "attention_mask") else value)
                  for key, value in inputs.items()}
        generation = dict(max_new_tokens=3, do_sample=False, eos_token_id=None,
                          pad_token_id=0, suppress_tokens=[IMAGE_ID])
        with torch.no_grad():
            cached = model.generate(**prefix, use_cache=True, **generation)
            uncached = model.generate(**prefix, use_cache=False, **generation)
        self.assertTrue(torch.equal(cached, uncached))
        # A different prefill must not retain visual features from the old image.
        other = {**prefix, "pixel_values": prefix["pixel_values"] * -0.7}
        with torch.no_grad():
            model.generate(**other, use_cache=True, **generation)
            after_other = model.generate(**prefix, use_cache=True, **generation)
        self.assertTrue(torch.equal(cached, after_other))

    def test_gradients_open_and_plugin_state_roundtrips(self):
        model, inputs, processor = fixture()
        controller = attach(model, processor)
        original = {n: p.detach().clone() for n, p in model.named_parameters()
                    if "abstractor_plugin." not in n}
        optimizer = torch.optim.AdamW(controller.plugin.parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs, use_cache=False).logits
            torch.nn.functional.cross_entropy(logits[:, -2], torch.tensor([12])).backward()
            optimizer.step()
        gradient = controller.plugin.layers[0].relation.q.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(gradient).all())
        for name, parameter in model.named_parameters():
            if name in original:
                self.assertIsNone(parameter.grad, name)
                self.assertTrue(torch.equal(parameter, original[name]), name)

        state = copy.deepcopy(controller.plugin.state_dict())
        with torch.no_grad():
            expected = model(**inputs, use_cache=False).logits.clone()
        controller.remove()
        restored = attach(model, processor)
        restored.plugin.load_state_dict(state, strict=True)
        with torch.no_grad():
            actual = model(**inputs, use_cache=False).logits
        self.assertTrue(torch.equal(expected, actual))

    def test_disabled_plugin_preserves_lora_gradients_and_next_update(self):
        def adapted():
            base, inputs, processor = fixture()
            targets = [name for name, module in base.named_modules()
                       if name.startswith("model.language_model.") and isinstance(module, torch.nn.Linear)]
            torch.manual_seed(71)
            model = get_peft_model(base, LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.05, bias="none", target_modules=targets))
            # Include nonzero B matrices, matching a trained adapter rather than a fresh no-op.
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "lora_B" in name:
                        parameter.normal_(mean=0.0, std=0.02)
            return model, inputs, processor

        plain, inputs, _ = adapted()
        augmented, inputs2, processor = adapted()
        self.assertTrue(torch.equal(inputs["pixel_values"], inputs2["pixel_values"]))
        controller = attach(augmented.get_base_model(), processor)
        open_output(controller)
        controller.set_enabled(False)
        controller.plugin.requires_grad_(False)
        frozen = {name: parameter.detach().clone() for name, parameter in plain.named_parameters()
                  if "lora_" not in name}

        outcomes = []
        for model in (plain, augmented):
            trainable = [(name, parameter) for name, parameter in model.named_parameters()
                         if "lora_" in name]
            optimizer = torch.optim.AdamW([p for _, p in trainable], lr=0.001)
            model.train()
            model.get_base_model().model.vision_tower.eval()
            model.get_base_model().model.multi_modal_projector.eval()
            torch.manual_seed(20261109)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(logits[:, -2], torch.tensor([12]))
            loss.backward()
            gradients = {n: p.grad.detach().clone() if p.grad is not None else None for n, p in trainable}
            optimizer.step()
            weights = {n: p.detach().clone() for n, p in trainable}
            outcomes.append((logits.detach(), gradients, weights))
            for name, parameter in model.named_parameters():
                if name in frozen:
                    self.assertIsNone(parameter.grad, name)
                    self.assertTrue(torch.equal(parameter, frozen[name]), name)

        self.assertTrue(torch.equal(outcomes[0][0], outcomes[1][0]))
        for field in (1, 2):
            self.assertEqual(outcomes[0][field].keys(), outcomes[1][field].keys())
            for name, expected in outcomes[0][field].items():
                actual = outcomes[1][field][name]
                if expected is None:
                    self.assertIsNone(actual, name)
                else:
                    self.assertTrue(torch.equal(expected, actual), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
