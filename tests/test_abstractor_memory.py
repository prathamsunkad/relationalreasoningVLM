"""Bounded native tests for relational memory; no downloads or full checkpoints."""

import copy
import unittest

import torch
import torch.nn.functional as F

from test_abstractor_plugin_native import IMAGE_ID, fixture
from pgmvlm.extensions.abstractor_memory import AbstractorMemory, install


def attach(model, processor=None):
    return install(model, processor, layer_index=2, width=16, heads=4, depth=2)


def open_output(controller):
    with torch.no_grad():
        torch.manual_seed(52)
        controller.plugin.output.weight.normal_(mean=0, std=0.04)


class NativeAbstractorMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_initial_noop_exact_bypass_and_original_flags(self):
        model, inputs, processor = fixture()
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        flags = {name: value.requires_grad for name, value in model.named_parameters()}
        with torch.no_grad():
            baseline = model(**inputs, use_cache=False).logits
        controller = attach(model, processor)
        self.assertIs(model.abstractor_plugin, controller.plugin)
        self.assertFalse(controller.spec['token_pooling'])
        with torch.no_grad():
            self.assertTrue(torch.equal(baseline, model(**inputs, use_cache=False).logits))
            open_output(controller)
            active = model(**inputs, use_cache=False).logits
            self.assertFalse(torch.equal(baseline[:, -1], active[:, -1]))
            controller.set_enabled(False)
            self.assertTrue(torch.equal(baseline, model(**inputs, use_cache=False).logits))
            controller.set_enabled(True)
            self.assertTrue(torch.equal(active, model(**inputs, use_cache=False).logits))
            controller.remove()
            self.assertTrue(torch.equal(baseline, model(**inputs, use_cache=False).logits))
        self.assertFalse(hasattr(model, 'abstractor_plugin'))
        for name, parameter in model.named_parameters():
            self.assertEqual(flags[name], parameter.requires_grad, name)
            self.assertTrue(torch.equal(before[name], parameter), name)

    def test_only_post_image_language_outputs_changed_with_native_tile_markers(self):
        model, inputs, processor = fixture()
        # The real processor inserts tile-address markers between <image> runs.
        inputs['input_ids'] = torch.tensor([[1] + [IMAGE_ID] * 4 + [15] + [IMAGE_ID] * 4 + [9, 10, 11, 12]])
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        attention = model.model.language_model.layers[2].self_attn
        captured = {}
        first = attention.register_forward_hook(lambda m, a, out: captured.update(native=out[0].detach().clone()))
        controller = attach(model, processor)
        open_output(controller)
        last = attention.register_forward_hook(lambda m, a, out: captured.update(updated=out[0].detach().clone()))
        with torch.no_grad():
            model(**inputs, use_cache=False)
        first.remove()
        last.remove()
        self.assertTrue(torch.equal(captured['native'][:, :10], captured['updated'][:, :10]))
        self.assertFalse(torch.equal(captured['native'][:, 10:], captured['updated'][:, 10:]))

    def test_future_answer_cannot_change_memory_or_earlier_logits(self):
        model, inputs, processor = fixture()
        controller = attach(model, processor)
        open_output(controller)
        memories = []
        handle = controller.plugin.register_forward_pre_hook(lambda m, args: memories.append(args[1].detach().clone()))
        altered = {key: value.clone() for key, value in inputs.items()}
        altered['input_ids'][0, -1] = 22
        with torch.no_grad():
            original = model(**inputs, use_cache=False).logits
            changed = model(**altered, use_cache=False).logits
        handle.remove()
        self.assertTrue(torch.equal(memories[0], memories[1]))
        self.assertTrue(torch.equal(original[:, :-1], changed[:, :-1]))

    def test_relational_memory_is_sensitive_to_image_content(self):
        model, inputs, processor = fixture()
        controller = attach(model, processor)
        open_output(controller)
        memories = []
        handle = controller.plugin.register_forward_pre_hook(lambda m, args: memories.append(args[1].detach().clone()))
        altered = {key: value.clone() for key, value in inputs.items()}
        altered['pixel_values'] = altered['pixel_values'].flip(1) * 2
        with torch.no_grad():
            model(**inputs, use_cache=False)
            model(**altered, use_cache=False)
        handle.remove()
        self.assertGreater(float((memories[0] - memories[1]).abs().max()), 1e-6)

    def test_training_opens_deep_gradients_and_preserves_frozen_vlm(self):
        model, inputs, processor = fixture()
        original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        controller = attach(model, processor)
        controller.capture_diagnostics()
        optimizer = torch.optim.AdamW(controller.plugin.parameters(), lr=0.01)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(**inputs, use_cache=False).logits[:, -2].float(), torch.tensor([12]))
            loss.backward()
            gradients = controller.diagnostics(include_gradients=True)['gradient_norms']
            self.assertGreater(gradients['output.weight'], 0)
            if step == 0:
                self.assertEqual(gradients['layers.0.relation.q.weight'], 0)
            else:
                for name in ('content_in.1.weight', 'layers.0.relation.q.weight',
                             'layers.1.relation.q.weight', 'row_symbols.weight',
                             'reader_query.1.weight', 'reader_key.weight', 'reader_value.weight'):
                    self.assertGreater(gradients[name], 0, name)
            optimizer.step()
            self.assertIsNone(controller._memory)
        diagnostics = controller.diagnostics()
        self.assertGreater(diagnostics['delta_to_input_rms'], 0)
        self.assertEqual(len(diagnostics['memory_attention_entropy']), 2)
        self.assertGreater(diagnostics['reader_attention_entropy'], 0)
        self.assertEqual(diagnostics['memory_tokens'], 8)
        self.assertEqual(diagnostics['query_tokens'], 4)
        for name, parameter in model.named_parameters():
            if name in original:
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)
                self.assertTrue(torch.equal(original[name], parameter), name)

    def test_save_restore_and_external_scale(self):
        model, inputs, processor = fixture()
        controller = attach(model, processor)
        open_output(controller)
        with torch.no_grad():
            unscaled = model(**inputs, use_cache=False).logits
        scale = controller.plugin.register_forward_hook(lambda m, args, output: output * 0.01)
        with torch.no_grad():
            expected = model(**inputs, use_cache=False).logits
        self.assertFalse(torch.equal(unscaled[:, -1], expected[:, -1]))
        clone, clone_inputs, clone_processor = fixture()
        cloned_controller = attach(clone, clone_processor)
        cloned_controller.plugin.load_state_dict(copy.deepcopy(controller.plugin.state_dict()), strict=True)
        other_scale = cloned_controller.plugin.register_forward_hook(lambda m, args, output: output * 0.01)
        with torch.no_grad():
            self.assertTrue(torch.equal(expected, clone(**clone_inputs, use_cache=False).logits))
        scale.remove()
        other_scale.remove()

    def test_cached_uncached_logits_and_generation_agree(self):
        model, inputs, processor = fixture()
        controller = attach(model, processor)
        open_output(controller)
        continuation = [17, 18]
        expected = []
        with torch.no_grad():
            for count in range(3):
                full = dict(inputs)
                full['input_ids'] = torch.cat((inputs['input_ids'], torch.tensor([continuation[:count]], dtype=torch.long)), dim=1)
                full['attention_mask'] = torch.ones_like(full['input_ids'])
                expected.append(model(**full, use_cache=False).logits[:, -1])
            result = model(**inputs, use_cache=True)
            self.assertTrue(torch.allclose(result.logits[:, -1], expected[0], atol=1e-6, rtol=1e-5))
            self.assertIsNotNone(controller._memory)
            self.assertFalse(controller._memory.requires_grad)
            for count, token in enumerate(continuation, 1):
                result = model(input_ids=torch.tensor([[token]]), past_key_values=result.past_key_values,
                               attention_mask=torch.ones(1, inputs['input_ids'].shape[1] + count, dtype=torch.long),
                               use_cache=True)
                self.assertTrue(torch.allclose(result.logits[:, -1], expected[count], atol=1e-6, rtol=1e-5))
            cached = model.generate(**inputs, max_new_tokens=3, do_sample=False, use_cache=True)
            uncached = model.generate(**inputs, max_new_tokens=3, do_sample=False, use_cache=False)
            self.assertTrue(torch.equal(cached, uncached))

    def test_foreign_cache_disable_failure_and_new_prefill_invalidate_memory(self):
        model, inputs, processor = fixture()
        foreign_model, foreign_inputs, _ = fixture()
        controller = attach(model, processor)
        open_output(controller)
        with torch.no_grad():
            foreign = foreign_model(**foreign_inputs, use_cache=True).past_key_values
            model(**inputs, use_cache=True)
            with self.assertRaisesRegex(ValueError, 'exact cache'):
                model(input_ids=torch.tensor([[17]]), past_key_values=foreign, use_cache=True)
            self.assertIsNone(controller._memory)
            self.assertFalse(controller._active)
            result = model(**inputs, use_cache=True)
            controller.set_enabled(False)
            self.assertIsNone(controller._memory)
            controller.set_enabled(True)
            with self.assertRaisesRegex(ValueError, 'after image prefill'):
                model(input_ids=torch.tensor([[17]]), past_key_values=result.past_key_values, use_cache=True)
            first = model(**inputs, use_cache=True)
            expected = controller._memory.clone()
            changed = dict(inputs, pixel_values=inputs['pixel_values'].flip(1))
            model(**changed, use_cache=True)
            self.assertFalse(torch.equal(expected, controller._memory))
            with self.assertRaisesRegex(ValueError, 'exact cache'):
                model(input_ids=torch.tensor([[17]]), past_key_values=first.past_key_values, use_cache=True)
            model(**inputs, use_cache=True)
            self.assertTrue(torch.equal(expected, controller._memory))
            model(**inputs, use_cache=False)
            self.assertIsNone(controller._memory)
            model(**inputs, use_cache=True)
            controller.remove()
            self.assertIsNone(controller._memory)

    def test_cached_decode_rejects_gradients_and_padded_masks(self):
        model, inputs, processor = fixture()
        controller = attach(model, processor)
        with torch.no_grad():
            result = model(**inputs, use_cache=True)
        with self.assertRaisesRegex(ValueError, 'inference-only'):
            model(input_ids=torch.tensor([[17]]), past_key_values=result.past_key_values, use_cache=True)
        with torch.no_grad():
            result = model(**inputs, use_cache=True)
            with self.assertRaisesRegex(ValueError, 'unpadded'):
                model(input_ids=torch.tensor([[17]]), past_key_values=result.past_key_values,
                      attention_mask=torch.zeros(1, 14, dtype=torch.long), use_cache=True)
        self.assertIsNone(controller._memory)

    def test_native_validation_and_dimension_guards(self):
        model, inputs, processor = fixture()
        with self.assertRaisesRegex(ValueError, 'first native'):
            install(model, processor, layer_index=4, width=16, heads=4, depth=2)
        controller = attach(model, processor)
        with self.assertRaisesRegex(ValueError, 'already installed'):
            attach(model, processor)
        invalid = dict(inputs, attention_mask=torch.zeros_like(inputs['attention_mask']))
        with self.assertRaisesRegex(ValueError, 'Padded'):
            model(**invalid, use_cache=False)
        self.assertIsNone(controller._memory)
        with self.assertRaisesRegex(ValueError, 'Invalid memory'):
            AbstractorMemory(32, width=15, heads=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
