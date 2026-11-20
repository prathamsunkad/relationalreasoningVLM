"""Offline native LFM fixture; synthetic processor avoids fetching a tokenizer."""
from types import SimpleNamespace
import numpy as np
import torch
from transformers import Lfm2VlConfig, Lfm2VlForConditionalGeneration


class Tokenizer:
    padding_side = 'left'
    model_max_length = 256
    def encode(self, text, **kwargs):
        return [12 + ord(text) - 65] if text in list('ABCDEFGH') else [10, 11]
    def decode(self, ids, **kwargs):
        return ''.join(chr(int(i) - 12 + 65) for i in ids if 12 <= int(i) < 20)


class Processor:
    tokenizer = Tokenizer()
    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        ids = [1] + [396] * 8 + [9, 10, 11]
        if not add_generation_prompt:
            ids += self.tokenizer.encode(messages[-1]['content'][0]['text']) + [2]
        image = messages[0]['content'][0]['image']
        pixels = np.asarray(image.resize((8, 8)), dtype=np.float32).reshape(16, 12) / 255.
        return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                    pixel_values=torch.tensor(np.stack([pixels, pixels])),
                    pixel_attention_mask=torch.ones(2, 16, dtype=torch.long), spatial_shapes=torch.tensor([[4, 4], [4, 4]]))


def tiny_lfm(seed=0):
    torch.manual_seed(seed)
    config = Lfm2VlConfig(image_token_id=396, downsample_factor=2, projector_hidden_size=64,
        text_config=dict(model_type='lfm2', hidden_size=32, intermediate_size=64, block_multiple_of=8,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         layer_types=['conv', 'conv', 'full_attention', 'conv'], vocab_size=512,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2),
        vision_config=dict(model_type='siglip2_vision_model', hidden_size=32, intermediate_size=64,
                           num_hidden_layers=1, num_attention_heads=4, patch_size=2,
                           num_patches=16, vision_use_head=False))
    return Lfm2VlForConditionalGeneration(config).eval(), Processor()
