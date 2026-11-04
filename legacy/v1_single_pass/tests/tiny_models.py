"""Random-weight, CPU-sized stand-ins for Qwen3-VL and InternVL3.5-HF (model + tokenizer + processor), so the whole
pipeline can be tested without downloading anything. They reproduce the *structure* the real checkpoints have
(module names, placeholder tokens, chat-template shape, digit-per-token tokenizer), not their quality."""
import os

import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

QWEN_SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
                 "<|image_pad|>", "<|video_pad|>"]
INTERN_SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<img>", "</img>", "<IMG_CONTEXT>", "<video>"]

QWEN_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{% if m['content'] is string %}{{ m['content'] }}{% else %}{% for c in m['content'] %}"
    "{% if c['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
INTERN_TEMPLATE = QWEN_TEMPLATE.replace("<|vision_start|><|image_pad|><|vision_end|>", "<IMG_CONTEXT>\n")

CORPUS = ["Answer: 1 2 3 4 5 6 7 8", "The first image shows a 3x3 abstract reasoning matrix with the bottom-right panel "
          "missing. The second image shows 8 candidate panels, numbered 1 through 8, arranged left to right and top to "
          "bottom. Determine which candidate completes the matrix by checking how shape type, size, color, count, and "
          "position, and line type and color, change across rows and columns (constant, progression, XOR, OR, AND, or "
          "consistent union). Do not explain your reasoning. Output only the final answer in this exact format and "
          "nothing else: Answer: <number> user assistant system"] * 4


def make_tokenizer(specials, extra=None, vocab_size=420):
    tok = Tokenizer(models.BPE())
    # Qwen's pre-tokenizer isolates every digit; mimic that so "Answer: 3" ends in a one-token digit.
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(r"\p{N}"), "isolated"),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False)])
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=specials,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS, tr)
    kw = dict(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    if extra:
        kw["extra_special_tokens"] = extra
    return PreTrainedTokenizerFast(**kw)


def tiny_qwen3vl(seed=0):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    tok = make_tokenizer(QWEN_SPECIALS)
    tok.chat_template = QWEN_TEMPLATE
    ids = {t: tok.convert_tokens_to_ids(t) for t in QWEN_SPECIALS}
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                         rope_parameters=dict(rope_type="default", rope_theta=10000.0, mrope_section=[2, 3, 3],
                                              mrope_interleaved=True),
                         tie_word_embeddings=True, pad_token_id=ids["<|endoftext|>"]),
        vision_config=dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=4, patch_size=16,
                           spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
                           num_position_embeddings=64, deepstack_visual_indexes=[0, 1]),
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
        tie_word_embeddings=True)
    torch.manual_seed(seed)
    model = Qwen3VLForConditionalGeneration(cfg).eval()
    ip = Qwen2VLImageProcessor(patch_size=16, size={"shortest_edge": 65536, "longest_edge": 16777216})
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    proc = Qwen3VLProcessor(image_processor=ip, tokenizer=tok, video_processor=Qwen3VLVideoProcessor(),
                            chat_template=QWEN_TEMPLATE)
    return model, proc


def tiny_internvl(seed=0, crop=True):
    from transformers import InternVLConfig, InternVLForConditionalGeneration, InternVLProcessor
    from transformers.models.got_ocr2.image_processing_got_ocr2 import GotOcr2ImageProcessor
    tok = make_tokenizer(INTERN_SPECIALS, extra={"start_image_token": "<img>", "end_image_token": "</img>",
                                                 "context_image_token": "<IMG_CONTEXT>", "video_token": "<video>"})
    tok.chat_template = INTERN_TEMPLATE
    ids = {t: tok.convert_tokens_to_ids(t) for t in INTERN_SPECIALS}
    cfg = InternVLConfig(
        text_config=dict(model_type="qwen3", vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         max_position_embeddings=4096, tie_word_embeddings=True,
                         pad_token_id=ids["<|endoftext|>"], eos_token_id=ids["<|im_end|>"]),
        vision_config=dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128,
                           image_size=[56, 56], patch_size=[14, 14]),
        image_token_id=ids["<IMG_CONTEXT>"], image_seq_length=4, downsample_ratio=0.5)
    torch.manual_seed(seed)
    model = InternVLForConditionalGeneration(cfg).eval()
    ip = GotOcr2ImageProcessor(size={"height": 56, "width": 56}, crop_to_patches=False, min_patches=1, max_patches=12)
    from transformers.models.internvl.video_processing_internvl import InternVLVideoProcessor
    proc = InternVLProcessor(image_processor=ip, tokenizer=tok, video_processor=InternVLVideoProcessor(),
                             image_seq_length=4, chat_template=INTERN_TEMPLATE)
    return model, proc
