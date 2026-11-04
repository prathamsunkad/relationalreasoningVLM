"""Model loading, LoRA wiring, answer-token bookkeeping and the two ways of reading an answer out of a VLM.

Supported (transformers 5.17.0, native classes, no trust_remote_code):
  Qwen3-VL           model_type "qwen3_vl"   e.g. Qwen/Qwen3-VL-2B-Instruct
  InternVL3.5 (-HF)  model_type "internvl"   e.g. OpenGVLab/InternVL3_5-2B-HF   (NOT the non-HF "internvl_chat" repo)
"""
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
MODEL_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
LLM_LINEARS = "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"


def detect_family(model_id):
    cfg = AutoConfig.from_pretrained(model_id)
    mt = cfg.model_type
    if mt == "qwen3_vl":
        return "qwen3vl"
    if mt == "internvl":
        return "internvl"
    if mt == "internvl_chat":
        raise SystemExit(f"{model_id} is the original (non-HF) InternVL repo. Use the '-HF' checkpoint, "
                         f"e.g. OpenGVLab/InternVL3_5-2B-HF, which transformers 5.x supports natively.")
    print(f"[warn] model_type {mt!r} not explicitly supported; using the generic LLM-only LoRA targets")
    return "generic"


def load_base(model_id, dtype="bf16", attn="sdpa", device=None, family=None):
    family = family or detect_family(model_id)
    kw = {} if attn == "default" else {"attn_implementation": attn}
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=DTYPES[dtype], **kw)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    if device is not None:
        model = model.to(device)
    return model, processor, family


# ----------------------------------------------------------------------------- LoRA
def lora_target_regex(family, lora_vision=False, lora_projector=False):
    """One regex (PEFT full-matches it against module names). Default: every linear layer of the language model
    only; the vision tower and projector stay frozen. Names verified against transformers 5.17.0."""
    parts = [rf".*language_model.*\.({LLM_LINEARS})"]
    if lora_vision:
        if family == "qwen3vl":
            parts.append(r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))")
        elif family == "internvl":
            parts.append(r".*vision_tower.*\.layer\.\d+\.(attention\.(q_proj|k_proj|v_proj|projection_layer)|mlp\.(fc1|fc2))")
    if lora_projector:
        if family == "qwen3vl":
            parts.append(r".*visual\.(merger|deepstack_merger_list\.\d+)\.(linear_fc1|linear_fc2)")
        elif family == "internvl":
            parts.append(r".*multi_modal_projector\.(linear_1|linear_2)")
    return "|".join(f"(?:{p})" for p in parts)


def add_lora(model, family, r=64, alpha=128, dropout=0.05, lora_vision=False, lora_projector=False, grad_ckpt=False):
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad_(False)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=lora_target_regex(family, lora_vision, lora_projector))
    model = get_peft_model(model, cfg)
    return model


def load_adapter(model, adapter_dir, trainable=False, grad_ckpt=False):
    from peft import PeftModel
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return PeftModel.from_pretrained(model, adapter_dir, is_trainable=trainable)


def param_report(model):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr, tot


# ----------------------------------------------------------------------------- answer tokens
class AnswerTokens:
    """Token-level facts about the answer string "Answer: <k>", resolved from the actual tokenizer (nothing is
    assumed): the shared prefix tokens, one token id per digit 1..8, and the full target sequence incl. <|im_end|>."""

    def __init__(self, tokenizer, eos_token="<|im_end|>"):
        eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        if eos_id is None or eos_id == tokenizer.unk_token_id:
            raise ValueError(f"end-of-turn token {eos_token!r} not in tokenizer")
        seqs = [tokenizer(f"Answer: {k}", add_special_tokens=False)["input_ids"] for k in range(1, 9)]
        if len({len(s) for s in seqs}) != 1:
            raise ValueError(f"'Answer: k' tokenizes to different lengths: {[len(s) for s in seqs]}")
        prefix = seqs[0][:-1]
        if any(s[:-1] != prefix for s in seqs):
            raise ValueError("'Answer: k' has no common token prefix")
        digit_ids = [s[-1] for s in seqs]
        if len(set(digit_ids)) != 8:
            raise ValueError("digits 1..8 do not map to 8 distinct final tokens")
        self.prefix = prefix
        self.digit_ids = digit_ids
        self.digit_pos = len(prefix)                       # index of the digit inside the response
        self.full = torch.tensor([s + [eos_id] for s in seqs], dtype=torch.long)   # (8, R)
        self.R = self.full.shape[1]
        self.eos_id = eos_id
        self.prefix_t = torch.tensor(prefix, dtype=torch.long)
        self.digit_ids_t = torch.tensor(digit_ids, dtype=torch.long)
        self.tokenizer = tokenizer

    def describe(self):
        d = self.tokenizer
        return (f"prefix={[d.decode([i]) for i in self.prefix]} digits={[d.decode([i]) for i in self.digit_ids]} "
                f"response_len={self.R}")


# ----------------------------------------------------------------------------- forward helpers
def to_model_inputs(batch, device, dtype):
    out = {}
    for k in MODEL_KEYS:
        if k in batch and batch[k] is not None:
            v = batch[k].to(device, non_blocking=True)
            out[k] = v.to(dtype) if k == "pixel_values" else v
    return out


def append_tokens(inputs, ids):
    """Append `ids` (B, n) on the right of a (left-padded) batch; keeps masks / mm_token_type_ids aligned."""
    ids = ids.to(inputs["input_ids"].device)
    out = dict(inputs)
    out["input_ids"] = torch.cat([inputs["input_ids"], ids], 1)
    out["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(ids)], 1)
    if "mm_token_type_ids" in inputs:
        out["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(ids)], 1)
    return out


def autocast_ctx(device, dtype):
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)


def teacher_forced_logits(model, batch, answer, device, dtype):
    """Forward over prompt + full answer; returns (R, B) shaped logits for the R response tokens: (B, R, V)."""
    targets = batch["targets"].to(device)
    resp = answer.full.to(device)[targets]                                   # (B, R)
    inputs = append_tokens(to_model_inputs(batch, device, dtype), resp)
    with autocast_ctx(device, dtype):
        out = model(**inputs, use_cache=False, logits_to_keep=answer.R + 1)
    return out.logits[:, :-1].float(), resp, targets


def lm_and_digit_losses(logits, resp, targets, answer):
    V = logits.shape[-1]
    lm = F.cross_entropy(logits.reshape(-1, V), resp.reshape(-1))
    dig = logits[:, answer.digit_pos][:, answer.digit_ids_t.to(logits.device)]           # (B, 8)
    d8 = F.cross_entropy(dig, targets)
    acc = (dig.argmax(-1) == targets).float().mean()
    return lm, d8, acc


@torch.no_grad()
def restricted_logits(model, batch, answer, device, dtype):
    """Paper-style readout: feed prompt + "Answer: ", take the next-token logits over exactly the 8 digit tokens.
    Returns (B, 8). The model can never produce an invalid answer, like the paper's 8-way softmax over candidates."""
    inputs = to_model_inputs(batch, device, dtype)
    B = inputs["input_ids"].shape[0]
    inputs = append_tokens(inputs, answer.prefix_t.unsqueeze(0).expand(B, -1))
    with autocast_ctx(device, dtype):
        out = model(**inputs, use_cache=False, logits_to_keep=1)
    return out.logits[:, -1].float()[:, answer.digit_ids_t.to(device)]


@torch.no_grad()
def generate_text(model, processor, batch, device, dtype, max_new_tokens=16, stop_ids=None):
    """Free greedy generation, exactly like pgm_infer.py (can produce invalid answers)."""
    inputs = to_model_inputs(batch, device, dtype)
    tok = processor.tokenizer
    # always stop at the end-of-turn token, whatever the checkpoint's generation_config says
    stops = sorted({i for i in list(stop_ids or []) + [tok.eos_token_id] if i is not None})
    with autocast_ctx(device, dtype):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                             top_k=None, eos_token_id=stops,
                             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    new = out[:, inputs["input_ids"].shape[1]:]
    return tok.batch_decode(new, skip_special_tokens=True)
