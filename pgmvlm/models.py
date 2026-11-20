"""Model loading, LoRA wiring, and per-candidate scoring: score(k) = logit("Yes") - logit("No") for the matrix completed
with candidate k; the 8 scores of a problem are compared with a softmax.

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


def detect_family(model_id, revision=None):
    cfg = AutoConfig.from_pretrained(model_id, **({'revision': revision} if revision else {}))
    mt = cfg.model_type
    if mt in ("lfm2_vl", "smolvlm", "qwen2_5_vl"):
        return {"lfm2_vl": "lfm25", "smolvlm": "smolvlm2", "qwen2_5_vl": "spatialevo"}[mt]
    if mt == "qwen3_vl":
        return "qwen3vl"
    if mt == "internvl":
        return "internvl"
    if mt == "internvl_chat":
        raise SystemExit(f"{model_id} is the original (non-HF) InternVL repo. Use the '-HF' checkpoint, "
                         f"e.g. OpenGVLab/InternVL3_5-2B-HF, which transformers 5.x supports natively.")
    print(f"[warn] model_type {mt!r} not explicitly supported; using the generic LLM-only LoRA targets")
    return "generic"


def load_base(model_id, dtype="bf16", attn="sdpa", device=None, family=None, revision=None):
    # The existing model IDs/defaults are untouched; new aliases pin our archived bases.
    from pgmvlm.registry import MODEL_REVISIONS
    if model_id in MODEL_REVISIONS:
        model_id, pinned = MODEL_REVISIONS[model_id]
        revision = revision or pinned
    family = family or detect_family(model_id, revision)
    kw = {} if attn == "default" else {"attn_implementation": attn}
    if revision:
        kw['revision'] = revision
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=DTYPES[dtype], **kw)
    processor = AutoProcessor.from_pretrained(model_id, **({'revision': revision} if revision else {}))
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
    targets = lora_target_regex(family, lora_vision, lora_projector)
    if family in ('lfm25', 'smolvlm2', 'spatialevo'):
        if lora_vision or lora_projector:
            raise ValueError('These model recipes support language-only LoRA')
        targets = [name for name, module in model.named_modules()
                   if name.startswith(('model.language_model.', 'model.text_model.', 'language_model.'))
                   and isinstance(module, torch.nn.Linear)]
        if not targets:
            raise ValueError('No audited language-model linear targets found')
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=targets)
    model = get_peft_model(model, cfg)
    return model


def load_adapter(model, adapter_dir, trainable=False, grad_ckpt=False):
    from peft import PeftModel
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return PeftModel.from_pretrained(model, adapter_dir, is_trainable=trainable)


FROZEN_BLOCKS = {"visual", "vision_tower", "vision_model", "multi_modal_projector", "connector"}


def set_train_mode(model):
    """model.train(), but vision tower / projector blocks without trainable parameters stay in eval mode (teammate recipe:
    frozen parts behave exactly as at inference, e.g. no dropout). Blocks that carry LoRA (--lora_vision/--lora_projector)
    are left in train mode."""
    model.train()
    for name, m in model.named_modules():
        if name.split(".")[-1] in FROZEN_BLOCKS and not any(p.requires_grad for p in m.parameters()):
            m.eval()
    return model


def param_report(model):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr, tot


# ----------------------------------------------------------------------------- score tokens
class ScoreTokens:
    """The two tokens whose logits make a candidate's score: the first token of "Yes" and of "No", resolved from the
    actual tokenizer. Real Qwen3 / InternVL3.5 tokenizers map both to single tokens; if a tokenizer splits them, the
    first tokens are used and they must differ."""

    def __init__(self, tokenizer, yes="Yes", no="No"):
        y = tokenizer(yes, add_special_tokens=False)["input_ids"]
        n = tokenizer(no, add_special_tokens=False)["input_ids"]
        if not y or not n or y[0] == n[0]:
            raise ValueError(f"{yes!r} -> {y} and {no!r} -> {n}: need distinct first tokens")
        self.yes_id, self.no_id = y[0], n[0]
        self.single = len(y) == 1 and len(n) == 1
        self.tokenizer = tokenizer

    def describe(self):
        d = self.tokenizer
        return (f"yes={d.decode([self.yes_id])!r} (id {self.yes_id}) no={d.decode([self.no_id])!r} (id {self.no_id}) "
                f"single-token={self.single}")


# ----------------------------------------------------------------------------- forward helpers
def to_model_inputs(batch, device, dtype):
    out = {}
    for k in MODEL_KEYS:
        if k in batch and batch[k] is not None:
            v = batch[k].to(device, non_blocking=True)
            out[k] = v.to(dtype) if k == "pixel_values" else v
    return out


def autocast_ctx(device, dtype):
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)


def candidate_scores(model, batch, st, device, dtype):
    """(B, 8) scores for a CandidateCollator batch of B problems (8*B sequences, problem-major).
    score = logit(Yes) - logit(No) at the first assistant position, computed in fp32 from the final (normed) hidden
    state: bf16 logits are rounded to ~0.06 at typical magnitudes, which made near-equal candidates tie exactly.
    Differentiable (training) and used as-is for evaluation (wrap in torch.no_grad())."""
    inputs = to_model_inputs(batch, device, dtype)
    with autocast_ctx(device, dtype):
        out = model(**inputs, use_cache=False, logits_to_keep=1, output_hidden_states=True)
    h = out.hidden_states[-1][:, -1].float()                                  # (8B, H), after the final norm
    head = model.get_output_embeddings()
    w = head.weight[[st.yes_id, st.no_id]].float()                            # (2, H)
    s = h @ (w[0] - w[1])
    if getattr(head, "bias", None) is not None:
        s = s + (head.bias[st.yes_id] - head.bias[st.no_id]).float()
    B = batch["targets"].shape[0]
    return s.view(B, -1)


def candidate_losses(scores, targets):
    """ce8: softmax over the 8 candidate scores + cross-entropy (the paper's loss).
    bce: each candidate on its own, Yes for the answer and No for the 7 others (positive weighted x7 to balance)."""
    targets = targets.to(scores.device)
    ce8 = F.cross_entropy(scores, targets)
    onehot = F.one_hot(targets, scores.shape[1]).float()
    bce = F.binary_cross_entropy_with_logits(scores, onehot, pos_weight=torch.tensor(scores.shape[1] - 1.0, device=scores.device))
    acc = (scores.argmax(-1) == targets).float().mean()
    return ce8, bce, acc
