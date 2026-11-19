"""Build / save / restore a RelationalVLM (shared by train_stage1.py and eval.py)."""
import torch

from backbones import Backbone, load_vlm
from relvlm import RelationalVLM, ScoreTokens
from slots import load_slot_ae

LLM_LINEARS = "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"


def add_lora(model, r, alpha, dropout):
    """LoRA on every linear layer of the language model only (same targets as the vlm_lora_pc baseline)."""
    from peft import LoraConfig, inject_adapter_in_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=rf".*language_model.*\.({LLM_LINEARS})")
    inject_adapter_in_model(cfg, model)          # in place: module objects (and our hooks) stay the same
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
        if "lora_" in n:
            p.data = p.data.float()
    return model


def build(model_id, slot_ckpt, cfg, device, dtype_name="bf16", attn="sdpa", lora=None):
    """cfg: RelationalVLM kwargs (inject, rel_tokens, abstractor_cfg, fuse_lambda, learn_lambda).
    lora: None or dict(r, alpha, dropout)."""
    model, processor, family = load_vlm(model_id, dtype_name, attn, device)
    model.requires_grad_(False)
    bb = Backbone(model, processor, family)
    ae, ae_ck = load_slot_ae(slot_ckpt, device)
    assert ae.cfg["in_dim"] == bb.vit_dim, f"slot ckpt in_dim {ae.cfg['in_dim']} != {family} ViT width {bb.vit_dim}"
    if ae_ck.get("backbone") and ae_ck["backbone"] != model_id:
        print(f"[warn] slot checkpoint was trained on {ae_ck['backbone']}, not {model_id}")
    if lora:
        add_lora(model, **lora)
    rv = RelationalVLM(bb, ae, **cfg).to(device)
    return rv, bb, ScoreTokens(processor.tokenizer)


def trainable_lora(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n}


def adapter_lora_cfg(adapter_dir):
    """dict(r, alpha, dropout) of a PEFT adapter saved with save_pretrained (e.g. a vlm_lora_pc run's best/)."""
    import json
    import os
    c = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    return dict(r=int(c["r"]), alpha=float(c["lora_alpha"]), dropout=float(c.get("lora_dropout", 0.0)))


def load_peft_adapter(model, adapter_dir, lora_cfg=None):
    """Copy a PEFT adapter (vlm_lora_pc's save_pretrained format: 'base_model.model.<name>.lora_A.weight') into the
    LoRA modules add_lora injected ('<name>.lora_A.default.weight'). Every adapter tensor must land on a LoRA
    parameter of the model and every LoRA parameter must be covered, with matching r and alpha; returns the count."""
    import os
    from safetensors.torch import load_file
    cfg = adapter_lora_cfg(adapter_dir)
    if lora_cfg and (int(lora_cfg["r"]) != cfg["r"] or float(lora_cfg["alpha"]) != cfg["alpha"]):
        raise RuntimeError(f"adapter {adapter_dir} has r={cfg['r']} alpha={cfg['alpha']}, the model was built with "
                           f"r={lora_cfg['r']} alpha={lora_cfg['alpha']}")
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    params = {n: p for n, p in model.named_parameters() if "lora_" in n}
    seen = set()
    for k, t in sd.items():
        n = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
        for ab in ("lora_A", "lora_B"):
            n = n.replace(f".{ab}.weight", f".{ab}.default.weight")
        if n not in params:
            raise RuntimeError(f"adapter tensor {k} -> {n} has no LoRA parameter in the model")
        if tuple(params[n].shape) != tuple(t.shape):
            raise RuntimeError(f"shape mismatch for {n}: model {tuple(params[n].shape)} adapter {tuple(t.shape)}")
        params[n].data.copy_(t.to(params[n].device, params[n].dtype))
        seen.add(n)
    if len(seen) != len(params):
        raise RuntimeError(f"adapter covers {len(seen)} of {len(params)} LoRA parameters, e.g. "
                           f"{sorted(set(params) - seen)[:3]}")
    return len(seen)


def checkpoint_payload(rv, bb, meta):
    keep_slots = getattr(rv, "finetune_slots", False) or getattr(rv, "save_slots", False)   # save_slots: loaded fine-tuned
    extra = dict(slot_ae=rv.slot_ae.state_dict()) if keep_slots else {}
    return dict(branch=rv.state_dict(), lora={n: p.detach().cpu() for n, p in trainable_lora(bb.model).items()},
                **extra, **meta)


def restore(rv, bb, ck, strict=True):
    missing, unexpected = rv.load_state_dict(ck["branch"], strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"branch state mismatch: missing {missing} unexpected {unexpected}")
    if ck.get("slot_ae"):
        rv.slot_ae.load_state_dict(ck["slot_ae"])
    if ck.get("lora"):
        params = dict(bb.model.named_parameters())
        for n, t in ck["lora"].items():
            if n not in params:
                raise RuntimeError(f"LoRA param {n} not in model (build with the same --lora_r)")
            params[n].data.copy_(t.to(params[n].device, params[n].dtype))
