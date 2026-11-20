"""Exp2 builder for OUR Slot Abstractor run (slot_abstractor/ at the repo root, trained by run_pgm_jsonl.py).

  SA_CKPT       the trained Slot Abstractor, e.g. .../weights/slot_abstractor_pgm_pgm20_1gpu_d24_lr8e-05_run0_best.pth.tar
                ({"abstractor_model_state_dict": 'module.'-prefixed weights, "args": the run's settings, ...})
  SA_SLOT_CKPT  its frozen slot autoencoder ({"slot_model_state_dict": ...}); default: the args' slot_checkpoint,
                looked up next to SA_CKPT (both live in weights/)
  SA_CODE_DIR   folder with slot_abstractor_pgm.py (default: <repo>/slot_abstractor)
  SA_ANTIALIAS  1 (default) / 0: how the 160 -> 80 px resize is done. It must match torchvision.transforms.Resize in
                the env the SA was trained in: torchvision >= 0.17 antialiases tensors (1), older versions do not (0).
                Check:  conda run -n slotabs python -c "import torchvision; print(torchvision.__version__)"
  SA_SEED       seed of the random slot initialisation (default 0)

What encode() returns, per completed matrix (8 context panels + one candidate):
  slots  (N, 9, 16, 32)  feat_slots of every panel: the slot-weighted CNN features the Abstractor reads
  rel    (N, 9, 16, 32)  the Abstractor's relation embeddings (its output before the mean and the linear scorer)
  score  (N,)            the SA's own score for this candidate; argmax over a puzzle's 8 = the SA's answer

The computation is the one in run_pgm_jsonl.py's score(): panels scaled to [-1, 1] and resized to 80 x 80, the
frozen slot autoencoder per panel, then scoring_model with the 8 context panels and one candidate. When the N rows
come as groups of 8 candidates of one puzzle (as in Exp2), the context panels go through slot attention once per
puzzle, exactly as in the original; test_sa_builder.py checks that the scores match the original code.
"""
import os
import sys
from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sa_interface import Capture  # noqa: E402

SA_CODE_DIR = os.environ.get("SA_CODE_DIR") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "slot_abstractor")
sys.path.insert(0, SA_CODE_DIR)
from slot_abstractor_pgm import SlotAttentionAutoEncoder, scoring_model  # noqa: E402

DEFAULTS = dict(img_size=80, depth=24, heads=8, mlp_dim=512, num_slots=16, num_iterations=3, hid_dim=32,
                apply_context_norm=True)


def _strip(sd):
    return OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())


class SlotAbstractor(nn.Module):
    def __init__(self, slot_model, abstractor, img_size):
        super().__init__()
        self.slot_model, self.abstractor, self.img_size = slot_model, abstractor, img_size


def build(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = SimpleNamespace(**{**DEFAULTS, **ck.get("args", {})})
    slot_path = os.environ.get("SA_SLOT_CKPT")
    if not slot_path:
        given = getattr(a, "slot_checkpoint", "slot_ae_pgm20.pth.tar")
        slot_path = given if os.path.isabs(given) and os.path.exists(given) else \
            os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), os.path.basename(given))
    if not os.path.exists(slot_path):
        raise SystemExit(f"slot autoencoder {slot_path} not found: export SA_SLOT_CKPT=<.../slot_ae_pgm20.pth.tar>")
    sck = torch.load(slot_path, map_location="cpu", weights_only=False)
    slot = SlotAttentionAutoEncoder((a.img_size, a.img_size), a.num_slots, a.num_iterations, a.hid_dim)
    slot.load_state_dict(_strip(sck.get("slot_model_state_dict", sck)))
    absr = scoring_model(a, a.hid_dim, a.depth, a.heads, a.mlp_dim, a.num_slots)
    absr.load_state_dict(_strip(ck["abstractor_model_state_dict"]))
    print(f"[sa_builder] Slot Abstractor {os.path.basename(ckpt_path)} (depth {a.depth}, {a.num_slots} slots x "
          f"{a.hid_dim}) + slot AE {os.path.basename(slot_path)}; antialias={os.environ.get('SA_ANTIALIAS', '1')}")
    return SlotAbstractor(slot, absr, a.img_size).to(device).eval()


def preprocess(panels, size):
    """uint8 (n, 160, 160) -> (n, 1, size, size): ToTensor (/255) -> 2x - 1 -> Resize(size), as in pgm_jsonl_data.py."""
    x = panels.float().div(255.0).mul(2).sub(1).unsqueeze(1)
    aa = os.environ.get("SA_ANTIALIAS", "1") == "1"
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=aa)


def _slots(model, imgs, device):
    """Frozen slot autoencoder on (n, 1, s, s) panels -> feat_slots, pos_slots (n, K, D)."""
    _, _, _, f, p, _ = model.slot_model(imgs, device)
    return f, p


@torch.no_grad()
def encode(model, panels):
    N = panels.shape[0]
    dev = next(model.parameters()).device
    seed = int(os.environ.get("SA_SEED", "0"))
    flat = panels.reshape(N, 9, *panels.shape[-2:])
    g8 = flat.reshape(N // 8, 8, 9, -1) if N % 8 == 0 else None
    grouped = g8 is not None and bool((g8[:, :, :8] == g8[:, :1, :8]).all())
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.manual_seed(seed)
        s = model.img_size
        if grouped:     # each puzzle's 8 context panels once, then the N candidates (as the original code)
            ctx = preprocess(flat.reshape(N // 8, 8, 9, *flat.shape[-2:])[:, 0, :8].reshape(-1, *flat.shape[-2:]), s)
            cf, cp = _slots(model, ctx, dev)
            af, ap = _slots(model, preprocess(flat[:, 8], s), dev)
            K, D = cf.shape[-2:]
            cf = cf.view(N // 8, 1, 8, K, D).expand(-1, 8, -1, -1, -1).reshape(N, 8, K, D)
            cp = cp.view(N // 8, 1, 8, K, D).expand(-1, 8, -1, -1, -1).reshape(N, 8, K, D)
            f = torch.cat([cf, af.unsqueeze(1)], 1)
            p = torch.cat([cp, ap.unsqueeze(1)], 1)
        else:
            f, p = _slots(model, preprocess(flat.reshape(-1, *flat.shape[-2:]), s), dev)
            K, D = f.shape[-2:]
            f, p = f.view(N, 9, K, D), p.view(N, 9, K, D)
        with Capture(model.abstractor, {"rel": "abstractors.abstractor"}) as cap:
            score = model.abstractor(f[:, :8], p[:, :8], f[:, 8:], p[:, 8:], dev)          # (N, 1): one candidate
    rel = cap["rel"]
    return {"slots": f, "rel": rel.reshape(N, 9, K, rel.shape[-1]), "score": score.reshape(N)}
