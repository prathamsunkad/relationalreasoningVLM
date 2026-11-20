"""How Exp2 talks to the pretrained Slot Abstractor (SA).

The SA code lives outside this repo, so Exp2 loads it through a small "builder" file that you write once
(start from sa_builder_template.py). The builder is a plain Python file with two functions:

    build(ckpt_path, device) -> model
        construct the SA exactly as it was trained and load its weights (eval mode, frozen)

    encode(model, panels) -> dict
        panels: uint8 tensor (N, 9, 160, 160) on `device`, the 8 context panels + one candidate in the
                bottom-right cell, row by row, straight from the PGM .npz (0-255, as stored)
        returns
          "slots": float (N, 9, K, D_obj)   object embeddings of every panel (slot attention output)
          "rel":   float (N, 9, K, D_rel)   relation embeddings after the Abstractor, one per slot
                   or    (N, R, D_rel)      if they are not per slot (then they are pooled into M tokens)
          "score": float (N,)               the SA's own score of this completed matrix (optional, but the
                                            complementarity check needs it)

Everything here runs under torch.no_grad(); the SA is never trained in Exp2.
"""
import importlib.util
import os

import torch
import torch.nn.functional as F


def load_builder(path):
    if not path or not os.path.exists(path):
        raise SystemExit(f"SA builder {path!r} not found: copy exp2/sa_builder_template.py, fill it in, pass --sa_builder")
    spec = importlib.util.spec_from_file_location("sa_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("build", "encode"):
        if not hasattr(mod, fn):
            raise SystemExit(f"{path} must define {fn}() (see exp2/sa_interface.py)")
    return mod


class FrozenSA:
    """The builder's model + a checked encode()."""

    def __init__(self, builder_path, ckpt, device):
        self.mod = load_builder(builder_path)
        self.model = self.mod.build(ckpt, device)
        if isinstance(self.model, torch.nn.Module):
            self.model.eval().requires_grad_(False)
        self.device = device
        self.dims = None

    @torch.no_grad()
    def __call__(self, panels):
        out = self.mod.encode(self.model, panels.to(self.device))
        n = panels.shape[0]
        s, r = out.get("slots"), out.get("rel")
        if s is None or r is None:
            raise RuntimeError("encode() must return 'slots' and 'rel'")
        if s.dim() != 4 or s.shape[:2] != (n, 9):
            raise RuntimeError(f"'slots' must be (N, 9, K, D), got {tuple(s.shape)}")
        if r.dim() not in (3, 4) or r.shape[0] != n:
            raise RuntimeError(f"'rel' must be (N, 9, K, D) or (N, R, D), got {tuple(r.shape)}")
        sc = out.get("score")
        if sc is not None:
            sc = sc.reshape(n).float()
        res = {"slots": s.float(), "rel": r.float(), "score": sc}
        if self.dims is None:
            self.dims = dict(d_obj=int(s.shape[-1]), d_rel=int(r.shape[-1]), rel_per_panel=r.dim() == 4,
                             K=int(s.shape[2]))
        return res


# ------------------------------------------------------------------ helpers for builders
def paper_preprocess(panels, size=80, invert=False):
    """The Slot Abstractor paper's PGM input: each panel resized to 80x80 and scaled to [-1, 1].
    panels uint8 (N, 9, H, W) -> float (N, 9, 1, size, size). invert=True flips black/white first, if your SA was
    trained on inverted panels (check: inspect should then report the SA's known accuracy)."""
    x = panels.float() / 255.0
    if invert:
        x = 1.0 - x
    n = x.shape[0]
    x = F.interpolate(x.reshape(n * 9, 1, *x.shape[-2:]), size=(size, size), mode="bilinear", antialias=True,
                      align_corners=False)
    return (x * 2 - 1).reshape(n, 9, 1, size, size)


class Capture:
    """Records the output of named submodules during one forward pass (for SA code whose forward returns only
    scores). Usage in a builder:
        cap = Capture(model, {"slots": "slot_attention", "rel": "abstractor"})
        with cap:  score = model(x)
        slots, rel = cap["slots"], cap["rel"]
    Find the module names with:  python exp2/exp2_sa_interleave.py inspect --list_modules ..."""

    def __init__(self, model, names, pick=None):
        self.model, self.names, self.pick = model, names, pick or {}
        mods = dict(model.named_modules())
        missing = [m for m in names.values() if m not in mods]
        if missing:
            raise SystemExit(f"modules {missing} not in the SA; run inspect --list_modules to see the names")
        self.mods, self.out, self.h = mods, {}, []

    def __enter__(self):
        self.out = {}
        for key, name in self.names.items():
            def hook(m, a, o, key=key):
                o = self.pick[key](o) if key in self.pick else (o[0] if isinstance(o, (tuple, list)) else o)
                self.out.setdefault(key, []).append(o)
            self.h.append(self.mods[name].register_forward_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self.h:
            h.remove()
        self.h = []

    def __getitem__(self, key):
        v = self.out[key]
        return v[0] if len(v) == 1 else v      # a module called several times gives a list
