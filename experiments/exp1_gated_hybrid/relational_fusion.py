"""
Gated relational fusion for InternVL (transformers-native "-HF" checkpoints).

Where it plugs in
-----------------
InternVL turns each 448x448 tile into 256 vision tokens:
    InternViT -> pixel shuffle (16x16 x 4*vit_dim) -> multi_modal_projector -> LLM width
and scatters those tokens into the <IMG_CONTEXT> placeholders. This module REPLACES the
projector with a wrapper that keeps the original projector and adds a relational path:

    v  = projector(x)                                  original vision tokens  [256, d]
    slots, A = SlotAttention(features of the panel)    K object slots per panel, A = token->slot assignment
    E  = Abstractor(slots of the 3x3 grid, once per candidate)   relational state per slot
    u  = up( A @ E )                                   relational info broadcast back to the panel's 256 tokens
    g  = sigmoid( W [LN(v); LN(u)] + b )               per-token, per-channel gate
    out = v + g * u                                    same 256 tokens -> decoder

`up` is zero-initialised, so at step 0 the model is exactly the pretrained InternVL.

Where the slots get their features (`source`, `slot_layer`) - the stage-0 ablation:
    vit, slot_layer=-1   last InternViT layer (the tokens the projector sees)       [default]
    vit, slot_layer=L    output of InternViT layer L (0 = patch embeddings, no self-attention)
    cnn                  a small CNN on the panel pixels, trained from scratch (as in the Slot Abstractor)
    resnet               frozen ImageNet ResNet-18 up to layer2 (needs torchvision and its weights)
The LLM always receives the last-layer tokens v; only u depends on the source. When the slot grid
differs from the 16x16 token grid, the assignment map A is average-pooled onto the token grid.

Optional reconstruction loss (`recon_weight` in train_hybrid.py): a spatial-broadcast decoder
rebuilds each panel's grayscale image from its slots - the usual way slot attention learns to
bind one object per slot. It is the same for every source, so the ablation stays fair.

Input convention (PGM): each puzzle is 16 images, one tile each (pass crop_to_patches=False
to the processor): the 8 context panels in reading order, then choices A-H. Tiles are grouped
16 at a time in batch order.

The abstractor is a compact relational cross-attention (queries/keys from slot features, values
from learned symbols). To reproduce the Slot Abstractor exactly, port its blocks into
SlotAttention / Abstractor.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_position(h, w, device, dtype):
    """[h*w, 4] grid coordinates (x, y, 1-x, 1-y), as in Slot Attention."""
    ys, xs = torch.meshgrid(torch.linspace(0, 1, h, device=device, dtype=dtype),
                            torch.linspace(0, 1, w, device=device, dtype=dtype), indexing="ij")
    g = torch.stack([xs, ys], -1).reshape(h * w, 2)
    return torch.cat([g, 1 - g], -1)


class SlotAttention(nn.Module):
    """Locatello et al. (2020). Returns slots [B, K, D] and token->slot assignment [B, N, K]."""

    def __init__(self, dim, n_slots=6, iters=3, hidden=256):
        super().__init__()
        self.n_slots, self.iters, self.scale = n_slots, iters, dim ** -0.5
        self.init = nn.Parameter(torch.randn(1, n_slots, dim) * dim ** -0.5)   # learned, deterministic start
        self.norm_in, self.norm_slot, self.norm_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, x):
        B, N, D = x.shape
        x = self.norm_in(x)
        k, v = self.k(x), self.v(x)
        slots = self.init.expand(B, -1, -1).to(x.dtype)
        for _ in range(self.iters):
            prev = slots
            q = self.q(self.norm_slot(slots))
            attn = torch.softmax(torch.einsum("bnd,bkd->bnk", k, q) * self.scale, dim=-1)   # tokens compete for slots
            w = attn / (attn.sum(1, keepdim=True) + 1e-8)
            upd = torch.einsum("bnk,bnd->bkd", w, v)
            slots = self.gru(upd.reshape(-1, D), prev.reshape(-1, D)).reshape(B, self.n_slots, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn


class RelationalLayer(nn.Module):
    """Relational cross-attention: attention pattern from object features X, values from abstract states E."""

    def __init__(self, dim, heads=4, qk_norm=True):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        # LayerNorm on the object features before the attention scores. Without it, slot features grow
        # during training, the attention saturates (each position attends to one key), swapping the
        # candidate stops changing the output and the gradients that could fix it vanish (collapse).
        self.nx = nn.LayerNorm(dim) if qk_norm else nn.Identity()
        self.qk = nn.Linear(dim, 2 * dim, bias=False)
        self.v, self.o = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, X, E):
        B, T, D = X.shape
        q, k = self.qk(self.nx(X)).view(B, T, 2, self.h, self.dh).unbind(2)
        v = self.v(self.n1(E)).view(B, T, self.h, self.dh)
        a = torch.softmax(torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.dh), -1)
        # diagnostics (no sync): mean top attention weight (1.0 = saturated), mean slot-feature norm
        self.diag = (a.detach().amax(-1).mean(), X.detach().norm(dim=-1).mean())
        E = E + self.o(torch.einsum("bhts,bshd->bthd", a, v).reshape(B, T, D))
        return E + self.mlp(self.n2(E))


class Abstractor(nn.Module):
    """Runs over the 9 cells x K slots of a completed grid. Symbols encode the cell (row, column)."""

    def __init__(self, dim, layers=2, heads=4, qk_norm=True):
        super().__init__()
        self.row, self.col = nn.Embedding(3, dim), nn.Embedding(3, dim)
        self.symbol = nn.Parameter(torch.randn(9, dim) * 0.02)
        self.layers = nn.ModuleList(RelationalLayer(dim, heads, qk_norm) for _ in range(layers))

    def forward(self, slots):                       # slots: [B, 9, K, D]
        B, C, K, D = slots.shape
        cell = torch.arange(9, device=slots.device)
        pos = (self.row(cell // 3) + self.col(cell % 3))[None, :, None]          # [1, 9, 1, D]
        X = (slots + pos).reshape(B, C * K, D)
        E = (self.symbol[None, :, None] + pos).expand(B, C, K, D).reshape(B, C * K, D)
        for layer in self.layers:
            E = layer(X, E)
        return E.view(B, C, K, D)




class CNNEncoder(nn.Module):
    """Slot-Attention-style CNN, trained from scratch: 128x128 panel -> 32x32 grid of features."""

    def __init__(self, dim, res=128, ch=64):
        super().__init__()
        self.res = res
        self.net = nn.Sequential(nn.Conv2d(3, ch, 5, 1, 2), nn.ReLU(), nn.Conv2d(ch, ch, 5, 2, 2), nn.ReLU(),
                                 nn.Conv2d(ch, ch, 5, 1, 2), nn.ReLU(), nn.Conv2d(ch, dim, 5, 2, 2))

    def forward(self, pixels):
        x = F.interpolate(pixels, size=(self.res, self.res), mode="area")
        return self.net(x).permute(0, 2, 3, 1)                         # [T, 32, 32, dim]


class ResNetEncoder(nn.Module):
    """Frozen ImageNet ResNet-18 up to layer2: 224x224 panel -> 28x28x128, then a trainable projection."""

    def __init__(self, dim):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2).eval()
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        self.proj = nn.Linear(128, dim)

    def train(self, mode=True):
        super().train(mode)
        self.trunk.eval()          # keep batch-norm statistics frozen
        return self

    def forward(self, pixels):     # InternVL pixels already use ImageNet mean/std
        with torch.no_grad():
            f = self.trunk(F.interpolate(pixels.float(), size=(224, 224), mode="bilinear", align_corners=False))
        return self.proj(f.permute(0, 2, 3, 1))                         # [T, 28, 28, dim]


class BroadcastDecoder(nn.Module):
    """Spatial-broadcast decoder: each slot paints a grayscale image and an alpha mask; masks compete."""

    def __init__(self, dim, res=64, ch=32):
        super().__init__()
        self.res = res
        self.pos = nn.Linear(4, dim)
        self.net = nn.Sequential(nn.Conv2d(dim, ch, 3, 1, 1), nn.ReLU(), nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(),
                                 nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(), nn.Conv2d(ch, 2, 1))

    def forward(self, slots):                                           # [M, K, D] -> image [M, 1, R, R]
        M, K, D = slots.shape
        R = self.res
        pos = self.pos(soft_position(R, R, slots.device, slots.dtype)).view(1, R, R, D)
        x = (slots.reshape(M * K, 1, 1, D) + pos).permute(0, 3, 1, 2)
        out = self.net(x).view(M, K, 2, R, R)
        alpha = out[:, :, 1:].softmax(1)
        return (alpha * out[:, :, :1]).sum(1), alpha


class RelationalBranch(nn.Module):
    """Encoder for the chosen source + slot attention per panel + abstractor over each
    candidate-completed grid + candidate score head (+ optional reconstruction decoder)."""

    def __init__(self, in_dim, rel_dim=256, n_slots=10, slot_iters=3, abs_layers=2, missing=8, panels=16,
                 source="vit", recon_res=64, qk_norm=True, **_):
        super().__init__()
        self.missing, self.panels, self.source = missing, panels, source
        if source == "vit":
            self.enc = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, rel_dim))
        elif source == "cnn":
            self.enc = CNNEncoder(rel_dim)
        elif source == "resnet":
            self.enc = ResNetEncoder(rel_dim)
        else:
            raise ValueError(f"unknown slot source {source!r}")
        self.pos = nn.Linear(4, rel_dim)
        self.slot_attn = SlotAttention(rel_dim, n_slots, slot_iters)
        self.abstractor = Abstractor(rel_dim, abs_layers, qk_norm=qk_norm)
        self.score_head = nn.Sequential(nn.LayerNorm(rel_dim), nn.Linear(rel_dim, rel_dim), nn.GELU(),
                                        nn.Linear(rel_dim, 1))       # stage-0 / auxiliary loss only
        self.decoder = BroadcastDecoder(rel_dim, recon_res)

    def encode(self, inp):
        """vit: inp [T, N, in_dim] on a square grid; cnn/resnet: inp = pixels [T, 3, H, W].
        Returns tokens [T, h*w, D] (with position) and (h, w)."""
        if self.source == "vit":
            T, N, _ = inp.shape
            h = w = int(round(N ** 0.5))
            f = self.enc(inp)
        else:
            g = self.enc(inp)
            T, h, w, _ = g.shape
            f = g.reshape(T, h * w, -1)
        return f + self.pos(soft_position(h, w, f.device, f.dtype)), (h, w)

    def forward(self, inp, out_tokens=None, need_u=True, recon_target=None, recon_panels=32):
        """Returns u [T, out_tokens, D] (or None), candidate logits [P, 8], assignment on the token grid
        (or None) and the reconstruction loss (or None)."""
        tokens, (h, w) = self.encode(inp)
        T = tokens.shape[0]
        P = T // self.panels
        slots, A = self.slot_attn(tokens)                              # [T, K, D], [T, h*w, K]
        K, D = slots.shape[1:]
        s = slots.view(P, self.panels, K, D)
        ctx, ch = s[:, :8], s[:, 8:]                                   # 8 context panels, 8 choices
        before, after = ctx[:, :self.missing], ctx[:, self.missing:]
        # one completed 3x3 grid per candidate, as in the Slot Abstractor: [P*8, 9, K, D]
        grids = torch.cat([before[:, None].expand(P, 8, *before.shape[1:]), ch[:, :, None],
                           after[:, None].expand(P, 8, *after.shape[1:])], 2).reshape(P * 8, 9, K, D)
        E = self.abstractor(grids).view(P, 8, 9, K, D)
        logits = self.score_head(E.mean((2, 3))).squeeze(-1)          # [P, 8]

        recon = None
        if recon_target is not None:                                   # a random subset of panels
            idx = torch.randperm(T, device=slots.device)[:recon_panels]
            img, _ = self.decoder(slots[idx])
            tgt = F.interpolate(recon_target[idx].float(), size=img.shape[-2:], mode="area").mean(1, keepdim=True)
            recon = F.mse_loss(img.float(), tgt)

        if not need_u:
            return None, logits, None, recon
        ctx_cells = [c for c in range(9) if c != self.missing]
        ctx_E = E[:, :, ctx_cells].mean(1)                             # context panels: averaged over candidates
        ch_E = E[:, :, self.missing]                                  # choice k: its own grid, at the missing cell
        panel_E = torch.cat([ctx_E, ch_E], 1).reshape(T, K, D)
        n_out = out_tokens or h * w
        if n_out != h * w:                                             # slot grid -> token grid
            gs = int(round(n_out ** 0.5))
            A = F.adaptive_avg_pool2d(A.view(T, h, w, K).permute(0, 3, 1, 2), gs).permute(0, 2, 3, 1)
            A = A.reshape(T, n_out, K)
        u = torch.einsum("tnk,tkd->tnd", A, panel_E)                  # broadcast slot states to the panel's tokens
        return u, logits, A, recon


class GatedRelationalProjector(nn.Module):
    def __init__(self, projector, in_dim, out_dim, missing=8, panels=16, rel_dim=256, n_slots=10,
                 slot_iters=3, abs_layers=2, gate="channel", n_branches=1, source="vit", slot_layer=-1,
                 recon_res=64, recon_panels=32, qk_norm=True, **_):
        super().__init__()
        self.projector = projector                  # the original InternVL projector (kept frozen)
        self.panels = panels
        self.config = dict(missing=missing, panels=panels, rel_dim=rel_dim, n_slots=n_slots, slot_iters=slot_iters,
                           abs_layers=abs_layers, gate=gate, n_branches=n_branches, source=source,
                           slot_layer=slot_layer, recon_res=recon_res, recon_panels=recon_panels, qk_norm=qk_norm)
        self.branches = nn.ModuleList(
            RelationalBranch(in_dim, rel_dim, n_slots, slot_iters, abs_layers, missing, panels, source, recon_res,
                             qk_norm=qk_norm)
            for _ in range(n_branches))
        self.up = nn.Linear(rel_dim, out_dim)
        nn.init.zeros_(self.up.weight), nn.init.zeros_(self.up.bias)  # identity at initialisation
        self.nv, self.nu = nn.LayerNorm(out_dim), nn.LayerNorm(out_dim)
        self.gate = nn.Linear(2 * out_dim, out_dim if gate == "channel" else 1)
        nn.init.zeros_(self.gate.bias)
        self.enabled = True        # False = plain InternVL (ablation)
        self.route = None          # per-puzzle branch index (cross-fitting); None = average all branches
        self.shuffle = False       # control: give each puzzle another puzzle's relational features
        self.compute_recon = False # stage 0: also compute the reconstruction loss
        self.stage0_only = False   # stage 0: compute candidate logits only, pass v through unchanged
        self.last = {}             # gate stats, candidate logits, reconstruction loss of the last forward
        self._pixels = None        # set by a hook on the vision tower
        self._layer_feats = None   # set by a hook on the chosen ViT layer
        self._shuffle_fn = None    # InternVL's pixel shuffle, set by install()

    def diagnostics(self):
        """Abstractor attention saturation and slot-feature size, averaged over the first branch's layers."""
        d = [l.diag for l in self.branches[0].abstractor.layers if hasattr(l, "diag")]
        if not d:
            return {}
        return {"attn_max": round(float(sum(x[0] for x in d) / len(d)), 4),
                "x_norm": round(float(sum(x[1] for x in d) / len(d)), 2)}

    # -- inputs for the slots ------------------------------------------------------------
    def slot_input(self, x):
        src, L = self.config["source"], self.config["slot_layer"]
        if src in ("cnn", "resnet"):
            if self._pixels is None:
                raise RuntimeError("no pixels captured: the vision-tower hook is missing (use install())")
            return self._pixels
        if L is None or L < 0:
            return x
        f = self._layer_feats
        if f is None:
            raise RuntimeError("no layer features captured: the layer hook is missing (use install())")
        T, n, C = f.shape
        s = int(math.isqrt(n))
        if s * s != n:             # drop the CLS token
            f, n = f[:, 1:], n - 1
            s = int(math.isqrt(n))
        f = self._shuffle_fn(f.reshape(T, s, s, C), scale_factor=0.5)
        return f.reshape(T, -1, f.shape[-1])

    def run_branches(self, inp, out_tokens, need_u=True):
        T = inp.shape[0]
        if T % self.panels:
            raise ValueError(f"{T} tiles is not a multiple of {self.panels}: give each panel exactly one tile "
                             "(processor(..., crop_to_patches=False)) and 16 images per puzzle")
        P = T // self.panels
        recon_t = self._pixels if (self.compute_recon and self.training) else None
        kw = dict(out_tokens=out_tokens, need_u=need_u, recon_target=recon_t,
                  recon_panels=self.config["recon_panels"])
        if len(self.branches) == 1:
            return self.branches[0](inp, **kw)
        if self.route is None:     # evaluation: average the branches
            outs = [b(inp, **{**kw, "recon_target": None}) for b in self.branches]
            avg = lambda xs: None if xs[0] is None else torch.stack(xs).mean(0)
            return tuple(avg(list(o)) for o in zip(*outs))
        route = torch.as_tensor(self.route, device=inp.device)
        assert len(route) == P, f"route has {len(route)} entries for {P} puzzles"
        xs = inp.view(P, self.panels, *inp.shape[1:])
        u = logits = A = None
        for i, b in enumerate(self.branches):
            sel = (route == i).nonzero().squeeze(-1)
            if len(sel) == 0:
                continue
            ui, li, Ai, _ = b(xs[sel].reshape(-1, *inp.shape[1:]), **{**kw, "recon_target": None})
            if logits is None:
                logits = li.new_zeros(P, 8)
                if need_u:
                    u = ui.new_zeros(P, self.panels, *ui.shape[1:])
                    A = Ai.new_zeros(P, self.panels, *Ai.shape[1:])
            logits[sel] = li
            if need_u:
                u[sel] = ui.view(len(sel), self.panels, *ui.shape[1:])
                A[sel] = Ai.view(len(sel), self.panels, *Ai.shape[1:])
        if need_u:
            u, A = u.reshape(T, *u.shape[2:]), A.reshape(T, *A.shape[2:])
        return u, logits, A, None

    # -- forward ---------------------------------------------------------------------------
    def stage0(self, x=None):
        """Candidate logits (and reconstruction loss) only, without computing u. For cnn/resnet the
        vision tower need not run: set `_pixels` and call with x=None."""
        inp = self.slot_input(x)
        if inp.dtype == torch.float16:
            inp = inp.float()
        _, logits, _, recon = self.run_branches(inp, None, need_u=False)
        self.last = {"candidate_logits": logits, "recon_loss": recon}
        return logits

    def forward(self, x):
        v = self.projector(x)
        if not self.enabled:
            return v
        if self.stage0_only:
            self.stage0(x)
            return v
        inp = self.slot_input(x)
        if inp.dtype == torch.float16:
            inp = inp.float()
        u, logits, A, recon = self.run_branches(inp, x.shape[1])
        if self.shuffle:
            P = x.shape[0] // self.panels
            u = u.view(P, self.panels, *u.shape[1:]).roll(1, 0).reshape(u.shape)
        u = self.up(u.to(v.dtype))
        g = torch.sigmoid(self.gate(torch.cat([self.nv(v), self.nu(u)], -1)))
        self.last = {"candidate_logits": logits, "recon_loss": recon, "gate_mean": g.mean().detach(),
                     "gate_per_panel": g.mean((1, 2)).detach(), "slot_assign": A.detach()}
        return v + g * u


def _vision_parts(model):
    inner = model.model if hasattr(model, "model") and hasattr(model.model, "multi_modal_projector") else model
    return inner, inner.vision_tower


def install(model, missing=8, **kw):
    """Wrap the projector of an InternVLForConditionalGeneration in place, attach the hooks the chosen
    slot source needs, and return the fusion module."""
    inner, tower = _vision_parts(model)
    proj = inner.multi_modal_projector
    in_dim = proj.layer_norm.normalized_shape[0]
    out_dim = model.config.text_config.hidden_size
    fusion = GatedRelationalProjector(proj, in_dim, out_dim, missing=missing, **kw)
    fusion.to(next(proj.parameters()).device)
    fusion._shuffle_fn = inner.pixel_shuffle
    n_layers = len(tower.encoder.layer)
    L = fusion.config["slot_layer"]
    if L is not None and L >= n_layers:          # the last layer is the projector's input
        fusion.config["slot_layer"] = L = -1
    if L is not None and L > n_layers:
        raise ValueError(f"slot_layer {L} > {n_layers} ViT layers")

    def grab_pixels(_m, args, kwargs):
        fusion._pixels = kwargs.get("pixel_values", args[0] if args else None)

    def grab_layer(_m, _inp, out):
        fusion._layer_feats = out[0] if isinstance(out, tuple) else out

    tower.register_forward_pre_hook(grab_pixels, with_kwargs=True)
    if fusion.config["source"] == "vit" and L is not None and L >= 0:
        (tower.embeddings if L == 0 else tower.encoder.layer[L - 1]).register_forward_hook(grab_layer)
    inner.multi_modal_projector = fusion
    return fusion


def new_parameters(fusion):
    """Everything except the wrapped original projector (frozen ResNet weights are excluded too)."""
    return [p for n, p in fusion.named_parameters() if not n.startswith("projector.") and p.requires_grad]


def fusion_state(fusion):
    return {k: v.detach().cpu() for k, v in fusion.state_dict().items() if not k.startswith("projector.")}
