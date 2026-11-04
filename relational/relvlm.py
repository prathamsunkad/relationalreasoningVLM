"""RelationalVLM: a frozen VLM + a relational-bottleneck branch (slot attention -> Abstractor) that feeds the LLM.

Native path (unchanged): image -> ViT -> merger/projector -> visual tokens -> LLM, text -> embeddings -> LLM.
Relational branch:      the same ViT forward's pre-merge patch tokens -> ROI-sample each panel to a P x P grid ->
                        frozen slot attention (Stage 0) -> z, m per slot -> Abstractor over all panels of the sequence
                        -> relation embeddings s (per slot) and r (pooled per sequence).

Injection (--inject): two separate training regimes, one per run
  interleave  r -> MLP -> M relation tokens written at the <|rel|> placeholders in the prompt (approach A, the diagram)
  fuse        every merged visual token inside a panel gets  lambda * v + (1 - lambda) * rel(v)  where rel(v) is the
              relation embedding of the slots that own that location (sum_k mask_k(x) s_k), projected to d_llm and
              RMS-matched to v. Tokens outside panels are unchanged. lambda = sigmoid(learned scalar). (approach B)
The LLM's answer is read from its own LM head: score = logit(Yes) - logit(No) (per-candidate tasks, as in the LoRA
baseline) or the logits of the choice letters. An optional aux head scores r directly (the paper's linear scorer).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from abstractor import RelationalAbstractor


def roi_sample(fmap, boxes, P):
    """fmap (h, w, C), boxes (n, 4) normalised [x0, y0, x1, y1] -> (n, P*P, C), bilinear, sample at cell centres."""
    h, w, C = fmap.shape
    n = boxes.shape[0]
    t = (torch.arange(P, device=fmap.device, dtype=torch.float32) + 0.5) / P
    gx = boxes[:, 0:1] + (boxes[:, 2:3] - boxes[:, 0:1]) * t[None]          # (n, P)
    gy = boxes[:, 1:2] + (boxes[:, 3:4] - boxes[:, 1:2]) * t[None]
    grid = torch.stack([gx[:, None, :].expand(n, P, P), gy[:, :, None].expand(n, P, P)], -1) * 2 - 1
    out = F.grid_sample(fmap.permute(2, 0, 1)[None].float(), grid.reshape(1, n * P, P, 2), mode="bilinear",
                        align_corners=False)                                   # (1, C, n*P, P)
    return out[0].reshape(C, n, P * P).permute(1, 2, 0)


def token_panel_coords(boxes, hm, wm):
    """For a hm x wm merged-token grid: which panel each token centre falls in (-1 = none) and its local [-1, 1] coords."""
    dev = boxes.device
    ys = (torch.arange(hm, device=dev, dtype=torch.float32) + 0.5) / hm
    xs = (torch.arange(wm, device=dev, dtype=torch.float32) + 0.5) / wm
    cy, cx = torch.meshgrid(ys, xs, indexing="ij")
    cx, cy = cx.reshape(-1), cy.reshape(-1)                                     # (T,)
    x0, y0, x1, y1 = (boxes[:, i:i + 1] for i in range(4))                      # (n, 1)
    inside = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)                    # (n, T)
    owner = torch.where(inside.any(0), inside.float().argmax(0), torch.full_like(cx, -1, dtype=torch.long))
    lx = ((cx - x0) / (x1 - x0).clamp_min(1e-6)) * 2 - 1
    ly = ((cy - y0) / (y1 - y0).clamp_min(1e-6)) * 2 - 1
    return owner, torch.stack([lx, ly], -1)                                     # (T,), (n, T, 2)


def _rms(x):
    return x.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()


class RelationalVLM(nn.Module):
    def __init__(self, backbone, slot_ae, inject="interleave", rel_tokens=1, abstractor_cfg=None, fuse_lambda=0.9,
                 learn_lambda=True):
        super().__init__()
        assert inject in ("interleave", "fuse")
        object.__setattr__(self, "bb", backbone)          # not submodules: keeps the HF model / frozen slots out of
        object.__setattr__(self, "slot_ae", slot_ae)      # this module's state_dict (saved separately)
        self.inject, self.M = inject, rel_tokens
        self.interleave, self.fuse = inject == "interleave", inject == "fuse"
        D, d = slot_ae.cfg["slot_dim"], backbone.d_llm
        self.P = slot_ae.cfg["grid"]
        self.cfg = dict(inject=inject, rel_tokens=rel_tokens, abstractor_cfg=abstractor_cfg or {},
                        fuse_lambda=fuse_lambda, learn_lambda=learn_lambda)
        self.abstractor = RelationalAbstractor(dim=D, **(abstractor_cfg or {}))
        self.aux_head = nn.Linear(D, 1)
        if self.interleave:
            self.tok_proj = nn.Sequential(nn.Linear(D, d), nn.GELU(), nn.Linear(d, rel_tokens * d))
            self.tok_gain = nn.Parameter(torch.ones(()))
            emb = backbone.model.get_input_embeddings().weight
            self.register_buffer("emb_rms", _rms(emb.detach()).mean().reshape(()), persistent=False)
        if self.fuse:
            self.fuse_proj = nn.Sequential(nn.Linear(D, d), nn.GELU(), nn.Linear(d, d))
            lam = min(max(fuse_lambda, 1e-4), 1 - 1e-4)
            self.lambda_logit = nn.Parameter(torch.tensor(math.log(lam / (1 - lam))), requires_grad=learn_lambda)
        self._ctx, self._res, self._handles = None, None, []
        self.attach()

    # ------------------------------------------------------------------ hooks
    def attach(self):
        bb = self.bb
        self._handles.append(bb.vision.register_forward_hook(self._vision_hook))
        if self.fuse and bb.family == "internvl":
            self._handles.append(bb.merged_module.register_forward_hook(self._merged_hook))
        if self.interleave:
            self._handles.append(bb.lm.register_forward_pre_hook(self._lm_hook, with_kwargs=True))

    def detach_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def lam(self):
        return torch.sigmoid(self.lambda_logit) if self.fuse else None

    def _vision_hook(self, module, args, out):
        ctx = self._ctx
        if ctx is None:
            return None
        maps = self.bb.feature_maps(out, ctx["grids"])
        self._res = self.relations(maps, ctx["boxes"])
        if self.fuse and ctx["llm"] and self.bb.family == "qwen3vl":
            return self.bb.replace_merged(out, self._fuse(self.bb.merged_tokens(out, ctx["grids"])))
        return None

    def _merged_hook(self, module, args, out):
        if self._ctx is None or not self._ctx["llm"]:
            return None
        return self.bb.replace_merged(out, self._fuse(self.bb.merged_tokens(out, self._ctx["grids"])))

    def _lm_hook(self, module, args, kwargs):
        ctx = self._ctx
        if ctx is None or not ctx["llm"]:
            return None
        e = kwargs["inputs_embeds"]
        mask = ctx["rel_mask"]
        toks = self.rel_tokens(self._res["r"])                                  # (n_seq, M, d)
        assert int(mask.sum()) == toks.shape[0] * toks.shape[1], \
            f"{int(mask.sum())} <|rel|> placeholders for {toks.shape[0]}x{toks.shape[1]} relation tokens"
        kwargs["inputs_embeds"] = e.masked_scatter(mask.unsqueeze(-1), toks.to(e.dtype))
        return args, kwargs

    # ------------------------------------------------------------------ branch
    def relations(self, maps, boxes):
        """maps: per image (h, w, C); boxes: per image (n_p, 4). One image per sequence."""
        P = self.P
        counts = [b.shape[0] for b in boxes]
        panels = torch.cat([roi_sample(fm, b, P) for fm, b in zip(maps, boxes)], 0)    # (N, P*P, C)
        gen = None
        if not self.training:      # slot init is random (it breaks slot symmetry); fixed seed => reproducible eval
            gen = torch.Generator(device=panels.device)
            gen.manual_seed(0)
        with torch.no_grad():
            z, m, attn = self.slot_ae.extract(panels, gen)                              # (N, K, D) x2, (N, K, P*P)
        n, Pmax, K, D = len(maps), max(counts), z.shape[1], z.shape[2]
        Z = z.new_zeros(n, Pmax, K, D)
        Mm = m.new_zeros(n, Pmax, K, D)
        mask = torch.zeros(n, Pmax, dtype=torch.bool, device=z.device)
        i = 0
        for j, c in enumerate(counts):
            Z[j, :c], Mm[j, :c], mask[j, :c] = z[i:i + c], m[i:i + c], True
            i += c
        s, r = self.abstractor(Z, Mm, mask)
        return dict(s=s, r=r, attn=attn, counts=counts, K=K, panel_mask=mask)

    def rel_tokens(self, r):
        t = self.tok_proj(r).view(r.shape[0], self.M, -1)
        return t / _rms(t) * (self.emb_rms * self.tok_gain)                     # on the scale of word embeddings

    def _fuse(self, merged):
        res, grids, boxes = self._res, self._ctx["grids"], self._ctx["boxes"]
        lam = self.lam()
        out, i = [], 0
        for j, (v, (_, (hm, wm)), b) in enumerate(zip(merged, grids, boxes)):
            c, K = res["counts"][j], res["K"]
            s = res["s"][j].view(-1, K, res["s"].shape[-1])[:c]                       # (c, K, D)
            attn = res["attn"][i:i + c]                                                 # (c, K, P*P)
            i += c
            rel_map = torch.einsum("ckn,ckd->cdn", attn, s).reshape(c, -1, self.P, self.P)
            owner, local = token_panel_coords(b, hm, wm)
            samp = F.grid_sample(rel_map, local[:, None], mode="bilinear", align_corners=False)[:, :, 0]  # (c, D, T)
            rel = samp.permute(0, 2, 1).gather(0, owner.clamp_min(0)[None, :, None].expand(1, -1, samp.shape[1]))[0]
            rel = self.fuse_proj(rel)
            rel = rel / _rms(rel) * _rms(v).detach()
            fused = lam * v.float() + (1 - lam) * rel
            out.append(torch.where((owner >= 0)[:, None], fused, v.float()).to(v.dtype))
        return out

    # ------------------------------------------------------------------ forward
    def forward(self, batch, device, dtype, llm=True):
        """Returns (h, aux): h (n_seq, d_llm) final normed LLM state at the answer position (None if llm=False);
        aux (n_seq,) the aux head's score of each sequence's pooled relation vector."""
        bb = self.bb
        inputs = bb.model_inputs(batch, device, dtype)
        boxes = [b.to(device) for b in batch["boxes"]]
        self._ctx = dict(grids=bb.grids(inputs, len(boxes)), boxes=boxes, llm=llm)
        if llm and self.interleave:
            self._ctx["rel_mask"] = inputs["input_ids"] == bb.rel_id
        try:
            ac = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and dtype != torch.float32 \
                else torch.autocast("cpu", enabled=False)
            with ac:
                if llm:
                    h = bb.core(**inputs, use_cache=False).last_hidden_state[:, -1].float()
                else:
                    bb.run_vision(inputs)
                    h = None
            aux = self.aux_head(self._res["r"].float()).squeeze(-1)
        finally:
            self._ctx = None
        return h, aux

    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items()}


# ---------------------------------------------------------------------- readout + losses
class ScoreTokens:
    """First token of "Yes"/"No" and of each choice letter, resolved from the real tokenizer."""

    def __init__(self, tokenizer, letters="ABCDEFGH"):
        first = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"][0]
        self.yes_id, self.no_id = first("Yes"), first("No")
        assert self.yes_id != self.no_id
        self.letter_ids = {c: first(c) for c in letters}


def example_logits(h, aux, batch, model, st):
    """Group per-sequence outputs into per-example logits.
    Returns list of dicts {llm: logits or None, aux: logits or None, loss: 'ce'|'bce', label}."""
    out = []
    W = model.get_output_embeddings().weight
    wdiff = (W[st.yes_id] - W[st.no_id]).float()
    yesno = h @ wdiff if h is not None else None
    for ex in batch["examples"]:
        idx, kind, label = ex["seq_idx"], ex["kind"], ex["label"]
        if kind == "image":
            raise ValueError(f"unlabelled example {ex['id']} ({ex['source']}): image-only sources are for Stage 0")
        rec = dict(loss="bce" if kind == "binary" else "ce", label=label, kind=kind)
        if kind == "cand":
            rec["llm"] = yesno[idx] if yesno is not None else None
            rec["aux"] = aux[idx] if aux is not None else None
        elif kind == "binary":
            rec["llm"] = yesno[idx] if yesno is not None else None
            rec["aux"] = aux[idx] if aux is not None else None
        else:   # letter: one sequence, logits of the choice letters
            ids = [st.letter_ids[c] for c in ex["choices"]]
            rec["llm"] = (h[idx[0]] @ W[ids].float().T) if h is not None else None
            rec["aux"] = None
        out.append(rec)
    return out


def loss_and_acc(recs, key):
    losses, correct, n = [], 0, 0
    for r in recs:
        x = r[key]
        if x is None:
            continue
        if r["loss"] == "ce":
            losses.append(F.cross_entropy(x[None], torch.tensor([r["label"]], device=x.device)))
            correct += int(int(x.argmax()) == r["label"])
        else:
            y = torch.tensor([float(r["label"])], device=x.device)
            losses.append(F.binary_cross_entropy_with_logits(x.reshape(1), y))
            correct += int((float(x.reshape(())) > 0) == bool(r["label"]))
        n += 1
    if not losses:
        return None, 0.0, 0
    return torch.stack(losses).mean(), correct / n, n
