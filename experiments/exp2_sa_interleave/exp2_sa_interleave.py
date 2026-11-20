#!/usr/bin/env python
"""Exp2: interleave a pretrained Slot Abstractor (SA) into the fine-tuned InternVL through two MLP connectors.

Built on the repo's InternVL fine-tuning recipe (lora/train_lora.py, pgmvlm/): the same .npz data, the same 8
per-candidate sequences (the matrix completed with candidate k + "Answer Yes or No"), the same score
logit(Yes) - logit(No), the same 8-way softmax cross-entropy, the same LoRA adapter format. What changes is the prompt:

  <image: completed matrix k>
  Object and relation features of the 9 panels of this completed matrix, row by row:
  panel 1: <|sa|><|sa|>          <- object token, relation token of panel 1
  ...
  panel 9: <|sa|><|sa|>          <- panel 9 = candidate k
  <the repo's PROMPT_TEXT: "... Is the bottom-right panel the correct completion of this matrix? Answer Yes or No.">

The frozen SA reads the same 9 panels (pixels from the .npz, its own preprocessing) and gives, per panel, K slots
(objects) and K relation embeddings (after its Abstractor). Two connectors turn them into LLM-sized tokens:
  object connector     per panel: DeepSets over its K slots          -> 1 token   (9 per sequence)
  relation connector   per panel: DeepSets over its K relation vecs  -> 1 token   (9 per sequence)
Both are RMS-normalised to the scale of the LLM's word embeddings with a learned gain (as relational/relvlm.py
does), and written into the <|sa|> placeholders by a forward pre-hook on the language model.

Stages
  inspect     check your SA builder: shapes, and the SA's own accuracy on validation problems (must match its
              known accuracy). --list_modules prints the SA's submodule names for the builder.
  complement  do SA and the fine-tuned VLM fail on different puzzles? Per-puzzle scores of both on the same
              problems, a 2x2 right/wrong table, and a cross-fitted ensemble. Run this first: if they rarely
              disagree, interleaving cannot help much.
  train --stage a   Exp2a: VLM + its fine-tuned LoRA frozen, SA frozen; only the two connectors train.
  train --stage b   Exp2b: connectors + LoRA (continued from the fine-tuned adapter), optional branch dropout.
        --control shuffle_puzzle   same training, but every puzzle gets another puzzle's SA features
                                   (the connectors can still learn a soft prompt: separates "SA content" from
                                   "extra trainable tokens")
        --control zero             tokens always zero = LoRA alone with the same prompt and steps (stage b)
  eval        sharded scoring in the repo's results_shard*.json format, with --ablate none | zero |
              shuffle_cand | shuffle_puzzle; merge with  python lora/eval_vlm.py --merge_dir <out>

Run from gap_hybrid_probes/ (or anywhere): the script finds the repo root two levels up.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402
import torch.nn as nn                                                      # noqa: E402
import torch.nn.functional as F                                            # noqa: E402
from torch.utils.data import DataLoader                                    # noqa: E402

from pgmvlm.common import TimeBudget, log_jsonl, lr_at, save_ckpt, seed_everything            # noqa: E402
from pgmvlm.data import (PROMPT_TEXT, CandidateCollator, PGMStream, filled_matrices, list_npz,  # noqa: E402
                         make_order, shard_files, subset_files)
from pgmvlm.evalcore import summarize, wilson                                                  # noqa: E402
from pgmvlm.models import (DTYPES, ScoreTokens, add_lora, candidate_losses, candidate_scores,   # noqa: E402
                           load_adapter, load_base, param_report, set_train_mode)
from sa_interface import FrozenSA                                                              # noqa: E402

SA_TOKEN = "<|sa|>"
ABLATIONS = ("none", "zero", "shuffle_cand", "shuffle_puzzle")


# ============================================================================ prompt and data
def sa_prompt(layout, n_rel):
    """Text with the <|sa|> placeholders, followed by the repo's PROMPT_TEXT. n_rel: relation tokens per sequence."""
    T = SA_TOKEN
    if layout == "interleave":
        assert n_rel == 9, "interleave needs one relation token per panel (per-slot relation embeddings)"
        body = ("Object and relation features of the 9 panels of this completed matrix, row by row:\n"
                + "\n".join(f"panel {i + 1}: {T}{T}" for i in range(9)))
    elif layout == "grouped":
        body = ("Features of the 9 panels of this completed matrix, row by row.\n"
                f"Objects: {T * 9}\nRelations: {T * n_rel}")
    elif layout == "obj":
        body = "Object features of the 9 panels of this completed matrix, row by row: " + T * 9
    elif layout == "rel":
        body = "Relation features of this completed matrix: " + T * n_rel
    else:
        raise ValueError(layout)
    return body + "\n\n" + PROMPT_TEXT


def sa_panels(items):
    """(8B, 9, 160, 160) uint8: per problem, the 8 context panels + candidate k, k = 0..7 (problem-major, like the
    8B sequences of CandidateCollator)."""
    out = [np.concatenate([it["context"], it["candidates"][k:k + 1]], 0) for it in items for k in range(8)]
    return torch.from_numpy(np.stack(out).astype(np.uint8))


class SACollator:
    """CandidateCollator with the SA prompt, plus the raw panels for the SA."""

    def __init__(self, processor, family, crop_to_patches, cell_size, text):
        self.processor, self.cell_size = processor, cell_size
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        self.text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.extra = {"crop_to_patches": bool(crop_to_patches)} if family == "internvl" else {}

    def __call__(self, items):
        images = []
        for it in items:
            images += filled_matrices(it["context"], it["candidates"], self.cell_size)
        batch = dict(self.processor(text=[self.text] * len(images), images=images, padding=True, return_tensors="pt",
                                    **self.extra))
        batch["sa_panels"] = sa_panels(items)
        batch["targets"] = torch.tensor([it["target"] for it in items], dtype=torch.long)
        batch["files"] = [it["file"] for it in items]
        return batch


class PlainCollator(CandidateCollator):
    """The repo's collator (original prompt, no placeholders) + the raw panels: for the complementarity check."""

    def __call__(self, items):
        batch = super().__call__(items)
        batch["sa_panels"] = sa_panels(items)
        return batch


def panels_only(items):
    return {"sa_panels": sa_panels(items), "targets": torch.tensor([it["target"] for it in items]),
            "files": [it["file"] for it in items]}


def read_list(path):
    return [l.split("\t")[0].strip() for l in open(path) if l.strip() and not l.startswith("#")]


def problem_files(npz_dir, file_list=None, cache_dir=None, limit=0):
    files = read_list(file_list) if file_list else list_npz(npz_dir, os.path.join(cache_dir, "files.txt") if cache_dir else None)
    return files[:limit] if limit else files


# ============================================================================ connectors and injection
class SetPool(nn.Module):
    """DeepSets: rho(mean_i phi(x_i)). (..., S, d_in) -> (..., n_out, d_out). Invariant to slot order."""

    def __init__(self, d_in, d_out, hidden, n_out=1):
        super().__init__()
        self.phi = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.rho = nn.Sequential(nn.GELU(), nn.Linear(hidden, n_out * d_out))
        self.n_out, self.d_out = n_out, d_out

    def forward(self, x):
        y = self.rho(self.phi(x).mean(-2))
        return y.reshape(*x.shape[:-2], self.n_out, self.d_out)


def _rms(x):
    return x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()


class Connectors(nn.Module):
    """The two MLP connectors: SA slots -> object tokens, SA relation embeddings -> relation tokens."""

    def __init__(self, d_obj, d_rel, rel_per_panel, d_llm, emb_rms, layout="interleave", rel_tokens=9, hidden=0):
        super().__init__()
        hidden = hidden or d_llm
        if not rel_per_panel and layout == "interleave":
            layout = "grouped"           # pooled relations have no panel to sit next to
        self.cfg = dict(d_obj=d_obj, d_rel=d_rel, rel_per_panel=rel_per_panel, d_llm=d_llm, emb_rms=float(emb_rms),
                        layout=layout, rel_tokens=rel_tokens, hidden=hidden)
        self.layout, self.rel_per_panel = layout, rel_per_panel
        self.obj = SetPool(d_obj, d_llm, hidden)                                         # per panel, over K slots
        self.rel = SetPool(d_rel, d_llm, hidden, n_out=1 if rel_per_panel else rel_tokens)
        self.gain_obj = nn.Parameter(torch.ones(()))
        self.gain_rel = nn.Parameter(torch.ones(()))
        self.register_buffer("emb_rms", torch.tensor(float(emb_rms)), persistent=False)

    @property
    def n_rel(self):
        return 9 if self.rel_per_panel else self.cfg["rel_tokens"]

    def n_tokens(self):
        return {"interleave": 18, "grouped": 9 + self.n_rel, "obj": 9, "rel": self.n_rel}[self.layout]

    def forward(self, sa):
        """sa dict from FrozenSA -> (N, T, d_llm) tokens in placeholder order."""
        o = self.obj(sa["slots"]).squeeze(-2)                                            # (N, 9, d)
        r = self.rel(sa["rel"])
        r = r.squeeze(-2) if self.rel_per_panel else r                                   # (N, 9 or M, d)
        o = o / _rms(o) * self.emb_rms * self.gain_obj
        r = r / _rms(r) * self.emb_rms * self.gain_rel
        if self.layout == "interleave":
            return torch.stack([o, r], 2).flatten(1, 2)                                  # o1 r1 o2 r2 ...
        if self.layout == "grouped":
            return torch.cat([o, r], 1)
        return o if self.layout == "obj" else r


class Injector:
    """Forward pre-hook on the language model: writes the connector tokens into the <|sa|> positions of
    inputs_embeds (after InternVL has inserted the image features). Same mechanism as relational/relvlm.py."""

    def __init__(self, model, sa_id):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.lm = base.model.language_model
        self.sa_id, self.mask, self.tokens = sa_id, None, None
        self.handle = self.lm.register_forward_pre_hook(self._hook, with_kwargs=True)

    def set(self, input_ids, tokens):
        self.mask, self.tokens = input_ids == self.sa_id, tokens

    def clear(self):
        self.mask, self.tokens = None, None

    def _hook(self, module, args, kwargs):
        if self.tokens is None:
            return None
        e = kwargs.get("inputs_embeds")
        if e is None:
            raise RuntimeError("the language model was called without inputs_embeds; cannot place the SA tokens")
        mask = self.mask.to(e.device)
        if mask.shape != e.shape[:2] or int(mask.sum()) != self.tokens.shape[0]:
            raise RuntimeError(f"{int(mask.sum())} <|sa|> placeholders for {self.tokens.shape[0]} SA tokens "
                               f"(mask {tuple(mask.shape)}, embeds {tuple(e.shape)})")
        kwargs["inputs_embeds"] = e.masked_scatter(mask.unsqueeze(-1), self.tokens.to(e.dtype))
        return args, kwargs


def ablate_sa(sa, B, mode):
    """Feature-level ablations. N = 8B rows, problem-major.
    shuffle_cand: each candidate's sequence gets the SA features of the next candidate of the same puzzle.
    shuffle_puzzle: each puzzle gets the next puzzle's features, candidate by candidate (needs B >= 2)."""
    if mode == "shuffle_cand":
        idx = [b * 8 + (k + 1) % 8 for b in range(B) for k in range(8)]
    elif mode == "shuffle_puzzle":
        if B < 2:
            raise SystemExit("shuffle_puzzle needs at least 2 problems per batch")
        idx = [((b + 1) % B) * 8 + k for b in range(B) for k in range(8)]
    else:
        return sa
    idx = torch.tensor(idx, device=sa["slots"].device)
    return {k: (v[idx] if torch.is_tensor(v) else v) for k, v in sa.items()}


# ============================================================================ model setup
class Exp2Model:
    def __init__(self, args, train_lora=False, lora_dir=None, conn_state=None, conn_cfg=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = DTYPES[args.dtype]
        model, processor, family = load_base(args.model_id, args.dtype, args.attn, self.device)
        tok = processor.tokenizer
        if SA_TOKEN not in tok.get_vocab():
            tok.add_tokens([SA_TOKEN], special_tokens=True)
        self.sa_id = tok.convert_tokens_to_ids(SA_TOKEN)
        if self.sa_id >= model.get_input_embeddings().weight.shape[0]:   # real checkpoints have spare rows
            model.resize_token_embeddings(len(tok))
        self.st = ScoreTokens(tok)
        emb_rms = float(_rms(model.get_input_embeddings().weight.detach().float()).mean())
        adapter = lora_dir or args.adapter
        if adapter:
            model = load_adapter(model, adapter, trainable=train_lora, grad_ckpt=args.grad_ckpt)
        elif train_lora:
            model = add_lora(model, family, args.lora_r, args.lora_alpha, args.lora_dropout, grad_ckpt=args.grad_ckpt)
        for n, p in model.named_parameters():
            p.requires_grad_(train_lora and "lora_" in n)
            if p.requires_grad:
                p.data = p.data.float()                       # LoRA in fp32, as in train_lora.py
        if args.grad_ckpt and not adapter and not train_lora:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        self.model, self.processor, self.family, self.adapter = model, processor, family, adapter
        self.sa = FrozenSA(args.sa_builder, args.sa_ckpt, self.device)
        self.sa(torch.zeros(2, 9, 160, 160, dtype=torch.uint8, device=self.device))      # learn the SA's shapes
        dims = self.sa.dims
        d_llm = int(model.get_input_embeddings().weight.shape[1])
        cfg = conn_cfg or dict(d_obj=dims["d_obj"], d_rel=dims["d_rel"], rel_per_panel=dims["rel_per_panel"],
                               d_llm=d_llm, emb_rms=emb_rms, layout=args.layout, rel_tokens=args.rel_tokens,
                               hidden=args.conn_hidden)
        self.conn = Connectors(**cfg).to(self.device)
        if conn_state is not None:
            self.conn.load_state_dict(conn_state)
        self.inj = Injector(model, self.sa_id)
        self._prev_sa = None
        self.text = sa_prompt(self.conn.layout, self.conn.n_rel)
        print(f"SA: K={dims['K']} slots, d_obj={dims['d_obj']}, d_rel={dims['d_rel']}, "
              f"relations {'per slot' if dims['rel_per_panel'] else 'pooled'} | connectors "
              f"{sum(p.numel() for p in self.conn.parameters()) / 1e6:.2f}M params, {self.conn.n_tokens()} tokens per "
              f"sequence, layout {self.conn.layout} | adapter {adapter or 'none'} | LoRA trainable {train_lora}")

    def collator(self, args):
        return SACollator(self.processor, self.family, args.crop_to_patches, args.panel_px, self.text)

    def scores(self, batch, ablate="none"):
        """(B, 8) VLM scores with the SA tokens, and the SA's own (B, 8) scores (or None)."""
        B = batch["targets"].shape[0]
        sa = self.sa(batch["sa_panels"])
        sa_own = sa["score"].view(B, 8) if sa["score"] is not None else None
        if ablate == "shuffle_puzzle" and B == 1:      # a last batch of one problem: use the previous batch's puzzle
            if self._prev_sa is None:
                raise SystemExit("shuffle_puzzle needs at least 2 problems in the first batch")
            cur, sa = sa, self._prev_sa
        else:
            cur, sa = sa, ablate_sa(sa, B, ablate)
        self._prev_sa = {k: (v[:8] if torch.is_tensor(v) else v) for k, v in cur.items()}
        toks = self.conn(sa)
        if ablate == "zero":
            toks = torch.zeros_like(toks)
        self.inj.set(batch["input_ids"].to(self.device), toks.reshape(-1, toks.shape[-1]))
        try:
            s = candidate_scores(self.model, batch, self.st, self.device, self.dtype)
        finally:
            self.inj.clear()
        return s, sa_own


@torch.no_grad()
def evaluate(m, loader, ablate="none", on_batch=None):
    m.model.eval()
    m.conn.eval()
    recs = []
    for batch in loader:
        s, sa_own = m.scores(batch, ablate)
        s = s.float().cpu()
        rows = []
        for i, (f, t) in enumerate(zip(batch["files"], batch["targets"].tolist())):
            p = int(s[i].argmax())
            r = dict(file=f, target=t, pred=p, correct=p == t, scores=[round(float(x), 4) for x in s[i]])
            if sa_own is not None:
                r["sa_scores"] = [round(float(x), 4) for x in sa_own[i].cpu()]
            rows.append(r)
        if on_batch:
            on_batch(rows)
        recs += rows
    return recs


# ============================================================================ stages
def stage_inspect(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sa = FrozenSA(args.sa_builder, args.sa_ckpt, device)
    if args.list_modules:
        for name, mod in sa.model.named_modules():
            print(f"{name or '(root)':50s} {type(mod).__name__}")
        return
    files = problem_files(args.val_dir, args.val_list, limit=args.n)
    loader = DataLoader(PGMStream(args.val_dir, files), batch_size=8, num_workers=args.num_workers, collate_fn=panels_only)
    correct, n, shown = 0, 0, False
    for b in loader:
        out = sa(b["sa_panels"])
        if not shown:
            print({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in out.items()})
            shown = True
        if out["score"] is None:
            print("encode() returns no 'score': the SA's own accuracy cannot be checked (and complement needs it)")
            return
        pred = out["score"].view(-1, 8).argmax(1).cpu()
        correct += int((pred == b["targets"]).sum())
        n += len(pred)
    lo, hi = wilson(correct, n)
    print(f"SA alone on {n} validation problems: {correct / n:.1%} (95% CI {lo:.1%}-{hi:.1%}); chance 12.5%")
    print("This must match the SA's known validation accuracy; if not, fix the preprocessing in your builder.")


def stage_complement(args):
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "records.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["file"] for l in open(path)}
    files = [f for f in problem_files(args.npz_dir, args.file_list, limit=args.limit) if f not in done]
    vlm_given = {}
    if args.vlm_results:          # per-problem scores from lora/eval_vlm.py (results_shard*.json)
        import glob
        for p in glob.glob(os.path.join(args.vlm_results, "results_shard*.json")):
            for r in json.load(open(p)):
                if "scores" in r:
                    vlm_given[r["file"]] = r["scores"]
        files = [f for f in files if f in vlm_given]
        print(f"VLM scores read for {len(vlm_given)} problems from {args.vlm_results}")
    print(f"{len(files)} problems to score ({len(done)} done)")
    if files:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = DTYPES[args.dtype]
        sa = FrozenSA(args.sa_builder, args.sa_ckpt, device)
        if vlm_given:
            model = None
            coll = panels_only
        else:
            model, processor, family = load_base(args.model_id, args.dtype, args.attn, device)
            st = ScoreTokens(processor.tokenizer)
            if args.adapter:
                model = load_adapter(model, args.adapter)
            model.eval()
            coll = PlainCollator(processor, family, args.crop_to_patches, cell_size=args.panel_px)
        loader = DataLoader(PGMStream(args.npz_dir, files), batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=coll)
        t0 = time.time()
        with open(path, "a") as fh, torch.no_grad():
            for i, b in enumerate(loader):
                sa_s = sa(b["sa_panels"])["score"]
                if sa_s is None:
                    raise SystemExit("encode() returns no 'score'; the complementarity check needs the SA's own scores")
                sa_s = sa_s.view(-1, 8).cpu()
                vlm = (candidate_scores(model, b, st, device, dtype).float().cpu() if model is not None else
                       torch.tensor([vlm_given[f] for f in b["files"]]))
                for f, t, a, v in zip(b["files"], b["targets"].tolist(), sa_s, vlm):
                    fh.write(json.dumps(dict(file=f, target=t, sa=[round(float(x), 4) for x in a],
                                             vlm=[round(float(x), 4) for x in v])) + "\n")
                fh.flush()
                if i % 50 == 0:
                    print(f"{(i + 1) * args.batch_size} / {len(files)}  ({time.time() - t0:.0f}s)", flush=True)
    complement_report(path, os.path.join(args.out_dir, "report.md"))


def complement_report(path, out_md):
    R = [json.loads(l) for l in open(path)]
    R.sort(key=lambda r: r["file"])
    if not R:
        print("no records")
        return
    t = np.array([r["target"] for r in R])
    sa = torch.tensor([r["sa"] for r in R]).float()
    vl = torch.tensor([r["vlm"] for r in R]).float()
    a, v = (sa.argmax(1).numpy() == t), (vl.argmax(1).numpy() == t)
    n = len(t)
    ls_sa, ls_vl = F.log_softmax(sa, 1), F.log_softmax(vl, 1)

    def ens_acc(w, idx):
        p = (w * ls_sa[idx] + (1 - w) * ls_vl[idx]).argmax(1).numpy()
        return float((p == t[idx]).mean())

    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    half = np.arange(n) % 2 == 0           # cross-fitting: pick the weight on one half, score it on the other
    A, B = np.where(half)[0], np.where(~half)[0]
    wA = max(grid, key=lambda w: ens_acc(w, A))
    wB = max(grid, key=lambda w: ens_acc(w, B))
    cross = (ens_acc(wA, B) * len(B) + ens_acc(wB, A) * len(A)) / n
    ci = lambda k: "{:.1%} [{:.1%}, {:.1%}]".format(k / n, *wilson(int(k), n))
    phi = float(np.corrcoef(a, v)[0, 1]) if a.std() > 0 and v.std() > 0 else float("nan")
    lines = [f"# Complementarity: Slot Abstractor vs fine-tuned VLM ({n} problems)", "",
             "| | accuracy [95% CI] |", "|---|---|",
             f"| SA alone | {ci(a.sum())} |", f"| VLM alone | {ci(v.sum())} |",
             f"| either right (oracle upper bound) | {ci((a | v).sum())} |",
             f"| ensemble, weight chosen on the other half (w_SA = {wA:.2f} / {wB:.2f}) | {cross:.1%} |", "",
             "| | VLM right | VLM wrong |", "|---|---|---|",
             f"| SA right | {int((a & v).sum())} | {int((a & ~v).sum())} |",
             f"| SA wrong | {int((~a & v).sum())} | {int((~a & ~v).sum())} |", "",
             f"SA accuracy on the puzzles the VLM gets wrong: {a[~v].mean():.1%} (n={int((~v).sum())}). "
             f"VLM accuracy on the puzzles the SA gets wrong: {v[~a].mean():.1%} (n={int((~a).sum())}). "
             f"Correlation of correctness (phi): {phi:.2f}.", "",
             "Reading: a large 'SA right, VLM wrong' cell and an ensemble clearly above max(SA, VLM) mean the SA knows "
             "things the VLM does not, which interleaving could pass on. If the ensemble is no better than the SA "
             "alone, the VLM adds nothing the SA lacks.", ""]
    txt = "\n".join(lines)
    print(txt)
    open(out_md, "w").write(txt)


def stage_train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    if args.stage == "a" and args.control == "zero":
        raise SystemExit("--control zero with --stage a trains nothing (connectors get no gradient)")
    train_lora = args.stage == "b"
    metrics = os.path.join(args.out_dir, "metrics.jsonl")
    state_path = os.path.join(args.out_dir, "state.pt")
    last_lora, best_lora = os.path.join(args.out_dir, "last_lora"), os.path.join(args.out_dir, "best_lora")
    resumed = os.path.exists(state_path)
    ck = torch.load(state_path, map_location="cpu", weights_only=False) if resumed else None
    init_conn = None
    if not resumed and args.init_from:            # stage b from a stage-a run: its best connectors
        init_conn = torch.load(args.init_from, map_location="cpu", weights_only=False)
    m = Exp2Model(args, train_lora=train_lora,
                  lora_dir=last_lora if (resumed and train_lora) else None,
                  conn_state=(ck or init_conn or {}).get("conn"), conn_cfg=(ck or init_conn or {}).get("conn_cfg"))

    cache = os.path.join(args.file_cache_dir, "train_files.txt") if args.file_cache_dir else None
    files = subset_files(list_npz(args.train_dir, cache), args.train_samples, seed=args.seed)
    glob_bs = args.batch_size * args.grad_accum
    max_steps = args.max_steps or max(1, int(math.ceil(args.epochs * len(files) / glob_bs)))
    order = make_order(len(files), max_steps * glob_bs, args.seed)
    print(f"train problems {len(files)} | global batch {glob_bs} | steps {max_steps} | "
          f"epochs {max_steps * glob_bs / len(files):.2f}")

    groups = [{"params": list(m.conn.parameters()), "lr": args.lr, "base": args.lr}]
    lora = [p for p in m.model.parameters() if p.requires_grad]
    if lora:
        groups.append({"params": lora, "lr": args.lora_lr, "base": args.lora_lr})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    tr, tot = param_report(m.model)
    print(f"trainable: connectors {sum(p.numel() for p in m.conn.parameters()) / 1e6:.2f}M, LoRA {tr / 1e6:.2f}M "
          f"(VLM total {tot / 1e6:.0f}M)")
    start_step, best_val = 0, -1.0
    if ck:
        opt.load_state_dict(ck["opt"])
        start_step, best_val = ck["step"], ck["best_val"]
        print(f"resumed at step {start_step} (best val {best_val:.4f})")
    if start_step >= max_steps:
        print("already finished")
        return
    meta = {**vars(args), "family": m.family, "max_steps": max_steps, "base_adapter": args.adapter}
    json.dump(meta, open(os.path.join(args.out_dir, "args.json"), "w"), indent=1)

    coll = m.collator(args)
    loader = DataLoader(PGMStream(args.train_dir, files, order=order, seed=args.seed), batch_size=args.batch_size,
                        sampler=range(start_step * glob_bs, len(order)), num_workers=args.num_workers, drop_last=True,
                        collate_fn=coll, pin_memory=m.device.type == "cuda",
                        prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = None
    if args.val_dir:
        vf = read_list(args.val_list) if args.val_list else subset_files(list_npz(args.val_dir), args.val_samples, seed=1234)
        val_loader = DataLoader(PGMStream(args.val_dir, vf[:args.val_samples]), batch_size=args.val_batch_size,
                                num_workers=min(2, args.num_workers), collate_fn=coll)
    run_mode = {"none": "none", "shuffle_puzzle": "shuffle_puzzle", "zero": "zero"}[args.control]
    val_modes = [run_mode] + [x for x in args.val_controls.split(",") if x and x != run_mode]

    def train_mode():
        if train_lora:
            set_train_mode(m.model)
        else:
            m.model.eval()                                # frozen VLM: no dropout anywhere
        m.conn.train()

    def save_state(step):
        if train_lora:
            m.model.save_pretrained(last_lora)
        save_ckpt(state_path, conn=m.conn.state_dict(), conn_cfg=m.conn.cfg, opt=opt.state_dict(), step=step,
                  best_val=best_val)

    def validate(step):
        nonlocal best_val
        if val_loader is None:
            return
        rec = {"step": step}
        for mode in val_modes:
            s = summarize(evaluate(m, val_loader, mode))
            rec[f"val_acc_{mode}"] = s["scores"]["acc"]
            rec[f"val_pred_hist_{mode}"] = s["scores"]["pred_hist"]
        log_jsonl(metrics, rec)
        print(f"[val] step {step}: " + "  ".join(f"{k[8:]} {v:.4f}" for k, v in rec.items() if k.startswith("val_acc_"))
              + f"  (n={s['n']}, chance 0.125)", flush=True)
        acc = rec[f"val_acc_{run_mode}"]
        if acc > best_val and step > 0:
            best_val = acc
            save_ckpt(os.path.join(args.out_dir, "best.pt"), conn=m.conn.state_dict(), conn_cfg=m.conn.cfg,
                      step=step, val_acc=acc, meta=meta)
            if train_lora:
                m.model.save_pretrained(best_lora)
            json.dump({"step": step, "val_acc": acc, "mode": run_mode}, open(os.path.join(args.out_dir, "best.json"), "w"))
        train_mode()

    if start_step == 0:
        validate(0)                                       # where the fine-tuned model starts with the new prompt
    budget, it, t0, seen = TimeBudget(args.time_limit_min), iter(loader), time.time(), 0
    agg, n_agg = defaultdict(float), 0
    rng = np.random.default_rng([args.seed, start_step])
    train_mode()
    for step in range(start_step, max_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, g["base"], args.warmup_steps, max_steps, schedule=args.lr_schedule)
        for _ in range(args.grad_accum):
            batch = next(it)
            mode = run_mode
            if mode == "none" and args.branch_dropout > 0 and rng.random() < args.branch_dropout:
                mode = "zero"
            scores, _ = m.scores(batch, mode)
            ce8, bce, acc = candidate_losses(scores.float(), batch["targets"].to(m.device))
            (ce8 / args.grad_accum).backward()
            agg["ce8"] += float(ce8) / args.grad_accum
            agg["train_acc"] += float(acc) / args.grad_accum
            seen += batch["targets"].numel()
        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        if not math.isfinite(float(gn)):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        done, n_agg = step + 1, n_agg + 1
        if done % args.log_every == 0 or done == max_steps:
            rec = {"step": done, **{k: v / n_agg for k, v in agg.items()}, "grad_norm": float(gn),
                   "lr": opt.param_groups[0]["lr"], "gain_obj": float(m.conn.gain_obj), "gain_rel": float(m.conn.gain_rel),
                   "problems_per_s": seen / (time.time() - t0)}
            log_jsonl(metrics, rec)
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
            agg.clear()
            n_agg = 0
        out_of_time = budget.exhausted()
        if done % args.val_every == 0 or done == max_steps:
            validate(done)
        if done % args.save_every == 0 or done == max_steps or out_of_time:
            save_state(done)
        if out_of_time and done < max_steps:
            print(f"time budget reached at step {done}; resubmit the same command to continue")
            return
    print(f"finished: best val {best_val:.4f}; {args.out_dir}/best.pt" + (" + best_lora/" if train_lora else ""))


def stage_eval(args):
    run = os.path.dirname(os.path.abspath(args.ckpt))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = ck.get("meta", {})
    for k in ("model_id", "sa_builder", "sa_ckpt", "crop_to_patches", "panel_px"):
        if meta.get(k) is not None and getattr(args, k) != meta[k]:
            print(f"[eval] {k} from the checkpoint: {meta[k]}")
            setattr(args, k, meta[k])
    args.adapter = meta.get("base_adapter", args.adapter)
    lora_dir = os.path.join(run, "best_lora") if meta.get("stage") == "b" else None
    m = Exp2Model(args, train_lora=False, lora_dir=lora_dir, conn_state=ck["conn"], conn_cfg=ck["conn_cfg"])
    out_dir = args.out_dir or os.path.join(run, f"eval_{args.ablate}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"results_shard{args.shard_idx}.json")
    files = shard_files(problem_files(args.npz_dir, args.file_list), args.shard_idx, args.num_shards, args.limit)
    results = json.load(open(out)) if os.path.exists(out) else []
    have = {r["file"] for r in results}
    todo = [f for f in files if f not in have]
    print(f"[eval] {len(files)} problems, {len(have)} done, {len(todo)} to go | ablate {args.ablate}")
    json.dump({"ckpt": args.ckpt, "lora": lora_dir, "base_adapter": args.adapter, "ablate": args.ablate,
               "npz_dir": args.npz_dir, "file_list": args.file_list}, open(os.path.join(out_dir, "eval_config.json"), "w"),
              indent=1)

    def flush(rows):
        results.extend(rows)
        tmp = out + ".tmp"
        json.dump(results, open(tmp, "w"))
        os.replace(tmp, out)

    if todo:
        loader = DataLoader(PGMStream(args.npz_dir, todo), batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=m.collator(args))
        evaluate(m, loader, args.ablate, on_batch=flush)
    s = summarize(results)
    if "scores" in s:
        lo, hi = s["scores"]["ci95"]
        print(f"[eval] {s['n']} problems: acc {s['scores']['acc']:.2%} (95% CI {lo:.2%}-{hi:.2%}) | ablate {args.ablate}")
    print(f"merge all shards:  python {os.path.relpath(os.path.join(ROOT, 'lora', 'eval_vlm.py'))} --merge_dir {out_dir}")


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["inspect", "complement", "train", "eval"])
    # models
    ap.add_argument("--model_id", default="OpenGVLab/InternVL3_5-1B-HF")
    ap.add_argument("--adapter", default=None, help="the fine-tuned LoRA adapter (PEFT dir), e.g. .../ivl1b_pc_recipe_e1/final_e6")
    ap.add_argument("--sa_builder", default=os.environ.get("SA_BUILDER"), help="your filled-in copy of sa_builder_template.py")
    ap.add_argument("--sa_ckpt", default=os.environ.get("SA_CKPT"), help="the Slot Abstractor checkpoint")
    # data (the repo's .npz layout: $DATA/{train_full,val_full,test_full}/neutral)
    ap.add_argument("--train_dir")
    ap.add_argument("--val_dir")
    ap.add_argument("--val_list", default=None, help="e.g. results/splits/val1000_shared.tsv")
    ap.add_argument("--npz_dir", help="complement / eval: the split to score")
    ap.add_argument("--file_list", default=None, help="complement / eval: e.g. results/splits/test5000.tsv")
    ap.add_argument("--file_cache_dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n", type=int, default=300, help="inspect: validation problems")
    ap.add_argument("--list_modules", action="store_true")
    ap.add_argument("--vlm_results", default=None, help="complement: reuse eval_vlm.py results_shard*.json instead of rescoring")
    # Exp2 design
    ap.add_argument("--stage", choices=["a", "b"], default="a")
    ap.add_argument("--layout", default="interleave", choices=["interleave", "grouped", "obj", "rel"])
    ap.add_argument("--rel_tokens", type=int, default=9, help="relation tokens when the SA's relations are pooled")
    ap.add_argument("--conn_hidden", type=int, default=0, help="connector width (0 = the LLM width)")
    ap.add_argument("--control", default="none", choices=["none", "shuffle_puzzle", "zero"])
    ap.add_argument("--branch_dropout", type=float, default=0.0, help="stage b: P(tokens zeroed) per micro-batch")
    ap.add_argument("--val_controls", default="zero", help="extra validation conditions, comma-separated")
    ap.add_argument("--init_from", default=None, help="stage b: a stage-a best.pt to start the connectors from")
    ap.add_argument("--ckpt", default=None, help="eval: a run's best.pt")
    ap.add_argument("--ablate", default="none", choices=ABLATIONS)
    ap.add_argument("--out_dir")
    # optimisation (the repo's InternVL recipe: global batch 16 problems, LoRA r16 / alpha 32 / lr 1e-4)
    ap.add_argument("--train_samples", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=2, help="problems per micro-batch (x8 sequences)")
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4, help="connectors")
    ap.add_argument("--lora_lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lr_schedule", default="cosine", choices=["constant", "cosine"])
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--val_samples", type=int, default=1000)
    ap.add_argument("--val_batch_size", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--time_limit_min", type=float, default=0)
    # system
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--crop_to_patches", type=int, default=1, help="must match the adapter's training (InternVL recipe: 1)")
    ap.add_argument("--panel_px", type=int, default=160, help="must match the adapter's training")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()
    need = {"inspect": ["sa_builder"] + ([] if args.list_modules else ["val_dir"]),
            "complement": ["sa_builder", "npz_dir", "out_dir"],
            "train": ["sa_builder", "train_dir", "out_dir"], "eval": ["ckpt", "npz_dir"]}[args.mode]
    miss = [k for k in need if not getattr(args, k)]
    if miss:
        ap.error("missing " + ", ".join("--" + k for k in miss))
    {"inspect": stage_inspect, "complement": stage_complement, "train": stage_train, "eval": stage_eval}[args.mode](args)


if __name__ == "__main__":
    main()
