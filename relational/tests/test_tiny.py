"""CPU tests with random-weight tiny Qwen3-VL / InternVL models (no downloads).   python tests/test_tiny.py
  1. geometry: Qwen merge-block -> raster reorder matches the library's own position ids; ROI sampling; token -> panel
  2. fuse with lambda = 1 reproduces the plain VLM exactly (the hooks do not disturb the native path)
  3. interleave / fuse: gradients reach the branch and nothing in the VLM; relation tokens change the output
  4. eval is deterministic (seeded slot init); aux scores depend on the candidate; the Abstractor learns a copy relation
  5. CLI: stage 0 -> stage 1 (+ resume) -> stage 2 (LoRA, init_from) -> sharded eval -> merge
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import tiny_models as T
from backbones import Backbone, window_to_raster
from data import Collator, FixedSet, MatrixSource, grid_layout
from relvlm import RelationalVLM, ScoreTokens, example_logits, loss_and_acc, roi_sample, token_panel_coords
from slots import SlotAutoencoder

CPU = torch.device("cpu")
FAMILIES = {"qwen3vl": T.tiny_qwen3vl, "internvl": lambda: T.tiny_internvl(crop=False)}


def fake_npz_dir(n=6, seed=0, easy=False):
    """Random problems in the real file layout. easy=True: the answer is a copy of context panel 2 (top-right cell) --
    a RELATION (identity between cells), which is what the relational bottleneck can see; plain appearance is not."""
    rng = np.random.default_rng(seed)
    d = tempfile.mkdtemp()
    for i in range(n):
        img = rng.integers(0, 255, (16, 160, 160), dtype=np.uint8)
        t = i % 8
        if easy:
            img[8 + t] = img[2]
        np.savez(os.path.join(d, f"p{i:03d}.npz"), image=img.reshape(160, 160, 16), target=np.int64(t))
    return d


def setup(fam, inject="interleave", M=2, **kw):
    model, proc = FAMILIES[fam]()
    model.requires_grad_(False)
    bb = Backbone(model, proc, fam)
    ae = SlotAutoencoder(bb.vit_dim, slot_dim=8, num_slots=4, iters=2, decoder_hidden=16, grid=4).eval()
    ae.requires_grad_(False)
    rv = RelationalVLM(bb, ae, inject=inject, rel_tokens=M,
                       abstractor_cfg=dict(layers=2, heads=2, head_dim=4, mlp_dim=16), **kw)
    return rv, bb, ScoreTokens(proc.tokenizer)


def batch_for(bb, rv, d, idx=(0, 1)):
    src = MatrixSource(d)
    col = Collator(bb.processor, bb.processor_kwargs(), rel_tokens=rv.M if rv.interleave else 0)
    return col([FixedSet(src, list(range(len(src))))[i] for i in idx])


# ----------------------------------------------------------------------------- 1. geometry
def test_window_to_raster():
    from transformers.vision_utils import get_vision_position_ids
    h, w = 6, 8
    pos = get_vision_position_ids(torch.tensor([[1, h, w]]), 2)            # (h*w, 2) block-major (row, col)
    r = window_to_raster(pos.float(), h, w, 2)
    gy, gx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert torch.equal(r[..., 0].long(), gy) and torch.equal(r[..., 1].long(), gx)


def test_roi_and_tokens():
    h, w = 10, 20
    xs = (torch.arange(w).float() + 0.5) / w
    fmap = xs[None, :, None].expand(h, w, 1).contiguous()                  # value = x coordinate
    box = torch.tensor([[0.25, 0.0, 0.75, 1.0]])
    out = roi_sample(fmap, box, 4)[0, :, 0].reshape(4, 4)
    want = 0.25 + 0.5 * (torch.arange(4).float() + 0.5) / 4
    assert torch.allclose(out[0], want, atol=1e-5), (out[0], want)
    _, boxes = grid_layout(3, 3)
    owner, local = token_panel_coords(torch.tensor(boxes), 16, 16)
    assert owner.max() == 8 and (owner == -1).any()                          # gaps between panels exist
    assert local[0][owner == 0].abs().max() <= 1.0 + 1e-6


def test_packed_sources():
    """zip-backed jsonl rows (all four kinds, per-candidate prompts) and a zipped matrix directory."""
    import zipfile
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "datasets"))
    from data import JsonlSource
    from packing import Packer
    d = tempfile.mkdtemp()
    pk = Packer(d, "train")
    a = pk.add_image("a.png", Image.new("RGB", (40, 30), (255, 0, 0)))
    b = pk.add_image("b.png", Image.new("RGB", (40, 30), (0, 255, 0)))
    pk.add_row({"id": "c", "candidates": [a, b], "prompts": ["p0", "p1"], "prompt": "", "answer": 1,
                "panels": [[0, 0, 0.5, 1], [0.5, 0, 1, 1]]})
    pk.add_row({"id": "y", "images": [a], "prompt": "q", "answer": "No"})
    pk.add_row({"id": "l", "images": [b], "prompt": "q", "choices": ["A", "B", "C"], "answer": "C"})
    pk.add_row({"id": "i", "images": [b]})
    pk.close()
    src = JsonlSource(os.path.join(d, "train.jsonl"))
    rng = np.random.default_rng(0)
    c, y, l, i = (src.get(k, rng) for k in range(4))
    assert c["kind"] == "cand" and c["label"] == 1 and [s["prompt"] for s in c["seqs"]] == ["p0", "p1"]
    assert c["seqs"][1]["image"].getpixel((0, 0)) == (0, 255, 0) and len(c["seqs"][0]["boxes"]) == 2
    assert y["kind"] == "binary" and y["label"] == 0
    assert l["kind"] == "letter" and l["label"] == 2 and l["choices"] == ["A", "B", "C"]
    assert i["kind"] == "image" and i["label"] is None
    md = fake_npz_dir(3)
    zpath = os.path.join(d, "m.zip")
    with zipfile.ZipFile(zpath, "w") as z:
        for f in sorted(os.listdir(md)):
            z.write(os.path.join(md, f), f"cfg/{f}")
    a1, a2 = MatrixSource(zpath).get(1, rng), MatrixSource(md).get(1, rng)
    assert a1["label"] == a2["label"] and np.array_equal(np.asarray(a1["seqs"][3]["image"]), np.asarray(a2["seqs"][3]["image"]))


# ----------------------------------------------------------------------------- 2.-4. model
def test_fuse_identity(fam):
    d = fake_npz_dir(2)
    rv, bb, st = setup(fam, inject="fuse")
    rv.eval()
    with torch.no_grad():
        rv.lambda_logit.fill_(40.0)                                          # sigmoid -> exactly 1.0 in fp32
        batch = batch_for(bb, rv, d)
        h, _ = rv(batch, CPU, torch.float32)
        ref = bb.core(**bb.model_inputs(batch, CPU, torch.float32), use_cache=False).last_hidden_state[:, -1]
    assert torch.allclose(h, ref.float(), atol=1e-5), (h - ref).abs().max()


def test_grads(fam, inject):
    d = fake_npz_dir(2)
    rv, bb, st = setup(fam, inject=inject, fuse_lambda=0.5)
    rv.train()
    batch = batch_for(bb, rv, d)
    h, aux = rv(batch, CPU, torch.float32)
    loss, _, _ = loss_and_acc(example_logits(h, aux, batch, bb.model, st), "llm")
    loss.backward()
    names = {n for n, p in rv.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any(n.startswith("abstractor.") for n in names), names
    if rv.interleave:
        assert any(n.startswith("tok_proj") for n in names) and "tok_gain" in names
    if rv.fuse:
        assert any(n.startswith("fuse_proj") for n in names) and "lambda_logit" in names
    assert all(p.grad is None for p in bb.model.parameters()), "the VLM must stay frozen"
    with torch.no_grad():                                                  # the injection actually reaches the LLM
        rv.eval()
        h1, _ = rv(batch, CPU, torch.float32)
        for n, p in rv.named_parameters():
            if n.startswith(("tok_proj", "fuse_proj")):
                p.add_(torch.randn_like(p))
        h2, _ = rv(batch, CPU, torch.float32)
    assert (h1 - h2).abs().max() > 1e-4


def test_eval_deterministic(fam, inject):
    d = fake_npz_dir(2)
    rv, bb, st = setup(fam, inject=inject)
    rv.eval()
    batch = batch_for(bb, rv, d)
    with torch.no_grad():
        a, x = rv(batch, CPU, torch.float32)
        b, y = rv(batch, CPU, torch.float32)
    assert torch.equal(a, b) and torch.equal(x, y)


def test_aux_depends_on_candidate(fam):
    """The 8 sequences of a problem differ only in the bottom-right cell: their aux scores must differ, and the aux loss
    must send gradient into the Abstractor."""
    d = fake_npz_dir(2)
    rv, bb, st = setup(fam, inject="interleave")
    rv.train()
    batch = batch_for(bb, rv, d)
    _, aux = rv(batch, CPU, torch.float32, llm=False)
    assert aux.view(2, 8).std(1).min() > 1e-6, aux
    loss, _, _ = loss_and_acc(example_logits(None, aux, batch, bb.model, st), "aux")
    loss.backward()
    assert sum(p.grad.abs().sum() for p in rv.abstractor.parameters() if p.grad is not None) > 0


def test_abstractor_learns_relation():
    """Pure Abstractor on synthetic slots: 9 panels x K slots, the correct candidate (panel 8) is a copy of panel 2,
    distractors are random. Appearance only reaches the Abstractor through who-attends-to-whom (the relational
    bottleneck), so solving this shows RCA + panel embeddings carry the relation."""
    from abstractor import RelationalAbstractor
    torch.manual_seed(0)
    K, D, B = 4, 16, 32
    ab = RelationalAbstractor(dim=D, heads=2, head_dim=8, layers=2, mlp_dim=32)
    head = torch.nn.Linear(D, 1)
    pos = torch.randn(K, D)
    opt = torch.optim.Adam([*ab.parameters(), *head.parameters()], lr=3e-3)
    mask = torch.ones(B * 8, 9, dtype=torch.bool)
    for it in range(300):
        ctx = torch.randn(B, 8, K, D)
        cands = torch.randn(B, 8, K, D)
        tgt = torch.randint(0, 8, (B,))
        cands[torch.arange(B), tgt] = ctx[:, 2]
        z = torch.cat([ctx[:, None].expand(B, 8, 8, K, D), cands[:, :, None]], 2).reshape(B * 8, 9, K, D)
        m = pos.expand(B * 8, 9, K, D)
        _, r = ab(z, m, mask)
        loss = torch.nn.functional.cross_entropy(head(r).view(B, 8), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = (head(r).view(B, 8).argmax(1) == tgt).float().mean()
    assert acc > 0.8, (float(loss), float(acc))


# ----------------------------------------------------------------------------- 5. CLI
def run(*cmd):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-5000:])
        raise RuntimeError(f"failed: {cmd[0]}")
    return r.stdout


def test_cli(fam):
    model, proc = FAMILIES[fam]()
    mdir = tempfile.mkdtemp()
    model.save_pretrained(mdir)
    proc.save_pretrained(mdir)
    d, out = fake_npz_dir(8, easy=True), tempfile.mkdtemp()
    common = ["--model_id", mdir, "--dtype", "fp32", "--num_workers", "0"]
    run("train_stage0.py", *common, "--out_dir", f"{out}/s0", "--train", f"pgm=matrix:{d}", "--val", f"pgm=matrix:{d}",
        "--val_samples", "4", "--grid", "4", "--slot_dim", "8", "--num_slots", "4", "--decoder_hidden", "16",
        "--batch_size", "2", "--max_steps", "4", "--val_every", "2", "--log_every", "2", "--warmup_steps", "1")
    assert os.path.exists(f"{out}/s0/best.pt") and os.path.exists(f"{out}/s0/slots_step2.png")
    s1 = [*common, "--slot_ckpt", f"{out}/s0/best.pt", "--train", f"pgm=matrix:{d}", "--val", f"pgm=matrix:{d}",
          "--val_samples", "4", "--layers", "2", "--heads", "2", "--head_dim", "4", "--mlp_dim", "16",
          "--batch_size", "2", "--grad_accum", "1", "--val_every", "2", "--log_every", "1", "--warmup_steps", "1",
          "--inject", "interleave", "--rel_tokens", "2", "--objective", "both"]
    run("train_stage1.py", *s1, "--out_dir", f"{out}/s1", "--max_steps", "2", "--fit_eval_samples", "4")
    assert any(json.loads(l).get("val") == "fit" for l in open(f"{out}/s1/metrics.jsonl"))
    o = run("train_stage1.py", *s1, "--out_dir", f"{out}/s1", "--max_steps", "4")          # resume 2 -> 4
    assert "resumed at step 2" in o
    run("train_stage1.py", *s1, "--out_dir", f"{out}/s2", "--max_steps", "2", "--lora_r", "4",
        "--init_from", f"{out}/s1/best.pt")
    ck = torch.load(f"{out}/s2/best.pt", weights_only=False)
    assert ck["lora"], "Stage 2 must save LoRA weights"
    for k in range(2):
        run("eval.py", "--ckpt", f"{out}/s2/best.pt", "--source", f"pgm=matrix:{d}", "--out_json",
            f"{out}/ev/shard{k}.json", "--shard_idx", str(k), "--num_shards", "2", "--dtype", "fp32", "--num_workers", "0")
    run("eval.py", "--merge_dir", f"{out}/ev", "--expected_n", "8")
    m = json.load(open(f"{out}/ev/merged.json"))
    assert m["unique"] == 8 and m["duplicates"] == 0 and "llm" in m and "aux" in m
    run("eval_baseline.py", "--model_id", mdir, "--source", f"pgm=matrix:{d}", "--out_json", f"{out}/bl/shard0.json",
        "--dtype", "fp32", "--num_workers", "0", "--max_examples", "4")
    b = json.load(open(f"{out}/bl/shard0.json"))
    assert b["summary"]["n"] == 4 and "llm" in b["summary"] and "aux" not in b["summary"]


if __name__ == "__main__":
    only = sys.argv[1:]
    tests = [("window_to_raster", test_window_to_raster, ()), ("roi_and_tokens", test_roi_and_tokens, ()),
             ("abstractor_learns_relation", test_abstractor_learns_relation, ()),
             ("packed_sources", test_packed_sources, ())]
    for fam in FAMILIES:
        tests += [(f"fuse_identity[{fam}]", test_fuse_identity, (fam,))]
        tests += [(f"grads[{fam},{inj}]", test_grads, (fam, inj)) for inj in ("interleave", "fuse")]
        tests += [(f"eval_deterministic[{fam},{inj}]", test_eval_deterministic, (fam, inj)) for inj in ("interleave", "fuse")]
        tests += [
                  (f"aux_depends[{fam}]", test_aux_depends_on_candidate, (fam,)), (f"cli[{fam}]", test_cli, (fam,))]
    failed = 0
    for name, fn, a in tests:
        if only and not any(o in name for o in only):
            continue
        try:
            fn(*a)
            print(f"PASS {name}", flush=True)
        except Exception as e:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {e}", flush=True)
    print(f"{failed} failed")
    sys.exit(1 if failed else 0)
