"""CPU tests on random-weight tiny Qwen3-VL / InternVL models (no downloads).  python tests/test_pipeline.py
What they check (per-candidate scoring):
  1. the 8 images of a problem are the context + candidate k in the bottom-right cell, in candidate order
  2. Yes/No score tokens resolve; each sequence ends with the assistant header
  3. scores of a batch of problems are problem-major: batched (B, 8) == each problem scored on its own
  4. left-padding equivalence: padding cannot change a score
  5. a few LoRA steps on 6 toy problems drive ce8 down and accuracy to 100% (the LoRA actually trains through the
     exact training code path); only LoRA params change; adapter reload reproduces the scores
  6. resume gives the same trajectory as an uninterrupted run (data order is a function of position)
  7. train -> eval (sharded) -> merge via the real CLI entry points; Table-5 tool; chained submit script
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import tiny_models as T
from pgmvlm.models import ScoreTokens, add_lora, candidate_losses, candidate_scores, to_model_inputs
from pgmvlm.data import (CandidateCollator, PGMStream, filled_matrices, list_npz, make_order, panels_to_grid_image,
                          to_panels)

FAMILIES = {"qwen3vl": T.tiny_qwen3vl, "internvl": T.tiny_internvl}
CPU = torch.device("cpu")


def fake_npz_dir(n=6, seed=0, const_target=None, easy=False):
    """Random problems in the real file layout. easy=True: the answer panel is all black and the 7 distractors all
    white, so 'is the bottom-right cell black?' decides it (a learnable toy for the training test)."""
    rng = np.random.default_rng(seed)
    d = tempfile.mkdtemp()
    for i in range(n):
        img = rng.integers(0, 255, (16, 160, 160), dtype=np.uint8)
        t = i % 8 if const_target is None else const_target
        if easy:
            img[8:] = 255
            img[8 + t] = 0
        else:
            img[8 + t] = img[2]
        np.savez(os.path.join(d, f"p{i:03d}.npz"), image=img.reshape(160, 160, 16), target=np.int64(t))   # real file layout
    return d


def batch_for(fam, proc, d, idx=(0, 1)):
    ds = PGMStream(d, list_npz(d))
    return CandidateCollator(proc, fam, crop_to_patches=False)([ds[i] for i in idx])


def test_to_panels_layout():
    """Real PGM files declare (160, 160, 16) but are panel-major: reshape, never transpose."""
    p = np.random.default_rng(0).integers(0, 255, (16, 160, 160), dtype=np.uint8)
    assert np.array_equal(to_panels(p.reshape(160, 160, 16)), p)
    assert np.array_equal(to_panels(p), p)
    try:
        to_panels(np.zeros((15, 160, 160)))
        raise AssertionError("wrong size must raise")
    except ValueError:
        pass


def test_filled_matrices_put_candidate_k_in_the_missing_cell():
    rng = np.random.default_rng(1)
    ctx = rng.integers(0, 255, (8, 160, 160), dtype=np.uint8)
    cand = rng.integers(0, 255, (8, 160, 160), dtype=np.uint8)
    mats = filled_matrices(ctx, cand)
    assert len(mats) == 8 and all(m.size == (520, 520) for m in mats)
    x = y = 10 + 2 * 170                                                   # bottom-right cell origin (pad 10, cell 160)
    for k, m in enumerate(mats):
        a = np.asarray(m)[:, :, 0]
        assert np.array_equal(a[y:y + 160, x:x + 160], cand[k]), k                  # candidate k in the missing cell
        assert np.array_equal(a[10:170, 10:170], ctx[0]) and np.array_equal(a[180:340, 350:510], ctx[5])
    ref = np.asarray(panels_to_grid_image(list(ctx) + [cand[3]], (3, 3)))
    assert np.array_equal(np.asarray(mats[3]), ref)
    big = filled_matrices(ctx, cand, cell_size=224)
    assert big[0].size == (3 * 224 + 40, 3 * 224 + 40)


def test_score_tokens_and_prompt():
    for fam, fn in FAMILIES.items():
        _, proc = fn()
        st = ScoreTokens(proc.tokenizer)
        assert st.yes_id != st.no_id
        assert proc.tokenizer.decode([st.yes_id]).startswith("Y") and proc.tokenizer.decode([st.no_id]).startswith("N")
        b = batch_for(fam, proc, fake_npz_dir(), idx=(0,))
        assert b["input_ids"].shape[0] == 8
        for row in b["input_ids"]:
            text = proc.tokenizer.decode(row.tolist())
            assert text.rstrip().endswith("<|im_start|>assistant"), text[-60:]
            assert "Answer Yes or No" in text


def test_internvl_crop_flag_is_honoured():
    """The InternVL processor defaults crop_to_patches=True and overrides the checkpoint's preprocessor_config (False);
    CandidateCollator must therefore pass the flag explicitly. Qwen ignores it."""
    _, proc = T.tiny_internvl()
    d = fake_npz_dir()
    ds = PGMStream(d, list_npz(d))
    off = CandidateCollator(proc, "internvl", crop_to_patches=False)([ds[0]])
    on = CandidateCollator(proc, "internvl", crop_to_patches=True)([ds[0]])
    assert off["pixel_values"].shape[0] == 8, off["pixel_values"].shape          # one tile per completed matrix
    n_img_tokens = lambda b: int((b["input_ids"] == proc.image_token_id).sum())
    assert n_img_tokens(off) == 8 * 4 and n_img_tokens(on) == on["pixel_values"].shape[0] * 4


def test_scores_are_problem_major():
    """(B, 8) scores of a batch == each problem scored alone, so row b / column k really is problem b / candidate k."""
    for fam, fn in FAMILIES.items():
        model, proc = fn()
        st = ScoreTokens(proc.tokenizer)
        d = fake_npz_dir()
        with torch.no_grad():
            both = candidate_scores(model, batch_for(fam, proc, d, (0, 1, 2)), st, CPU, torch.float32)
            alone = torch.cat([candidate_scores(model, batch_for(fam, proc, d, (i,)), st, CPU, torch.float32) for i in range(3)])
        assert both.shape == (3, 8)
        assert torch.allclose(both, alone, atol=1e-4), (fam, (both - alone).abs().max().item())
        # and the score is exactly logit(Yes) - logit(No) at the last position
        b = batch_for(fam, proc, d, (0,))
        with torch.no_grad():
            full = model(**to_model_inputs(b, CPU, torch.float32)).logits[:, -1]
        assert torch.allclose(full[:, st.yes_id] - full[:, st.no_id], alone[0], atol=1e-4)


def test_left_padding_does_not_change_scores():
    for fam, fn in FAMILIES.items():
        model, proc = fn()
        st = ScoreTokens(proc.tokenizer)
        b = batch_for(fam, proc, fake_npz_dir(), idx=(0,))
        with torch.no_grad():
            ref = candidate_scores(model, b, st, CPU, torch.float32)
        pad, n = 7, b["input_ids"].shape[0]
        padded = dict(b)
        padded["input_ids"] = torch.cat([torch.full((n, pad), proc.tokenizer.pad_token_id), b["input_ids"]], 1)
        padded["attention_mask"] = torch.cat([torch.zeros(n, pad, dtype=torch.long), b["attention_mask"]], 1)
        if "mm_token_type_ids" in b:
            padded["mm_token_type_ids"] = torch.cat([torch.zeros(n, pad, dtype=torch.long), b["mm_token_type_ids"]], 1)
        with torch.no_grad():
            got = candidate_scores(model, padded, st, CPU, torch.float32)
        err = (ref - got).abs().max().item()
        assert err < 1e-3, f"{fam}: left padding changed scores by {err}"


def test_losses():
    s = torch.tensor([[0.0, 3.0, -1.0, 0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0, 0, 0, 0]])
    t = torch.tensor([1, 3])
    ce8, bce, acc = candidate_losses(s, t)
    assert abs(ce8.item() - F.cross_entropy(s, t).item()) < 1e-6 and acc.item() == 0.5
    onehot = F.one_hot(t, 8).float()
    manual = -(7 * onehot * F.logsigmoid(s) + (1 - onehot) * F.logsigmoid(-s)).mean()     # pos_weight 7 on the answer
    assert abs(bce.item() - manual.item()) < 1e-5, (bce.item(), manual.item())


def _train_steps(fam, fn, steps, d, n_problems=6):
    model, proc = fn(seed=0)
    with torch.no_grad():       # random tiny embeddings are ~0.02-scale, which caps the reachable logit range;
        model.get_input_embeddings().weight.mul_(8)       # scale them so the scores can actually separate
    st = ScoreTokens(proc.tokenizer)
    model = add_lora(model, fam, r=8, alpha=16, dropout=0.0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-3)
    batch = batch_for(fam, proc, d, idx=tuple(range(n_problems)))
    model.train()
    hist = []
    for s in range(steps):
        ce8, bce, acc = candidate_losses(candidate_scores(model, batch, st, CPU, torch.float32), batch["targets"])
        ce8.backward()
        opt.step(); opt.zero_grad()
        hist.append((ce8.item(), acc.item()))
    return model, proc, st, batch, hist


def test_lora_trains_and_base_is_frozen():
    """Toy where the answer is the matrix whose bottom-right cell is black: LoRA must learn to pick it through the exact
    training code path; base weights stay bit-identical; an adapter reload reproduces the scores."""
    from pgmvlm.models import load_adapter
    d = fake_npz_dir(easy=True)
    for fam, fn in FAMILIES.items():
        base_model, _ = fn(seed=0)
        with torch.no_grad():
            base_model.get_input_embeddings().weight.mul_(8)
        before = {k: v.clone() for k, v in base_model.state_dict().items()}
        model, proc, st, batch, hist = _train_steps(fam, fn, 60, d)
        assert hist[-1][0] < 0.3 * hist[0][0], f"{fam}: ce8 {hist[0][0]:.3f} -> {hist[-1][0]:.3f}"
        assert hist[-1][1] == 1.0, f"{fam}: accuracy {hist[0][1]:.2f} -> {hist[-1][1]:.2f}"
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trainable and all("lora_" in n for n in trainable)
        after = {k.replace("base_model.model.", "").replace(".base_layer", ""): v for k, v in model.state_dict().items() if "lora_" not in k}
        for k, v in before.items():
            assert torch.equal(v, after[k]), f"{fam}: base weight {k} changed"
        model.eval()
        with torch.no_grad():
            sc = candidate_scores(model, batch, st, CPU, torch.float32)
        assert (sc.argmax(-1) == batch["targets"]).all(), (fam, sc.argmax(-1), batch["targets"])
        td = tempfile.mkdtemp()
        model.save_pretrained(td)
        fresh, _ = fn(seed=0)
        with torch.no_grad():
            fresh.get_input_embeddings().weight.mul_(8)
        re = load_adapter(fresh, td).eval()
        with torch.no_grad():
            sc2 = candidate_scores(re, batch, st, CPU, torch.float32)
        assert torch.allclose(sc, sc2, atol=1e-4), (sc - sc2).abs().max()
        print(f"   {fam}: ce8 {hist[0][0]:.3f} -> {hist[-1][0]:.3f}, acc {hist[0][1]:.2f} -> {hist[-1][1]:.2f}")


def test_data_order_is_position_deterministic():
    d = fake_npz_dir()
    files = list_npz(d)
    order = make_order(len(files), 20, seed=3)
    s1 = PGMStream(d, files, order=order, seed=3, shuffle_candidates=True)
    s2 = PGMStream(d, files, order=order, seed=3, shuffle_candidates=True)
    for pos in (0, 7, 19):
        a, b = s1[pos], s2[pos]
        assert a["file"] == b["file"] and a["target"] == b["target"] and np.array_equal(a["candidates"], b["candidates"])
    plain = PGMStream(d, files, order=order, seed=3, shuffle_candidates=False)
    for pos in range(10):
        x, y = plain[pos], s1[pos]
        assert x["file"] == y["file"] and np.array_equal(x["candidates"][x["target"]], y["candidates"][y["target"]])
    assert len(set(order[:6].tolist())) == 6                                   # first epoch is a permutation


def test_preflight_runs():
    d = fake_npz_dir()
    for fam in FAMILIES:
        env = dict(os.environ, TINY_FAMILY=fam)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "run_tiny.py"), "--model_id", "tiny_" + fam, "--dtype", "fp32",
                            "--attn", "default", "preflight", "--npz_dir", d, "--batch_size", "2", "--steps", "2", "--lora_r", "4",
                            "--crop_to_patches", "0", "--lora_vision", "--lora_projector"], capture_output=True, text=True, env=env)
        assert r.returncode == 0 and "preflight OK" in r.stdout, r.stdout[-1500:] + r.stderr[-2500:]
        assert "image tokens>" in r.stdout and "score tokens" in r.stdout and "16 sequences for 2 problems" in r.stdout


def test_end_to_end_scripts():
    """train (interrupted + resumed) and eval (sharded + merged) via the real CLI entry points on tiny models."""
    d_tr, d_va = fake_npz_dir(8, 1), fake_npz_dir(4, 2)
    for fam in FAMILIES:
        out = tempfile.mkdtemp()
        env = dict(os.environ, TINY_FAMILY=fam)
        base = [sys.executable, os.path.join(ROOT, "tests", "run_tiny.py")]
        common = ["--model_id", "tiny_" + fam, "--dtype", "fp32", "--num_workers", "0", "--attn", "default"]
        tr = common + ["train", "--train_dir", d_tr, "--val_dir", d_va, "--out_dir", out, "--max_steps", "6",
                       "--batch_size", "2", "--grad_accum", "2", "--lr", "3e-3", "--warmup_steps", "1", "--val_every", "3",
                       "--val_samples", "4", "--val_batch_size", "2", "--save_every", "2", "--log_every", "2",
                       "--lora_r", "8", "--lora_alpha", "16", "--crop_to_patches", "0", "--fit_eval_samples", "4"]
        r = subprocess.run(base + tr + ["--time_limit_min", "0.0001"], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        assert "time budget reached" in r.stdout, r.stdout[-800:]
        r = subprocess.run(base + tr, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        assert "resumed from step" in r.stdout and "finished" in r.stdout, r.stdout[-1200:]
        assert "[fit] step 6: acc on 4 training problems" in r.stdout, r.stdout[-1200:]
        assert os.path.isdir(os.path.join(out, "final")) and os.path.isdir(os.path.join(out, "best"))
        res = tempfile.mkdtemp()
        for sh in (0, 1):
            ev = common + ["eval", "--adapter", os.path.join(out, "best"), "--npz_dir", d_va, "--out_dir", res,
                           "--shard_idx", str(sh), "--num_shards", "2", "--batch_size", "2", "--crop_to_patches", "0"]
            r = subprocess.run(base + ev, capture_output=True, text=True, env=env)
            assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        r = subprocess.run(base + ["--model_id", "x", "eval", "--merge_dir", res, "--expected_n", "4"], capture_output=True, text=True, env=env)
        assert r.returncode == 0 and "4 problems from 2 shards" in r.stdout, r.stdout + r.stderr[-1500:]
        import json
        s = json.load(open(os.path.join(res, "summary.json")))
        assert s["complete"] and s["duplicates"] == 0 and "scores" in s and s["scores"]["n"] == 4
        # resume of an eval shard skips finished work
        r = subprocess.run(base + common + ["eval", "--adapter", os.path.join(out, "best"), "--npz_dir", d_va, "--out_dir", res,
                                           "--shard_idx", "0", "--num_shards", "2", "--batch_size", "2", "--crop_to_patches", "0"],
                           capture_output=True, text=True, env=env)
        assert "0 to go" in r.stdout, r.stdout[-600:]
        print(f"   {fam}: train(interrupt+resume) -> eval(2 shards) -> merge OK")


def test_resume_matches_uninterrupted():
    """Interrupt after one step, resume, finish: the final adapter must equal an uninterrupted run's (same data order,
    same optimizer state, same LR schedule). LoRA dropout is 0 so the comparison is exact."""
    from safetensors.torch import load_file
    d_tr = fake_npz_dir(8, 1)
    for fam in FAMILIES:
        env = dict(os.environ, TINY_FAMILY=fam)
        base = [sys.executable, os.path.join(ROOT, "tests", "run_tiny.py"), "--model_id", "tiny_" + fam, "--dtype", "fp32",
                "--num_workers", "0", "--attn", "default"]
        def cmd(out):
            return base + ["train", "--train_dir", d_tr, "--out_dir", out, "--max_steps", "6", "--batch_size", "2",
                           "--grad_accum", "2", "--lr", "3e-3", "--warmup_steps", "2", "--val_every", "1000", "--val_samples", "0",
                           "--save_every", "1000", "--log_every", "3", "--lora_r", "8", "--lora_alpha", "16",
                           "--lora_dropout", "0.0", "--crop_to_patches", "0"]
        a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
        r = subprocess.run(cmd(a), capture_output=True, text=True, env=env); assert r.returncode == 0, r.stderr[-2000:]
        r = subprocess.run(cmd(b) + ["--time_limit_min", "0.0001"], capture_output=True, text=True, env=env)
        assert r.returncode == 0 and "time budget reached at step 1" in r.stdout, r.stdout[-600:] + r.stderr[-1500:]
        r = subprocess.run(cmd(b), capture_output=True, text=True, env=env); assert r.returncode == 0, r.stderr[-2000:]
        assert "resumed from step 1" in r.stdout
        wa = load_file(os.path.join(a, "final", "adapter_model.safetensors"))
        wb = load_file(os.path.join(b, "final", "adapter_model.safetensors"))
        assert wa.keys() == wb.keys()
        err = max((wa[k] - wb[k]).abs().max().item() for k in wa)
        # Not exactly 0 on some runs: CPU kernels are not bit-deterministic on every op, and Adam turns ~1e-9 gradient
        # noise on near-zero-gradient entries into ~1e-5 weight differences. A real resume bug (wrong data order, lost
        # optimizer state, wrong LR) moves the weights by >= 1e-2 over these 6 steps at lr 3e-3.
        assert err < 1e-3, f"{fam}: resumed run differs from uninterrupted by {err}"
        print(f"   {fam}: resumed vs uninterrupted max |dW| = {err:.2e}")


def test_extending_epochs_keeps_lr_schedule():
    """Finish a run, then resume it with more steps: the LR must continue at the value it ended on (no jump back up),
    unless --reset_schedule is given."""
    import json
    d_tr = fake_npz_dir(8, 1)
    env = dict(os.environ, TINY_FAMILY="qwen3vl")
    base = [sys.executable, os.path.join(ROOT, "tests", "run_tiny.py"), "--model_id", "tiny_qwen3vl", "--dtype", "fp32",
            "--num_workers", "0", "--attn", "default"]
    def run(out, steps, extra=()):
        r = subprocess.run(base + ["train", "--train_dir", d_tr, "--out_dir", out, "--max_steps", str(steps), "--batch_size", "2",
                                   "--grad_accum", "1", "--lr", "1e-3", "--warmup_steps", "1", "--val_samples", "0",
                                   "--log_every", "1", "--save_every", "1000", "--lora_r", "4", "--crop_to_patches", "0",
                                   "--lr_schedule", "cosine", *extra],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        return r.stdout
    def lrs(out):
        return {r["step"]: r["lr"] for r in map(json.loads, open(os.path.join(out, "metrics.jsonl"))) if "lr" in r}
    a = tempfile.mkdtemp()
    run(a, 4)
    first = lrs(a)
    out = run(a, 8)
    assert "resumed from step 4" in out and "continuing at the final LR" in out, out[-800:]
    ext = lrs(a)
    assert ext[5] <= first[4], (first, ext)                            # no jump back up when resuming
    assert all(abs(ext[s] - 1e-4) < 1e-9 for s in range(5, 9)), ext   # holds at the schedule's floor (0.1 x lr)
    assert json.load(open(os.path.join(a, "args.json")))["schedule_steps"] == 4
    b = tempfile.mkdtemp()                                           # opt-in: redraw the schedule over the new total
    run(b, 4)
    run(b, 8, ["--reset_schedule"])
    assert lrs(b)[5] > 3e-4, lrs(b)


def test_constant_schedule():
    from pgmvlm.common import lr_at
    assert [round(lr_at(s, 1e-4, 20, 15000, schedule="constant"), 12) for s in (0, 9, 19)] == [5e-6, 5e-5, 1e-4]
    assert all(lr_at(s, 1e-4, 20, 15000, schedule="constant") == 1e-4 for s in (20, 7500, 14999, 30000))
    assert lr_at(14999, 1e-4, 20, 15000, schedule="cosine") < 1.1e-5            # cosine still decays to 0.1 x lr


def test_cosine_decay_from_a_later_step():
    from pgmvlm.common import lr_at
    f = lambda s: lr_at(s, 1e-4, 20, 60000, schedule="cosine", decay_start=45000)
    assert f(30000) == 1e-4 and f(44999) == 1e-4 and f(45000) == 1e-4           # held, then no jump at the start
    assert abs(f(52500) - 5.5e-5) < 1e-12 and abs(f(60000) - 1e-5) < 1e-12     # halfway, floor at the end
    lrs = [f(s) for s in range(45000, 60001, 500)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))                            # only goes down


def test_extend_with_decay_start_continues_without_jump():
    """Constant-LR run to step 4, then extend to step 8 with a cosine decay starting at 4: step 5 is still the full LR,
    later steps decrease monotonically, and nothing ever goes above it."""
    import json
    d_tr = fake_npz_dir(8, 1)
    env = dict(os.environ, TINY_FAMILY="qwen3vl")
    base = [sys.executable, os.path.join(ROOT, "tests", "run_tiny.py"), "--model_id", "t", "--dtype", "fp32",
            "--num_workers", "0", "--attn", "default", "train", "--train_dir", d_tr, "--batch_size", "2", "--grad_accum", "1",
            "--lr", "1e-3", "--warmup_steps", "1", "--val_samples", "0", "--log_every", "1", "--save_every", "1000",
            "--lora_r", "4", "--crop_to_patches", "0"]
    out = tempfile.mkdtemp()
    r = subprocess.run(base + ["--out_dir", out, "--max_steps", "4"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    r = subprocess.run(base + ["--out_dir", out, "--max_steps", "8", "--lr_schedule", "cosine", "--decay_start", "4"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    assert "held until step 4, then cosine decay" in r.stdout, r.stdout[-800:]
    lr = {x["step"]: x["lr"] for x in map(json.loads, open(os.path.join(out, "metrics.jsonl"))) if "lr" in x}
    assert lr[4] == 1e-3 and lr[5] == 1e-3, lr                              # end of constant phase, start of decay
    assert lr[5] > lr[6] > lr[7] > lr[8] >= 1e-4, lr


def test_warm_restart_schedule():
    from pgmvlm.common import lr_restart
    f = lambda s: lr_restart(s, 1e-4, 60000, 500, 90000, 1e-5, 1e-5)
    assert abs(f(60000) - (1e-5 + 9e-5 / 500)) < 1e-12                        # starts at the old floor, no jump
    assert abs(f(60499) - 1e-4) < 1e-12 and abs(f(60500) - 1e-4) < 1e-12     # peak after the warmup
    warm = [f(s) for s in range(60000, 60500)]
    assert all(a < b for a, b in zip(warm, warm[1:]))                        # warmup only rises
    dec = [f(s) for s in range(60500, 90001, 250)]
    assert all(a >= b for a, b in zip(dec, dec[1:])) and abs(f(90000) - 1e-5) < 1e-12   # then only falls, to the floor
    g = lambda s: lr_restart(s, 5e-5, 60000, 500, 75000, 1e-5, 5e-6)
    assert abs(g(60499) - 5e-5) < 1e-12 and abs(g(75000) - 5e-6) < 1e-12


def test_extend_with_warm_restart():
    """Finish a run, then continue it with --lr_schedule restart: the LR starts near --cycle_from, rises to --lr over
    --cycle_warmup updates, then decays to --min_lr; a missing --cycle_start is refused."""
    import json
    d_tr = fake_npz_dir(8, 1)
    env = dict(os.environ, TINY_FAMILY="qwen3vl")
    base = [sys.executable, os.path.join(ROOT, "tests", "run_tiny.py"), "--model_id", "t", "--dtype", "fp32",
            "--num_workers", "0", "--attn", "default", "train", "--train_dir", d_tr, "--batch_size", "2", "--grad_accum", "1",
            "--warmup_steps", "1", "--val_samples", "0", "--log_every", "1", "--save_every", "1000",
            "--lora_r", "4", "--crop_to_patches", "0"]
    out = tempfile.mkdtemp()
    r = subprocess.run(base + ["--out_dir", out, "--max_steps", "4", "--lr", "1e-3"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    r = subprocess.run(base + ["--out_dir", out, "--max_steps", "10", "--lr_schedule", "restart"], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "needs --cycle_start" in (r.stdout + r.stderr)
    r = subprocess.run(base + ["--out_dir", out, "--max_steps", "10", "--lr", "2e-3", "--lr_schedule", "restart",
                               "--cycle_start", "4", "--cycle_warmup", "2", "--cycle_from", "1e-4", "--min_lr", "5e-5"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    assert "warm restart at step 4" in r.stdout and "resumed from step 4" in r.stdout, r.stdout[-800:]
    lr = {x["step"]: x["lr"] for x in map(json.loads, open(os.path.join(out, "metrics.jsonl"))) if "lr" in x}
    assert lr[5] < lr[6] and abs(lr[6] - 2e-3) < 1e-12, lr                   # warmup reaches the new peak
    assert lr[6] >= lr[7] > lr[8] > lr[9] > lr[10] >= 5e-5, lr                  # then decays toward the floor


def test_frozen_blocks_stay_in_eval_mode():
    from pgmvlm.models import set_train_mode
    for fam, fn in FAMILIES.items():
        for vision in (False, True):
            model, _ = fn()
            model = add_lora(model, fam, r=4, alpha=8, dropout=0.05, lora_vision=vision)
            set_train_mode(model)
            blocks = {n: m for n, m in model.named_modules() if n.split(".")[-1] in ("visual", "vision_tower")}
            assert blocks, fam
            lm = [m for n, m in model.named_modules() if n.endswith("language_model")]
            assert lm and all(m.training for m in lm), fam
            # frozen vision tower -> eval; vision tower carrying LoRA -> left in train mode
            assert all(m.training == vision for m in blocks.values()), (fam, vision, {n: m.training for n, m in blocks.items()})


def test_val_list_selects_exact_problems():
    import json
    d_tr, d_va = fake_npz_dir(4, 1), fake_npz_dir(6, 2)
    lst = os.path.join(tempfile.mkdtemp(), "val.tsv")
    open(lst, "w").write("# header\np001.npz\t1\np004.npz\t4\n")
    out = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "run_tiny.py"), "--model_id", "t", "--dtype", "fp32",
                        "--num_workers", "0", "--attn", "default", "train", "--train_dir", d_tr, "--val_dir", d_va,
                        "--val_list", lst, "--out_dir", out, "--max_steps", "2", "--batch_size", "2", "--grad_accum", "1",
                        "--val_every", "2", "--val_batch_size", "2", "--lora_r", "4", "--crop_to_patches", "0"],
                       capture_output=True, text=True, env=dict(os.environ, TINY_FAMILY="qwen3vl"))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    assert "on 2 problems" in r.stdout, r.stdout[-800:]
    a = json.load(open(os.path.join(out, "args.json")))
    assert a["lr_schedule"] == "constant" and a["lora_r"] == 4 and a["weight_decay"] == 0.01 and a["lr"] == 1e-4


def test_train_fraction_flag_and_table5_and_submit_script():
    import json
    d_tr = fake_npz_dir(8, 1)
    out = tempfile.mkdtemp()
    env = dict(os.environ, TINY_FAMILY="qwen3vl")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "run_tiny.py"), "--model_id", "t", "--dtype", "fp32", "--num_workers", "0",
                        "--attn", "default", "train", "--train_dir", d_tr, "--out_dir", out, "--train_fraction", "0.5", "--epochs", "2",
                        "--batch_size", "2", "--grad_accum", "1", "--val_samples", "0", "--lora_r", "4", "--crop_to_patches", "0"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "using 4 (50.0%)" in r.stdout and "steps 4" in r.stdout, r.stdout[-700:]
    # table5 layout
    def res(acc):
        dd = tempfile.mkdtemp()
        json.dump([dict(file=f"f{i}", target=0, pred=0 if i < acc * 100 else 1, correct=i < acc * 100,
                        scores=[0.1] * 8) for i in range(100)], open(os.path.join(dd, "results_shard0.json"), "w"))
        return dd
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_table.py"), "--table5", "--row",
                        f"demo:20={res(0.4)},100={res(0.8)}"], capture_output=True, text=True)
    assert r.returncode == 0 and "| demo | 40.0" in r.stdout and "80.0" in r.stdout and "71.25" in r.stdout, r.stdout + r.stderr
    # submit script with a stub sbatch: chain structure and dependencies
    stub = tempfile.mkdtemp()
    log = os.path.join(stub, "calls.txt")
    open(os.path.join(stub, "sbatch"), "w").write(f'#!/bin/bash\necho "$@" >> {log}\nN=$(wc -l < {log})\necho $((100+N))\n')
    os.chmod(os.path.join(stub, "sbatch"), 0o755)
    r = subprocess.run(["bash", os.path.join(ROOT, "slurm", "submit_table5.sh")], capture_output=True, text=True,
                       env=dict(os.environ, PATH=stub + ":" + os.environ["PATH"], MODEL_ID="m/x", MODEL_TAG="qw", PARTS="2"))
    assert r.returncode == 0, r.stderr
    calls = open(log).read().strip().splitlines()
    assert len(calls) == 2 * (2 + 1), calls                                    # 2 fractions x (2 train parts + 1 eval)
    assert "--dependency" not in calls[0]
    # settings travel as environment variables, so only the job names / dependencies are visible in the sbatch arguments
    assert "--job-name=qw_neutral_f20_e1" in calls[0] and calls[0].endswith("train.sbatch")
    assert "--dependency=afterany:101" in calls[1] and "--dependency=afterany:102" in calls[2]
    assert "--job-name=eval_qw_neutral_f20_e1" in calls[2] and calls[2].endswith("eval_array.sbatch")
    assert "--job-name=qw_neutral_f100_e1" in calls[3] and "--dependency" not in calls[3]


def test_load_base_from_saved_checkpoint():
    """The real loading path (AutoConfig -> family, AutoModelForImageTextToText, AutoProcessor) on a saved tiny checkpoint."""
    from pgmvlm.models import detect_family, load_base
    for fam, fn in FAMILIES.items():
        model, proc = fn()
        d = tempfile.mkdtemp()
        model.save_pretrained(d); proc.save_pretrained(d)
        assert detect_family(d) == fam
        m2, p2, f2 = load_base(d, dtype="fp32", attn="sdpa", device=torch.device("cpu"))
        assert f2 == fam
        nd = fake_npz_dir()
        b = CandidateCollator(p2, f2, crop_to_patches=False)([PGMStream(nd, list_npz(nd))[0]])
        kw = to_model_inputs(b, "cpu", torch.float32)
        with torch.no_grad():
            assert torch.equal(model(**kw).logits, m2(**kw).logits)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n{len(tests)} tests passed")
