"""CPU tests on random-weight tiny Qwen3-VL / InternVL models (no downloads).  python tests/test_pipeline.py
What they check:
  1. label bookkeeping: the supervised tokens are exactly "Answer: k<|im_end|>", digit ids are single tokens
  2. left-padding equivalence: padded batch logits == unpadded logits (so batching cannot change answers)
  3. loss really is computed only over the response (prompt tokens do not change the loss)
  4. a handful of LoRA steps on 6 problems drives loss down and accuracy up (the LoRA actually trains)
  5. only LoRA params receive gradients; base weights are unchanged
  6. resume gives the same trajectory as an uninterrupted run (data order is a function of position)
  7. restricted readout + free generation both run; adapter reload reproduces logits
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
from model_utils import (AnswerTokens, add_lora, append_tokens, lm_and_digit_losses, restricted_logits,
                         teacher_forced_logits, to_model_inputs, generate_text)
from pgm_vlm_data import PGMStream, VLMCollator, list_npz, extract_answer, make_order, to_panels

FAMILIES = {"qwen3vl": T.tiny_qwen3vl, "internvl": T.tiny_internvl}


def fake_npz_dir(n=6, seed=0, const_target=None):
    rng = np.random.default_rng(seed)
    d = tempfile.mkdtemp()
    for i in range(n):
        img = rng.integers(0, 255, (16, 160, 160), dtype=np.uint8)
        # make the answer recoverable: the correct candidate is a copy of context panel 2
        t = i % 8 if const_target is None else const_target
        img[8 + t] = img[2]
        np.savez(os.path.join(d, f"p{i:03d}.npz"), image=img.reshape(160, 160, 16), target=np.int64(t))   # real file layout
    return d


def batch_for(fam, proc, d, idx=(0, 1), shuffle=False):
    ds = PGMStream(d, list_npz(d), shuffle_candidates=shuffle)
    return VLMCollator(proc, fam, crop_to_patches=False)([ds[i] for i in idx])


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


def test_answer_tokens():
    for fam, fn in FAMILIES.items():
        _, proc = fn()
        a = AnswerTokens(proc.tokenizer)
        tok = proc.tokenizer
        assert a.R == 5 and a.digit_pos == 3, (a.R, a.digit_pos)
        assert tok.decode(a.full[2].tolist()).startswith("Answer: 3"), tok.decode(a.full[2].tolist())
        assert a.full[0, -1].item() == tok.convert_tokens_to_ids("<|im_end|>")
        assert [tok.decode([i]) for i in a.digit_ids] == list("12345678")


def test_collator_prompt_ends_with_generation_header():
    for fam, fn in FAMILIES.items():
        _, proc = fn()
        d = fake_npz_dir()
        b = batch_for(fam, proc, d)
        text = proc.tokenizer.decode(b["input_ids"][0].tolist())
        assert text.rstrip().endswith("<|im_start|>assistant"), text[-60:]
        assert "Answer: <number>" in text


def test_preflight_runs():
    d = fake_npz_dir()
    for fam in FAMILIES:
        env = dict(os.environ, TINY_FAMILY=fam)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "run_tiny.py"), "--model_id", "tiny_" + fam, "--dtype", "fp32",
                            "--attn", "default", "preflight", "--npz_dir", d, "--batch_size", "2", "--steps", "2", "--lora_r", "4",
                            "--crop_to_patches", "0", "--lora_vision", "--lora_projector"], capture_output=True, text=True, env=env)
        assert r.returncode == 0 and "preflight OK" in r.stdout, r.stdout[-1500:] + r.stderr[-2500:]
        assert "image tokens>" in r.stdout and "answer tokens" in r.stdout


def test_internvl_crop_flag_is_honoured():
    """The InternVL processor defaults crop_to_patches=True and overrides the checkpoint's preprocessor_config (False);
    VLMCollator must therefore pass the flag explicitly. Qwen ignores it."""
    _, proc = T.tiny_internvl()
    d = fake_npz_dir()
    ds = PGMStream(d, list_npz(d))
    on = VLMCollator(proc, "internvl", crop_to_patches=True)([ds[0]])
    off = VLMCollator(proc, "internvl", crop_to_patches=False)([ds[0]])
    assert off["pixel_values"].shape[0] == 2 and on["pixel_values"].shape[0] > 2, (on["pixel_values"].shape, off["pixel_values"].shape)
    n_img_tokens = lambda b: int((b["input_ids"] == proc.image_token_id).sum())
    assert n_img_tokens(on) == on["pixel_values"].shape[0] * 4 and n_img_tokens(off) == 2 * 4


def test_left_padding_does_not_change_logits():
    for fam, fn in FAMILIES.items():
        model, proc = fn()
        d = fake_npz_dir()
        a = AnswerTokens(proc.tokenizer)
        b = batch_for(fam, proc, d, idx=(0,))
        inp = to_model_inputs(b, "cpu", torch.float32)
        with torch.no_grad():
            ref = model(**append_tokens(inp, a.prefix_t[None]), logits_to_keep=1).logits[0, -1]
        pad = 7
        pid = proc.tokenizer.pad_token_id
        padded = dict(inp)
        padded["input_ids"] = torch.cat([torch.full((1, pad), pid), inp["input_ids"]], 1)
        padded["attention_mask"] = torch.cat([torch.zeros(1, pad, dtype=torch.long), inp["attention_mask"]], 1)
        if "mm_token_type_ids" in inp:
            padded["mm_token_type_ids"] = torch.cat([torch.zeros(1, pad, dtype=torch.long), inp["mm_token_type_ids"]], 1)
        with torch.no_grad():
            got = model(**append_tokens(padded, a.prefix_t[None]), logits_to_keep=1).logits[0, -1]
        err = (ref - got).abs().max().item()
        assert err < 1e-3, f"{fam}: left padding changed logits by {err}"


def test_loss_only_on_response():
    for fam, fn in FAMILIES.items():
        model, proc = fn()
        d = fake_npz_dir()
        a = AnswerTokens(proc.tokenizer)
        b = batch_for(fam, proc, d)
        with torch.no_grad():
            logits, resp, targets = teacher_forced_logits(model, b, a, torch.device("cpu"), torch.float32)
            assert logits.shape == (2, a.R, model.config.text_config.vocab_size), logits.shape
            lm, d8, acc = lm_and_digit_losses(logits, resp, targets, a)
            # reference: full-sequence logits, shift manually, average over the last R target tokens
            inp = append_tokens(to_model_inputs(b, "cpu", torch.float32), resp)
            full = model(**inp).logits.float()
            ref = F.cross_entropy(full[:, -a.R - 1:-1].reshape(-1, full.shape[-1]), inp["input_ids"][:, -a.R:].reshape(-1))
        assert abs(lm.item() - ref.item()) < 1e-4, (lm.item(), ref.item())


def _train_steps(fam, fn, steps, d, n_problems=6):
    model, proc = fn(seed=0)
    with torch.no_grad():       # random tiny embeddings are ~0.02-scale, which caps the reachable logit range (loss floor
        model.get_input_embeddings().weight.mul_(8)       # ~4.7 for 420 tokens); scale them so the loss can actually fall
    a = AnswerTokens(proc.tokenizer)
    model = add_lora(model, fam, r=8, alpha=16, dropout=0.0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-3)
    ds = PGMStream(d, list_npz(d), shuffle_candidates=False)
    batch = VLMCollator(proc, fam, crop_to_patches=False)([ds[i] for i in range(n_problems)])
    model.train()
    hist = []
    for s in range(steps):
        logits, resp, targets = teacher_forced_logits(model, batch, a, torch.device("cpu"), torch.float32)
        lm, d8, acc = lm_and_digit_losses(logits, resp, targets, a)
        lm.backward()
        opt.step(); opt.zero_grad()
        hist.append((lm.item(), d8.item(), acc.item()))
    return model, proc, a, batch, hist


def test_readout_consistency():
    """Teacher-forced digit logits (training) and the restricted readout (evaluation) must be the same numbers."""
    for fam, fn in FAMILIES.items():
        model, proc = fn()
        d = fake_npz_dir()
        a = AnswerTokens(proc.tokenizer)
        b = batch_for(fam, proc, d, idx=(0, 1, 2))
        with torch.no_grad():
            logits, resp, targets = teacher_forced_logits(model, b, a, torch.device("cpu"), torch.float32)
            tf = logits[:, a.digit_pos][:, a.digit_ids_t]
            ro = restricted_logits(model, b, a, torch.device("cpu"), torch.float32)
        assert torch.allclose(tf, ro, atol=1e-4), (fam, (tf - ro).abs().max().item())


def test_lora_trains_and_base_is_frozen():
    """Constant-answer toy (every problem's answer is '5'): LoRA must learn the format and the digit, through the
    exact same code path as real training; then the restricted readout, free generation and an adapter reload agree."""
    d = fake_npz_dir(const_target=4)
    for fam, fn in FAMILIES.items():
        base_model, _ = fn(seed=0)
        with torch.no_grad():
            base_model.get_input_embeddings().weight.mul_(8)
        before = {k: v.clone() for k, v in base_model.state_dict().items()}
        model, proc, a, batch, hist = _train_steps(fam, fn, 60, d)
        assert hist[-1][0] < 0.2 * hist[0][0], f"{fam}: lm loss {hist[0][0]:.3f} -> {hist[-1][0]:.3f}"
        assert hist[-1][2] == 1.0, f"{fam}: digit accuracy {hist[0][2]:.2f} -> {hist[-1][2]:.2f}"
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trainable and all("lora_" in n for n in trainable)
        after = {k.replace("base_model.model.", "").replace(".base_layer", ""): v for k, v in model.state_dict().items() if "lora_" not in k}
        for k, v in before.items():
            assert torch.equal(v, after[k]), f"{fam}: base weight {k} changed"
        model.eval()
        lg = restricted_logits(model, batch, a, torch.device("cpu"), torch.float32)
        assert (lg.argmax(-1) == 4).all(), (fam, lg.argmax(-1))
        texts = generate_text(model, proc, batch, torch.device("cpu"), torch.float32, max_new_tokens=8, stop_ids=[a.eos_id])
        assert [extract_answer(t) for t in texts] == [4] * len(texts), (fam, texts)
        from model_utils import load_adapter
        td = tempfile.mkdtemp()
        model.save_pretrained(td)
        fresh, _ = fn(seed=0)
        with torch.no_grad():
            fresh.get_input_embeddings().weight.mul_(8)
        re = load_adapter(fresh, td).eval()
        lg2 = restricted_logits(re, batch, a, torch.device("cpu"), torch.float32)
        assert torch.allclose(lg, lg2, atol=1e-4), (lg - lg2).abs().max()
        print(f"   {fam}: lm {hist[0][0]:.3f} -> {hist[-1][0]:.3f}, digit acc {hist[0][2]:.2f} -> {hist[-1][2]:.2f}; generated {texts[0]!r}")


def test_data_order_is_position_deterministic():
    d = fake_npz_dir()
    files = list_npz(d)
    order = make_order(len(files), 20, seed=3)
    s1 = PGMStream(d, files, order=order, seed=3, shuffle_candidates=True)
    s2 = PGMStream(d, files, order=order, seed=3, shuffle_candidates=True)
    for pos in (0, 7, 19):
        a, b = s1[pos], s2[pos]
        assert a["file"] == b["file"] and a["target"] == b["target"] and np.array_equal(a["candidates"], b["candidates"])
    # shuffled candidates: the answer panel is the same image wherever it moved
    plain = PGMStream(d, files, order=order, seed=3, shuffle_candidates=False)
    for pos in range(10):
        x, y = plain[pos], s1[pos]
        assert x["file"] == y["file"] and np.array_equal(x["candidates"][x["target"]], y["candidates"][y["target"]])
    assert len(set(order[:6].tolist())) == 6                                   # first epoch is a permutation


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
        assert s["complete"] and s["duplicates"] == 0 and "logits" in s and "generate" in s
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
        json.dump([dict(file=f"f{i}", target=0, pred_logits=0 if i < acc * 100 else 1, correct_logits=i < acc * 100,
                        probs=[0.1] * 8) for i in range(100)], open(os.path.join(dd, "results_shard0.json"), "w"))
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
    from model_utils import detect_family, load_base
    for fam, fn in FAMILIES.items():
        model, proc = fn()
        d = tempfile.mkdtemp()
        model.save_pretrained(d); proc.save_pretrained(d)
        assert detect_family(d) == fam
        m2, p2, f2 = load_base(d, dtype="fp32", attn="sdpa", device=torch.device("cpu"))
        assert f2 == fam
        nd = fake_npz_dir()
        b = VLMCollator(p2, f2, crop_to_patches=False)([PGMStream(nd, list_npz(nd))[0]])
        kw = to_model_inputs(b, "cpu", torch.float32)
        with torch.no_grad():
            assert torch.equal(model(**kw).logits, m2(**kw).logits)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n{len(tests)} tests passed")
