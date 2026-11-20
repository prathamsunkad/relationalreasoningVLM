"""CPU tests for Exp2 on the repo's random-weight tiny InternVL + a random stand-in SA (no downloads, ~1-2 min).
    python gap_hybrid_probes/exp2/test_exp2.py
Checks:
  1. every sequence has exactly as many <|sa|> placeholders as the connectors make tokens
  2. the hook writes the connector tokens at exactly those positions (and nowhere else)
  3. batched scores == each problem scored alone (problem-major order is kept through SA, connectors, VLM)
  4. the tokens matter: zero / shuffle_cand / shuffle_puzzle change the scores; gradients reach the connectors
     through the frozen VLM
  5. stage a trains ONLY the connectors (LoRA and base unchanged); stage b also changes LoRA
  6. a timed-out run resumes; eval writes the repo's results format, eval_vlm.py merges it; complement reports
"""
import glob
import json
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests"), HERE]

import pgmvlm.models as MU                       # noqa: E402
import tiny_models as T                          # noqa: E402
from test_pipeline import fake_npz_dir           # noqa: E402


def fake_load_base(model_id, dtype="bf16", attn="sdpa", device=None, family=None):
    model, proc = T.tiny_internvl(seed=0)
    proc.tokenizer.padding_side = "left"
    return (model.to(device) if device is not None else model), proc, "internvl"


MU.load_base = fake_load_base
import exp2_sa_interleave as E                   # noqa: E402  (after the patch)

DUMMY = os.path.join(HERE, "sa_builder_dummy.py")


def run(*argv):
    sys.argv = ["exp2_sa_interleave.py", *argv]
    E.main()


def common(adapter):
    return ["--model_id", "tiny", "--adapter", adapter, "--sa_builder", DUMMY, "--dtype", "fp32", "--crop_to_patches", "0",
            "--num_workers", "0"]


def args_ns(adapter, **kw):
    import argparse
    a = argparse.Namespace(model_id="tiny", adapter=adapter, sa_builder=DUMMY, sa_ckpt=None, dtype="fp32", attn="sdpa",
                           grad_ckpt=False, layout="interleave", rel_tokens=9, conn_hidden=32, crop_to_patches=0,
                           panel_px=160, lora_r=8, lora_alpha=16, lora_dropout=0.0)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def lora_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def main():
    torch.manual_seed(0)
    td = tempfile.mkdtemp()
    # a "fine-tuned" adapter: random LoRA on the tiny model (B matrices non-zero so it changes the model)
    base, _, fam = fake_load_base("tiny")
    pm = MU.add_lora(base, fam, r=8, alpha=16, dropout=0.0)
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.05)
    adapter = os.path.join(td, "adapter")
    pm.save_pretrained(adapter)
    d_tr, d_va = fake_npz_dir(16, seed=1), fake_npz_dir(6, seed=2)
    from pgmvlm.data import PGMStream
    ds = PGMStream(d_va, sorted(os.listdir(d_va)))

    # ---------------------------------------------------------------- 1-4 on the model directly
    m = E.Exp2Model(args_ns(adapter))
    coll = E.SACollator(m.processor, m.family, 0, 160, m.text)
    b3 = coll([ds[i] for i in range(3)])
    nt = m.conn.n_tokens()
    per_seq = (b3["input_ids"] == m.sa_id).sum(1)
    assert per_seq.tolist() == [nt] * 24, per_seq.tolist()
    print(f"1. {nt} <|sa|> placeholders in each of the 24 sequences")

    seen = {}
    h = m.inj.lm.register_forward_pre_hook(lambda mod, a, kw: seen.update(e=kw["inputs_embeds"].detach()), with_kwargs=True)
    with torch.no_grad():
        sa = m.sa(b3["sa_panels"])
        toks = m.conn(sa).reshape(-1, m.conn.cfg["d_llm"])
        m.scores(b3)
    h.remove()
    mask = b3["input_ids"] == m.sa_id
    assert torch.allclose(seen["e"][mask], toks, atol=1e-5)
    emb = m.model.get_input_embeddings()(b3["input_ids"])
    text = (~mask) & (b3["input_ids"] != m.model.config.image_token_id)
    assert torch.allclose(seen["e"][text], emb[text], atol=1e-5), "text embeddings changed outside the placeholders"
    print("2. the tokens land exactly at the placeholders; other text embeddings untouched")

    with torch.no_grad():
        s3, sa3 = m.scores(b3)
        s1 = torch.cat([m.scores(coll([ds[i]]))[0] for i in range(3)])
    assert torch.allclose(s3, s1, atol=1e-3), (s3 - s1).abs().max()
    assert sa3.shape == (3, 8)
    print("3. batched scores == per-problem scores", f"(max diff {(s3 - s1).abs().max():.2e})")

    with torch.no_grad():
        diffs = {a: float((m.scores(b3, a)[0] - s3).abs().max()) for a in ("zero", "shuffle_cand", "shuffle_puzzle")}
    assert all(v > 1e-4 for v in diffs.values()), diffs
    m.conn.zero_grad()
    s, _ = m.scores(b3)
    s.sum().backward()
    g = {n: (p.grad is not None and float(p.grad.abs().sum()) > 0) for n, p in m.conn.named_parameters()}
    assert all(g.values()), g
    assert not any(p.grad is not None for p in m.model.parameters()), "frozen VLM got gradients"
    print("4. ablations change the scores", {k: f"{v:.3g}" for k, v in diffs.items()}, "; gradients reach every connector")

    # ---------------------------------------------------------------- 5-6 through the CLI
    out_a = os.path.join(td, "run_a")
    tr = ["train", "--stage", "a", "--train_dir", d_tr, "--val_dir", d_va, "--val_samples", "6", "--max_steps", "6",
          "--batch_size", "2", "--grad_accum", "1", "--val_every", "3", "--save_every", "3", "--warmup_steps", "1",
          "--conn_hidden", "32", "--lr", "1e-2", "--log_every", "2"]
    run(*tr, "--out_dir", out_a, "--time_limit_min", "0.0001", *common(adapter))      # stops after 1 step
    st = torch.load(os.path.join(out_a, "state.pt"), weights_only=False)
    assert st["step"] == 1, st["step"]
    run(*tr, "--out_dir", out_a, *common(adapter))                                     # resumes to the end
    st = torch.load(os.path.join(out_a, "state.pt"), weights_only=False)
    assert st["step"] == 6 and os.path.exists(os.path.join(out_a, "best.pt"))
    assert not os.path.exists(os.path.join(out_a, "last_lora")), "stage a must not save a LoRA"
    ck = torch.load(os.path.join(out_a, "best.pt"), weights_only=False)
    m0 = E.Exp2Model(args_ns(adapter))
    changed = any(not torch.allclose(ck["conn"][k], v) for k, v in m0.conn.state_dict().items())
    assert changed, "connectors did not train"
    m_a = E.Exp2Model(args_ns(adapter), conn_state=ck["conn"], conn_cfg=ck["conn_cfg"])
    la, l0 = lora_state(m_a.model), lora_state(m0.model)
    assert all(torch.equal(la[k], l0[k]) for k in l0), "stage a changed the LoRA"
    vals = [json.loads(l) for l in open(os.path.join(out_a, "metrics.jsonl")) if "val_acc_none" in l]
    assert vals[0]["step"] == 0 and "val_acc_zero" in vals[0]
    print("5a. stage a: connectors trained, LoRA untouched; resume after time-out OK; step-0 validation logged")

    out_b = os.path.join(td, "run_b")
    run(*tr, "--stage", "b", "--init_from", os.path.join(out_a, "best.pt"), "--branch_dropout", "0.5", "--lora_lr", "1e-2",
        "--out_dir", out_b, *common(adapter))
    assert os.path.isdir(os.path.join(out_b, "best_lora")) and os.path.isdir(os.path.join(out_b, "last_lora"))
    m_b = E.Exp2Model(args_ns(adapter), lora_dir=os.path.join(out_b, "last_lora"))
    lb = lora_state(m_b.model)
    assert any(not torch.allclose(lb[k], l0[k]) for k in l0), "stage b did not change the LoRA"
    print("5b. stage b: LoRA continued from the adapter and changed; best_lora/ saved")

    res = os.path.join(td, "eval_b")
    for sh in (0, 1):
        run("eval", "--ckpt", os.path.join(out_b, "best.pt"), "--npz_dir", d_va, "--out_dir", res, "--shard_idx", str(sh),
            "--num_shards", "2", "--batch_size", "2", *common(adapter))
    sys.path.insert(0, os.path.join(ROOT, "lora"))
    from eval_vlm import merge
    summ = merge(res, expected_n=6)
    assert summ["complete"] and summ["n"] == 6
    r0 = json.load(open(glob.glob(os.path.join(res, "results_shard0.json"))[0]))[0]
    assert set(r0) >= {"file", "target", "pred", "correct", "scores", "sa_scores"}
    run("eval", "--ckpt", os.path.join(out_b, "best.pt"), "--npz_dir", d_va, "--out_dir", os.path.join(td, "eval_sp"),
        "--ablate", "shuffle_puzzle", "--batch_size", "5", *common(adapter))
    print("6a. eval: shards written in the repo format, merged by lora/eval_vlm.py; ablations run")

    out_c = os.path.join(td, "comp")
    run("complement", "--npz_dir", d_va, "--out_dir", out_c, "--batch_size", "2", *common(adapter))
    assert len(open(os.path.join(out_c, "records.jsonl")).readlines()) == 6
    assert os.path.exists(os.path.join(out_c, "report.md"))
    run("inspect", "--val_dir", d_va, "--n", "6", "--sa_builder", DUMMY, "--num_workers", "0")
    run("inspect", "--list_modules", "--sa_builder", DUMMY)
    print("6b. complement and inspect run")
    print("\nall Exp2 tests passed")


if __name__ == "__main__":
    main()
