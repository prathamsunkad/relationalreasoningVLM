"""CPU test of exp2/sa_builder.py against the ORIGINAL Slot Abstractor scoring (slot_abstractor/run_pgm_jsonl.py).
Random weights in the real checkpoint formats, a 2-layer Abstractor so it runs on a CPU (~1 min):
    python gap_hybrid_probes/exp2/test_sa_builder.py
Checks:
  1. the builder loads checkpoints saved the way run_pgm_jsonl.py / pretrain_slot_ae_pgm.py save them
  2. its per-candidate scores equal the original 16-panel scoring of the same puzzles (torchvision transform,
     slot AE per panel, scoring_model over 8 candidates) -> same SA answer
  3. puzzle-grouped and row-by-row encoding agree; shapes are (N, 9, 16, 32)
  4. one Exp2 training step and an eval run with the real SA on the tiny InternVL
The slot initialisation noise is switched off (sigma = 0) so the two code paths can be compared exactly.
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

torch.Tensor.cuda = lambda self, *a, **k: self          # the SA code builds its position grid with .cuda(); CPU test only

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SA_DIR = os.path.join(ROOT, "slot_abstractor")
sys.path[:0] = [SA_DIR, HERE]

from slot_abstractor_pgm import SlotAttentionAutoEncoder, scoring_model   # noqa: E402
from torchvision.transforms import transforms                              # noqa: E402
from PIL import Image                                                      # noqa: E402


def main():
    td = tempfile.mkdtemp()
    torch.manual_seed(0)
    args = dict(img_size=80, depth=2, heads=8, mlp_dim=64, num_slots=16, num_iterations=3, hid_dim=32,
                apply_context_norm=True, slot_checkpoint="weights/slot_ae_pgm20.pth.tar")
    slot = SlotAttentionAutoEncoder((80, 80), 16, 3, 32)
    slot.slot_attention.slots_sigma.data.zero_()                           # deterministic slots
    absr = scoring_model(SimpleNamespace(**args), 32, 2, 8, 64, 16)
    torch.save({"slot_model_state_dict": slot.state_dict()}, os.path.join(td, "slot_ae_pgm20.pth.tar"))
    ck = os.path.join(td, "slot_abstractor_pgm_test_d2_lr8e-05_run0_best.pth.tar")
    torch.save({"abstractor_model_state_dict": {"module." + k: v for k, v in absr.state_dict().items()},
                "epoch": 0, "args": args}, ck)                             # DDP 'module.' keys, as run_pgm_jsonl.py saves
    os.environ.update(SA_CKPT=ck, SA_CODE_DIR=SA_DIR)
    import sa_builder as B
    from sa_interface import FrozenSA
    sa = FrozenSA(os.path.join(HERE, "sa_builder.py"), ck, torch.device("cpu"))
    print("1. builder loaded the checkpoints (slot AE found next to the SA checkpoint)")

    # original path: 16 panels per puzzle through the torchvision transform of pgm_jsonl_data.py
    rng = np.random.default_rng(0)
    P = 3
    puzzles = rng.integers(0, 256, (P, 16, 160, 160), dtype=np.uint8)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda X: 2 * X - 1.0), transforms.Resize((80, 80))])
    img = torch.stack([torch.stack([tf(Image.fromarray(p, mode="L")) for p in pz]) for pz in puzzles])  # (P,16,1,80,80)
    slot.eval()
    absr.eval()
    with torch.no_grad():
        feats, poss = [], []
        for i in range(16):
            _, _, _, f, p, _ = slot(img[:, i], torch.device("cpu"))
            feats.append(f)
            poss.append(p)
        f, ps = torch.stack(feats, 1), torch.stack(poss, 1)
        orig = absr(f[:, :8], ps[:, :8], f[:, 8:], ps[:, 8:], torch.device("cpu"))                    # (P, 8)
    rows = torch.from_numpy(np.stack([np.concatenate([pz[:8], pz[8 + k:9 + k]]) for pz in puzzles for k in range(8)]))
    out = sa(rows)
    mine = out["score"].view(P, 8)
    err = float((mine - orig).abs().max())
    assert err < 1e-4, err
    assert torch.equal(mine.argmax(1), orig.argmax(1))
    print(f"2. builder scores == original scoring (max diff {err:.1e}); same answers {orig.argmax(1).tolist()}")

    single = torch.cat([sa(rows[i:i + 1])["score"] for i in range(len(rows))])
    assert float((single - out["score"]).abs().max()) < 1e-4
    assert tuple(out["slots"].shape) == (P * 8, 9, 16, 32) and tuple(out["rel"].shape) == (P * 8, 9, 16, 32)
    print("3. grouped == row-by-row; slots and rel are (N, 9, 16, 32)")

    # 4. through Exp2 on the tiny InternVL
    import test_exp2 as TE                    # patches the repo's load_base with the tiny model
    from test_pipeline import fake_npz_dir
    base, _, fam = TE.fake_load_base("tiny")
    pm = TE.MU.add_lora(base, fam, r=8, alpha=16, dropout=0.0)
    adapter = os.path.join(td, "adapter")
    pm.save_pretrained(adapter)
    d_tr, d_va = fake_npz_dir(8, seed=1), fake_npz_dir(4, seed=2)
    common = ["--model_id", "tiny", "--adapter", adapter, "--sa_builder", os.path.join(HERE, "sa_builder.py"),
              "--sa_ckpt", ck, "--dtype", "fp32", "--crop_to_patches", "0", "--num_workers", "0"]
    out_a = os.path.join(td, "run_a")
    TE.run("train", "--stage", "a", "--train_dir", d_tr, "--val_dir", d_va, "--val_samples", "4", "--max_steps", "2",
           "--batch_size", "2", "--grad_accum", "1", "--val_every", "2", "--warmup_steps", "1", "--conn_hidden", "32",
           "--out_dir", out_a, *common)
    TE.run("eval", "--ckpt", os.path.join(out_a, "best.pt"), "--npz_dir", d_va, "--out_dir", os.path.join(td, "ev"),
           "--batch_size", "2", *common)
    TE.run("complement", "--npz_dir", d_va, "--out_dir", os.path.join(td, "comp"), "--batch_size", "2", *common)
    print("4. Exp2 train / eval / complement run with the real Slot Abstractor code")
    print("\nall SA builder tests passed")


if __name__ == "__main__":
    main()
