"""
Pretrain the PGM slot-attention autoencoder (reconstruction only, no labels) on the
panels of your training subset, so that Slot Abstractor never sees data outside the 20%.

The released code loads slot_attention_autoencoder_16slots_dspritesdecoder_morewarmup_
lrdecay_pgm_wholetrainset_neutral_run_1_best.pth.tar, which is not in the repository and,
by its name, was trained on the whole 1.2M-question neutral training set.

Architecture = slot_abstractor_pgm.SlotAttentionAutoEncoder (16 slots, hid 32, 3 iters,
80x80, as in run_train_pgm_job.slurm). Schedule = linear warmup then exponential decay,
the formula left commented out in train_slot_abstractor_pgm.py, with that parser's
defaults (lr 4e-4, 10k warmup, x0.5 per 100k steps). The authors' exact AE settings are
not published here: treat these as our choice and report them.

Saves {'slot_model_state_dict': ...}, the format the training scripts load.

  python pretrain_slot_ae_pgm.py --jsonl_root $DATA --layout layout.json \
      --out weights/slot_ae_pgm20.pth.tar --steps 300000 --max_hours 7.5 --resume
"""
import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pgm_jsonl_data import JsonlPGM
from slot_abstractor_pgm import SlotAttentionAutoEncoder

p = argparse.ArgumentParser()
p.add_argument("--jsonl_root", required=True)
p.add_argument("--layout", required=True)
p.add_argument("--train_file", default="train.jsonl")
p.add_argument("--val_file", default="validation.jsonl")
p.add_argument("--n_train", type=int, default=0, help="first N training questions (0 = all)")
p.add_argument("--out", default="weights/slot_ae_pgm20.pth.tar")
p.add_argument("--img_size", type=int, default=80)
p.add_argument("--num_slots", type=int, default=16)
p.add_argument("--num_iterations", type=int, default=3)
p.add_argument("--hid_dim", type=int, default=32)
p.add_argument("--puzzles_per_batch", type=int, default=4, help="x16 panels per step")
p.add_argument("--lr", type=float, default=4e-4)
p.add_argument("--warmup_steps", type=int, default=10000)
p.add_argument("--decay_rate", type=float, default=0.5)
p.add_argument("--decay_steps", type=int, default=100000)
p.add_argument("--steps", type=int, default=300000)
p.add_argument("--eval_every", type=int, default=5000)
p.add_argument("--n_val", type=int, default=200, help="validation puzzles for reconstruction MSE")
p.add_argument("--num_workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
p.add_argument("--resume", action="store_true")
p.add_argument("--max_hours", type=float, default=0)
p.add_argument("--seed", type=int, default=0)
opt = p.parse_args()

torch.manual_seed(opt.seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    torch.cuda.set_device(0)
os.makedirs(os.path.dirname(os.path.abspath(opt.out)), exist_ok=True)
last_path = opt.out.replace(".pth.tar", "") + "_last.pth.tar"
log_path = opt.out.replace(".pth.tar", "") + "_log.csv"

train_ds = JsonlPGM(opt.jsonl_root, opt.train_file, opt.layout, opt.img_size, n=opt.n_train)
val_ds = JsonlPGM(opt.jsonl_root, opt.val_file, opt.layout, opt.img_size, n=opt.n_val)
print(f"pretraining on panels of {len(train_ds)} training questions; device={device}", flush=True)

model = SlotAttentionAutoEncoder((opt.img_size, opt.img_size), opt.num_slots, opt.num_iterations,
                                 opt.hid_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
step, best = 0, float("inf")
if opt.resume and os.path.exists(last_path):
    ck = torch.load(last_path, map_location="cpu")
    model.load_state_dict(ck["slot_model_state_dict"])
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    step, best = ck["step"], ck["best_val_mse"]
    print(f"resumed at step {step}, best val MSE {best:.5f}", flush=True)


def lr_at(i):
    lr = opt.lr * (i / opt.warmup_steps) if i < opt.warmup_steps else opt.lr
    return lr * opt.decay_rate ** (i / opt.decay_steps)


@torch.no_grad()
def val_mse():
    model.eval()
    tot, n = 0.0, 0
    for img, _ in DataLoader(val_ds, batch_size=opt.puzzles_per_batch, num_workers=opt.num_workers):
        x = img.flatten(0, 1).to(device)
        recon = model(x, device)[0]
        tot += F.mse_loss(recon, x, reduction="sum").item() / x[0].numel()
        n += x.shape[0]
    model.train()
    return tot / n


@torch.no_grad()
def save_recon(path):
    """Top row: 8 validation panels; bottom row: their reconstructions."""
    import numpy as np
    from PIL import Image
    model.eval()
    x = val_ds[0][0][:8].to(device)                       # 8 panels of the first validation puzzle
    r = model(x, device)[0]
    model.train()
    to8 = lambda t: ((t[:, 0].float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    top, bot = np.concatenate(list(to8(x)), 1), np.concatenate(list(to8(r)), 1)
    Image.fromarray(np.concatenate([top, np.full((4, top.shape[1]), 128, np.uint8), bot])).save(path)


def save(path, extra=False):
    d = {"slot_model_state_dict": model.state_dict(), "step": step, "best_val_mse": best,
         "args": vars(opt)}
    if extra:
        d["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(d, path)


t_job, t_last, done = time.time(), time.time(), False
model.train()
while not done:
    loader = DataLoader(train_ds, batch_size=opt.puzzles_per_batch, shuffle=True, drop_last=True,
                        num_workers=opt.num_workers, pin_memory=use_cuda)
    for img, _ in loader:
        x = img.flatten(0, 1).to(device)                    # (puzzles*16, 1, H, W)
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        recon = model(x, device)[0]
        loss = F.mse_loss(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.5f} lr {lr_at(step):.2e}", flush=True)
        if step % opt.eval_every == 0 or step >= opt.steps:
            v = val_mse()
            if v < best:
                best = v
                save(opt.out)
            save(last_path, extra=True)
            new = not os.path.exists(log_path)
            with open(log_path, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["step", "train_mse", "val_mse", "best_val_mse", "sec"])
                w.writerow([step, f"{loss.item():.5f}", f"{v:.5f}", f"{best:.5f}", f"{time.time() - t_last:.0f}"])
            save_recon(opt.out.replace(".pth.tar", "") + "_recon.png")
            print(f"step {step}: val MSE {v:.5f} (best {best:.5f})", flush=True)
            t_last = time.time()
            elapsed_h = (time.time() - t_job) / 3600
            if step >= opt.steps:
                done = True
                break
            if opt.max_hours and elapsed_h * (1 + opt.eval_every / max(step, 1)) > opt.max_hours:
                print("stopping before walltime; rerun with --resume", flush=True)
                done = True
                break
print(f"finished at step {step}; best checkpoint {opt.out} (val MSE {best:.5f})", flush=True)
