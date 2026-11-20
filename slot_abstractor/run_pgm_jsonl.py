"""
Train / test Slot Abstractor on PGM from the shared JSONL + contact-sheet package.

Same model, loss, optimiser and data transform as train_slot_abstractor_pgm.py /
test_slot_abstractor_pgm.py (imports slot_abstractor_pgm.py unchanged; frozen
pretrained slot encoder; abstractor trained with Adam at a constant LR and
cross-entropy). Bookkeeping differences, all of which leave the model's computation unchanged:
  * reads train.jsonl / validation.jsonl / test.jsonl through pgm_jsonl_data.JsonlPGM;
  * validation/test accuracy is exact across GPUs (no sampler padding, all-reduced
    counts) and only rank 0 saves checkpoints;
  * the frozen slot encoder runs under torch.no_grad() (same outputs, less memory);
  * per-epoch CSV log, `last` checkpoint, --resume and --max_hours for walltime limits;
  * test mode writes per-question predictions.
Works under srun (uses SLURM_* variables) or as a single process.

  srun python run_pgm_jsonl.py --mode train --jsonl_root $DATA --layout layout.json \
       --slot_checkpoint weights/slot_ae_pgm20.pth.tar --tag pgm20 --max_hours 7.5 --resume
  srun python run_pgm_jsonl.py --mode test  --jsonl_root $DATA --layout layout.json \
       --slot_checkpoint weights/slot_ae_pgm20.pth.tar --model_checkpoint weights/<..>_best.pth.tar
"""
import argparse
import csv
import json
import math
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from pgm_jsonl_data import JsonlPGM
from slot_abstractor_pgm import SlotAttentionAutoEncoder, scoring_model

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["train", "test"], required=True)
p.add_argument("--jsonl_root", required=True, help="extracted pgm-neutral-...-v1 folder")
p.add_argument("--layout", required=True)
p.add_argument("--train_file", default="train.jsonl")
p.add_argument("--val_file", default="validation.jsonl")
p.add_argument("--test_file", default="test.jsonl")
p.add_argument("--n_train", type=int, default=0, help="first N training rows (0 = all 240k)")
p.add_argument("--n_val", type=int, default=0)
p.add_argument("--n_test", type=int, default=0)
p.add_argument("--test_offset", type=int, default=0, help="test mode: skip the first K test rows (evaluate in parts)")
p.add_argument("--skip_val", action="store_true", help="test mode: evaluate only the test split")
p.add_argument("--slot_checkpoint", required=True, help="pretrained slot autoencoder (.pth.tar)")
p.add_argument("--model_checkpoint", default=None, help="test mode: trained abstractor checkpoint")
p.add_argument("--save_path", default="weights/")
p.add_argument("--tag", default="pgm20")
p.add_argument("--resume", action="store_true", help="continue from <save_path>/<name>_last.pth.tar")
p.add_argument("--max_hours", type=float, default=0, help="stop cleanly before this walltime (0 = off)")
p.add_argument("--margin_min", type=float, default=5, help="minutes kept free before --max_hours for saving")
p.add_argument("--snapshot_every", type=int, default=0, help="also keep <name>_ep<E>.pth.tar every E epochs (0 = off)")
# model / optimisation: defaults = run_train_pgm_job.slurm
p.add_argument("--img_size", type=int, default=80)
p.add_argument("--batch_size", type=int, default=24, help="per GPU")
p.add_argument("--depth", type=int, default=24)
p.add_argument("--heads", type=int, default=8)
p.add_argument("--mlp_dim", type=int, default=512)
p.add_argument("--num_slots", type=int, default=16)
p.add_argument("--num_iterations", type=int, default=3)
p.add_argument("--hid_dim", type=int, default=32)
p.add_argument("--learning_rate", type=float, default=0.00008)
p.add_argument("--num_epochs", type=int, default=100)
p.add_argument("--apply_context_norm", type=bool, default=True)
p.add_argument("--log_interval", type=int, default=100)
p.add_argument("--num_workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--image_key", default="images")
p.add_argument("--answer_key", default="answer")
p.add_argument("--tf32", action="store_true", help="TF32 matmuls on H100/A100 (faster, tiny numeric change)")
opt = p.parse_args()

# ---------------------------------------------------------------- distributed setup
env = os.environ
if all(k in env for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):          # torchrun
    rank, local_rank, world_size = int(env["RANK"]), int(env["LOCAL_RANK"]), int(env["WORLD_SIZE"])
elif "SLURM_STEP_ID" in env and int(env.get("SLURM_STEP_NUM_TASKS", env.get("SLURM_NTASKS", 1))) > 1:
    rank, local_rank = int(env["SLURM_PROCID"]), int(env.get("SLURM_LOCALID", 0))   # srun, several tasks
    world_size = int(env.get("SLURM_STEP_NUM_TASKS", env["SLURM_NTASKS"]))
else:                                                                     # plain `python` (incl. inside salloc)
    rank, local_rank, world_size = 0, 0, 1
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", str(29500 + int(os.environ.get("SLURM_JOBID", "0")[-3:] or 0)))
use_cuda = torch.cuda.is_available()
if opt.tf32:                                     # PyTorch default: TF32 on for convs, off for matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
if use_cuda:
    if torch.cuda.device_count() == 1:      # cluster binds one GPU per task
        local_rank = 0
    torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")
dist.init_process_group("nccl" if use_cuda else "gloo", rank=rank, world_size=world_size)
torch.manual_seed(opt.seed)
is_main = rank == 0


def log(msg):
    if is_main:
        print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


log(f"world_size={world_size} device={device} args={vars(opt)}")


def make(split_file, n, return_index=False, offset=0):
    return JsonlPGM(opt.jsonl_root, split_file, opt.layout, opt.img_size, n=n, offset=offset,
                    image_key=opt.image_key, answer_key=opt.answer_key, return_index=return_index)


def shard_loader(ds):
    """Exact sharding for evaluation: every example seen once, no padding."""
    return DataLoader(Subset(ds, list(range(rank, len(ds), world_size))), batch_size=opt.batch_size,
                      shuffle=False, num_workers=opt.num_workers, pin_memory=use_cuda)


# ---------------------------------------------------------------- models
slot_model = SlotAttentionAutoEncoder((opt.img_size, opt.img_size), opt.num_slots, opt.num_iterations, opt.hid_dim)
ckp = torch.load(opt.slot_checkpoint, map_location="cpu")
state = ckp.get("slot_model_state_dict", ckp)
slot_model.load_state_dict(OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items()))
slot_model = slot_model.to(device).eval()
for prm in slot_model.parameters():
    prm.requires_grad = False

abstractor = scoring_model(opt, opt.hid_dim, opt.depth, opt.heads, opt.mlp_dim, opt.num_slots)
ddp = DDP(abstractor.to(device), device_ids=[local_rank] if use_cuda else None)
dev_arg = local_rank if use_cuda else device


def score(img):
    """Exactly the original per-panel slot extraction + scoring."""
    feats, poss = [], []
    with torch.no_grad():
        for idx in range(img.shape[1]):
            _, _, _, feat_slots, pos_slots, _ = slot_model(img[:, idx], dev_arg)
            feats.append(feat_slots)
            poss.append(pos_slots)
    f, ps = torch.stack(feats, 1), torch.stack(poss, 1)
    return ddp(f[:, :8], ps[:, :8], f[:, 8:], ps[:, 8:], dev_arg)


def evaluate(ds, keep_preds=False):
    ddp.eval()
    correct = torch.zeros(2, device=device)
    preds = []
    with torch.no_grad():
        for batch in shard_loader(ds):
            img, target = batch[0].to(device), batch[1].to(device)
            pred = score(img).argmax(1)
            correct[0] += (pred == target).sum()
            correct[1] += target.numel()
            if keep_preds:
                preds += list(zip(batch[2].tolist(), pred.tolist(), target.tolist()))
    dist.all_reduce(correct)
    return (correct[0] / correct[1]).item() * 100, int(correct[1].item()), preds


name = f"slot_abstractor_pgm_{opt.tag}_d{opt.depth}_lr{opt.learning_rate:g}_run{opt.seed}"
os.makedirs(opt.save_path, exist_ok=True)

# ---------------------------------------------------------------- test
if opt.mode == "test":
    ck = torch.load(opt.model_checkpoint, map_location="cpu")
    ddp.load_state_dict(ck["abstractor_model_state_dict"])
    out = {}
    part = f".off{opt.test_offset}" if (opt.test_offset or opt.n_test) else ""
    splits = [("test", opt.test_file, opt.n_test, opt.test_offset)]
    if not opt.skip_val:
        splits.insert(0, ("validation", opt.val_file, opt.n_val, 0))
    for split, f, n, off in splits:
        ds = make(f, n, return_index=True, offset=off)
        acc, total, preds = evaluate(ds, keep_preds=True)
        gathered = [None] * world_size
        dist.all_gather_object(gathered, preds)
        if is_main:
            rows = sorted(r for g in gathered for r in g)
            pth = os.path.join(opt.save_path, f"{os.path.basename(opt.model_checkpoint)}.seed{opt.seed}{part if split == 'test' else ''}.{split}_preds.csv")
            with open(pth, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["id", "pred", "target", "correct"])
                for i, pr, t in rows:
                    w.writerow([ds.ids[i], "ABCDEFGH"[pr], "ABCDEFGH"[t], int(pr == t)])
        out[split] = {"accuracy": acc, "n": total, "correct": int(round(acc * total / 100)), "offset": off}
        log(f"{split}: accuracy {acc:.2f}% on {total} questions")
    if is_main:
        with open(os.path.join(opt.save_path, f"{os.path.basename(opt.model_checkpoint)}.seed{opt.seed}{part}.eval.json"), "w") as fh:
            json.dump(dict(out, checkpoint=opt.model_checkpoint, slot_checkpoint=opt.slot_checkpoint), fh, indent=1)
    dist.destroy_process_group()
    raise SystemExit(0)

# ---------------------------------------------------------------- train
class ListSampler(torch.utils.data.Sampler):
    """Fixed list of indices (lets an epoch resume from a given batch)."""

    def __init__(self, idx):
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


train_ds, val_ds = make(opt.train_file, opt.n_train), make(opt.val_file, opt.n_val)
sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=opt.seed)
n_batches = math.ceil(len(sampler) / opt.batch_size)
log(f"train {len(train_ds)} questions ({n_batches} batches/GPU, global batch {opt.batch_size * world_size}), "
    f"validation {len(val_ds)}")

optimizer = torch.optim.Adam(ddp.parameters(), lr=opt.learning_rate)
ce = nn.CrossEntropyLoss()
start_epoch, start_batch, max_val_acc = 0, 0, 0.0
carried_stats, carried_sec = None, 0.0
last_path = os.path.join(opt.save_path, name + "_last.pth.tar")
best_path = os.path.join(opt.save_path, name + "_best.pth.tar")
csv_path = os.path.join(opt.save_path, name + "_log.csv")
if opt.resume and os.path.exists(last_path):
    ck = torch.load(last_path, map_location="cpu")
    ddp.load_state_dict(ck["abstractor_model_state_dict"])
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    max_val_acc = ck.get("max_val_acc", 0.0)
    if ck.get("next_batch") is not None:                      # stopped in the middle of an epoch
        same_layout = (ck.get("world_size"), ck.get("batch_size")) == (world_size, opt.batch_size)
        start_epoch = ck["epoch"]
        if same_layout:
            start_batch, carried_sec = ck["next_batch"], ck.get("epoch_sec", 0.0)
            carried_stats = torch.tensor(ck["stats"], device=device) if is_main else None
        else:
            log("GPU count or batch size changed since the mid-epoch checkpoint: restarting that epoch")
    else:
        start_epoch = ck["epoch"] + 1
    log(f"resumed from {last_path}: epoch {start_epoch}, batch {start_batch}, best val {max_val_acc:.2f}")


def save(path, epoch, next_batch=None, stats=None, epoch_sec=0.0):
    torch.save({"abstractor_model_state_dict": ddp.state_dict(),   # 'module.' keys, as the original
                "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                "next_batch": next_batch, "stats": stats, "epoch_sec": epoch_sec,
                "world_size": world_size, "batch_size": opt.batch_size,
                "max_val_acc": max_val_acc, "args": vars(opt)}, path)


def out_of_time():
    """All ranks agree on whether the walltime budget (minus a safety margin) is used up."""
    if not opt.max_hours:
        return False
    flag = torch.tensor(float(time.time() - t_job > opt.max_hours * 3600 - opt.margin_min * 60), device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


t_job, stopped = time.time(), False
for epoch in range(start_epoch, opt.num_epochs):
    t0 = time.time() - carried_sec
    sampler.set_epoch(epoch)
    idx = list(iter(sampler))[start_batch * opt.batch_size:]
    loader = DataLoader(train_ds, batch_size=opt.batch_size, sampler=ListSampler(idx),
                        num_workers=opt.num_workers, pin_memory=use_cuda)
    ddp.train()
    stats = torch.zeros(3, device=device)                  # correct, total, loss sum
    if carried_stats is not None:
        stats += carried_stats
    for k, (img, target) in enumerate(loader):
        b = start_batch + k
        img, target = img.to(device), target.to(device)
        scores = score(img)
        loss = ce(scores, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stats += torch.stack([(scores.argmax(1) == target).sum().float(),
                              torch.tensor(float(target.numel()), device=device),
                              loss.detach() * target.numel()])
        if b % opt.log_interval == 0:
            log(f"epoch {epoch} batch {b}/{n_batches} loss {loss.item():.4f} "
                f"[{(time.time() - t_job) / 60:.1f} min in this job]")
        if (k + 1) % 20 == 0 and b + 1 < n_batches and out_of_time():
            part = stats.clone()
            dist.all_reduce(part)
            if is_main:
                save(last_path, epoch, next_batch=b + 1, stats=part.tolist(), epoch_sec=time.time() - t0)
            dist.barrier()
            log(f"time budget reached in epoch {epoch} at batch {b + 1}/{n_batches}: saved; "
                f"rerun with --resume to continue from there")
            stopped = True
            break
    if stopped:
        break
    start_batch, carried_stats = 0, None
    dist.all_reduce(stats)
    train_acc, train_loss = (stats[0] / stats[1]).item() * 100, (stats[2] / stats[1]).item()
    val_acc, _, _ = evaluate(val_ds)
    improved = val_acc > max_val_acc
    if improved:
        max_val_acc = val_acc
    if is_main:
        if improved:
            save(best_path, epoch)
        save(last_path, epoch)
        if opt.snapshot_every and (epoch + 1) % opt.snapshot_every == 0:
            snap = os.path.join(opt.save_path, f"{name}_ep{epoch + 1}.pth.tar")
            save(snap, epoch)
            log(f"snapshot after {epoch + 1} epochs -> {snap}")
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["epoch", "train_loss", "train_acc", "val_acc", "best_val_acc", "epoch_sec"])
            w.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}", f"{val_acc:.2f}",
                        f"{max_val_acc:.2f}", f"{time.time() - t0:.0f}"])
    log(f"epoch {epoch}: train acc {train_acc:.2f}  val acc {val_acc:.2f}  best {max_val_acc:.2f}"
        f"{'  (saved best)' if improved else ''}  [{time.time() - t0:.0f}s]")
    if epoch + 1 < opt.num_epochs and out_of_time():
        log(f"time budget reached after epoch {epoch}; rerun with --resume to continue")
        stopped = True
        break
if not stopped and is_main:
    log(f"finished: {opt.num_epochs} epochs, best validation accuracy {max_val_acc:.2f}")
dist.destroy_process_group()
