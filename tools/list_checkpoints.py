"""Write results/checkpoints_inventory.csv: every checkpoint and LoRA adapter on PACE, with what it is.

    python3 tools/list_checkpoints.py      # run on PACE; reads only the run folders (no torch needed)
"""
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "/storage/ice1/4/7/psunkad3"          # = ~psunkad3/scratch; use this path (the home-folder link is private)
REL = f"{BASE}/relvlm_runs"
LORA = f"{BASE}/vlm_lora_pc_runs"


def kind(run):
    for p, k in [("s0_", "Stage 0 slot attention"), ("aux_", "branch alone (Stage 1a)"), ("align_", "alignment (connector)"),
                 ("lora_lf_", "LoRA-first hybrid"), ("lora_w5d_", "Stage 2 hybrid, dropout 0.5"),
                 ("lora_w5_", "Stage 2 hybrid, no dropout"), ("lora_mix", "mixture hybrid (H5)"), ("lora_q_", "full-data hybrid (H4)"),
                 ("s1_", "frozen-LLM Stage 1"), ("llm_ovf", "frozen-LLM memorisation (H3)"), ("overfit", "frozen-LLM memorisation (H1)")]:
        if run.startswith(p):
            return k
    return "other"


FILE_NOTE = {
    "best.pt": "best validation checkpoint (use this)",
    "final.pt": "end of training",
    "last.pt": "latest (Stage 0 resume point)",
    "state.pt": "resume state incl. optimizer (large; only to continue training)",
}


def step_of(run_dir, fname):
    if fname == "best.pt" and os.path.exists(f"{run_dir}/best.json"):
        b = json.load(open(f"{run_dir}/best.json"))
        acc = b.get("llm_acc", b.get("aux_acc", b.get("best_val_mse")))
        return b.get("step"), acc
    if "step" in fname:
        try:
            return int(fname.split("step")[-1].split(".")[0]), None
        except ValueError:
            pass
    return None, None


rows = []
for run_dir in sorted(glob.glob(f"{REL}/*/")):
    run = os.path.basename(run_dir.rstrip("/"))
    if run in ("snapshots", "auto", "logs"):
        continue
    for f in sorted(os.listdir(run_dir)):
        if not f.endswith(".pt"):
            continue
        p = run_dir + f
        st, acc = step_of(run_dir, f)
        note = FILE_NOTE.get(f) or ("copy of the aligned checkpoint this run started from" if f.startswith("init_") else
                                    "snapshot" if "step" in f else "checkpoint")
        rows.append(dict(path=p, group="relational", kind=kind(run),
                         run=run, file=f, step=st, val=acc, size_mb=round(os.path.getsize(p) / 2**20, 1), what=note))
for p in sorted(glob.glob(f"{REL}/snapshots/*.pt")) + sorted(glob.glob(f"{REL}/auto/snap/*.pt")):
    f = os.path.basename(p)
    st, _ = step_of("", f)
    rows.append(dict(path=p, group="relational", kind="branch snapshot" if "/snapshots/" in p else "alignment snapshot (passed on)",
                     run=f.split("_step")[0].replace("init_", ""), file=f, step=st, val=None,
                     size_mb=round(os.path.getsize(p) / 2**20, 1), what="frozen copy used to start later runs"))
for run_dir in sorted(glob.glob(f"{LORA}/*/")):
    run = os.path.basename(run_dir.rstrip("/"))
    for d in sorted(os.listdir(run_dir)):
        a = f"{run_dir}{d}/adapter_model.safetensors"
        if not os.path.exists(a):
            continue
        st, acc = None, None
        j = f"{run_dir}{d}.json"
        if os.path.exists(j):
            b = json.load(open(j))
            st, acc = b.get("step"), b.get("val_acc")
        rows.append(dict(path=run_dir + d, group="lora", kind="LoRA adapter (PEFT)", run=run, file=d, step=st, val=acc,
                         size_mb=round(os.path.getsize(a) / 2**20, 1),
                         what={"best": "best validation adapter", "final": "end of training", "last": "latest"}.get(
                             d, "best adapter of epoch " + d[-1] if d.startswith("best_e") else
                             "adapter at the end of epoch " + d[-1] if d.startswith("final_e") else "adapter")))

out = f"{ROOT}/results/checkpoints_inventory.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "entries ->", out)
