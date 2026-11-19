# Checkpoints and adapters: where they are and how to load them

Every saved model lives on PACE under `/storage/ice1/4/7/psunkad3/` (42 GB, not in git). The full list, 435 files
with run, kind, step, validation accuracy and size, is [`results/checkpoints_inventory.csv`](../results/checkpoints_inventory.csv)
(rebuilt by `python3 tools/list_checkpoints.py`). The ones you are most likely to need are below.

Always use the `/storage/ice1/4/7/psunkad3/...` paths: `~psunkad3/scratch` is a link inside a private home folder.

## 1. Getting access (Pratham runs this once per teammate)

The run folders are already world-readable; only the top folder blocks entry. Granting a teammate permission to
pass through it (not to list it) is enough:

```bash
setfacl -m u:<teammate_gt_username>:x /storage/ice1/4/7/psunkad3      # repeat per teammate
getfacl /storage/ice1/4/7/psunkad3                                   # check
# the same grant also opens the PGM data (/storage/ice1/4/7/psunkad3/pgm_data is world-readable) and relvlm_data
```

A teammate then checks with `ls /storage/ice1/4/7/psunkad3/relvlm_runs/s0_qwen_pgm/`. If space allows, copying the
needed files to their own scratch is safer than reading live folders: the four runs still training rewrite their
`best.pt` / `state.pt` until they finish.

## 2. The key checkpoints

`R = /storage/ice1/4/7/psunkad3/relvlm_runs`, `L = /storage/ice1/4/7/psunkad3/vlm_lora_pc_runs`.

**LoRA adapters (PEFT format: `adapter_config.json` + `adapter_model.safetensors`)**

| Adapter | Path | Result |
|---|---|---|
| Qwen, full data, epoch 4 (main baseline) | `$L/qwen_pc_recipe_e1/final_e4` | 81.50% (200k test) |
| Qwen, full data, epoch 5 | `$L/qwen_pc_recipe_e1/final_e5` | 82.06% (200k test) |
| Qwen, full data, best val | `$L/qwen_pc_recipe_e1/best` | 82.1% val (step 72k) |
| InternVL, full data, epoch 4 / epoch 6 | `$L/ivl1b_pc_recipe_e1/final_e4`, `.../final_e6` | 57.73% / 60.31% (200k test) |
| **Qwen, 5% data (start of every LoRA-first run)** | `$L/qwen_pc_neutral5pct_recipe/best` | 53.4% val (step 14k), 52.6% test |
| Qwen, 1% data | `$L/qwen_pc_neutral1pct_recipe/best` | 24.4% val |
| InternVL, 5% / 1% data | `$L/ivl1b_pc_neutral5pct_recipe/best`, `$L/ivl1b_pc_neutral1pct_recipe/best` | 37.7% / 22.8% val |

Every epoch's adapters are also there (`best_eN`, `final_eN`, each `best_eN.json` with its step and accuracy).

**Stage 0 slot attention (`best.pt`: `config` + `model`)**

| Slots | Path | Used by |
|---|---|---|
| Qwen, PGM 240k, 32-dim | `$R/s0_qwen_pgm/best.pt` | Every 32-dim branch: H1–H3, H7–H12, B1–B6 |
| Qwen, PGM 240k, 256-dim | `$R/s0_qwen_pgm_d256/best.pt` | H4, H6 |
| Qwen, image mixture, 256-dim | `$R/s0_qwen_mix20_d256/best.pt` | H5, B4 |
| Qwen, PGM 60k only, 32-dim | `$R/s0_qwen_pgm60k/best.pt` | H13 |
| InternVL, PGM 240k, 32-dim | `$R/s0_ivl1b_pgm/best.pt` | Early InternVL runs |

**Branches (slots fine-tuned + Abstractor + linear head; `best.pt` or snapshot)**

| Branch | Path | Branch alone |
|---|---|---|
| 10 epochs (H7–H9) | `$R/aux_ft_q_L6_lr3e-4_ep10/best.pt` | 37.1% val, 35.3% test |
| 50-epoch run at step 204k (H10, H11) | `$R/snapshots/aux_ft_q_L6_lr3e-4_ep50_step204000.pt` | 40.9% val, 39.8% test |
| 50-epoch run at step 276k (H12) | `$R/snapshots/aux_ft_q_L6_lr3e-4_ep50_step276000.pt` | 47.6% val, 45.0% test |
| Settled copy, step 238k (A1, A2 reference) | `$R/aux_ft_q_L6_lr3e-4_settle/best.pt` | 44.8% val, 43.3% test |
| 50-epoch run, best so far (**still training**) | `$R/aux_ft_q_L6_lr3e-4_ep50/best.pt` | 51.7% val (step 442k, changes) |
| 60k-only at step 43k (H13 early set) | `$R/snapshots/aux_ft60k_q_L6_lr3e-4_step43000.pt` | 28.9% val, 26.8% test |
| 60k-only (**still training**, ends ~Nov 19 night) | `$R/aux_ft60k_q_L6_lr{3e-4,1e-3}/best.pt` | 29.4% / 30.8% val |

**Aligned checkpoints passed on to Stage 2 / LoRA-first** (branch + trained connector, LLM untouched)

| From | Path |
|---|---|
| 10-epoch branch | `$R/lora_w5_q_int_L6ft/init_*.pt`, `$R/lora_w5_q_fuse_L6ft/init_*.pt` (copies inside each run) |
| Step 204k / 276k / 60k-early branch | `$R/auto/snap/init_align_q_{int,fuse}_L6ft{50,276k,c60e}_lr3e-4_step*.pt` |

**Hybrids (`best.pt`: branch + LoRA, loads with `relational/eval.py`)**

| Model | Path | Test (5,000) |
|---|---|---|
| **H12 LoRA-first, interleave / fusion (best result)** | `$R/lora_lf_q_int_ft276k/best.pt`, `$R/lora_lf_q_fuse_ft276k/best.pt` | 56.5% / 56.4% |
| H11 LoRA-first and its controls | `$R/lora_lf_q_{int,fuse}_ft50[_ctrl]/best.pt` | 55.1% / 54.9% (controls 53.2% / 52.9%) |
| H9 Stage 2 with dropout | `$R/lora_w5d_q_{int,fuse}_L6ft/best.pt` | 55.2% / 54.1% |
| H10 Stage 2, stronger branch | `$R/lora_w5d_q_{int,fuse}_L6ft50/best.pt` | 55.1% / 55.3% |
| H13 (60k-only branch) | `$R/lora_{w5,w5d,lf}_q_{int,fuse}_c60e/best.pt` | see `STATUS.md` |

Every other run (all 114) has `best.pt` and usually `final.pt` in `$R/<run>/`; `state.pt` files hold the optimizer
too and are only for continuing training.

## 3. Loading them

Run inside the PACE container with the repo's `requirements.txt` installed (login nodes have no torch).

**Score a hybrid or branch on PGM puzzles** (the same command A2 used):

```bash
cd relational
python eval.py --ckpt /storage/ice1/4/7/psunkad3/relvlm_runs/lora_lf_q_int_ft276k/best.pt \
  --source pgm=matrix:$DATA/test_full/neutral --val_list ../results/splits/test5000.tsv --out_json out/lf276_int.json
#   --branch_off          the same model with the branch switched off
#   --readout aux         the branch alone (its linear head; no LLM needed in the answer)
#   --lora_from <adapter> LoRA alone through the same code: e.g. a fusion checkpoint + --branch_off + the 5% adapter
```

**Load a relational checkpoint in Python** (what `eval.py` does):

```python
import sys, torch
sys.path.insert(0, "relational")
from factory import build, restore

ck = torch.load(path, map_location="cpu", weights_only=False)   # keys: branch, lora, (slot_ae), model_id, slot_ckpt, cfg, args, step, best
# ck["slot_ckpt"] is the Stage 0 file it was built on, saved as /home/hice1/psunkad3/scratch/...; teammates must point it at /storage:
ck["slot_ckpt"] = ck["slot_ckpt"].replace("/home/hice1/psunkad3/scratch", "/storage/ice1/4/7/psunkad3")
rv, bb, score_tokens = build(ck["model_id"], ck["slot_ckpt"], ck["cfg"], "cuda", "bf16",
                             ck.get("args", {}).get("attn", "sdpa"), ck.get("lora_cfg"))
restore(rv, bb, ck)        # branch weights, fine-tuned slots if saved, and the LoRA tensors
# rv = the relational branch wrapped around the VLM; bb.model = Qwen3-VL with LoRA; rv.branch_off = True switches the branch off
```

`relational/eval.py` reads `ck["slot_ckpt"]` as saved; on another account, apply the same `.replace(...)` there
(line `rv, bb, st = build(ck["model_id"], ck["slot_ckpt"], ...)`) or create the path.

**Load slot attention alone:**

```python
from slots import load_slot_ae
ae, ck = load_slot_ae("/storage/ice1/4/7/psunkad3/relvlm_runs/s0_qwen_pgm/best.pt", "cuda")   # frozen SlotAutoencoder; ck["config"]
```

**Load a LoRA adapter for the baseline code:**

```python
from pgmvlm.models import load_base, load_adapter
model, processor, family = load_base("Qwen/Qwen3-VL-2B-Instruct", "bf16", device="cuda")
model = load_adapter(model, "/storage/ice1/4/7/psunkad3/vlm_lora_pc_runs/qwen_pc_recipe_e1/final_e4")
# or standard PEFT: PeftModel.from_pretrained(base_model, adapter_dir)
```

Score with the candidate scoring in `pgmvlm/models.py` (`candidate_scores`), or run the full 200k test with
`bash slurm/run_test_eval.sh Qwen/Qwen3-VL-2B-Instruct qwen_pc_recipe_e1 final_e4 _check`, or one puzzle with
`tools/demo_predict.py` (see the front-page README).

**Start new training from a checkpoint:** `train_stage1.py --init_from <aligned .pt>` (branch, connector and fine-tuned
slots), `--lora_from <adapter dir>` (a PEFT adapter as the starting LoRA), `--slot_ckpt <Stage 0 best.pt>`.
`python3 tools/rerun_cmd.py <run>` shows how each existing run used them.
