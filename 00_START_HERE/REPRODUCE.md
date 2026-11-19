# Reproducing the results

Everything ran on Georgia Tech PACE ICE, one GPU per job (A100, H100, H200 or L40S), inside the PACE container
`pytorch-2.1.0.sif` with the packages in `requirements.txt` (torch 2.6, transformers 5.17, peft 0.21.2). Jobs are
2–5 hour Slurm segments that resume from `state.pt` and resubmit themselves until `final.pt` exists.

## 1. Setup

```bash
git clone https://github.com/prathamsunkad/relationalreasoningVLM.git && cd relationalreasoningVLM
# edit the defaults in env.sh (or export your own values first): DATA, RUNS, RELVLM_RUNS, HF_HOME, CONTAINER
source env.sh
pip install --target $HOME/.local/lib/python3.10/site-packages -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

- **PGM data:** DeepMind's Procedurally Generated Matrices, `neutral` regime, as `.npz` under `$DATA/{train_full,val_full,test_full}/neutral/`. `tools/` has the preparation scripts used for the 240k training subset and the file caches.
- **Models:** `Qwen/Qwen3-VL-2B-Instruct` and `OpenGVLab/InternVL3_5-1B-HF` at the revisions pinned in `env.sh`, downloaded into `$HF_HOME` (jobs run with `HF_HUB_OFFLINE=1`).
- **Mixture data (H5, B4):** build with `relational/datasets/` (see its README) into `$RELVLM_DATA`.
- **Login nodes have no torch and no apptainer:** run everything, tests included, through Slurm.

**Paths inside scripts.** `relational/slurm/*.sbatch` read `REPO`, `RUNS`, `DATA`, `VAL_LIST` and `CONTAINER` from the
environment, with the original PACE paths as defaults; their `#SBATCH -o` log path and `HF_HOME` are written out and
need editing on another account. The scripts in `experiments/` are kept exactly as they ran (useful as a record);
change the `RUNS=`, `REPO=` and adapter paths at their top before reusing them.

## 2. Check the install

```bash
sbatch slurm/tests.sbatch                                      # baseline code: 22 CPU tests on tiny random models
REPO=$RELVLM_REPO sbatch relational/slurm/tests.sbatch          # relational code: CPU tests on tiny random models
```

## 3. Which checkpoints each step needs

Nothing large is in git, so a step can only be re-run once the checkpoint it starts from exists (re-train it, or get
a copy from `$RELVLM_RUNS` / `$RUNS` on PACE).

| To run | You need first |
|---|---|
| Zero-shot (Z1, Z2) | The base models only |
| LoRA baselines (L2–L4) | The base models only |
| Stage 0 (`s0_*`) | The base models only |
| Branch alone (B1–B7), H1–H6 | A Stage 0 slot checkpoint: `s0_qwen_pgm/best.pt` (32-dim) or `s0_qwen_pgm_d256/best.pt` (256-dim); `s0_qwen_pgm60k/best.pt` for H13 |
| Alignment (H7, H10, H12, H13) | A branch-alone checkpoint: `aux_ft_q_L6_lr3e-4_ep10/best.pt` (H7); 50-epoch snapshots `aux_ft_q_L6_lr3e-4_ep50_step{204000,276000}.pt` (H10–H12); `aux_ft60k_q_L6_lr3e-4_step43000.pt` (H13). Provenance in `results/relational/snapshots/` |
| Stage 2 (H8–H10, H13) | The alignment checkpoint at the step where it levelled off (copied into the run folder as `init_*.pt`; step in the job log) |
| LoRA-first (H11–H13) | The same alignment checkpoint, plus the LoRA-alone 5% adapter `vlm_lora_pc_runs/qwen_pc_neutral5pct_recipe/best` |
| A1, A2 scoring | The finished runs' `best.pt` (and the LoRA adapters) |
| A1, A2 analysis, all figures | Nothing: the records in `results/` are enough (section 6) |

## 4. Re-running any relational run

Every run folder in `results/relational/run_records/` has the complete `args.json`. This prints the full command,
with every argument spelled out, plus the job-log lines saying when and why it was submitted:

```bash
python3 tools/rerun_cmd.py lora_lf_q_int_ft276k
python3 tools/rerun_cmd.py lora_lf_q_int_ft276k --remap /home/hice1/psunkad3/scratch=$HOME/scratch
```

Run the printed `python train_stage1.py ...` from `relational/` inside the container on one GPU, or hand the
flags to the Slurm wrapper, which adds resume and auto-resubmit:

```bash
REPO=$RELVLM_REPO RUNS=$RELVLM_RUNS MODEL_ID=Qwen/Qwen3-VL-2B-Instruct SLOT=$RELVLM_RUNS/s0_qwen_pgm/best.pt \
  TAG=my_run EPOCHS=4 EXTRA="<flags from the job log>" AUTO_RESUBMIT=1 SB_OPTS="-J my_run -t 02:00:00" \
  sbatch --export=ALL -J my_run -t 02:00:00 relational/slurm/stage1.sbatch
```

Stage 0 runs use `relational/slurm/stage0.sbatch` the same way (`TAG`, `MODEL_ID`, `EXTRA`; their flags are in the
job log, e.g. `s0_qwen_pgm60k`: `--train_samples 60000 --max_steps 100000`).

**The recipes as they were launched** (each script's header explains its settings):

| Experiment | Script in `experiments/auto/` |
|---|---|
| H7, H10, H12 alignment (all 240k puzzles, lr 3e-4) | `launch_align.sh <int\|fuse> <branch .pt> <new tag> <note>` |
| H10, H13 Stage 2 (dropout p) | `launch_s2wd.sh <int\|fuse> <aligned .pt> <new tag> <p> <note>` |
| H11 LoRA-first (p = 0) and its controls (p = 1.0) | `launch_lf.sh <int\|fuse> <new tag> <p> <note>` |
| H12, H13 LoRA-first | `launch_lf2.sh <int\|fuse> <aligned .pt> <new tag> <note>` |
| H13 slots, branch, 60k alignment | `launch_s0_60k.sh`; `launch_aux60k.sh <lr> <new tag> <note>`; `launch_align60k.sh <int\|fuse> <branch .pt> <new tag> <note>` |
| Hand-offs (wait for levelling, then launch the next stage) | `auto_stage2.py` (H10), `auto_lf276.py` (H12), `auto_c60.py` (H13; `--branch <pt> --suffix c60e` for the early set) |
| Test scoring as runs finish | `wait_5k.sh <job> <time limit> <run>=<eval names> ...` |

"Levelled off" (when an alignment checkpoint is passed on): at least 3 validations, and the best not up more than
1 point over the last two.

## 5. Re-scoring a model on the test or validation puzzles

```bash
# A2: 5,000 test puzzles. Names map to checkpoints inside the script (one `ev` line per model)
sbatch experiments/diag_5k/diag_5k.sbatch lf276_int lf276_int_off
# A1: 1,000 validation puzzles
sbatch experiments/diag_overlap/diag_overlap.sbatch
```

Both call `relational/eval.py`, which writes one record per puzzle (all 8 answer scores). `--branch_off` scores a
hybrid with its branch switched off (fusion λ = 1, interleave gain = 0); `--readout aux` scores the branch alone;
`--lora_from <adapter>` scores LoRA alone through the same code.

## 6. Rebuilding every table and figure from the repo alone

```bash
bash tools/unpack_results.sh
python3 experiments/diag_5k/analyze5k.py --dir results/relational/test5000          # A2 report.md
python3 experiments/diag_overlap/analyze.py --dir results/relational/overlap_val1000 # A1 report.md
sbatch slurm/plots.sbatch                                                            # results/plots/*.png (needs matplotlib: runs in the container)
```

The rebuilt A2 report matches the committed one line for line (checked Nov 19).

## 7. Baselines

Zero-shot, LoRA training and the 200k-test evaluation are described in the front-page [`README.md`](../README.md)
(quickstart) and [`docs/experiments.md`](../docs/experiments.md); e.g.

```bash
bash slurm/run_test_eval.sh Qwen/Qwen3-VL-2B-Instruct qwen_pc_recipe_e1 final_e4 _check    # 28 shards + merge
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=my_run,REGIME=neutral,TRAIN_SAMPLES=60000,EPOCHS=4 slurm/train.sbatch
```
