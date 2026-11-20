# Reproducibility pointers

A one-page summary for the report. The details are in [`REPRODUCE.md`](REPRODUCE.md), [`CHECKPOINTS.md`](CHECKPOINTS.md)
and [`EXPERIMENTS.md`](EXPERIMENTS.md).

## 1. Repository commit / tag

| What | Value |
|---|---|
| Repository | https://github.com/prathamsunkad/relationalreasoningVLM |
| Results snapshot | `main` at commit `e256309` (Nov 19, 2026): all results through H13 (early branch), A1, A2, and the A3 ablation code |
| Tag | `results-2026-11-19`, on the commit that adds this page. It contains no code changes after `e256309` |
| Code that produced the results | `pgmvlm/`, `lora/`, `zeroshot/`, `relational/` and `experiments/` exactly as they are at that tag. Later branches (`merged-yiran`, `pranav`) change or add code and are **not** part of these results |
| Software | PACE container `pytorch-2.1.0.sif` (Python 3.10) + `requirements.txt`: torch 2.6.0+cu124, torchvision 0.21.0, transformers 5.17.0, peft 0.21.2 |
| Base models (pinned in `env.sh`) | `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d8`; `OpenGVLab/InternVL3_5-1B-HF` @ `9191dbccf312` |
| Data | DeepMind PGM, `neutral` regime. Exact split files: `results/splits/` (240k training list, 60k subset, 1,000 validation puzzles, 5,000 test puzzles `test5000.tsv`) |

## 2. Exact commands for the main results

Run from the repo root on PACE after `source env.sh`. All scripts are the path-free copies in `slurm_portable/`.

| Result | Command |
|---|---|
| Check the install (CPU tests) | `sbatch slurm_portable/slurm/tests.sbatch` and `sbatch slurm_portable/relational/slurm/tests.sbatch` |
| Zero-shot PGM, 200k test (12.85% Qwen / 13.75% InternVL) | `sbatch slurm_portable/zeroshot/slurm/pgm_eval_qwenfulltest.sbatch`; `sbatch slurm_portable/zeroshot/slurm/pgm_eval_internvl_test.sbatch` |
| LoRA alone, full data (81.50% Qwen, 200k test) | `sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=my_run,REGIME=neutral,TRAIN_SAMPLES=0,EPOCHS=4 slurm_portable/slurm/train.sbatch`, then `bash slurm_portable/slurm/run_test_eval.sh Qwen/Qwen3-VL-2B-Instruct my_run final_e4 _check` |
| LoRA alone, 5% (52.6% test) | Same with `TRAIN_SAMPLES=60000` |
| Any relational run, exact flags | `python3 tools/rerun_cmd.py <run> --remap /home/hice1/psunkad3/scratch=$HOME/scratch` (rebuilt from the run's saved `args.json`) |
| **H12, best result** (56.5% interleave / 56.4% fusion, test) | 1. Alignment: `bash slurm_portable/experiments/auto/launch_align.sh int <step-276k branch .pt> align_q_int_L6ft276k_lr3e-4 "<note>"`. 2. LoRA-first from the aligned checkpoint at its levelled-off step (3,000): `bash slurm_portable/experiments/auto/launch_lf2.sh int <aligned .pt> lora_lf_q_int_ft276k "<note>"`. Use `fuse` for fusion |
| H12 control (branch off, 53.2% / 52.9%) | `bash slurm_portable/experiments/auto/launch_lf.sh int lora_lf_q_int_ft50_ctrl 1.0 "<note>"` |
| A2: score models on the 5,000 test puzzles | `sbatch slurm_portable/experiments/diag_5k/diag_5k.sbatch lf276_int lf276_int_off`, then `python3 experiments/diag_5k/analyze5k.py --dir $RELVLM_RUNS/diag_5k` |
| A1: overlap and averaging, 1,000 validation puzzles | `sbatch slurm_portable/experiments/diag_overlap/diag_overlap.sbatch`, then `python3 experiments/diag_overlap/analyze.py --dir $RELVLM_RUNS/diag_overlap` |
| A3: relation-token ablation | `CKPT_RUNS=$SHARED_ROOT/relvlm_runs bash slurm_portable/experiments/diag_ablate/submit_ablate.sh lora_lf_q_int_ft276k`, then `python3 experiments/diag_ablate/analyze_ablate.py --dir $RELVLM_RUNS/diag_ablate` |
| Rebuild every table and figure **without a GPU** | `bash tools/unpack_results.sh`; `python3 experiments/diag_5k/analyze5k.py --dir results/relational/test5000`; `python3 experiments/diag_overlap/analyze.py --dir results/relational/overlap_val1000`; `sbatch slurm/plots.sbatch` |

The H12 chain needs the branch first: Stage 0 slots (`relational/slurm/stage0.sbatch`), then the 50-epoch
branch-alone run `aux_ft_q_L6_lr3e-4_ep50` up to step 276k (`relational/slurm/stage1.sbatch --objective aux`). Every
step's flags are in `results/relational/job_log/night_queue_2026-11-08.txt`. Re-training from scratch is optional:
teammates with access can start from our saved checkpoints (section 3).

## 3. Where to find logs, checkpoints, outputs and figures

| Item | In the repo | On PACE (shared, read-only) |
|---|---|---|
| Run settings | `results/{lora,relational}/run_records/<run>/args.json` | `<run folder>/args.json` |
| Training and validation curves | `results/*/run_records/<run>/metrics.jsonl.gz` | `<run folder>/metrics.jsonl` |
| Job log: every submission, its flags and reason | `results/relational/job_log/night_queue_2026-11-08.txt` | `relvlm_runs/night_queue_2026-11-08.txt` |
| Slurm logs | Not in git | `relvlm_runs/logs/`, `vlm_lora_pc_runs/logs/` |
| Checkpoints (LoRA adapters, slots, branches, aligned, hybrids) | `results/checkpoints_inventory.csv` (435 files, 42 GB); paths and loading code in [`CHECKPOINTS.md`](CHECKPOINTS.md) | `/storage/ice1/4/7/psunkad3/vlm_lora_pc_runs/` (8.7 GB), `/storage/ice1/4/7/psunkad3/relvlm_runs/` (38 GB). Access is granted per user with `setfacl` |
| Branch snapshots used by H10–H13 | `results/relational/snapshots/` (provenance) | `relvlm_runs/snapshots/*.pt` |
| Per-puzzle outputs (every candidate's score) | `results/zeroshot/per_problem/`, `results/lora/test_200k/`, `results/relational/{test5000,overlap_val1000,branchoff_val1000,ablation_test5000}/` | the matching `diag_*` folders |
| Reports | `results/lora/lora_pgm_results.md`, `results/relational/test5000/report.md` (A2), `overlap_val1000/report.md` (A1), `ablation_test5000/report.md` (A3, when done) | |
| Figures | `results/plots/*.png` (12 relational figures), `results/lora/*_curves_*.png` | |
| Write-ups | `docs/relational_deep_dives.md`, `docs/relational_experiment_log.md`, `docs/experiments.md` | |

## 4. Known limitations

**Reproducibility**
- **One seed** (seed 0) for every run, so no run-to-run variance is measured. On 1,000 validation puzzles the margin is about ±3 points; on the 5,000 test puzzles about ±1.4.
- **GPU type changes results slightly.** Jobs ran in bf16 on whatever GPU was free (A100, H100, H200, L40S). Rounding differs by type and flips about 0.5% of answers, so gaps under about 1 point between models scored in different jobs are not settled. Same-job comparisons (each control, every A3 condition) are not affected.
- **Resumed training is not bit-identical.** Runs are 2–5 hour Slurm segments that resume from `state.pt`, often on a different GPU type, so a re-run follows a slightly different path.
- **Checkpoint selection used the validation puzzles**, so validation numbers are optimistic. Quote the test numbers.
- **The 5% hybrids H8–H12 use a branch trained on all 240k puzzles**, 4× the data LoRA saw. H13 removes this and so far finds no gain.
- **Some hand-offs depended on timing.** The aligned checkpoint passed to Stage 2 or LoRA-first is the one at the step where alignment "levelled off" (rule in `REPRODUCE.md`). The step used for each run is in the job log; a re-run may level off at a different step.
- **Paths.** The original scripts hard-code `/home/hice1/psunkad3/scratch`. Use the `slurm_portable/` copies, or `--remap` with `rerun_cmd.py`. Some saved checkpoints store the old slot path; `CHECKPOINTS.md` shows the one-line fix.
- **Not in git:** the data, the model weights, the checkpoints and the Slurm logs. Without PACE access to the shared folders, everything must be re-trained.
- **Still running at this tag:** the 50-epoch and 60k-only branch-alone runs, the H13 full-branch recipes, and A3. `STATUS.md` lists them; their results are not in this snapshot.

**Compute**
- Each job used one GPU with at least 40 GB (A100 40/80 GB, H100, H200, L40S); Blackwell nodes don't work with torch 2.6/cu124. Login nodes have no torch or apptainer, so everything, tests included, runs through Slurm.
- LoRA alone, Qwen: about 6.6 puzzles/s. Full data (240k × 4 epochs, 60k steps) ≈ 40 GPU-hours; 5% (15k steps) ≈ 10 GPU-hours.
- 200k test evaluation: 28 shards per adapter.
- The 50-epoch branch behind H10–H12: 750k steps, run as more than a week of chained 2-hour segments and still training. Reproducing H12 from scratch is dominated by this run. The later stages are short: alignment 10k steps, LoRA-first 3,500 steps.
- A3: about 2–2.5 GPU-hours per model for 5–6 conditions on 5,000 puzzles.
- Storage: 47 GB of checkpoints in total (inventory above); 66 MB of results in git.
