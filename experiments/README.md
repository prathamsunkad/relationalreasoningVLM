# experiments/: the scripts that ran H7–H13, A1 and A2

These are the exact files used on PACE, kept unchanged as a record. They hard-code the original paths
(`/home/hice1/psunkad3/scratch/relvlm_runs`, `.../relvlm_ft`, the LoRA adapter under `.../vlm_lora_pc_runs`) and append
each submission to the job log and `priority.json` (now copied to `results/relational/job_log/`). To reuse one,
change the `RUNS=`, `REPO=` (= this repo's `relational/`) and adapter paths at its top. Each script's header comment
explains the recipe. Earlier experiments (H1–H6, Stage 0, branch alone) were launched directly with
`relational/slurm/stage{0,1}.sbatch`; their flags are in the job log and in each run's `args.json`.

| Path | What |
|---|---|
| `auto/launch_align.sh` | Alignment: branch and LLM frozen, connector only, 10k steps, lr 3e-4 (H7, H10, H12) |
| `auto/launch_s2wd.sh` | Stage 2 from an aligned checkpoint, LoRA from scratch, branch dropout p (H10, H13) |
| `auto/launch_lf.sh`, `auto/launch_lf2.sh` | LoRA-first from the LoRA-alone 5% adapter; `launch_lf.sh` also makes the branch-off controls (H11; H12, H13) |
| `auto/launch_s0_60k.sh`, `launch_aux60k.sh`, `launch_align60k.sh`, `wait_s0_60k.sh` | Clean 60k pipeline: slots, branch alone and alignment on the 60k only (H13) |
| `auto/auto_stage2.py`, `auto_lf276.py`, `auto_c60.py` | Hand-offs: wait for an alignment to level off, launch the next stage, start the test scoring (H10, H12, H13) |
| `auto/wait_5k.sh`, `wait_brset.sh` | Submit the scoring job once the listed runs have `final.pt` |
| `auto/monitor.sh`, `ptshapes.py` | Progress monitor; parameter count of a checkpoint without torch |
| `diag_branchoff/diag_branchoff.sbatch` | H8 check: score step-5k checkpoints with the branch on and off |
| `diag_overlap/diag_overlap.sbatch`, `analyze.py`, `rules.py` | A1: LoRA alone vs branch alone on the 1,000 validation puzzles |
| `diag_5k/diag_5k.sbatch`, `analyze5k.py`; `diag_5kw/` | A2: 5,000 test puzzles, one `ev <name> <ckpt and flags>` line per model; paired McNemar tests, averaging, per attribute |

Results of these jobs live in `results/relational/` (`test5000/`, `overlap_val1000/`, `branchoff_val1000/`).
