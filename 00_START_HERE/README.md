# Start here: where everything is

This folder is the map of the repo. Read this page first, then:

| File | What it answers |
|---|---|
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Every experiment we ran (zero-shot, LoRA baselines, Stage 0, branch alone, H1–H13, A1, A2): its runs, the code and launcher, where its results and figure are, and its headline number |
| [`STATUS.md`](STATUS.md) | What is finished, what is still training, what launches by itself next, and the open decisions (as of Nov 19, 2026) |
| [`DATA.md`](DATA.md) | Every dataset (PGM and the mixture), the exact splits and subsets, the mixture weights, and how the code loads them |
| [`CONDITIONS_AND_ASSUMPTIONS.md`](CONDITIONS_AND_ASSUMPTIONS.md) | Every recipe's hyperparameters, how results were measured, the assumptions and untested choices, and the caveats |
| [`REPRODUCE.md`](REPRODUCE.md) | Setup on PACE, data, which checkpoints each step needs, the exact command for any run, and how to rebuild every table and figure |

The full write-ups are in [`../docs/`](../docs): [`relational_deep_dives.md`](../docs/relational_deep_dives.md)
(one section per experiment H1–H13, A1, A2 with every setting and result),
[`relational_experiment_log.md`](../docs/relational_experiment_log.md) (the same work as one table, numbered 1–23),
and [`experiments.md`](../docs/experiments.md) (zero-shot and LoRA baselines).

## The project in four lines

- **Task:** Procedurally Generated Matrices (PGM, neutral regime): pick the missing panel of a 3×3 matrix from 8 candidates (chance 12.5%). Models: Qwen3-VL-2B (main) and InternVL3.5-1B.
- **Baselines:** zero-shot is at chance (12.9% / 13.8%). LoRA fine-tuning with per-candidate Yes/No scoring reaches 81.5% (Qwen) and 57.7% (InternVL) on the 200k test set with 240k training puzzles.
- **Our method:** a relational branch (slot attention → Abstractor → connector) feeding the VLM, either as extra prompt tokens ("interleave") or blended into the image tokens ("fusion").
- **Result so far:** at 5% data (60k puzzles), LoRA-first with the relational branch reaches **56.5%** vs 52.6% for LoRA alone (+3.3 to +3.5 over an equal-steps control, p < 1e-11), mostly on counting puzzles. At full data, and on the task mixture, the branch adds nothing. A branch trained only on the 60k (early checkpoint) gave no gain.

## Folder map

| Path | What is in it |
|---|---|
| `00_START_HERE/` | This guide |
| `README.md` | Front page: results table and quickstart |
| `env.sh` | Every path (data, runs, container, HF cache) and the pinned model revisions. `source env.sh` first |
| **Code** | |
| `pgmvlm/` | Shared core for the baselines: PGM loading and rendering, model loading, LoRA and Yes/No candidate scoring, evaluation loop |
| `zeroshot/` | Zero-shot inference for PGM and CLEVR-ART, plus their Slurm scripts |
| `lora/` | LoRA training (`train_lora.py`, resumable) and sharded 200k test evaluation (`eval_vlm.py`) |
| `relational/` | The relational branch: slot attention (`slots.py`), Abstractor (`abstractor.py`), the VLM wrapper with interleave / fusion (`relvlm.py`), training (`train_stage0.py`, `train_stage1.py`), evaluation (`eval.py`), dataset builders (`datasets/`), Slurm scripts (`slurm/`), CPU tests (`tests/`). Its own [`README.md`](../relational/README.md) explains the stages and flags |
| `experiments/` | The exact launchers and chains that ran H7–H13 (`auto/`), and the scoring jobs and analyzers for A1, A2 and the branch-off check (`diag_*`). Kept verbatim, with the original PACE paths |
| `slurm/` | Baseline Slurm jobs (training, 200k test evaluation, tests) and `plots.sbatch` (redraws the figures) |
| `tools/` | Data prep, plots (`plot_curves.py` for LoRA, `plot_relational.py` for the branch), `rerun_cmd.py` (exact command of any relational run), `unpack_results.sh` |
| `tests/` | CPU tests for the baseline code on tiny random models |
| `share/` | Builder for the adapter bundle shared with collaborators |
| `legacy/vlm_lora_pc/` | The original LoRA code that produced every LoRA result (since reorganised into `pgmvlm/` + `lora/`) |
| `data_manifests/` | Gzipped manifests of every mixture dataset (prompts, answers, image references, panel boxes) and the loader check report; images are not included |
| **Results (no weights, no data)** | |
| `results/zeroshot/` | Zero-shot summary (`zeroshot_summary.csv`) and per-problem outputs for every PGM and CLEVR-ART run (`per_problem/*.json.gz`) |
| `results/lora/` | LoRA results table (`lora_pgm_results.md`), curves, validation-loss JSON; `run_records/<run>/` (config + training log for every LoRA run, including 1% and 5% data); `test_200k/<eval>/` (summary + config of every test evaluation; per-problem records for the final adapters) |
| `results/relational/run_records/<run>/` | One folder per relational run (114): `args.json` (every setting), `metrics.jsonl.gz` (training and validation log), `best.json`, `done.json` / `stopped.json`, Stage 0 `log.jsonl.gz` and final slot picture |
| `results/relational/test5000/` | A2: per-puzzle scores of every model on the 5,000 test puzzles (`*.json.gz`), the puzzle list (`test5000.tsv`), rules per puzzle, and the analysis (`report.md`) |
| `results/relational/overlap_val1000/` | A1: overlap and averaging check on the 1,000 validation puzzles, with `report.md` |
| `results/relational/branchoff_val1000/` | The H8 branch-off check |
| `results/relational/job_log/` | `night_queue_2026-11-08.txt`: every Slurm submission (job id, run, all flags, why); `priority.json` (scheduling tiers); run inventory |
| `results/relational/snapshots/` | Provenance of each branch snapshot (which run and step it came from) |
| `results/splits/` | Exact PGM file lists: the shared 240k selection, the 60k and 12k subsets, validation, the 200k test list, the 1,000 validation and 5,000 test puzzles |
| `results/plots/` | All figures (listed in `EXPERIMENTS.md`) |
| `docs/` | Write-ups (above) and the protocol comparison with the Slot Abstractor paper |

## Not in the repo

| What | Where it is on PACE | Why |
|---|---|---|
| PGM data (`.npz`) | `$DATA` = `~psunkad3/scratch/pgm_data` | ~1.2M files; download from DeepMind's PGM release |
| Mixture datasets | `$RELVLM_DATA` = `~psunkad3/scratch/relvlm_data` | Built by `relational/datasets/` (registry in its README) |
| Checkpoints (`*.pt`), LoRA adapters, branch snapshots | `$RELVLM_RUNS` = `~psunkad3/scratch/relvlm_runs`, `$RUNS` = `~psunkad3/scratch/vlm_lora_pc_runs` | Several GB each. Ask Pratham for read access or a copy |
| Slurm logs | `$RELVLM_RUNS/logs`, `$RUNS/logs` | Large; the job log and `metrics.jsonl` hold what matters |

## Naming used everywhere

| Term | Meaning |
|---|---|
| int / interleave | Branch output enters the prompt as relation tokens |
| fuse / fusion | Branch output is blended into the image tokens with weight 1 − λ |
| Stage 0 | Slot attention trained on images only (`s0_*` runs) |
| Stage 1a, "branch alone", `aux_*` | Abstractor with its own linear answer head, no LLM |
| Stage 1, alignment, `align_*` | Branch and LLM frozen; only the connector trains |
| Stage 2, `lora_w5_*` / `lora_w5d_*` | LoRA from scratch together with the branch; `w5d` = with branch dropout 0.5 |
| LoRA-first, `lora_lf_*` | Start from the finished LoRA-alone adapter, attach the aligned branch, train 3,500 steps; `_ctrl` = branch always off |
| `ft`, `ft50`, `ft276k`, `c60e`, `c60` | Which branch: 10-epoch; 50-epoch run at step 204k; at step 276k; 60k-only branch at step 43k; finished 60k-only branch |
| 5% / 60k, full / 240k | Seed-0 60k subset of PGM neutral / the 240k shared training set (20% of PGM neutral) |
| val / test | 1,000 shared validation puzzles (±3 points) / 5,000 test puzzles from the 200k test set (±1.4 points) |
