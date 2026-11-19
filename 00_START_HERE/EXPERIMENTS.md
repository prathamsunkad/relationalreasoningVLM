# Every experiment, and where to find it

One row per experiment. "Runs" are folder names under `results/relational/run_records/` (relational) or
`results/lora/run_records/` (LoRA). Every run's full settings are in its `args.json`; its job id, flags and reason
are in `results/relational/job_log/night_queue_2026-11-08.txt` (search for the run name), and
`python3 tools/rerun_cmd.py <run>` prints the exact command. Figures are in `results/plots/`. Every launcher or Slurm script named below has a path-free copy at the same
path under `slurm_portable/`; use that one on any account.
The deep dive for each H / A row is the section of the same name in [`docs/relational_deep_dives.md`](../docs/relational_deep_dives.md).

Accuracies: val = 1,000 shared validation puzzles (±3 points); test = 5,000 test puzzles (±1.4) unless marked 200k. Chance = 12.5%.

## Baselines

| # | Experiment | Runs / outputs | Code | Results | Headline |
|---|---|---|---|---|---|
| Z1 | Zero-shot PGM, 200k test | `Qwen2BInstruct_PGM_test/results_v2`, `results_v3_fixed`; `InternVL_PGM_test/results_v2` | `zeroshot/pgm_infer.py`, `zeroshot/slurm/` | `results/zeroshot/zeroshot_summary.csv`, `per_problem/` | Qwen 12.85%, InternVL 13.75%: chance |
| Z2 | Zero-shot CLEVR-ART (RMTS, identity rules), generate vs prefill | `*_CLEVRART_test/results_*` | `zeroshot/clevr_art_infer.py` | same | RMTS ≈ 49–51% (chance 50%); identity rules 21.6–25.0% (chance 25%) |
| L1 | LoRA v1 (single-pass format) | removed; see `docs/experiments.md` §1 | git history | `docs/experiments.md` | Stalled near 20% |
| L2 | LoRA v2, per-candidate Yes/No, full data (240k × 4 epochs) | `qwen_pc_recipe_e1`, `ivl1b_pc_recipe_e1` | `lora/train_lora.py`, `slurm/train.sbatch` | `results/lora/lora_pgm_results.md`, `test_200k/` | **Qwen 81.50%, InternVL 57.73%** (200k test); epoch 5 / 6: 82.06% / 60.31% |
| L3 | LoRA at 5% and 1% data | `*_pc_neutral5pct_recipe`, `*_pc_neutral1pct_recipe` | same | `run_records/`; 5% test in A2 | Qwen 5%: 53.4% val, **52.6% test** (the number every 5% hybrid is compared with) |
| L4 | LoRA memorising 500 puzzles | `overfit_pc_qwen`, `overfit_pc_ivl1b` | same | `run_records/` | 100% within ~300 steps |

## Relational branch: building blocks

| # | Experiment | Runs | Launcher | Results / figure | Headline |
|---|---|---|---|---|---|
| S0 | Stage 0 slot attention (images only) | `s0_qwen_pgm`, `s0_qwen_pgm_d256`, `s0_qwen_mix20[_d256]`, `s0_ivl1b_*`, `*_long`, `s0_qwen_pgm60k` | `relational/slurm/stage0.sbatch`; `experiments/auto/launch_s0_60k.sh` | `stage0_slots.png`; final slot picture in each run folder | Reconstruction MSE: Qwen PGM 32-dim 0.331, 256-dim 0.098; 60k-only 0.326 |
| B1 | Branch alone memorising 500 puzzles | `aux_overfit_*`, `aux_ovf_q_*` | `relational/slurm/stage1.sbatch` (`--objective aux`) | log #5 | 6 layers at 3e-4: 94.6%; 24 layers at 1e-4: 27.6% |
| B2 | Branch alone, first unseen-data runs | `aux_60k_qwen_*`, `aux_full_*_L6_lr3e-4_ep4`, `aux_full_qwen`, `aux_full_ivl1b` | same | log #6 | 240k × 4 epochs: 24.4% (Qwen), 24.7% (InternVL) |
| B3 | Sweep at 60k × 4 epochs (slots, 256-dim, depth, rate) | `aux_60k4_q_*`, `aux_full10_q_d256_lr1e-3` | same | log #7 | All 15.9–22.8%: no setting separates |
| B4 | Branch alone on the task mixture | `aux_mix30_q`, `aux_mix50_q`, `aux_mix30_noCART_q`, `aux_mix30_q_lr3e-4` | same | log #8 | PGM at chance; synthetic tasks 99.7–100% |
| B5 | 10-epoch branch, frozen vs fine-tuned slots | `aux_full_qwen_*_ep10`, `aux_ft_q_*_ep10` | same | `branch_alone_curves.png`; log #9 | Frozen 30.2%, **fine-tuned 37.1%** |
| B6 | 50-epoch branch (still training) and its settled copy | `aux_ft_q_L6_lr3e-4_ep50`, `aux_ft_q_L24_lr8e-5_ep50`, `aux_ft_q_L6_lr3e-4_settle` | same, chained; `sb_opts` per run | `branch_alone_curves.png`; snapshots in `results/relational/snapshots/` | 6 layers 51.5% val so far; snapshots at 204k (40.9%), 276k (47.6%); settled 44.8% val, 43.3% test |
| B7 | 60k-only branch (H13 step 2, still training) | `aux_ft60k_q_L6_lr3e-4`, `aux_ft60k_q_L6_lr1e-3` | `experiments/auto/launch_aux60k.sh`, `wait_s0_60k.sh` | `branch_alone_curves.png` | 29.4% / 30.8% val, flat since ~75k steps |

## Hybrids (branch + VLM)

| # | Experiment | Runs | Launcher | Figure | Headline |
|---|---|---|---|---|---|
| H1 | Frozen-LLM memorisation, rounds 1–2 (24 layers) | `overfit_*`, `overfit2_*` | `relational/slurm/stage1.sbatch` | `H1_H3_memorisation.png` | Best 32.6% of the 500 training puzzles |
| H2 | Frozen-LLM Stage 1, full data, 1 epoch | `s1_qwen_*`, `s1_ivl1b_*` | same | `H2_H6_frozen_llm.png` | 16–19% val |
| H3 | Frozen-LLM memorisation, 6 layers | `llm_ovf_q_*` | same | `H1_H3_memorisation.png` | Interleave 85% (3e-4), fusion 100% (1e-3) |
| H4 | Full-data hybrids + LoRA from scratch (1 / 9 / 144 tokens, fusion, control) | `lora_q_*` | same | `H4_full_data_hybrids.png` | 79.5–81.4% val vs LoRA alone 81.6%; control 81.3% |
| H5 | Mixture hybrids + LoRA | `lora_mix30_q_*` | same | `H5_mixture_hybrids.png` | Family score 87.2 / 87.2, control 87.6 |
| H6 | Frozen-LLM Stage 1 toward 10 epochs, 256-dim (stopped) | `s1_q_*_ep10` | same | `H2_H6_frozen_llm.png` | Best 22.6–27.2% val |
| H7 | Alignment from the 10-epoch branch | `align_q_{int,fuse}_L6ft_lr{3e-4,1e-3}` | `experiments/auto/launch_align.sh` | `H7_alignment.png` | 35–37.5% val = the branch's 37.1% |
| H8 | Stage 2 at 5%, no dropout + branch-off check | `lora_w5_q_*_L6ft` | `relational/slurm/stage1.sbatch`; `experiments/diag_branchoff/` | `H8_H9_H10_stage2.png` | Branch off 13–16% at step 5k (shortcut); test 49.1% / 38.8% |
| H9 | Stage 2 at 5%, branch dropout 0.5 | `lora_w5d_q_*_L6ft` | same (`--branch_dropout 0.5`) | `H8_H9_H10_stage2.png` | **Test 55.2% / 54.1%**; branch off 52.5% / 51.7% |
| H10 | Stronger branch (step 204k): alignment + Stage 2 with dropout | `align_q_*_L6ft50_lr3e-4`, `lora_w5d_q_*_L6ft50` | `experiments/auto/auto_stage2.py`, `launch_s2wd.sh` | `H7_alignment.png`, `H8_H9_H10_stage2.png` | Test 55.1% / 55.3%; branch off 49.5% / 49.3% |
| H11 | LoRA-first, step-204k branch, + branch-off controls | `lora_lf_q_*_ft50`, `lora_lf_q_*_ft50_ctrl` | `experiments/auto/launch_lf.sh` | `H11_H12_H13_lora_first.png` | Test 55.1% / 54.9% vs controls 53.2% / 52.9% |
| H12 | LoRA-first, step-276k branch (with its alignment) | `align_q_*_L6ft276k_lr3e-4`, `lora_lf_q_*_ft276k` | `experiments/auto/auto_lf276.py`, `launch_lf2.sh` | `H11_H12_H13_lora_first.png`, `gain_vs_branch_strength.png` | **Test 56.5% / 56.4%**, +3.3 / +3.5 over control; ties averaging |
| H13 | Clean 60k pipeline; early branch (step 43k) through the H8, H9, H12 recipes | `s0_qwen_pgm60k`, `aux_ft60k_*`, `align_q_*_L6ftc60e_lr3e-4`, `lora_{w5,w5d,lf}_q_*_c60e` | `experiments/auto/auto_c60.py --branch ... --suffix c60e`, `launch_align60k.sh`, `launch_s2wd.sh`, `launch_lf2.sh` | `H13_stage2_60k_branch.png`, `H11_H12_H13_lora_first.png` | Branch alone 26.8%; H12 recipe 53.7% / 53.3%, identical with the branch off: no gain |

## Analyses

| # | Analysis | Job / script | Results | Headline |
|---|---|---|---|---|
| A1 | Overlap and averaging (LoRA alone vs branch alone), 1,000 val puzzles | `experiments/diag_overlap/diag_overlap.sbatch`, `analyze.py`, `experiments/auto/wait_brset.sh` | `results/relational/overlap_val1000/report.md` | 5%: averaging +2.2 to +3.5; full data: −0.5 to −1.3 |
| A2 | 5,000-puzzle test scoring of every model (8 jobs, 3 GPU types) | `experiments/diag_5k/diag_5k.sbatch`, `diag_5kw/`, `analyze5k.py`, `experiments/auto/wait_5k.sh` | `results/relational/test5000/report.md` | All test numbers above; figure `A2_test5000.png` |

## Figures in `results/plots/`

| File | Shows |
|---|---|
| `H1_H3_memorisation.png` | Accuracy on the 500 training puzzles: H1 (24 layers) vs H3 (6 layers) vs branch alone |
| `H2_H6_frozen_llm.png` | Branch trained through the frozen LLM (H2, H6) vs branch alone |
| `H4_full_data_hybrids.png` | Full-data hybrids vs LoRA alone |
| `H5_mixture_hybrids.png` | Mixture hybrids: PGM, CLEVR, VSR |
| `H7_alignment.png` | All eight alignments against each branch's own accuracy |
| `H8_H9_H10_stage2.png` | Stage 2 at 5% vs LoRA alone |
| `H11_H12_H13_lora_first.png` | LoRA-first with three branches vs the control |
| `H13_stage2_60k_branch.png` | Stage 2 recipes with the 60k-only branch |
| `branch_alone_curves.png` | Branch-alone validation for the branches behind the hybrids |
| `stage0_slots.png` | Stage 0 reconstruction error |
| `A2_test5000.png` | Test accuracy of every scored model, with 95% intervals |
| `gain_vs_branch_strength.png` | LoRA-first gain over control against the branch's own test accuracy |
| `../lora/*_curves_*.png` | LoRA baseline training and validation curves |
