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
| B6 | 50-epoch branch (still training) and its settled copy | `aux_ft_q_L6_lr3e-4_ep50`, `aux_ft_q_L24_lr8e-5_ep50`, `aux_ft_q_L6_lr3e-4_settle` | same, chained; `sb_opts` per run | `branch_alone_curves.png`; snapshots in `results/relational/snapshots/` | 6 layers finished: best 67.2% val (746k) = 64.8% test, final 65.3% test; snapshots at 204k (40.9%), 276k (47.6%), **636k (63.9% val, 62.0% test; used in H14, H15)**; settled 44.8% val, 43.3% test; 24 layers 52.3% val at 600k |
| B7 | 60k-only branch (H13 step 2) | `aux_ft60k_q_L6_lr3e-4`, `aux_ft60k_q_L6_lr1e-3` | `experiments/auto/launch_aux60k.sh`, `wait_s0_60k.sh` | `branch_alone_curves.png` | 29.4% / 30.8% val (peak at 71k), then overfits (training 54% vs val 27% at 150k); best snapshot `aux_ft60k_best` = **28.1% test** |
| B8 | 5% branch smoke runs: fewer layers (2 / 4), Abstractor dropout (0.1 / 0.3), weight decay (0.05 / 0.1), row-swap augmentation (50%), lr 3e-4, frozen slots; 60k × 20 epochs = 75k steps | `aux60s_*` (9 runs) | `experiments/auto/launch_aux60k_smoke.sh` (`--abs_dropout`, `--row_swap_p`) | `run_records/aux60s_*` | Best 31.6% val (dropout 0.3) vs the old 30.8%: no clear gain; frozen slots 26.6% |

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
| H13 | Clean 60k pipeline; early branch (step 43k) through the H8, H9, H12 recipes | `s0_qwen_pgm60k`, `aux_ft60k_*`, `align_q_*_L6ftc60e_lr3e-4`, `lora_{w5,w5d,lf}_q_*_c60e` | `experiments/auto/auto_c60.py --branch ... --suffix c60e`, `launch_align60k.sh`, `launch_s2wd.sh`, `launch_lf2.sh` | `H13_stage2_60k_branch.png`, `H11_H12_H13_lora_first.png` | Early branch (26.8% test): H12 recipe 53.7% / 53.3%, identical with the branch off. Finished branch (28.1% test, `*_c60`): H12 recipe **53.8% / 53.4%** vs control 53.2% / 52.9% (n.s.); H9 recipe 50.4% / 52.1%; H8 recipe 43.5% / 46.8%: no gain at matched data |
| H14 | LoRA-first with the step-636k branch (62.0% test alone) on the 5% LoRA | `align_q_*_L6ft636k_lr3e-4`, `lora_lf_q_{int,fuse}_ft636k`; Stage 2 + dropout `lora_w5d_q_*_L6ft636k` (training) | `experiments/auto/auto_p636.py`, `launch_lf2.sh`, `launch_s2wd.sh` | `results/relational/test5000/report.md` | **Test 63.2% / 62.9%** vs control 53.2% / 52.9% (p < 1e-59); +1.2 / +0.9 over the branch alone |
| H15 | The same on the 20% LoRA after epochs 1 and 4, on all 240k puzzles (matched data), with controls | `lora_lf20e{1,4}_q_{int,fuse}_ft636k`, `lora_lf20e{1,4}_q_{int,fuse}_ctrl` | `experiments/auto/launch_lf20.sh` | same | **Epoch 1: 65.5% / 65.9%** vs control 60.2%, LoRA alone 55.2%, branch 62.0% (p < 1e-14 vs branch). Epoch 4: 81.4% / 81.3% vs LoRA alone 81.2% (n.s.) |
| H16 | Per-epoch 5%: LoRA-first with the 60k-only branch on the 5% LoRA after epochs 1 and 2 (rerun that keeps each epoch's adapter) | `qwen_pc_neutral5pct_ep`, `lora_lf5e{1,2}_q_int_c60`, `_ctrl` | `lora/train_lora.py --keep_steps`, `experiments/auto/launch_lf5ep.sh`, `wait_ep5.sh` | `run_records/` | Test: epoch 1 36.0% vs control 36.0% (p = 1); epoch 2 44.1% vs 43.6% (p = 0.38); LoRA alone 31.1 / 40.3 / 45.7 / 51.3% after epochs 1-4. Number puzzles at epoch 1: +7.1 (p = 3e-4) |
| H17 | H15 epoch 1 with the Abstractor unfrozen (lr 3e-5, slots frozen), plus H15 and its control re-scored in the same job | `lora_lfu20e1_q_int_ft636k_a3e-5` | `experiments/auto/launch_lfu20.sh` | `test5000/lfu20e1_*`, `lf20e1_*_rs` | Test 66.0% vs H15 65.8% (same job, p = 0.5); vs control 60.6% (+5.4); the adapted Abstractor alone is unchanged (62.0%) |

## Analyses

| # | Analysis | Job / script | Results | Headline |
|---|---|---|---|---|
| A1 | Overlap and averaging (LoRA alone vs branch alone), 1,000 val puzzles | `experiments/diag_overlap/diag_overlap.sbatch`, `analyze.py`, `experiments/auto/wait_brset.sh` | `results/relational/overlap_val1000/report.md` | 5%: averaging +2.2 to +3.5; full data: −0.5 to −1.3 |
| A1b | Overlap and averaging on the 5,000 test puzzles: step-636k branch vs LoRA alone at 5% and at 20% after epochs 1 / 2 / 4 | `experiments/diag_5k/diag_5k.sbatch` (`br636k`, `lora20e*`, `lora5p0`), `experiments/diag_overlap/analyze.py --dir` | `results/relational/overlap_test5000_p636k/report.md` | Averaging +10.2 (5%), +8.7 (ep 1), +2.3 (ep 2), −0.1 (ep 4); branch-only puzzles 979 / 874 / 528 / 238; number puzzles 70.9% branch vs 39.5% LoRA |
| A1c | Per-attribute gains at matched data: each H13 / H15 / H16 hybrid vs its equal-steps control on the 5,000 test puzzles (a puzzle counts under every attribute it has a rule on; exact McNemar) | `experiments/diag_overlap/by_attribute_matched.py` (reads the per-puzzle files in `results/relational/test5000/`) | `results/relational/test5000/by_attribute_matched.md` | Number (counting) puzzles: H15 epoch 1 +22.8 / +24.7 (p = 1e-17 / 1e-21) vs +3 to +5 on other attributes; H15 epoch 4 +2.6 / +1.3; H16 epoch 1 +7.1 (p = 3e-4), epoch 2 +2.8; H13 +0.0 / +2.0. In-distribution only: train and test share all shapes |
| A2 | 5,000-puzzle test scoring of every model (8 jobs, 3 GPU types) | `experiments/diag_5k/diag_5k.sbatch`, `diag_5kw/`, `analyze5k.py`, `experiments/auto/wait_5k.sh` | `results/relational/test5000/report.md` | All test numbers above; figure `A2_test5000.png` |
| A3 | Relation-token ablation (interleave runs of H4, H7–H12, H14, H15): score each model with its tokens zeroed, removed, shuffled across candidates, shuffled across puzzles, and reordered (multi-token runs) | `experiments/diag_ablate/diag_ablate.sbatch`, `submit_ablate.sh`, `analyze_ablate.py`; `relational/eval.py --rel_ablate` | `results/relational/ablation_test5000/report.md` | H12 LoRA-first: 56.4% → 52.5% zeroed, 47.6% shuffled (p < 1e-15); H14: 63.4% → 51.7% zeroed, 32.2% shuffled across candidates; H15 epoch 1: 65.7% → 57.7%, 18.8%; H15 epoch 4: within 2 points; H4 full data: within ±0.7 (tokens ignored); frozen-LLM alignments: 14.6% zeroed |
| C1 | Accuracy at equal training FLOPs (analytic count: ViT 7.73e11 per image, LLM 1.06e12 per sequence; branch 6.18e12 and LoRA 2.31e13 per puzzle) | — | `docs/relvlm_results_H1_H15.md` (Compute) | At 5.5e18 FLOPs: LoRA 53.4% vs branch 28.7% (5%); the branch needs 4–8× more total compute for equal accuracy |
| E1 | Hybrid vs a calibrated score ensemble of the same two checkpoints (2-fold cross-fitted weight), McNemar | — | `experiments/diag_overlap/ens_vs_hybrid.py` | `results/relational/ensemble_vs_hybrid_test5000/report.txt` | H12 and H14 tie the ensemble; H15 epoch 1 beats it by +1.4 / +1.8 (p ≤ 6e-4) |

## InternVL + standalone Slot Abstractor (Nov 25)

| # | Experiment | Runs / outputs | Code | Results | Headline |
|---|---|---|---|---|---|
| I1 | Shared 20% Slot Abstractor vs InternVL LoRA alone (inspect + overlap, no training) | `exp2_runs/complement_test5000_{5pct,e1,e2,e4,e6}` | `experiments/internvl_sa/run_exp2_container.sbatch` + `experiments/exp2_sa_interleave/` | `results/internvl_sa/` | SA 61.0% test; InternVL LoRA 38.3% (5%), 57.9 / 60.3% (20% ep 4 / 6); ensemble up to 67.0% |
| H15-IVL | H15 on InternVL (Yiran): epoch-6 InternVL LoRA + shared Slot Abstractor, LoRA-first interleave, 15,000 updates on 2 GPUs | `internvl_epoch6_interleave_20pct_two_gpu` (+ epoch 4 planned) | `tools/train_distributed_interleave.py`, `slurm_portable/slurm/distributed_interleave_*.sbatch` | `results/internvl_interleave/2026-11-24-two-gpu/` | In progress: 5,832 updates / 93,312 questions. Selected min-CE checkpoint: 66.8%, CE 1.0007 at update 3,500; highest accuracy 67.2% at 1,500; latest 66.7%, CE 1.0575 at 5,500. Native-input averaging control: 66.7% on the same 1,000 validation IDs (see `2026-11-25-average-control/`) |
| I2 | Slot Abstractor retrained on our 5% list (for H16 on InternVL) | `sa5/` | `experiments/internvl_sa/sa5/` | — | Running |
| I3 | InternVL 5% LoRA rerun keeping each epoch | `ivl1b_pc_neutral5pct_ep` | `lora/train_lora.py --keep_steps` | `results/lora/run_records/ivl1b_pc_neutral5pct_ep` | Running |

## Teammates' experiments

Each folder's README has the commands, settings and full results.

| # | Who | Experiment | Code | Results | Headline |
|---|---|---|---|---|---|
| T1 | Yiran | Answer-letter LoRA: LFM2.5, SmolVLM2, SpatialEvo on CLEVR-ART and PGM | `lora/` (answer-letter protocol), `configs/vlm/` | `results/vlm/README.md`, `results/vlm/run_records/` | PGM 200k test: LFM2.5 76.69%, SpatialEvo 61.87%; CLEVR-ART 89–100% |
| T2 | Yiran | LFM relational probes (5 insertion points), RCA plugin, relational memory; diagnostics | `pgmvlm/extensions/`, `docs/VLM_INTEGRATION.md` | `results/vlm/` | 12.1–13.6% PGM val: at chance |
| T3 | Pranav | Slot Abstractor from scratch on the same 240k puzzles | `slot_abstractor/` | `slot_abstractor/results/` | 62.8% val (epoch 64); paper 71.25% test; test not run |
| T4 | Pranav | Letter-answer LoRA from the contact sheet; perception-vs-reasoning probes | `probes/` | `probes/README.md` | Qwen 75.33%, InternVL 52.31% (200k test); pilot: answer found 78% from panels vs 10% from the sheet |
| T5 | Pranav | Exp1: gated relational hybrid at InternVL's projector | `experiments/exp1_gated_hybrid/` | its README | Stage 0 branch alone 20.4–21.6% held-out |
| T6 | Pranav | Exp2: Slot Abstractor tokens interleaved into fine-tuned InternVL | `experiments/exp2_sa_interleave/` | its README | Not run yet |
| T7 | Kun-Lin | I-RAVEN transfer of the PGM-trained LFM2.5 adapter | `iraven_eval/` | `iraven_eval/results/` | 23.1% over 14,000 puzzles (chance 12.5%); 19.5–26.0% per configuration |

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
