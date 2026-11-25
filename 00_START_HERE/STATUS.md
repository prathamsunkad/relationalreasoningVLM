# Status (Nov 25, 2026, 3 pm EST)

**Finished since Nov 24:** H16 (the 5%-only branch on the 5% LoRA after epochs 1, 2, 4), H17 (H15 with the Abstractor
unfrozen), the Stage 2 arm of H14, the 5% branch smoke runs (B8), the 6-layer 50-epoch branch (B6), the hybrid-vs-ensemble
check (E1), and InternVL phase 1 (I1). **Training:** the Slot Abstractor retrained on our 5% (I2) and the InternVL 5% LoRA
kept per epoch (I3), both for H16 on InternVL. Every experiment, run and test score with its exact settings:
[`docs/relvlm_experiments.xlsx`](../docs/relvlm_experiments.xlsx); narrative write-up:
[`docs/relvlm_results_H1_H15.md`](../docs/relvlm_results_H1_H15.md).

## Finished experiments (headline numbers)

| # | Result | Test (5,000) or val |
|---|---|---|
| Zero-shot | Near chance on PGM | 12.48% Qwen (final run, 187,315 puzzles), 13.75% InternVL (200k test) |
| LoRA alone, 20% | Main baseline | 81.50% Qwen (200k test); on `test5000`: epoch 1 / 2 / 4 = 55.2 / 68.1 / 81.2% |
| LoRA alone, 5% | Low-data baseline | 52.6% test; rerun per epoch: 31.1 / 40.3 / 45.7 / 51.3% after epochs 1-4 |
| H1-H3, H6 | Branch trained through a frozen LLM: memorises only with 6 layers, generalises slowly | 16-27% val |
| H4, H5 | Hybrids at full data and on the mixture: no gain | 81.3-81.4% val vs 81.6%; family 87.2 vs control 87.6 |
| H7 | Alignment: the frozen LLM reads the branch within 1-3k steps | = branch accuracy (step 636k: 62.1% test) |
| H8 / H9 / H10 | Stage 2 at 5%: shortcut, fixed by dropout, flat with a stronger branch | 49.1/38.8; 55.2/54.1; 55.1/55.3% |
| H11 / H12 | LoRA-first at 5%, steps 204k / 276k branch | 55.1/54.9; 56.5/56.4% (controls 53.2/52.9); H12 ties a calibrated ensemble |
| H13 | Everything at matched 5% data (60k-only branch, 28.1% alone) | LoRA-first 53.8/53.4% vs control 53.2/52.9 (n.s.) |
| A1 / A1b | The branch solves puzzles LoRA misses; averaging gain shrinks as LoRA trains | A1b: +10.2 (5%), +8.7 / +2.3 / -0.1 (20% epochs 1 / 2 / 4) |
| H14 | Step-636k branch on the 5% LoRA | LoRA-first 63.2/62.9% vs control 53.2/52.9 (+1.2/+0.9 over the branch alone); Stage 2 arm 62.4/61.7%, branch off 33.5/20.0% |
| **H15** | Same on the 20% LoRA (matched data) | **Epoch 1: 65.5/65.9%** vs control 60.2, LoRA 55.2, branch 62.0, ensemble 64.1. Epoch 4: 81.4/81.3% vs 81.2 (n.s.) |
| H16 | 60k-only branch on weaker 5% LoRAs | Epoch 1: 36.0 vs control 36.0; epoch 2: 44.1 vs 43.6 (n.s.); number puzzles at epoch 1: +7.1 (p = 3e-4) |
| H17 | H15 epoch 1 with the Abstractor unfrozen (3e-5) | 66.0% vs 65.8% frozen (p = 0.5); the Abstractor barely moved |
| B6 | 50-epoch branch, 6 layers, finished | Step 746k (best val, 67.2%): 64.8% test; final 750k: 65.3% test |
| B8 | 5% branch smoke runs (fewer layers, dropout, row swap, frozen slots) | Best 31.6% val vs the old 30.8%: no clear gain; limited by data |
| E1 | Hybrid vs calibrated ensemble of the same two checkpoints | Ties for H12 and H14; H15 epoch 1 beats it by +1.4/+1.8 (p <= 6e-4) |
| A3 | Relation-token ablation (19 runs) | H14 / H15 ep 1: shuffled across candidates 32.2 / 18.8%; H15 ep 4 and H17 as expected |
| **I1** | InternVL: shared 20% Slot Abstractor vs InternVL LoRA alone | SA 61.0%; InternVL LoRA 38.3% (5%), 40.1 / 48.4 / 57.9 / 60.3% (20% ep 1/2/4/6); ensembles 61.2-67.0%: complementary even at epoch 6 |

Pairs are interleave / fusion.

## Running (Nov 25, 3 pm)

| Run(s) | What it is | Progress |
|---|---|---|
| `ivl1b_pc_neutral5pct_ep` (I3) | InternVL 5% LoRA rerun keeping each epoch | ~1k / 15k steps |
| `sa5_ae`, `sa5_train`, `sa5_test` (I2) | Slot Abstractor retrained on our 5%: slot AE (300k steps), Abstractor (80 epochs), test | AE at ~20k steps; ~1.5-2 days in total |
| `aux_ft_q_L24_lr8e-5_ep50` | 50-epoch branch, 24 layers | **Paused** Nov 25 at ~636k to free a GPU; resumable |

## Open decisions

| Decision | What it would answer |
|---|---|
| H16 on InternVL (connectors + LoRA-first on the 5% epochs, with controls) | A low-data claim on InternVL. Runs only if the 5% Slot Abstractor beats the InternVL 5% LoRA (38.3%) |
| Add the teammate's InternVL H15 results | Scored on test5000 with the same paired checks |
| Stage 2 at 20% (LoRA from scratch + branch + dropout, Qwen and InternVL) | Whether the branch speeds up LoRA learning epoch by epoch |
| Longer LoRA-first on H15 epoch 1; seeds; single-job re-score | Robustness of the headline result |

## Known caveats

- One seed everywhere.
- H8-H12 and H14 use a branch trained on all 240k labelled puzzles (4x the 5% LoRA's data); H13, H15, H16 are at matched data.
- InternVL uses a different relational module (the paper's pixel-based Slot Abstractor) from the Qwen H-series (ViT-feature branch).
- Scoring jobs ran on different GPU types; cross-job gaps under ~1 point are not settled. `diag_5k/analyze5k.py` currently fails on the newest result names, so `results/relational/test5000/report.md` lags; the spreadsheet is computed from the per-puzzle files directly.
- Validation puzzles were also used to choose checkpoints; quote the test numbers.
