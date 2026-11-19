# Experiment log: VLMs on PGM (Procedurally Generated Matrices)

All PGM numbers are on the **neutral** regime. Chance = 12.5% (8 answer choices). Test = the full 200,000-problem
neutral test set. Validation = the 1,000 shared validation problems (same IDs as the teammate runs), unless noted.
Models: Qwen3-VL-2B-Instruct (`89644892`), InternVL3.5-1B-HF (`9191dbcc`).

## 0. Zero-shot (`zeroshot/`)

Base models, no fine-tuning. Input: 3x3 context grid image + labelled 2x4 candidate grid image; the model generates
`Answer: <number>`, parsed as the last digit 1-8. Full table: `results/zeroshot/zeroshot_summary.csv`.

| Task | Chance | Qwen3-VL-2B | InternVL3.5-1B |
|---|---|---|---|
| PGM neutral, 200k test | 12.5% | 12.85% | 13.75% |
| CLEVR-ART RMTS (10k, prefilled answer) | 50% | 48.6% | 49.1% |
| CLEVR-ART identity rules (10k, prefilled answer) | 25% | 21.6% | 25.0% |

Both models are at chance on all three tasks. (`results_v3_fixed` for Qwen PGM is incomplete, 187k of 200k; the complete run is
`results_v2`.) Free generation failed to produce parseable answers in two CLEVR-ART configurations (0% = all invalid);
the prefilled variants are the valid numbers.

## 1. LoRA v1: single-pass format: failed (code removed from this repo; original at `~/scratch/vlm_lora`, also in git history)

Input: matrix image + image of the 8 candidates with number tags; the model outputs `Answer: k`.
Trained on 703k problems (an incomplete extraction of the 1.2M training set).

| # | Experiment | Setup | Result |
|---|---|---|---|
| 1 | Smoke test, both models | 3,000 problems, 94 steps; LR 2e-4 cosine, LoRA r=64, batch 32 | val 13.1% / 12.5% |
| 2 | Full training, both models | 2-epoch plan on 703k, ~23k steps each; `lm` loss; candidate shuffling | collapsed to one answer for every problem; val ~12-14% |
| 3 | Overfit test (500 problems x 40 epochs) | Qwen LLM-only; Qwen vision+LLM LoRA; InternVL vision+LLM LoRA | ~12-13% even on the training problems |

Data, labels and training code were checked and are correct; the format gave the model no usable learning signal.

## 2. LoRA v2: per-candidate scoring (`lora/`, `pgmvlm/`)

Each problem becomes 8 completed 3x3 matrices (candidate k in the missing cell, no tags), one sequence each, prompt
"...Is the bottom-right panel the correct completion of this matrix? Answer Yes or No."; score = logit(Yes) - logit(No);
8-way softmax + cross-entropy over the 8 scores (`ce8`). From experiment 6 on: the shared 20% training set (240,000
problems, `tools/make_train20pct.py`), 1 epoch = 15,000 updates at 16 problems/update.

| # | Experiment | Setup | Result |
|---|---|---|---|
| 4 | Overfit test, both models | 500 problems x 20 epochs, LR 2e-4 cosine, r=64 | 100% on the training problems (val ~16%) |
| 5 | H200 speed test | 30 steps | InternVL 10.7, Qwen 6.7 problems/s |
| 6 | Initial regime, 1 epoch | LR 2e-4 cosine -> 2e-5, warmup 200, r=64/alpha=128, batch 32, no weight decay, bf16 scores | val InternVL 18.9%, Qwen 19.8%; gradient spikes while LR > ~1.2e-4 |
| 7 | Initial regime, InternVL epoch 2 | constant 2e-5 (end of the cosine) | val 18.9% -> 20.3% |

## 3. LoRA v2 + teammate recipe (main results)

LR 1e-4 constant (20 warmup updates), LoRA r=16/alpha=32/dropout 0.05 on all language-model linear layers (vision
tower + projector frozen and in eval mode), AdamW, weight decay 0.01, grad clip 1.0, 16 problems/update, fp32 scores.
Per-epoch numbers: `results/lora/lora_pgm_results.csv`; curves: `results/lora/*_curves_*.png`; validation loss of the
saved checkpoints: `results/lora/val_loss_*.json` (computed with `tools/val_loss.py`; it was not logged in training).

| Epoch | LR schedule | Qwen3-VL-2B test (val / val loss) | InternVL3.5-1B test (val / val loss) |
|---|---|---|---|
| 1 | constant 1e-4 | 55.87% (56.7% / 1.153) | 40.23% (40.1% / 1.536) |
| 2 | constant 1e-4 | 68.61% (68.9% / 0.838) | 48.53% (49.6% / 1.309) |
| 3 | constant 1e-4 | 75.33% (73.2% / 0.674) | 52.31% (51.5% / 1.210) |
| 4 | cosine 1e-4 -> 1e-5 | 81.50% (81.6% / 0.475) | 57.73% (57.6% / 1.054) |
| 5 | Qwen: warm restart 1e-5 -> 5e-5 (500 updates), cosine -> 5e-6 | **82.06%** (81.6% / 0.509) | - |
| 5-6 | InternVL: warm restart 1e-5 -> 1e-4 (500 updates), cosine -> 1e-5 over 2 epochs | - | **60.31%** (61.2% / 1.045) |

Convergence: training loss still falls in the last epoch for both models, but validation loss does not: Qwen's rises
(0.475 -> 0.509) with flat validation accuracy, while its test accuracy still gained 0.56 points (81.50% -> 82.06%);
InternVL's is flat (1.054 -> 1.045) while test accuracy gained 2.58 points (57.73% -> 60.31%). Both are close to the
limit of this setup.

Reference points (same 240k training set, same 200k test set): Slot Abstractor paper, 20% data: 71.25%; teammate
LFM2.5-VL-1.6B (single image + letter, 1 epoch): 43.8%; teammate SpatialEvo-3B (0.41 epoch): 30.0%.

## 4. Relational branch (`relational/`, `experiments/`)

Slot Attention -> Abstractor relational branch feeding the VLM (Stage 0 slots, Stage 1a branch alone, alignment,
Stage 2 with LoRA, LoRA-first), same backbones and the same per-candidate readout as section 3. Best result: at 5%
data, LoRA-first with the branch reaches 56.5% on 5,000 test puzzles vs 52.6% for LoRA alone (+3.3 to +3.5 over an
equal-steps control); no gain at full data or on the task mixture. Full write-ups:
[`relational_deep_dives.md`](relational_deep_dives.md) (H1-H13, A1, A2) and
[`relational_experiment_log.md`](relational_experiment_log.md); index and status in [`../00_START_HERE/`](../00_START_HERE).

## Infrastructure notes

- GPU nodes excluded from jobs (`BAD_NODES` in `env.sh`): `atl1-1-03-013-3-0` (CUDA device unavailable),
  `atl1-1-03-018-2-0`, `atl1-1-03-017-30-0`, `atl1-1-03-017-16-0` (thermally throttled to ~1/3 speed).
- The first job of the epoch-5/6 chains ended as TIMEOUT after ~14 h although training stops itself at ~4.75 h: the
  process seems to hang on exit after saving. The chained jobs resumed correctly; worth investigating if it recurs.
