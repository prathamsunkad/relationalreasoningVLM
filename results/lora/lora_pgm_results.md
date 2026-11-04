# LoRA fine-tuning on PGM neutral: results

Setup: per-candidate scoring (8 completed 3x3 matrices per problem, score = logit(Yes) - logit(No), 8-way softmax CE),
LoRA r=16 / alpha=32 / dropout 0.05 on all language-model linear layers (vision tower + projector frozen), AdamW,
weight decay 0.01, 16 problems per update, bf16 model + fp32 LoRA and scores, 1 H200.
Data: shared 20% PGM neutral training set (240,000 problems; 1 epoch = 15,000 updates).
Validation: the 1,000 shared validation problems, every 1,000 updates. Test: full 200,000-problem neutral test set.
LR: constant 1e-4 (20 warmup updates) for epochs 1-3; epoch 4 cosine decay 1e-4 -> 1e-5. Chance = 12.5%, random-guess loss = ln 8 = 2.079.

## Qwen3-VL-2B

| Epoch | LR | Test acc (200k) | 95% CI | Test loss | Val acc (end of epoch) | Best val in epoch | Train loss (epoch mean) | Train loss (last 1k updates) | Train acc (last 1k) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | constant 1e-4 | **55.87%** | 55.65-56.08 | 1.160 | 56.7% | 56.7% (step 15000) | 1.573 | 1.198 | 55.1% |
| 2 | constant 1e-4 | **68.61%** | 68.40-68.81 | 0.844 | 68.9% | 68.9% (step 30000) | 0.986 | 0.853 | 67.9% |
| 3 | constant 1e-4 | **75.33%** | 75.14-75.51 | 0.656 | 73.2% | 74.0% (step 44000) | 0.728 | 0.666 | 74.4% |
| 4 | cosine 1e-4 -> 1e-5 | **81.50%** | 81.33-81.67 | 0.468 | 81.6% | 81.6% (step 60000) | 0.517 | 0.428 | 83.4% |

Curves: `plots/qwen_curves_ep0-4.png`. Adapters: `qwen_pc_recipe_e1/final_e1` ... `final_e4`.

## InternVL3.5-1B

| Epoch | LR | Test acc (200k) | 95% CI | Test loss | Val acc (end of epoch) | Best val in epoch | Train loss (epoch mean) | Train loss (last 1k updates) | Train acc (last 1k) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | constant 1e-4 | **40.23%** | 40.02-40.45 | 1.546 | 40.1% | 40.1% (step 15000) | 1.795 | 1.556 | 39.6% |
| 2 | constant 1e-4 | **48.53%** | 48.31-48.75 | 1.309 | 49.6% | 50.1% (step 29000) | 1.410 | 1.309 | 49.2% |
| 3 | constant 1e-4 | **52.31%** | 52.10-52.53 | 1.205 | 51.5% | 53.0% (step 44000) | 1.230 | 1.193 | 52.7% |
| 4 | cosine 1e-4 -> 1e-5 | **57.73%** | 57.51-57.95 | 1.054 | 57.6% | 57.8% (step 59000) | 1.080 | 1.011 | 59.9% |

Curves: `plots/internvl_curves_ep0-4.png`. Adapters: `ivl1b_pc_recipe_e1/final_e1` ... `final_e4`.

## Reference points (same 240k training set, same 200k test set)

| Model | Test acc |
|---|---|
| Slot Abstractor paper, 20% data (task-specific model, many epochs) | 71.25% |
| Teammate LFM2.5-VL-1.6B, 1 epoch (single image + letter) | 43.8% |
| Teammate SpatialEvo-3B, 0.41 epoch | 30.0% |
| Earlier setup (2e-4 cosine, r=64, batch 32), 1 epoch, both models | ~19-20% (validation) |

