# Exp1: gated relational hybrid at InternVL's projector

A relational branch (slot attention, then an Abstractor over each candidate-completed grid) is added to
InternVL3.5-1B's projector tokens through a gated residual:

    v' = v + sigmoid(W [LN v; LN u] + b) ⊙ u

The up-projection of u starts at zero, so training starts from the unchanged model. Input: the 16 panels cropped
from the contact sheet (`probes/layout.json`); answer: the letter A–H.

## Stages

- **Stage 0:** the branch alone, plus a slot reconstruction loss.
- **Stage 1:** alignment, with the LLM frozen.
- **Stage 2:** LoRA plus the branch.
- **Baseline:** LoRA on the same crops for the same number of steps.

Two branches are trained on halves A and B of the data (cross-fitting), so no puzzle is scored by a branch trained
on it.

## Setup

Paths come from the repo root's `env_local.sh`: `SHEET_DATA`, `CONDA_ENV`, `PANEL_CACHE`, and `PROBES` / `LAYOUT`
(`pgm_gap.py` and `layout.json` live in `probes/`). Run everything from this folder: `cd experiments/exp1_gated_hybrid; mkdir -p logs`.

## Stage-0 search (successive halving)

There are 7 slot sources, numbered by variant index:

| Index | Variant | Slot source |
|---|---|---|
| 0 | `vit_L0` | InternViT layer 0 (patch embeddings) |
| 1 | `vit_L6` | InternViT layer 6 |
| 2 | `vit_L12` | InternViT layer 12 |
| 3 | `vit_L18` | InternViT layer 18 |
| 4 | `vit_last` | InternViT last layer |
| 5 | `cnn` | a CNN |
| 6 | `resnet` | a ResNet |

The learning rate is flat after warmup, so a rung is the same job run to a larger `--max_steps`; it resumes from
its checkpoint.

```bash
sbatch build_cache.sbatch                                                    # once (CPU), panel cache
for i in 0 1 2 3 4 5 6; do sbatch -J sh_$i stage0_sh.sbatch $i 20000; done   # rung 1
python rank_stage0.py ckpt/sh_v2/seed0 --keep 3                              # who advances
for i in 1 3 4; do sbatch -J sh_$i stage0_sh.sbatch $i 60000; done           # rung 2 (continues)
sbatch -J sh_3_s1 stage0_sh.sbatch 3 60000 1                                 # second seed of a finalist
python rank_stage0.py ckpt/sh_v2/seed0 --choose                              # writes stage0_choice.txt
bash run_hybrid.sh all                                                       # stage 0 halves A/B, 1, 2, baseline, eval
```

**Settings in `stage0_sh.sbatch`:**
- `QK_NORM=1` (the collapse fix);
- `RECON=1.0`;
- `LR=1e-4`;
- validation = the last 10,000 rows of `train.jsonl`.

**Other scripts:**
- `collapse_check.sbatch` and `lr_check.sbatch`: the 6,000-step CNN diagnostics that chose those settings.
- `stage0_ablation.sbatch`, `run_stage0_ablation.sh`: the earlier fixed-budget ablation.

Training logs (`ckpt/.../train_log.jsonl`) record:
- `ce` and `acc`;
- `spread`: 0 means all 8 candidates got the same score (collapse);
- `attn_max` and `x_norm`.

## Results so far (stage 0, branch alone, no LLM)

### Collapse and its fix

| Run | Outcome |
|---|---|
| First fixed-budget ablation (60k puzzles, 2 epochs, no fix) | all 7 slot sources near chance: L6 16.2, L18 15.9, last 14.4, L0 13.1, L12 13.1, resnet 13.0, cnn 12.4% |
| Overfit test (512 puzzles) | loss plateaus ~1,500 steps, then memorises; validation stays at chance, so the branch can fit but did not generalise |
| Rung 1 of successive halving, no fix | 5 of 7 variants collapsed (`spread` = 0); best L18 17.3%, L12 16.4% |
| `collapse_check` (CNN, 6k steps) | `qk_norm` (LayerNorm before the Abstractor's attention) removes the collapse; reconstruction weight 0.1 brings it back |
| `lr_check` (CNN, 6k steps) | learning rate 1e-4 kept; lower rates did not help further |

Settings since then: `qk_norm` on, reconstruction weight 1.0, LR 1e-4 constant after 500 warmup steps.

### Rung 1 with the fix

20k steps; held-out set = the last 10,000 rows of `train.jsonl`; chance loss is ln 8 = 2.079.

| Rank | Slot source | Best val loss | Val accuracy |
|---|---|---|---|
| 1 | InternViT layer 18 | **1.9231** | 21.6% |
| 2 | InternViT last layer | 1.9281 | 21.6% |
| 3 | InternViT layer 12 (timed out at 15,850 steps) | 1.9287 | 20.8% |
| 4 | InternViT layer 6 | 1.9328 | 21.1% |
| 5 | ResNet | 1.9395 | 20.4% |
| 6 | CNN | 1.9609 | 21.5% |
| 7 | InternViT layer 0 (patch embeddings) | 1.9893 | 20.4% |

Accuracy intervals overlap (about ±0.8 points at n = 10,000), so the ranking is by loss.

- **Self-attention does not seem to dilute object information.** Layer 0, before any self-attention, is the worst
  source, and deeper layers are better. Tentative: one seed, short budget.
- **The order may still change.** The ViT variants level off after 12.5–15k steps; the CNN, ResNet and layer 0 are
  still improving.
- **All sources sit near 21%.** With the source making little difference, the Abstractor itself (2 layers, vs 24 in
  the Slot Abstractor) may be the bottleneck.

**Running now:** rung 2, variants 1–6 continued to 60k steps.

## Next steps

1. Rank rung 2 (`python rank_stage0.py ckpt/sh_v2/seed0 --keep 2`), then run a second seed of the 2 finalists.
2. If all sources stay near 21%, test a deeper Abstractor before choosing.
3. Run the final stage 0 on halves A/B with the chosen source, then stage 1, stage 2 and the matched LoRA-on-crops
   baseline.
4. Evaluate on test with the controls: branch off, and slot tokens shuffled across candidates.

## Outputs (not committed)

`ckpt/`, `pgm_panel_cache/`, `stage0_choice.txt`, `logs/` (see `.gitignore`).
