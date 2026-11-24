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

## Full pipeline (`pipeline.sh`)

One command runs the whole recipe as chained 6-hour jobs; every step resumes, and finished steps are skipped.
All settings are in `pipeline_config.sh` (an exported variable overrides a default).

```bash
bash pipeline.sh print        # settings, output folder, which steps are finished
bash pipeline.sh smoke        # interactive GPU, ~15 min: every step for a few steps, in ckpt/pipeline_smoke
rm -r ckpt/pipeline_smoke
bash pipeline.sh submit       # the real run (repeat it any time: it only submits unfinished steps)
bash pipeline.sh status       # step / total, ETA, best held-out loss and accuracy per step
bash pipeline.sh report       # after the eval step: the test table, also in $RUN/report_test.md
SLOT_SOURCE=vit SLOT_LAYER=12 bash pipeline.sh submit    # the same with slots from InternViT layer 12
```

| Step | What trains | Length | Settings |
|---|---|---|---|
| `s0A`, `s0B` | the branch alone, on half A / half B (cross-fitting) | ≤ 300k steps each (~42 epochs of a half, ~7 h) | 24 Abstractor layers, CNN slots (10 × 256, 3 iterations), `qk_norm`, reconstruction 1.0, LR 1e-4 constant after 500 warmup, batch 16; after 12 validations without a new best (or at the end) the LR decays to 0 over 30k steps |
| `base` | LoRA on the 16 crops, no branch | 3 epochs | r16 / α32, LR 1e-4 cosine, 16 puzzles per step |
| `s1` | fusion head only; LLM and `base`'s LoRA frozen | 0.25 epoch | LR 1e-3; each puzzle goes to the branch that did not train on it |
| `s2` | LoRA + fusion head + branch (except its CNN trunk) | 1 epoch | LoRA 1e-4, head 1e-3, branch 3e-5, score-head loss 0.2, **branch dropout 0.5** |
| `ctrl` | `base`'s LoRA, no branch | 1 epoch (= `s2`) | same LoRA settings |
| `eval` | — | first 50k test puzzles | `base`, `ctrl`, `s1`, `s2`, `s2` branch off, `s2` shuffled relational features |

**Why this recipe:**
- **LoRA first, then the branch** (`RECIPE=lorafirst`): the repo's H11/H12 runs did best this way, and trained jointly a strong branch let the LLM take a shortcut (H8). `RECIPE=joint` runs the original order: fresh LoRA trained together with the branch, with `base` as the control.
- **Branch dropout 0.5** switches the relational path off for a random half of the puzzles in stage 2, so the LLM keeps solving on its own (H9).
- **The branch is the deep one** (24 layers), trained long: the depth test ranked 24 > 12 > 6 > 2 and 24 was still improving, and the Slot Abstractor needed ~50 epochs to level off.
- **The same puzzles everywhere.** Every arm trains on the same pool (`TRAIN_LIMIT`, minus the last 10k rows of `train.jsonl`, the held-out set every stage validates on). `ctrl` gets exactly as many LoRA steps as `s2`.
- **Stage 1 stops if a branch learned too little** (held-out accuracy below `MIN_BRANCH_ACC=0.25`): raise `S0_STEPS` and resubmit, or set `MIN_BRANCH_ACC=0`.

**What the result has to show** (`pipeline_report.py`, paired McNemar tests on the same puzzles):
1. `s2` > `ctrl`: it beats LoRA with the same number of steps.
2. `s2` > `s2` shuffled: it uses its own puzzle's relational features, not just extra capacity.
3. `s2` > ensemble (the control's letter scores + the branch's 8-way scores): the fusion does more than averaging.

**Useful variables:**
- `TRAIN_LIMIT=60000`: the 5% scale, where H11/H12 found the gain (it disappeared at full data, H4).
- `SEED=1`: a second seed in its own folder.
- `JOBS_*`: copies per chain. A copy that finds its step finished exits at once.
- `TEST_LIMIT=0`: all 200k test puzzles.
- `USE_CACHE=0`: read the PNG sheets instead of the panel cache.

**New `train_hybrid.py` options** used by the pipeline:
- `--lora_from`: stage 1/2 start from a trained baseline's LoRA.
- `--branch_dropout`: stage 2.
- `--max_hours`: save and exit cleanly before the job limit.
- `--patience` and `--decay_steps`: plateau, then LR decay, then stop.

`run_hybrid.sh` is the earlier single-job version (no LoRA-first, no dropout).

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
bash pipeline.sh submit                                                      # then the full recipe (above)
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

### Depth test (CNN slots, 20k steps)

| Abstractor layers | Best val loss | Val accuracy | Puzzles/s |
|---|---|---|---|
| 2 | 1.9609 | 21.5% | — |
| 6 | 1.9560 | 20.8% | 295 |
| 12 | 1.9428 | 21.1% | 253 |
| 24 | **1.9171** | **23.4%** | 195 |

None collapsed. 24 layers was still improving at 20k steps (about 1.4 epochs), so the pipeline uses 24 layers and
trains much longer.

## Next steps

1. Pick the slot source: CNN (default) or `vit_L12` (`SLOT_SOURCE=vit SLOT_LAYER=12`), from the 24-layer runs.
2. `bash pipeline.sh smoke`, then `bash pipeline.sh submit`.
3. Second seed (`SEED=1`) and the 5% scale (`TRAIN_LIMIT=60000`).

## Outputs (not committed)

`ckpt/`, `pgm_panel_cache/`, `stage0_choice.txt`, `logs/` (see `.gitignore`).
