# relationalreasoningVLM

Abstract visual reasoning (PGM, CLEVR-ART, I-RAVEN) with small vision-language models on Georgia Tech PACE ICE:
zero-shot baselines, LoRA fine-tuning, relational modules (Slot Attention -> Abstractor -> connector, LFM probes, RCA
plugin) added to the VLM, a standalone Slot Abstractor baseline, and perception-vs-reasoning probes. Models:
`Qwen/Qwen3-VL-2B-Instruct`, `OpenGVLab/InternVL3_5-1B-HF` (candidate scoring); `LiquidAI/LFM2.5-VL-1.6B`, SmolVLM2,
SpatialEvo (answer-letter prediction).

**New here? Start with [`00_START_HERE/`](00_START_HERE):** where everything is, every experiment with its runs,
code and results ([`EXPERIMENTS.md`](00_START_HERE/EXPERIMENTS.md)), data ([`DATA.md`](00_START_HERE/DATA.md)), every setting and assumption
([`CONDITIONS_AND_ASSUMPTIONS.md`](00_START_HERE/CONDITIONS_AND_ASSUMPTIONS.md)), saved models
([`CHECKPOINTS.md`](00_START_HERE/CHECKPOINTS.md)), current status ([`STATUS.md`](00_START_HERE/STATUS.md))
and how to reproduce it ([`REPRODUCE.md`](00_START_HERE/REPRODUCE.md)).

## Results (PGM neutral, chance 12.5%)

**Baselines, 200,000 test problems, 240k training puzzles (20% of PGM neutral):**

| Model | Zero-shot | LoRA, 4 epochs (last with cosine LR decay) | After a warm-restart cycle |
|---|---|---|---|
| Qwen3-VL-2B | 12.85% | 81.50% | **82.06%** (epoch 5) |
| InternVL3.5-1B | 13.75% | 57.73% | **60.31%** (epoch 6) |

Slot Abstractor paper with the same 20% of the training data: 71.25%.

Zero-shot CLEVR-ART (out-of-distribution test, 10,000 puzzles each, reply prefilled with "Answer:"): RMTS 48.6% (Qwen) /
49.1% (InternVL), chance 50%; identity rules 21.6% / 25.0%, chance 25%. Both models are at chance.

**Relational branch, Qwen3-VL-2B, 5,000 test problems:**

| Setting | LoRA alone | Best hybrid | Equal-steps control | Notes |
|---|---|---|---|---|
| 5% data (60k puzzles) | 52.6% | **56.5%** (LoRA-first, interleave; fusion 56.4%) | 53.2% / 52.9% | +3.3 / +3.5, p < 1e-11; gain mostly on counting ("number" +14.5); ties averaging LoRA alone with the branch |
| Full data (240k) | 81.6% (val) | 81.3-81.4% (val) | 81.3% (non-relational, val) | No gain |
| Task mixture (30% PGM) | not run | family score 87.2 | 87.6 (non-relational) | No gain; synthetic tasks saturate at 100% |
| 5%, branch trained on the 60k only (early checkpoint) | 52.6% | 53.7% | 53.2% | No gain; the model ignores this weaker branch |

**Do the hybrids use the relation tokens? (A3, 5,000 test problems)** Zeroing them costs the best hybrid 3.9 points and
shuffling them across candidates 8.8 (56.4% → 52.5% / 47.6%, p < 1e-15); full-data hybrids are unaffected (±0.7, they
ignore the branch); frozen-LLM models fall to 14.6%. [`report.md`](results/relational/ablation_test5000/report.md)

**Answer-letter LoRA, LFM2.5 / SmolVLM2 / SpatialEvo** (one contact-sheet image per puzzle; details and records in
[`results/vlm/README.md`](results/vlm/README.md)):

| Model | CLEVR-ART RMTS | CLEVR-ART identity rules | PGM neutral (200k test) |
|---|---|---|---|
| LFM2.5-VL-1.6B | 99.39% | 93.07% | **76.69%** (checkpoint at epoch 6.73) |
| SmolVLM2 | 89.05% | 68.54% | not finished |
| SpatialEvo | 100.00% | 95.79% | 61.87% (6 epochs) |

LFM relational probes (five insertion points, RCA plugin), frozen LLM or with LoRA: 12.1-13.6% on PGM validation, i.e.
at chance; follow-up diagnostics (depth, readout, gradients) are in [`results/vlm/`](results/vlm/README.md) and
[`docs/VLM_INTEGRATION.md`](docs/VLM_INTEGRATION.md).

**Other baselines and transfer:**

| What | Result | Where |
|---|---|---|
| Slot Abstractor trained from scratch on the same 240k puzzles | 62.8% PGM validation (epoch 64); test not run yet; paper 71.25% | [`slot_abstractor/`](slot_abstractor) |
| LFM2.5 PGM adapter evaluated zero-shot on I-RAVEN (14,000 puzzles) | 23.1% (chance 12.5%) | [`iraven_eval/`](iraven_eval) |
| Base Qwen3-VL-2B / InternVL3.5-1B, no adapter, scored with the LoRA runs per-candidate Yes/No protocol (PGM neutral test; partial: 128,574 of 200,000) | 14.57% / 13.74% (chance 12.5%) | [`pgm_zeroshot_pc/`](pgm_zeroshot_pc) |
| Letter-answer LoRA from one contact-sheet image, 240k, 3 epochs | Qwen3-VL-2B 75.33%, InternVL3.5-1B 52.31% (200k test) | [`probes/README.md`](probes/README.md) |
| Perception pilot: find the answer panel among the 8 choices, no rule needed (Qwen, zero-shot, 50 puzzles) | 78% from separate panels vs 10% from the contact sheet | [`probes/`](probes) |
| Exp1: gated relational branch at InternVL's projector, branch alone (stage 0, 7 slot sources) | 20.4–21.6% on 10,000 held-out puzzles; InternViT layer 18 best by loss | [`experiments/exp1_gated_hybrid/`](experiments/exp1_gated_hybrid) |
| Exp2: the Slot Abstractor's tokens interleaved into fine-tuned InternVL | not run yet | [`experiments/exp2_sa_interleave/`](experiments/exp2_sa_interleave) |

Figures: [`results/plots/`](results/plots). Full write-ups: [`docs/relational_deep_dives.md`](docs/relational_deep_dives.md)
(one section per experiment, H1-H13, A1, A2), [`docs/experiments.md`](docs/experiments.md) (zero-shot and LoRA, including
the approaches that failed).

## Layout

| Path | What |
|---|---|
| `00_START_HERE/` | the guide: map, experiment index, data, conditions and assumptions, status, reproduction |
| `pgmvlm/` | shared core: PGM loading and rendering (`data.py`), model loading, LoRA and Yes/No candidate scoring (`models.py`), evaluation loop (`evalcore.py`), LR schedules (`common.py`) |
| `lora/` | `train_lora.py` (resumable LoRA training), `eval_vlm.py` (sharded test evaluation + merge) |
| `zeroshot/` | zero-shot inference for PGM and CLEVR-ART + their job scripts |
| `relational/` | relational branch: slots, Abstractor, interleave / fusion wrapper, Stage 0-2 training, evaluation, dataset builders, Slurm scripts, tests (self-contained; run from that folder) |
| `experiments/` | the exact launchers, hand-off chains and scoring jobs used for H7-H13, A1, A2, A3 (`auto/`, `diag_*`); Exp1 gated hybrid at InternVL's projector (`exp1_gated_hybrid/`); Exp2 Slot Abstractor tokens interleaved into InternVL (`exp2_sa_interleave/`) |
| `slot_abstractor/` | the standalone Slot Abstractor baseline on PGM (slot autoencoder pretraining, training, test, run logs, results) |
| `probes/` | perception-vs-reasoning probes: attribute reading, rule identification, the same puzzle asked as matching and as reasoning |
| `iraven_eval/` | I-RAVEN test set build and evaluation of the PGM-trained LFM2.5 adapter |
| `configs/vlm/` | portable recipes for the answer-letter models (LFM2.5, SmolVLM2, SpatialEvo) and the LFM relational probes |
| `tools/` | data preparation, preflight, plots, validation loss of checkpoints, single-problem demo, `rerun_cmd.py` |
| `slurm/` | baseline training, evaluation (`run_test_eval.sh`: 28 shards, retries, merge), tests, plots |
| `slurm_portable/` | path-free copies of every Slurm job and launcher (paths from `env.sh`); use these on any account |
| `tests/` | CPU tests on tiny random models (`sbatch slurm/tests.sbatch`) |
| `results/` | every run's settings and training log, per-problem test outputs, tables, figures (no weights, no data); `results/vlm/` holds the answer-letter and LFM probe records |
| `docs/` | experiment write-ups, protocol comparison with the Slot Abstractor paper, the answer-letter / LFM integration guide (`VLM_INTEGRATION.md`), checkpoint recovery |
| `share/` | building the adapter bundle shared with collaborators |
| `legacy/` | the original LoRA code that produced the LoRA results |
| `data_manifests/` | manifests of the mixture datasets (no images) |

Data, model weights and run outputs are not in git; every checkpoint's path on PACE and how to load it is in
[`00_START_HERE/CHECKPOINTS.md`](00_START_HERE/CHECKPOINTS.md). `env.sh` holds every path (shared data and checkpoints,
your own outputs, container, HF cache), the pinned base-model revisions and the GPU nodes to avoid.

## Quickstart (PACE ICE)

```bash
cd relationalreasoningVLM && source env.sh
sbatch slurm_portable/slurm/tests.sbatch                    # CPU tests, ~10 min (slurm_portable/: path-free copies)
# one test problem through a fine-tuned adapter (prints the 8 candidate scores)
apptainer exec --nv $CONTAINER python tools/demo_predict.py --npz_dir $DATA/test_full/neutral --seed 7 \
    --model "Qwen=Qwen/Qwen3-VL-2B-Instruct@$SHARED_ROOT/vlm_lora_pc_runs/qwen_pc_recipe_e1/final_e4" --out_dir /tmp/demo
# full 200k test evaluation of an adapter (runs on the login node, submits and waits on Slurm jobs)
bash slurm_portable/slurm/run_test_eval.sh Qwen/Qwen3-VL-2B-Instruct qwen_pc_recipe_e1 final_e4 _check
# training (resumable; chain jobs with --dependency=afterany)
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=my_run,REGIME=neutral,TRAIN_SAMPLES=0,EPOCHS=1 slurm_portable/slurm/train.sbatch
```

Environment: the PACE container `pytorch-2.1.0.sif` plus the packages in `requirements.txt`
(torch 2.6, transformers 5.17, peft 0.21.2).

## How the fine-tuned models are used

They never see the 8 options at once. For each candidate k, the model gets the full 3x3 matrix with candidate k in the
missing cell and the question "Is the bottom-right panel the correct completion of this matrix? Answer Yes or No."
The score is logit(Yes) - logit(No); the answer is the candidate with the highest score. Free-text generation is not
what the adapters were trained for (only the Yes-vs-No logit difference is), so always use the scores.

## Additional VLM protocols and LFM architectures

The existing training and evaluation commands now also support answer-letter prediction for LFM2.5, SmolVLM2 and SpatialEvo, plus the original five LFM probes, RCA plugin and relational memory. Start with [the integration guide](docs/VLM_INTEGRATION.md), [portable recipes](configs/vlm), and [historical results](results/vlm/README.md). Candidate scoring remains the default.
