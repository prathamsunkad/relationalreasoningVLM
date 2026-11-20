# relationalreasoningVLM

Abstract visual reasoning (PGM, CLEVR-ART) with small vision-language models on Georgia Tech PACE ICE: zero-shot
baselines, LoRA fine-tuning with per-candidate scoring, and a relational branch (Slot Attention -> Abstractor ->
connector) feeding the VLM. Models: `Qwen/Qwen3-VL-2B-Instruct` (main) and `OpenGVLab/InternVL3_5-1B-HF`.

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

**Relational branch, Qwen3-VL-2B, 5,000 test problems:**

| Setting | LoRA alone | Best hybrid | Equal-steps control | Notes |
|---|---|---|---|---|
| 5% data (60k puzzles) | 52.6% | **56.5%** (LoRA-first, interleave; fusion 56.4%) | 53.2% / 52.9% | +3.3 / +3.5, p < 1e-11; gain mostly on counting ("number" +14.5); ties averaging LoRA alone with the branch |
| Full data (240k) | 81.6% (val) | 81.3-81.4% (val) | 81.3% (non-relational, val) | No gain |
| Task mixture (30% PGM) | not run | family score 87.2 | 87.6 (non-relational) | No gain; synthetic tasks saturate at 100% |
| 5%, branch trained on the 60k only (early checkpoint) | 52.6% | 53.7% | 53.2% | No gain; the model ignores this weaker branch |

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
| `experiments/` | the exact launchers, hand-off chains and scoring jobs used for H7-H13, A1, A2 |
| `tools/` | data preparation, preflight, plots, validation loss of checkpoints, single-problem demo, `rerun_cmd.py` |
| `slurm/` | baseline training, evaluation (`run_test_eval.sh`: 28 shards, retries, merge), tests, plots |
| `slurm_portable/` | path-free copies of every Slurm job and launcher (paths from `env.sh`); use these on any account |
| `tests/` | CPU tests on tiny random models (`sbatch slurm/tests.sbatch`) |
| `results/` | every run's settings and training log, per-problem test outputs, tables, figures (no weights, no data) |
| `docs/` | experiment write-ups, protocol comparison with the Slot Abstractor paper |
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
