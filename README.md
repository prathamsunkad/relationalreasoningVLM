# pgm-vlm-reasoning

Abstract visual reasoning (PGM, CLEVR-ART) with small vision-language models on Georgia Tech PACE ICE:
zero-shot baselines, LoRA fine-tuning with per-candidate scoring, and a relational (Slot Attention -> Abstractor)
branch in progress. Models: `Qwen/Qwen3-VL-2B-Instruct` and `OpenGVLab/InternVL3_5-1B-HF`.

## Results (PGM neutral, 200,000 test problems, chance 12.5%)

| Model | Zero-shot | LoRA, 4 epochs (last with cosine LR decay) | After a warm-restart cycle |
|---|---|---|---|
| Qwen3-VL-2B | 12.85% | 81.50% | **82.06%** (epoch 5) |
| InternVL3.5-1B | 13.75% | 57.73% | **60.31%** (epoch 6) |

Slot Abstractor paper with the same 20% of the training data: 71.25%. Full log of every experiment, including the
approaches that failed: [`docs/experiments.md`](docs/experiments.md). Tables and curves: [`results/`](results/).

## Layout

| Path | What |
|---|---|
| `pgmvlm/` | shared core: PGM loading and rendering (`data.py`), model loading, LoRA and Yes/No candidate scoring (`models.py`), evaluation loop (`evalcore.py`), LR schedules (`common.py`) |
| `lora/` | `train_lora.py` (resumable LoRA training), `eval_vlm.py` (sharded test evaluation + merge) |
| `tools/` | data preparation, preflight, plots (`plot_curves.py`), validation loss of checkpoints (`val_loss.py`), single-problem demo (`demo_predict.py`), tables |
| `slurm/` | training, evaluation (`run_test_eval.sh`: 28 shards, retries, merge), tests |
| `zeroshot/` | zero-shot inference for PGM and CLEVR-ART + their job scripts |
| `relational/` | relational branch (work in progress; self-contained, run from that folder) |
| `tests/` | CPU tests on tiny random models (`sbatch slurm/tests.sbatch`) |
| `results/` | result tables, plots, validation-loss JSON (no weights, no data) |
| `docs/` | experiment log, protocol comparison with the Slot Abstractor paper |
| `legacy/v1_single_pass/` | the first LoRA approach, kept as the documented negative result |
| `share/` | building the adapter bundle shared with collaborators |

Data, model weights and run outputs are not in git. `env.sh` holds every path (data, runs, container, HF cache),
the pinned base-model revisions and the GPU nodes to avoid.

## Quickstart (PACE ICE)

```bash
cd ~/scratch/pgm-vlm-reasoning && source env.sh
sbatch slurm/tests.sbatch                                   # 22 CPU tests, ~10 min
# one test problem through a fine-tuned adapter (prints the 8 candidate scores)
apptainer exec --nv $CONTAINER python tools/demo_predict.py --npz_dir $DATA/test_full/neutral --seed 7 \
    --model "Qwen=Qwen/Qwen3-VL-2B-Instruct@$RUNS/qwen_pc_recipe_e1/final_e4" --out_dir /tmp/demo
# full 200k test evaluation of an adapter (runs on the login node, submits and waits on Slurm jobs)
bash slurm/run_test_eval.sh Qwen/Qwen3-VL-2B-Instruct qwen_pc_recipe_e1 final_e4 _check
# training (resumable; chain jobs with --dependency=afterany)
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=my_run,REGIME=neutral,TRAIN_SAMPLES=0,EPOCHS=1 slurm/train.sbatch
```

Environment: the PACE container `pytorch-2.1.0.sif` plus the packages in `requirements.txt`
(torch 2.6, transformers 5.17, peft 0.21.2).

## How the fine-tuned models are used

They never see the 8 options at once. For each candidate k, the model gets the full 3x3 matrix with candidate k in the
missing cell and the question "Is the bottom-right panel the correct completion of this matrix? Answer Yes or No."
The score is logit(Yes) - logit(No); the answer is the candidate with the highest score. Free-text generation is not
what the adapters were trained for (only the Yes-vs-No logit difference is), so always use the scores.
