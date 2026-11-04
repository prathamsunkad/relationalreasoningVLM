# vlm_lora_pc: per-candidate LoRA fine-tuning of Qwen3-VL / InternVL3.5 on PGM

Version 2 of `../vlm_lora`, which is left untouched. Models: `Qwen/Qwen3-VL-2B-Instruct` and `OpenGVLab/InternVL3_5-1B-HF`.
Runs go to `~/scratch/vlm_lora_pc_runs` (v1 used `vlm_lora_runs`).

## Why a v2
In v1 the model saw one image of the matrix and one image of the 8 tagged candidates, and had to output `Answer: k`.
Every v1 run collapsed: the loss went to ln 8 within ~20 steps, and the model gave the same answer to every problem.
The overfit test failed for both models, LLM-only and with vision LoRA: after 40 epochs on 500 problems, accuracy on
those same problems was still ~12.5%. The data, the labels and the training code were checked and are correct. The
single-pass "find the tagged panel" formulation gave no usable learning signal.

## What v2 does (the Slot Abstractor / WReN formulation)
For every problem the model gets **8 sequences**, one per answer choice k:
* **image**: the full 3x3 matrix, with the 8 context panels and **candidate k in the bottom-right cell** (no number tags)
* **prompt**: "...Is the bottom-right panel the correct completion of this matrix? Answer Yes or No."
* **score(k) = logit("Yes") − logit("No")** at the first assistant position. This uses the model's own LM head, so there
  are no new parameters.
* **loss** `--loss ce8` (default): softmax over the 8 scores + cross-entropy with the correct choice (the paper's loss).
  `bce`: each candidate on its own (Yes for the answer, No for the others, answer weighted ×7). `both`: the sum of the two.
* **prediction**: argmax of the 8 scores. It is always a valid choice, and chance is 12.5%.

Each sequence asks one local question ("is this pattern consistent?"), and the 8 near-identical matrices give the
model a direct contrast between the right answer and the wrong ones.

**Cost**: 8 sequences per problem, each with one image instead of two (Qwen: ~256 image tokens + ~80 text tokens).
That is roughly 4× the compute per problem of v1. `--batch_size` counts **problems** (default 2 = 16 sequences);
`--grad_accum 16` gives a global batch of 32 problems.

## Training recipe (adopted from the teammate LoRA runs, 2026-10-17)
Defaults in `train_lora.py` follow the recipe that converged on PGM for the teammate (LFM2.5-VL-1.6B: 43.8% test after one
epoch on the same 240k set; SpatialEvo-3B: 30.0% after 0.41 epoch). Their *format* differs (one contact-sheet image + answer
letter); only the optimisation recipe is taken over here, per-candidate scoring stays.

| | default now | first v2 run (2026-10-16) |
|---|---|---|
| LR | 1e-4, **constant** after 20 warmup updates (`--lr_schedule constant`) | 2e-4 cosine to 2e-5, 200 warmup |
| LoRA | r=16, alpha=32, dropout 0.05, LLM linear layers only | r=64, alpha=128 |
| weight decay / grad clip | 0.01 / 1.0 | 0 / 1.0 |
| batch | 16 problems per update (2 x 8 accumulation) = 15,000 updates per 240k epoch | 32 (7,500 updates) |
| scores | logit(Yes) - logit(No) computed in **fp32** from the final hidden state | from bf16 logits (near-equal scores tied) |
| frozen vision tower / projector | **eval mode** during training (`set_train_mode`) | train mode |
| validation | the 1,000 shared validation problems (`--val_list`, used by `slurm/train.sbatch` automatically) every 1,000 updates | 1,000 random val problems every 500 |

Why: in the first v2 run both models had gradient spikes (up to 100+) and lost their progress while the LR was above ~1.2e-4,
and only improved steadily once the cosine took it lower (InternVL 18.9%, Qwen 19.8% after one epoch).

## Changes vs v1
| | v1 | v2 |
|---|---|---|
| model input per problem | 1 sequence, 2 images (matrix + tagged candidates) | 8 sequences, 1 image each (completed matrix, no tags) |
| output read | digit token after `Answer: ` | logit(Yes) − logit(No) |
| loss | `lm` / `digit8` | `ce8` / `bce` / `both` |
| readouts | restricted logits + free generation | candidate scores only |
| new flags | | `--panel_px` (draw panels larger = more visual tokens per panel) |
| kept | | `--fit_eval_samples`, collapse warning, `--lora_vision/--lora_projector`, resume, sharded eval, Table-5 tools |
| result keys | `logits` / `generate` | `scores` (`acc`, `pred_hist`, `answer_says_yes`, `distractor_says_yes`) |

## Run order
```bash
cd ~/scratch/vlm_lora_pc
sbatch slurm/tests.sbatch                                     # CPU tests with tiny random models, ~10 min
# preflight per model: prints the input, zero-shot scores, memory and speed (1 GPU, a few minutes)
python tools/preflight.py --model_id Qwen/Qwen3-VL-2B-Instruct --npz_dir $DATA/train_full/neutral --batch_size 2
# overfit test FIRST: 500 problems x 20 epochs; [fit] must clearly climb above 12.5%
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=overfit_pc_qwen slurm/overfit.sbatch
sbatch --export=ALL,MODEL_ID=OpenGVLab/InternVL3_5-1B-HF,TAG=overfit_pc_ivl1b slurm/overfit.sbatch
grep "\[fit\]\|\[val\]" ~/scratch/vlm_lora_pc_runs/logs/overfit_*.log
# zero-shot baseline with the same Yes/No readout
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=qwen_pc_zeroshot,REGIME=neutral,ADAPTER=none slurm/eval_array.sbatch
# a real run
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=qwen_pc_neutral_100k,REGIME=neutral,TRAIN_SAMPLES=100000 slurm/train.sbatch
sbatch --export=ALL,MODEL_ID=...,TAG=qwen_pc_neutral_100k,REGIME=neutral,ADAPTER=best slurm/eval_array.sbatch
python eval_vlm.py --merge_dir ~/scratch/vlm_lora_pc_runs/results/qwen_pc_neutral_100k_neutral --expected_n 200000
```
If the overfit test still stays at chance, try `EXTRA="--panel_px 224"` (more visual tokens per panel) and
`EXTRA="--lora_vision --lora_projector"`. Training and evaluation must use the same `--panel_px` and `--crop_to_patches`.
Sanity check after a real run: `EXTRA="--context_mode blank --max_examples 2000"` on the eval script should fall to ~12.5%.

## Data notes
* Panels are decoded with `reshape(16,160,160)`. The files declare (160,160,16), but the bytes are panel-major; a
  transpose scrambles the panels. `to_panels` enforces this and the tests check it.
* The extracted training set is incomplete: 703,581 of 1.2M problems. The extraction stopped on a disk-quota error.
  Free space and rerun `tools/extract_pgm.py` (it skips files that already exist) before any 100% run.

## Files
`pgm_vlm_data.py` (filled matrices, `CandidateCollator`) | `model_utils.py` (`ScoreTokens`, `candidate_scores`, `candidate_losses`) |
`train_lora.py` | `eval_vlm.py` | `evalcore.py` | `tools/{extract_pgm,preflight,make_table,check_data,visualize_input}.py` |
`slurm/{tests,overfit,train,eval_array}.sbatch`, `slurm/submit_table5.sh` | `tests/` | `PROTOCOL_COMPARISON.md`
