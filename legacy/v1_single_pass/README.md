# vlm_lora: LoRA fine-tuning + paper-style evaluation of Qwen3-VL and InternVL3.5 on PGM

Fine-tunes a VLM on one PGM regime with LoRA and evaluates it so the numbers line up with the Slot Abstractors paper
(per-regime test accuracy, and the 20% vs 100% data experiment). Same image layout and prompt as your `pgm_infer.py`
(3x3 context grid image, 2x4 labelled candidate grid image, `Answer: <number>`), so zero-shot and fine-tuned numbers are
directly comparable. Models: `Qwen/Qwen3-VL-2B-Instruct` and `OpenGVLab/InternVL3_5-1B-HF` (1.1B: 0.3B InternViT + 0.8B Qwen3).
**`PROTOCOL_COMPARISON.md`: how the paper trained/reported and how this maps onto it.**

## Status: what is verified, what is not
| Verified here (CPU; random-weight stand-ins built from the real transformers 5.17.0 classes; 13 tests) | NOT verified (needs your cluster + real checkpoints) |
|---|---|
| train -> interrupt -> resume -> sharded eval -> merge -> table, both model families | Loading the real Qwen3-VL-2B / InternVL3.5-1B-HF weights (no Hugging Face access from here) |
| Real loading path (AutoConfig / AutoModelForImageTextToText / AutoProcessor) on saved tiny checkpoints | Real tokenizers giving single-token digits after `Answer: ` (the code asserts this at startup and fails loudly) |
| Supervised tokens are exactly `Answer: k<|im_end|>` (vs an independent full-sequence loss) | Real chat templates; `peft` inside your container; torchvision for the video processor |
| Left-padded batches == unpadded logits; training logits == evaluation readout | GPU memory, speed, bf16 on your GPU; every accuracy number |
| LoRA trains only `lora_*`; base weights bit-identical; resume matches an uninterrupted run | `extract_pgm.py` against your real file names (tested only on a fake tarball with `PGM_neutral_train_0.npz` style names) |
| `--train_fraction`, Table-5 output, chained submit script (stub `sbatch`) | The Slurm scripts themselves (never submitted) |

## 0. Prepare the data (you only have test_full so far)
```bash
DATA=/home/hice1/psunkad3/scratch/pgm_data
# dry run first: prints files found per split + example names; confirm the names look right
python tools/extract_pgm.py --tar /path/to/neutral.tar.gz --regime neutral --out_root $DATA --splits train,val --dry_run
python tools/extract_pgm.py --tar /path/to/neutral.tar.gz --regime neutral --out_root $DATA --splits train,val
# (or --src_dir <folder with the extracted files>; add --train_fraction 0.2 if scratch is tight)
```
Result: `$DATA/train_full/neutral`, `$DATA/val_full/neutral` next to your `test_full/neutral`. Extract the **full** train set once;
the 20% run then uses `--train_fraction 0.2` (a random, nested subset chosen at training time).

## 1. Run order
```bash
pip install --user peft                # inside the container's python (tested with peft 0.21.2); transformers must be 5.17.0
python tests/test_pipeline.py          # a few minutes on CPU, no downloads
# ALWAYS first, per model: shows the exact prompt, tokens, memory, speed
python tools/preflight.py --model_id Qwen/Qwen3-VL-2B-Instruct   --npz_dir $DATA/train_full/neutral --batch_size 8
python tools/preflight.py --model_id OpenGVLab/InternVL3_5-1B-HF --npz_dir $DATA/train_full/neutral --batch_size 8
# zero-shot baseline (2 shards ~ 14k problems is enough)
sbatch --array=0-1 --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=qwen_zeroshot,REGIME=neutral,ADAPTER=none slurm/eval_array.sbatch
# a single training run (any size)
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=qwen_neutral_100k,REGIME=neutral,TRAIN_SAMPLES=100000 slurm/train.sbatch
```
Both HF models must be in the offline cache first (`huggingface-cli download <id>` on a login node). Edit the placeholder paths
at the top of the `.sbatch` files.

## 1b. Overfit test (run this before any long training run)
The first full runs (`qwen_neutral_p59_e2`, `ivl1b_neutral_p59_e2`) collapsed: the loss went to ln 8 within ~20 steps and every
validation check scored exactly the frequency of one label, i.e. the model answered the same number for every problem. Before
spending GPU-days again, check that the model can at least memorise a small set:
```bash
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=overfit_qwen slurm/overfit.sbatch
sbatch --export=ALL,MODEL_ID=OpenGVLab/InternVL3_5-1B-HF,TAG=overfit_ivl1b slurm/overfit.sbatch
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=overfit_qwen_llmonly,EXTRA= slurm/overfit.sbatch   # original LLM-only LoRA
grep "\[fit\]\|\[val\]" $RUNS/logs/overfit_*.log
```
`[fit]` = accuracy on the 500 training problems, `[val]` = held-out accuracy, both with the predicted-position histogram
(`COLLAPSED` is printed when one position gets >80% of the predictions). `[fit]` near 100% means the inputs are distinguishable
and the problem is generalisation; `[fit]` stuck near 12.5% means a perception/resolution bottleneck.
Any run can add `--fit_eval_samples N` to get the `[fit]` line.

## 2. The paper's 20% vs 100% experiment (Table 5)
```bash
MODEL_ID=Qwen/Qwen3-VL-2B-Instruct   MODEL_TAG=qwen  REGIME=neutral PARTS=6 bash slurm/submit_table5.sh
MODEL_ID=OpenGVLab/InternVL3_5-1B-HF MODEL_TAG=ivl1b REGIME=neutral PARTS=6 bash slurm/submit_table5.sh
# then, for each result dir:  python eval_vlm.py --merge_dir $RUNS/results/<TAG>_neutral --expected_n 200000
python tools/make_table.py --table5 --row "Qwen3-VL-2B:20=$RUNS/results/qwen_neutral_f20_e1_neutral,100=$RUNS/results/qwen_neutral_f100_e1_neutral"
```
It submits, for 20% and for 100%, `PARTS` chained training jobs (each resumes and stops itself before 5 h) and then the 28-shard
test evaluation. `EPOCHS_20` / `EPOCHS_100` set the passes over each subset (default 1 each). The paper does not say how many
epochs its 20% run used, so also run `EPOCHS_20=5` (same number of optimiser steps as 100% x 1 epoch) to separate "less data" from "fewer steps".
Pick `PARTS` from measured speed: hours = problems x epochs / (samples_per_s x 3600); 100% x 1 epoch = 1.2M problems.
For other regimes repeat with `REGIME=interpolation` etc. (they need their own extracted train/val/test folders).

## 3. Evaluate, merge, tabulate
```bash
sbatch --export=ALL,MODEL_ID=...,TAG=<run>,REGIME=neutral,ADAPTER=best slurm/eval_array.sbatch     # best | last | final | none
python eval_vlm.py --merge_dir $RUNS/results/<run>_neutral --expected_n 200000
python tools/make_table.py --metric logits --row "Qwen3-VL-2B zero-shot:neutral=..." --row "Qwen3-VL-2B LoRA:neutral=...,interp=..."
```

## What it does
* **Data**: your `panels_to_grid_image`, prompt and `extract_answer` verbatim (`pgm_vlm_data.py`). Candidate order is shuffled per sample in training
  (answer remapped); evaluation never shuffles.
* **Loss**: only `Answer: k<|im_end|>` is supervised. `--loss digit8` (default; the paper's 8-way CE over digit logits) | `lm` (all 5 answer tokens, the digit gets 1/5 of the loss) | `both`.
* **LoRA**: all language-model linear layers, r=64, alpha=128, dropout 0.05, fp32 adapters over a bf16 base. `--lora_vision`, `--lora_projector` extend it.
* **Two readouts**: `logits` (next-token logits restricted to the 8 digit tokens = the paper's choice among 8; headline) and `generate` (free greedy text,
  exactly your script, with invalid-output rate).
* **Resumable** training (data order is a function of seed and position; optimiser state saved), `last/ best/ final/` adapters, `--time_limit_min`.
* **Sharded eval** with your sorted-file sharding; per-shard resume; merge gives accuracy + 95% Wilson interval, prediction/target histograms.
* **Sanity**: `--context_mode blank|shuffle` at eval (should fall to ~12.5%).

## Model specifics (transformers 5.17.0 source + model cards)
| | Qwen3-VL-2B-Instruct | InternVL3.5-1B-HF |
|---|---|---|
| Loading | `AutoModelForImageTextToText`, no remote code | same (native `InternVLForConditionalGeneration`; the card's `trust_remote_code=True` is not needed; use the **-HF** repo) |
| Size | ~2B | 1.1B (0.3B vision, 0.8B Qwen3 LLM, hidden 1024, 28 layers) |
| Vision tokens for our 2 images | 256 + 242 = 498 | `--crop_to_patches 1` (default) 1024; `0`: 512 (candidate grid squashed to 448x448) |
| Extra input | `mm_token_type_ids` (passed through) | none |
| LoRA names | `language_model.layers.N.{q,k,v,o,gate,up,down}_proj`; vision `visual.blocks.N.{attn.qkv,attn.proj,mlp.linear_fc1/2}` | same LLM names; vision `vision_tower.encoder.layer.N.{attention.q/k/v_proj, attention.projection_layer, mlp.fc1/2}` |
InternVL's processor forces `crop_to_patches=True` (overriding the checkpoint's False), so the collator passes it explicitly; keep it identical for training and evaluation.

## What is complete and what is not
**Done:** data extraction helper; LoRA SFT for both models; two readouts; sharded/resumable eval; per-regime and Table-5 tables with the paper's rows;
20%/100% submit script; preflight; tests.

**Not done / limitations (be aware):**
1. Nothing has run on the real checkpoints or on a GPU (see the status table); first real run may need small fixes.
2. Single GPU only. The paper used 4 GPUs at global batch 96; no DDP/DeepSpeed here, and no flash-attn path is tested.
3. The abstractor / slot-abstractor is **not** in this repo. This is the LoRA-only baseline; Stage 1b (injecting per-candidate relation tokens into the VLM) is still unwritten.
4. PGM only. No V-PROM, RAVEN, ART, SVRT, CLEVR-ART.
5. No automatic sweep over the 8 regimes: you extract each regime's data and run `submit_table5.sh` / `train.sbatch` per regime.
6. Hyperparameters (lr 2e-4, r 64, batch 32, 1 epoch) are my reasonable defaults, not tuned; no sweep. Budgets are far below the paper's 50-115 epochs, so absolute numbers are not compute-matched.
7. Checkpoint selection uses 1000 validation problems (about +-1.5 points of noise); report `final` as well as `best`. The paper's PGM selection rule is not stated.
8. No seed aggregation tool: run seeds yourself (`--seed`); the table's interval is test-sampling error only. LoRA dropout > 0 makes a resumed run not bit-identical.
9. No per-rule/meta-label breakdown (`meta_target` is not read), no auxiliary losses (deliberately, to match the paper).
10. Labels on the candidate grid use `DejaVuSans-Bold.ttf` if present, else Pillow's default font (same as your `pgm_infer.py`); train and evaluate in the same environment so the tags look identical.
11. No vLLM / fast generation; the 200K-problem test run is 28 array tasks of ~7K problems each; measure the time on a shard first.
12. Slurm scripts use placeholder paths and have never been submitted.

## Files
`pgm_vlm_data.py` | `model_utils.py` | `train_lora.py` | `eval_vlm.py` | `evalcore.py` | `tools/{extract_pgm,preflight,make_table,check_data,visualize_input}.py` | `slurm/{train.sbatch,overfit.sbatch,eval_array.sbatch,submit_table5.sh}` | `tests/` | `PROTOCOL_COMPARISON.md`
