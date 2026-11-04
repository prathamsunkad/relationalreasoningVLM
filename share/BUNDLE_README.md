# PGM LoRA adapters for Qwen3-VL-2B and InternVL3.5-1B (v1)

LoRA adapters that teach two small vision-language models to solve **PGM** (Procedurally Generated Matrices) abstract
reasoning problems, plus everything needed to run them on Georgia Tech **PACE ICE**: the inference code, a one-job
setup, an installation check and a full test-set evaluation.

| Model | Adapter | PGM neutral test accuracy (200,000 problems, chance 12.5%) |
|---|---|---|
| Qwen3-VL-2B-Instruct | `adapters/qwen3vl2b_pgm_ep5` (final) | TEST_QWEN_EP5 (epoch 4: 81.50%) |
| InternVL3.5-1B | `adapters/internvl35_1b_pgm_ep6` (final) | TEST_INTERNVL_EP6 (epoch 4: 57.73%) |

For comparison, the Slot Abstractor paper reports 71.25% with the same 20% of the training data. Every epoch's adapter
is included; per-epoch accuracies and training curves are in `results/`.

## Quick start (4 steps, all from inside this folder)

**0. Copy the folder into your scratch and enter it** (it needs ~12 GB once set up):
```bash
cp -r <shared path>/pgm-lora-v1 ~/scratch/ && cd ~/scratch/pgm-lora-v1
sha256sum -c SHA256SUMS --quiet && echo "copy OK"
```

**1. Install (one job, CPU, ~20-40 min).** Installs the exact Python packages into `env/` and downloads the two base
models at the exact revisions into `models/`. Nothing outside this folder is changed; your own Python packages are
neither used nor touched.
```bash
sbatch setup.sbatch            # log: setup_<jobid>.log, ends with "setup done"
```

**2. Check the installation (one GPU job, ~5 min).** Scores 10 sample problems and compares every score with the
values from the original training environment.
```bash
sbatch check.sbatch            # log: check_<jobid>.log, should end with "ALL CHECKS PASSED"
```

**3. Use it.** One problem (needs a GPU: inside an `salloc --gres=gpu:1 -p ice-gpu` session, or wrap it in sbatch):
```bash
bash run.sh predict sample_problems/<file>.npz              # both models, final adapters
bash run.sh predict my_problem.npz qwen 4                   # one model, a specific epoch's adapter
```
It prints the 8 candidate scores, the model's answer and the correct answer, and saves a figure of the 8 inputs.

Full test-set evaluation (28 GPU jobs in parallel), then the accuracy:
```bash
sbatch --export=ALL,MODEL=qwen,EPOCH=5,DATA_DIR=/path/to/test_full/neutral,OUT_DIR=$PWD/results_qwen_ep5 eval_test.sbatch
bash run.sh merge $PWD/results_qwen_ep5
```
`DATA_DIR` is a folder of PGM `.npz` files (16 panels of 160x160 + `target`), such as the 200,000-problem neutral
test split. The data is not included in this bundle.

## How the models are used (important)

The models never see the 8 answer choices together. For each candidate k (1..8) the model gets **one image: the full
3x3 matrix with candidate k placed in the empty bottom-right cell** (no labels), and the question:

> The image shows a 3x3 abstract reasoning matrix. In a correct matrix, shape type, size, color, count, and position,
> and line type and color, follow consistent rules across rows and columns (constant, progression, XOR, OR, AND, or
> consistent union).
>
> Is the bottom-right panel the correct completion of this matrix? Answer Yes or No.

The score of candidate k is **logit("Yes") - logit("No")** at the first answer position; the answer is the candidate
with the highest score (softmax over the 8 scores = probabilities). `code/pgmvlm/data.py` (`filled_matrices`,
`CandidateCollator`) and `code/pgmvlm/models.py` (`candidate_scores`) implement exactly this; `run.sh predict` uses
them. Free-text generation is not what the adapters were trained for, so always use the scores.

## Contents

| Path | What |
|---|---|
| `run.sh` | the entry point (predict / eval-shard / merge / check); runs inside the shared PACE container |
| `setup.sbatch`, `check.sbatch`, `eval_test.sbatch` | install, verify, full evaluation |
| `requirements-lock.txt` | the exact package versions (torch 2.6.0+cu124, transformers 5.17.0, peft 0.21.2, ...) |
| `adapters/` | LoRA adapters (r=16, alpha=32, language model only): `qwen3vl2b_pgm_ep1..ep5`, `internvl35_1b_pgm_ep1..ep4, ep6` (no separate InternVL epoch-5 checkpoint; epochs 5-6 were one run); `*_bestval` = best validation checkpoint where it differs from the final one |
| `code/` | inference and evaluation code |
| `results/` | per-epoch test/validation accuracy and loss, training curves |
| `sample_problems/` | 10 PGM test problems + the reference scores used by the check |

Base models (downloaded by setup): `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203`,
`OpenGVLab/InternVL3_5-1B-HF` @ `9191dbccf312b537016f041b25d61c72e7c5c9f3`. Container:
`/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif` (Python 3.10.12, numpy 1.22.2).

## Training details

Data: a fixed 20% subset of the PGM neutral training set (240,000 problems); validation on 1,000 official validation
problems. LoRA on all linear layers of the language model (vision tower and projector frozen), AdamW, weight decay
0.01, 16 problems per update, 8-way cross-entropy over the 8 candidate scores. Learning rate: constant 1e-4 for epochs
1-3, cosine 1e-4 -> 1e-5 in epoch 4, then a warm restart (Qwen: 1e-5 -> 5e-5 -> 5e-6 over epoch 5; InternVL:
1e-5 -> 1e-4 -> 1e-5 over epochs 5-6). Details and all numbers: `results/lora_pgm_results.md`.

## Troubleshooting

- **setup fails while downloading**: compute nodes need internet access to pypi.org, download.pytorch.org and
  huggingface.co; re-submitting `setup.sbatch` resumes. It needs ~12 GB of free space in this folder.
- **check fails with different answers / large differences**: compare the versions printed at the top of
  `check_<jobid>.log` with `requirements-lock.txt`; make sure the run used `run.sh` (it ignores your `~/.local`
  packages). Small differences (< 0.3) are normal on a different GPU type and are accepted.
- **"no adapter for ..."**: the epoch does not exist for that model (see `adapters/`).
- **Data licence**: PGM (DeepMind) is for non-commercial research; the adapters are trained on it.
