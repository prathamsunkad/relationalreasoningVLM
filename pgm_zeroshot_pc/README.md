# pgm_zeroshot_pc: zero-shot Qwen3-VL-2B and InternVL3.5-1B on the PGM neutral test set, per-candidate protocol

This folder scores the **base models, with no LoRA adapter**, on the 200,000-problem PGM neutral test set. It uses
the same input, output and scoring as the LoRA runs in the `pgm-lora-v1` bundle (`code/lora/eval_vlm.py`), so these
numbers are the zero-shot starting point for the 82.06% (Qwen, epoch 5) and 60.31% (InternVL, epoch 6) LoRA results.

The existing zero-shot numbers in `zeroshot/` and `results/zeroshot/` (12.85% / 13.75%) use a different protocol:
one context-grid image plus a 2x4 candidate image, with the model generating `Answer: <n>`. This folder uses the
per-candidate Yes/No readout the LoRA adapters were trained and tested with.

**Status: partial.** 18 of 28 shards (128,574 of 200,000 problems) are done; the other 10 are queued on ICE. When
they finish, `jobs/merge_zeroshot.sbatch` merges all 28 with the bundle's own `--merge_dir --expected_n 200000`.
This folder is then updated with the full records.

## Results so far (128,574 problems, shards 0-17)

| Model (zero-shot, no adapter) | Problems | Accuracy | 95% Wilson interval | Answer scored Yes > No | Distractors scored Yes > No | LoRA (bundle, 200k test) |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct (`89644892`) | 128,574 | 14.57% | 14.37-14.76% | 92.3% | 91.4% | 82.06% (ep5) |
| InternVL3.5-1B-HF (`9191dbcc`) | 128,574 | 13.74% | 13.55-13.92% | 82.1% | 82.2% | 60.31% (ep6) |
| Chance | | 12.5% | | | | |

- Both base models are barely above chance. They score Yes on almost every candidate: the correct one and the 7
  distractors get a positive score about equally often. So the pick depends on small score differences.
- The predicted positions are spread evenly, so there is no collapse onto one answer slot
  (`pred_hist` in `summary.json`).
- 0 errors and 0 duplicate problems. One run with no sampling, so no seeds are involved.
- GPUs: NVIDIA L40S for 15 shards, A100-PCIE-40GB for 3 (shards 6, 9, 10). The repo documents that about 0.5% of answers
  flip across GPU types (`CONDITIONS_AND_ASSUMPTIONS.md`).

## Input and output (identical to the LoRA runs)

For each problem and each of the 8 candidates, the bundle builds one image: the 3x3 matrix with that candidate placed
in the empty bottom-right cell (160 px panels, `context_mode=normal`, `crop_to_patches=1`). It then asks the bundle's
Yes/No question. The candidate's score is logit(Yes) - logit(No) on the first answer token, and the prediction is the
argmax over the 8 scores. Everything comes from the bundle unchanged: code, pinned packages (`env/`: torch 2.6.0+cu124,
transformers 5.17.0, peft 0.21.2), base models, container (`ngc-images/pytorch-2.1.0.sif`) and offline Hugging Face
settings. The only difference from `run.sh eval-shard` is that `--adapter` is omitted. The bundle's code documents
that case as "the base model scored with the same Yes/No readout".

## Data

The original PGM `.npz` test files were not readable from this account, so the test set was rebuilt from the Hugging Face
mirror `HuggingFaceM4/PGM`, revision `195a03a7e3bc6d6a349b55986afba5319929c3bf` (9 test parquet files, 6.2 GB). The rebuilt
set has ids 1-200,000, all unique.

- Each problem becomes the bundle's `.npz` format: `image` is (16,160,160) uint8, the 8 context panels then the 8 choices,
  and `target` is the answer index. Files are named `PGM_neutral_test_<id>.npz`.
- Problems are sharded with the bundle's own rule: sorted file names, every 28th file per shard.
- **Check 1:** the bundle's 10 `sample_problems/*.npz` are pixel- and target-identical to the rebuilt ones.
- **Check 2:** every target agrees with the answer letter in the team's 20% contact-sheet `test.jsonl`, matched by
  `mirror_sample_id` (0 mismatches).
- Both checks are recorded in `data/packs.json`.
- Scratch has an inode quota, so each shard is stored as one compressed pack. Each task unpacks its pack into node-local
  `/tmp`. Neither the data nor the packs (3.8 GB) are in git.

## Files

| Path | What it does |
|---|---|
| `jobs/prep_data.sbatch`, `scripts/build_packs.py` | CPU job: downloads the mirror at the pinned revision, rebuilds the 200k problems, runs both checks, writes 28 shard packs and `packs.json` |
| `jobs/eval_zeroshot.sbatch` | GPU array (28 tasks): unpacks one shard, then runs the bundle `eval_vlm.py` without `--adapter` for InternVL and then Qwen; resumable |
| `jobs/merge_zeroshot.sbatch`, `scripts/summ.py` | CPU job queued with `--dependency=afterany` on the array: bundle merge with `--expected_n 200000`, plus a short text summary |
| `results/<model>_zeroshot/per_problem.json.gz` | one record per problem: `file`, `target`, `pred`, `correct`, the 8 `scores` (sorted by problem id) |
| `results/<model>_zeroshot/summary.json` | the bundle `summarize()` output (accuracy, Wilson CI, histograms, Yes rates), plus `shards_included` and `complete` |
| `results/<model>_zeroshot/eval_config.json` | the configuration `eval_vlm.py` recorded |
| `results/run_logs/*.log.gz` | Slurm logs. `prep-7309700` failed (an argument-name clash in `np.savez_compressed`, fixed before the successful run `prep-7309742`) |
| `data/packs.json` | data source, revision, count and the two check results |

`summary.json` for the partial results was computed on the login node with the bundle's own `wilson()` and `summarize()`
functions, taken from `pgmvlm/evalcore.py` unchanged. It has the same fields `eval_vlm.py --merge_dir` writes.

## Reproduce (ICE)

The scripts use `$HOME/scratch/pgm-zeroshot-eval` as the working folder and `$HOME/scratch/pgm-lora-v1` as the bundle
(already set up with its `setup.sbatch` and `check.sbatch`). Copy `jobs/` and `scripts/` there, then:

```bash
sbatch jobs/prep_data.sbatch                                     # about 15 min on ice-cpu
sbatch jobs/eval_zeroshot.sbatch                                 # 28 tasks, about 35 min each on an L40S
sbatch --dependency=afterany:<array job id> jobs/merge_zeroshot.sbatch
```

Note: the ICE login notice prohibits Qwen models on ICE. The Qwen3-VL-2B run was done at the team's own decision.
