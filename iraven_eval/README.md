# iraven_eval: PGM-trained LFM2.5 LoRA adapter on the I-RAVEN test set

This folder evaluates the PGM-trained adapter `lfm25-pgm-best` (bundle `pgm-best-lora-2026-11-01-v1`,
LoRA r16/α32 on `LiquidAI/LFM2.5-VL-1.6B` @ `919fde3d`) **zero-shot on I-RAVEN**, with no further training.
Each I-RAVEN puzzle is rendered into the same 700×1016 contact sheet and prompt the adapter was trained on.

## Results

| I-RAVEN configuration | Correct / 2,000 | Accuracy | 95% Wilson interval |
|---|---|---|---|
| center_single | 519 | 26.0% | 24.1–27.9% |
| distribute_nine | 506 | 25.3% | 23.4–27.3% |
| distribute_four | 484 | 24.2% | 22.4–26.1% |
| in_distribute_four_out_center_single | 475 | 23.8% | 21.9–25.7% |
| in_center_single_out_center_single | 451 | 22.6% | 20.8–24.4% |
| left_center_single_right_center_single | 408 | 20.4% | 18.7–22.2% |
| up_center_single_down_center_single | 389 | 19.5% | 17.8–21.2% |
| **All (I-RAVEN test)** | **3,232 / 14,000** | **23.1%** | **22.4–23.8%** |
| PGM neutral test, first 1,000 (pipeline check) | 772 / 1,000 | 77.2% | 74.5–79.7% |

- Chance is 12.5% (8 choices). The bundle's published accuracy on the full 200k PGM neutral test is 75.535%,
  so the pipeline check reproduces the adapter's in-distribution behaviour; the drop to 23.1% is the transfer gap.
- All 15,000 outputs were a valid single letter A–H (0 invalid). The model over-predicts "F" on I-RAVEN
  (2,533 predictions vs 1,750 correct F answers).
- One run, greedy decoding, no seeds involved at inference.
- `spatialevo-pgm-best` (Qwen2.5-VL based) is **not evaluated**: the ICE login notice prohibits Qwen on that system.

## Data

There is no hosted official I-RAVEN, so it was generated with the authors' generator, unmodified:
[husheng12345/SRAN](https://github.com/husheng12345/SRAN) `I-RAVEN/`, commit `9530ad6190ae00bf708460e76a965af87df8ecaa`,
Python 2.7 with its pinned packages (numpy 1.16.6, scipy 1.2.3, opencv-python 4.2.0.32, pillow 6.2.2),
`--seed 1234 --num-samples 10000`, standard 6:2:2 split (`k % 10` in {8, 9} = test) → 2,000 test puzzles × 7 configurations.
The generator's own solver check reported accuracy 1.0 on every configuration.

Rendering: the 16 panels (160×160) are pasted into `template/pgm_contact_sheet_template.png`, a real PGM sheet with its
16 slots blanked. Re-rendering real PGM sheets from their own panels this way is pixel-identical to the originals.
The prompt is the bundle's `prompts.json` (`matrix-3x3-options-2x4-v1`); answer letter = `'ABCDEFGH'[target]`.

Generated data, sheets and weights are not in git (≈270 MB of sheets). `data/` holds the manifest and its description,
with a sha256 per sheet, so a regenerated copy can be checked against it.

## Files

| Path | What it does |
|---|---|
| `jobs/gen_iraven.sbatch` | CPU job: builds the py2.7 env in node `/tmp`, clones SRAN, runs the generator, packs the test split |
| `scripts/pack_test.py` | packs each configuration's test puzzles into one `.npz` (+ the XML annotations as `.tar.gz`) |
| `jobs/setup_env.sbatch`, `scripts/download_base.py` | Python 3.11 venv with the bundle's pinned versions; downloads the base model at the pinned revision |
| `scripts/build_sheets.py`, `template/` | renders the contact sheets (one `.tar` of PNGs per configuration) and writes `test.jsonl` in the PGM manifest schema |
| `jobs/eval.sbatch`, `scripts/eval_adapter.py` | GPU job: the PGM 1,000-question check, then the full I-RAVEN test. Loads the adapter through the bundle's own `load_adapter.py` (sha256-checked), greedy decoding, `max_new_tokens=16`; resumable |
| `results/*.summary.json` | overall and per-configuration accuracy, Wilson intervals, invalid counts, prediction/answer histograms |
| `results/*.predictions.jsonl.gz` | one line per question: `id` (contains the configuration), `answer`, `raw_output`, `prediction` (parsed letter, or null if invalid), `latency_seconds` |
| `results/run_logs/*.log.gz` | the Slurm logs of the four jobs (generation, environment, the two evaluation jobs) |
| `data/` | `test.jsonl.gz` (I-RAVEN test manifest), `dataset-description.json`, data `README.md` |

## Reproduce (ICE)

The job scripts are kept exactly as run; they use `$HOME/iraven_eval` as the working folder and read the bundle from
`$HOME/scratch/pgm-best-lora-2026-11-01-v1`. Copy `jobs/`, `scripts/` and `template/` there, then:

```bash
cd ~/iraven_eval && mkdir -p logs
g=$(sbatch --parsable jobs/gen_iraven.sbatch)
e=$(sbatch --parsable jobs/setup_env.sbatch)
sbatch --dependency=afterok:$g:$e --export=ALL,KEY=lfm25-pgm-best jobs/eval.sbatch
```

Timing: generation 55 min (ice-cpu). Sheet rendering took ≈2.5 h inside the first eval job. Inference: 1.4 s/question
on a V100, 0.29 s/question on an A40. The first eval job timed out after 7,200 I-RAVEN questions; a resubmitted job
(`-C cuda13 -t 08:00:00`) resumed from the predictions file and finished the rest.
