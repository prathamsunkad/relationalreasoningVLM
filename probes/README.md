# probes: is the VLM gap on PGM about perception or about reasoning?

Tests on the **contact-sheet** version of PGM neutral: one grayscale image per puzzle, with the 3×3 matrix (the
bottom-right cell missing) and choices A–H. The model answers with a letter, scored by its log-probabilities for
A–H. `pgm_gap.py` is the shared VLM wrapper; `experiments/exp1_gated_hybrid/` imports it from here too.

| Script | Question |
|---|---|
| `pgm_gap.py`, `run_gap.sh`, `infer_layout.py` | Is the gap about perceiving the panels or reasoning over them? (gap tests) |
| `pgm_attr_probe.py` + `attr_probe.sbatch` | Synthetic PGM-style puzzles with known attributes: can the model read the attributes, and does knowing them make the rule easy? |
| `pgm_real_probe.py` + `real_probe.sbatch` | Real PGM, using each puzzle's rules: attribute order, rule identification, solving with and without being told the rule |
| `perceive_vs_reason.py` + `pvr.sbatch` | The same puzzle asked twice: with the answer shown (perceive) and normally (reason) |

## Setup

1. **Paths:** the repo root's `env_local.sh` (`SHEET_DATA`, `CONDA_ENV`). The job scripts here source it. The
   repo's `env.sh` also sets `DATA`, but to the `.npz` files, so pass `--data $SHEET_DATA` when you run a script by hand.
2. **Environment:** `torch`, `torchvision`, `transformers`, `peft`, `pillow`, `numpy`. `pgm_real_probe.py rules`
   also needs `pyarrow` and `huggingface_hub`.
3. **Panel layout:** `layout.json` in this folder (pixel boxes of the 16 panels; missing cell = 8). Exp1 reads it
   from here as well (`LAYOUT` in `env_local.sh`). To remake it: `python infer_layout.py --data $SHEET_DATA --out layout.json`,
   then check `layout_check.png`.
4. **Run everything from this folder** (`cd probes; mkdir -p logs`).

Jobs are written for PACE ICE (H200, 6 h limit) and exclude the slow node `atl1-1-03-017-30-0`.

## Gap tests (`pgm_gap.py`)

```bash
bash run_gap.sh pilot      # 50 validation puzzles per model + debug images: check those first
sbatch run_gap.sh full     # everything, both models
```

| Stage | Tests |
|---|---|
| `image --variant sheet\|crops` | solving from the contact sheet vs from the 16 panels as separate upscaled images |
| `match --variant sheet\|crops` | perception only: the answer panel is shown, find the identical choice (no rule) |
| `samediff` | are two panels identical? AUC, broken down by how many pixels differ |
| `probe` | a linear same/different probe on frozen vision features (encoded vs reported) |
| `describe`, `text` | the VLM describes each panel as JSON, then solves from the 16 descriptions only (`--rules` adds the rule) |
| `solve` | a small relational network trained on the descriptions: is the information in them sufficient? (CPU) |
| `report` | one table per model with 95% Wilson intervals |

Each stage resumes; per-puzzle results go to `runs/<mode>/`, tables to `runs/<mode>/report_validation.md`.

## Attribute-learning probes

```bash
# synthetic PGM-style puzzles with exact per-panel attributes
python pgm_attr_probe.py synth --out attr_probe --n 1600 --template_jsonl $SHEET_DATA/validation.jsonl --data $SHEET_DATA --layout layout.json
sbatch -J probe_base attr_probe.sbatch OpenGVLab/InternVL3_5-1B-HF

# real PGM: rules from relation_structure (login node, needs internet once)
python pgm_real_probe.py rules --data $SHEET_DATA --split validation      # -> real_probe/rules_validation.json
sbatch -J real_base real_probe.sbatch OpenGVLab/InternVL3_5-1B-HF

# perceive vs reason
sbatch -J pvr_base pvr.sbatch OpenGVLab/InternVL3_5-1B-HF
python perceive_vs_reason.py report        # pvr/report_validation.md
```

| Probe | Conditions | Reading |
|---|---|---|
| `pgm_attr_probe.py` | `attrs`: shape type, number, position, line type, which is darker / larger; `infer`: images, ground-truth text, images + text, images + rule, text + rule, sheet | attributes read well and `gt_text` ≫ `images` → attribute perception is the bottleneck |
| `pgm_real_probe.py` | `order`: which of a row's 3 panels has the middle value (progression rows, chance 33%); `ruleid`: which attribute / relation; `infer`: crops, crops + rule, sheet, sheet + rule | `crops_rule` ≫ `crops` → it can read values; finding the rule was the problem |
| `perceive_vs_reason.py` | PERCEIVE vs REASON on the same puzzles, `--input crops\|sheet` | REASON on puzzles it PERCEIVED correctly = reasoning when perception is not the problem |

A fine-tuned model must be a full Hugging Face model folder; merge a LoRA adapter into the base model first.
`pgm_real_probe.py rules` reads only the `id`, `relation_structure` and `target` columns of the pinned
`HuggingFaceM4/PGM` revision, and stops unless its answers agree with the JSONL for ≥ 98% of puzzles. This download
path was only tested against a local parquet copy.

## Results so far

PGM neutral, chance 12.5%, letter answers from the contact sheet. This is a different question format from the
repo's Yes/No numbers, so the two are not directly comparable.

**Contact-sheet baselines** (200k test):

| Model | Setting | Accuracy |
|---|---|---|
| Qwen3-VL-2B + LoRA | 240k, epochs 1 / 2 / 3 | 55.87% / 68.61% / **75.33%** |
| InternVL3.5-1B + LoRA | 240k, epochs 1 / 2 / 3 | 40.23% / 48.53% / **52.31%** |
| LFM2.5 + LoRA | epoch 1 → 6 | 69.60% → **75.54%** (val 75.9%) |
| SpatialEvo + LoRA | epoch 1 → 5.07 | 58.19% → **61.87%** (val 65.3%) |
| Qwen3-VL-2B, InternVL3.5-1B | zero-shot | near chance |

**Gap pilot** (zero-shot, 50 validation puzzles, ±12 points):
- **Perception from the contact sheet is a barrier in itself.** Qwen3-VL-2B finds the answer panel among the 8
  choices (no rule needed) **78%** of the time from separate panels (`match --variant crops`), but **10%** from the
  contact sheet (`match --variant sheet`).
- Solving zero-shot is near chance, so the base models fail at more than perception from the sheet.

**Not run yet:** the full gap run and the three probes. Next: `pvr.sbatch`, `real_probe.sbatch` and
`attr_probe.sbatch` on the base and fine-tuned InternVL (and Qwen), then `bash run_gap.sh full`.

## Outputs (not committed)

`runs/`, `attr_probe/`, `real_probe/`, `pvr/`, `logs/` (see `.gitignore`). Copy finished reports into this README
or a run record under `results/`.
