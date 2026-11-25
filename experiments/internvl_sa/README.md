# InternVL3.5-1B + the standalone Slot Abstractor (started Nov 25, 2026)

The Slot Abstractor here is the paper's architecture (Mondal et al., 2024) trained by Pranav on the same 240k puzzles
(`slot_abstractor/`; best 62.8% val, epoch 63): its own slot autoencoder on 80x80 pixel panels, depth 24. It is not the
ViT-feature branch of the Qwen H-series. It is connected to InternVL through Pranav's Exp2
(`experiments/exp2_sa_interleave/`).

| File | What it is |
|---|---|
| `run_exp2_container.sbatch` | Runs `exp2_sa_interleave.py` in the PACE pytorch-2.1.0 container (no conda env needed; `einops`, `colorlog`, `transformers`, `peft` from `~/.local/lib/python3.10`). Every argument is passed through |
| `sa5/pgm_npz_data.py` | Drop-in for `slot_abstractor/pgm_jsonl_data.JsonlPGM` that reads the original 16-panel `.npz` files from a list (same tensors as the authors' loader; resize antialiased) |
| `sa5/make_code.sh` | Rebuilds the exact code + lists the 5% Slot Abstractor jobs ran from |
| `sa5/sa5_job.sbatch` | `ae` (slot AE, 300k steps), `train` (Abstractor, depth 24, lr 8e-5, batch 96, 80 epochs), `test` (test5000), all on our 5% list |

## Results so far

- **Inspect (val1000):** the shared 20% Slot Abstractor scores 61.0% (95% CI 57.9-64.0%) through Exp2's inputs, vs its
  recorded 62.8%: same model, consistent within slot-initialisation noise (`results/internvl_sa/inspect_val1000.txt`).
- **Overlap with InternVL LoRA alone (test5000, `results/internvl_sa/complement_test5000_*`):**

| InternVL LoRA | LoRA alone | Slot Abstractor alone | Ensemble (cross-fitted weight) |
|---|---|---|---|
| 5% (`ivl1b_pc_neutral5pct_recipe/best`) | 38.3% | 61.0% | 61.2% |
| 20%, epoch 1 | 40.1% | 61.0% | 61.2% |
| 20%, epoch 2 | 48.4% | 61.0% | 62.4% |
| 20%, epoch 4 | 57.9% | 60.8% | 65.3% |
| 20%, epoch 6 | 60.3% | 61.0% | 67.0% |

Unlike Qwen (whose 20% LoRA reaches 81%), even the fully trained InternVL LoRA is complementary to the relational module.

## Running

- `ivl1b_pc_neutral5pct_ep`: InternVL 5% LoRA rerun keeping each epoch (for H16 on InternVL).
- `sa5`: the Slot Abstractor retrained on our 5% (slot AE, then Abstractor, then test). Gate: the H16 hybrids run only if
  it beats the InternVL 5% LoRA (38.3% test).
- H15 on InternVL was run by a teammate (not rerun here).
