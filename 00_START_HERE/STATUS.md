# Status (Nov 19, 2026, 3:20 pm EST)

**Finished:** zero-shot and LoRA baselines; Stage 0; the branch-alone sweeps; H1–H12; the early-branch part of H13;
A1 and A2. **Training:** four branch-alone runs. **Next, automatically:** the H8, H9 and H12 recipes on the finished
60k-only branch, then their test scoring.

## Finished experiments (headline numbers)

| # | Result | Test (5,000) or val |
|---|---|---|
| Zero-shot | At chance on PGM | 12.85% Qwen, 13.75% InternVL (200k test) |
| LoRA alone, full data | Main baseline | 81.50% Qwen, 57.73% InternVL (200k test) |
| LoRA alone, 5% | Low-data baseline | 52.6% test |
| H1–H3, H6 | Branch trained through a frozen LLM: memorises only with 6 layers, generalises slowly | 16–27% val |
| H4, H5 | Hybrids at full data and on the mixture: no gain | 81.3–81.4% val vs 81.6%; family 87.2 vs control 87.6 |
| H7 | Alignment: the frozen LLM reads the branch within 1k steps | = branch accuracy |
| H8 | Stage 2 without dropout: shortcut | 49.1% / 38.8% |
| H9 | Stage 2 with dropout 0.5 | 55.2% / 54.1% |
| H10 | Stronger branch in Stage 2: LoRA weakens | 55.1% / 55.3% |
| H11 | LoRA-first, step-204k branch | 55.1% / 54.9% (controls 53.2% / 52.9%) |
| **H12** | **LoRA-first, step-276k branch: best result** | **56.5% / 56.4%** (+3.3 / +3.5 over control) |
| H13 (early) | 60k-only branch at step 43k: no gain in any recipe | H12 recipe 53.7% / 53.3%, branch off identical |
| A1 | Branch knows things LoRA doesn't at 5%, not at full data | averaging +2.2 to +3.5 at 5% |
| A2 | Test scoring of all of the above | `results/relational/test5000/report.md` |

Pairs are interleave / fusion.

## Still training

| Run | What it is | Step | Best val (step) | Expected finish |
|---|---|---|---|---|
| `aux_ft60k_q_L6_lr3e-4` | 60k-only branch, lr 3e-4 (H13) | 135k / 150k | 29.4% (101k) | ~10 pm Nov 19 |
| `aux_ft60k_q_L6_lr1e-3` | 60k-only branch, lr 1e-3 (H13) | 136k / 150k | 30.8% (71k) | ~10 pm Nov 19 |
| `aux_ft_q_L6_lr3e-4_ep50` | 50-epoch branch, 6 layers (source of the H10–H12 branches) | 441k / 750k | **51.5%** (424k) | ~6 days |
| `aux_ft_q_L24_lr8e-5_ep50` | 50-epoch branch, 24 layers | 379k / 750k | 48.4% (366k) | ~9 days |

The 60k-only branches have plateaued: training accuracy keeps rising while validation has been flat since about
step 75k. Their cosine decay runs from step 135k to the end.

## What launches by itself next

`experiments/auto/auto_c60.py` (running on the PACE login node) waits for both 60k-only branches to finish, then:

1. snapshots the better one to `snapshots/aux_ft60k_best.pt`;
2. aligns it on the 60k only (`align_q_{int,fuse}_L6ft60k_lr3e-4`);
3. once each alignment levels off, launches the H8, H9 and H12 recipes: `lora_w5_q_*_c60`, `lora_w5d_q_*_c60`, `lora_lf_q_*_c60`;
4. scores them on the 5,000 test puzzles (`diag_5k`, `diag_5kw`) and re-runs the analysis.

Expected: H12-recipe test numbers around midday Nov 20, H8/H9 about a day later.

## Open decisions

| Decision | What it would answer |
|---|---|
| Longer LoRA-alone baseline at 5% | Whether the gains over LoRA alone survive a fully converged baseline (its loss was still falling) |
| Single-job re-score of every model (~6 GPU-hours) | Removes the bf16 GPU-type caveat on cross-job gaps under ~1 point, including the H12-vs-averaging tie |
| Relational vs same-size control inside LoRA-first | Whether the relational bottleneck matters where the branch actually helps |
| Higher branch dropout for stronger branches | Whether Stage 2 can keep LoRA intact with a strong branch (H10) |
| A joint stage where the Abstractor adapts | Whether a hybrid can beat simply averaging the two models (H12 ties it) |
| LoRA-first from a newer 50-epoch checkpoint (now 51.5% val) | Whether the gain keeps growing with branch strength |
| Held-out-rule PGM split | Whether the branch helps on rules never seen in training |

## Known caveats

- One seed everywhere.
- 5% hybrids H8–H12 use a branch trained on all 240k labelled puzzles (4× what LoRA saw); H13 removes this.
- A2 scored models in 8 jobs on 3 GPU types (A100, H200, L40S). bf16 rounding differs by GPU type, so cross-job gaps under ~1 point are not settled.
- Validation puzzles were also used to choose checkpoints; quote the test numbers.
