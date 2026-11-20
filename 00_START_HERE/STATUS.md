# Status (Nov 20, 2026, 12:40 am EST)

**Finished:** zero-shot and LoRA baselines; Stage 0; the branch-alone sweeps; H1–H12; the early-branch part of H13;
A1, A2 and A3. **Training:** the two 50-epoch branch-alone runs, and the H8, H9 and H12 recipes on the finished 60k-only
branch (H13; interleave since 12:07 am, fusion after its alignment levels off). **Next, automatically:** their test scoring.

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
| A3 | Relation-token ablation: LoRA-first H12 loses 3.9 points zeroed, 8.8 shuffled; full-data H4 unaffected | `results/relational/ablation_test5000/report.md` |

Pairs are interleave / fusion.

## Still training

| Run | What it is | Step | Best val (step) | Expected finish |
|---|---|---|---|---|
| `aux_ft_q_L6_lr3e-4_ep50` | 50-epoch branch, 6 layers (source of the H10–H12 branches) | 468k / 750k | **54.4%** (460k) | ~5 days |
| `aux_ft_q_L24_lr8e-5_ep50` | 50-epoch branch, 24 layers | 396k / 750k | 49.7% (386k) | ~8 days |
| `lora_{w5,w5d,lf}_q_int_c60` | H13: H8, H9, H12 recipes on the finished 60k-only branch | started 12:07 am Nov 20 | | H12 recipe ~midday Nov 20 |
| `align_q_fuse_L6ft60k_lr3e-4` | H13: fusion alignment of the finished 60k-only branch | 3k / 10k | 29.5% (2k) | fusion hybrids launch when it levels off |

The finished 60k-only branches (H13): `aux_ft60k_q_L6_lr1e-3` best 30.8% val (step 71k, used), `aux_ft60k_q_L6_lr3e-4`
29.4% (101k); both ended at about 27% val after overfitting. Interleave alignment levelled at 29.6% (step 2k).

## A3: relation-token ablation (finished Nov 19, 9:27 pm)

14 interleave runs (H4, H7–H12), each scored on the 5,000 test puzzles with its relation tokens left alone, zeroed,
removed, shuffled across a puzzle's candidates, shuffled across puzzles, and (9- and 144-token runs) reordered; all
conditions of a run in one job on one GPU. Full table and McNemar tests: `results/relational/ablation_test5000/report.md`.

| Runs | Normal | Zeroed | Shuffled across candidates | Reading |
|---|---|---|---|---|
| H4 full data (5 runs, control included) | 80.2–81.0% | −0.7 to +0.5 | −0.4 to +0.2 | Tokens ignored |
| Frozen-LLM alignments (H7, H10, H12) | 34.9–45.6% | 14.6% | 8.1–9.1% | Fully dependent; shuffled is below chance |
| H8 Stage 2, no dropout | 48.8% | −16.9 | −24.5 | Shortcut through the branch |
| H9 / H10 Stage 2 with dropout | 54.9% / 54.6% | −2.2 / −5.4 | −8.8 / −21.4 | Uses the tokens; more with the stronger branch |
| H11 / H12 LoRA-first | 54.8% / 56.4% | −1.6 / −3.9 | −2.9 / −8.8 | Uses the tokens (H12: p < 1e-15); zeroed H12 ≈ its control |

Zeroed and removed agree within 0.6 points in every run (the 3-point gap in the 200-puzzle check was noise).
Reordering had no effect, but was only tested on H4 runs, which ignore the tokens.

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
