# Status (Nov 24, 2026, 8 pm EST)

**Finished since Nov 20:** H13 on the finished 60k-only branch; A1b (overlap on the 5,000 test puzzles with the step-636k
branch); H14 (LoRA-first with the step-636k branch at 5%); H15 (the same on the 20% LoRA after epochs 1 and 4, matched
data); A3 extended to H14 and H15. **Training:** the 5% branch smoke runs, the per-epoch 5% LoRA and its hybrids, Stage 2
with the step-636k branch, the unfrozen-Abstractor check, and the two 50-epoch branches. Full write-up with every setting:
[`docs/relvlm_results_H1_H15.md`](../docs/relvlm_results_H1_H15.md).

## Finished experiments (headline numbers)

| # | Result | Test (5,000) or val |
|---|---|---|
| Zero-shot | At chance on PGM | 12.85% Qwen, 13.75% InternVL (200k test) |
| LoRA alone, full data (20%) | Main baseline | 81.50% Qwen (200k test); on `test5000`: epoch 1 / 2 / 4 = 55.2 / 68.1 / 81.2% |
| LoRA alone, 5% | Low-data baseline | 52.6% test (52.8% rescored with the H14 job) |
| H1–H3, H6 | Branch trained through a frozen LLM: memorises only with 6 layers, generalises slowly | 16–27% val |
| H4, H5 | Hybrids at full data and on the mixture: no gain | 81.3–81.4% val vs 81.6%; family 87.2 vs control 87.6 |
| H7 | Alignment: the frozen LLM reads the branch within 1k steps | = branch accuracy (step 636k: 62.1% test) |
| H8 / H9 / H10 | Stage 2 at 5%: shortcut, fixed by dropout, flat with a stronger branch | 49.1/38.8; 55.2/54.1; 55.1/55.3% |
| H11 / H12 | LoRA-first at 5%, steps 204k / 276k branch | 55.1/54.9; **56.5/56.4%** (controls 53.2/52.9) |
| H13 | Everything at matched 5% data (60k-only branch, 28.1% test alone) | LoRA-first 53.8/53.4% vs control 53.2/52.9 (n.s.); H9 recipe 50.4/52.1; H8 recipe 43.5/46.8 |
| A1 / **A1b** | The branch solves puzzles LoRA misses; averaging gain shrinks as LoRA trains | A1b (636k branch, 62.0%): +10.2 (5%), +8.7 (20% ep 1), +2.3 (ep 2), −0.1 (ep 4) |
| **H14** | LoRA-first with the step-636k branch on the 5% LoRA | **63.2 / 62.9%** vs control 53.2/52.9 (p < 1e-59); only +1.2/+0.9 over the branch alone |
| **H15** | Same on the 20% LoRA (matched data) | **Epoch 1: 65.5 / 65.9%** vs control 60.2, LoRA 55.2, branch 62.0 (p < 1e-14). Epoch 4: 81.4 / 81.3% vs LoRA 81.2 (n.s.) |
| A3 | Relation-token ablation (now 18 runs) | H14 / H15 ep 1: zeroed −11.7 / −8.0, shuffled across candidates −31.2 / −46.9; H15 ep 4: at most 2 points |

Pairs are interleave / fusion.

## Still training (Nov 24, 8 pm)

| Run(s) | What it is | Progress | Latest val |
|---|---|---|---|
| `aux60s_*` (9 runs) | Smoke runs: can a smaller or regularised 5% branch beat the 60k-only branch's 30.8% peak? | 40–52k / 75k | 21–28% (old run ~27–28% at the same steps) |
| `qwen_pc_neutral5pct_ep` | 5% LoRA rerun keeping the adapter after each epoch (`--keep_steps`) | ~10.5k / 15k | 43.7% |
| `lora_lf5e{1,2}_q_int_c60`, `_ctrl` | LoRA-first with the 60k-only branch on the 5% LoRA after epochs 1 and 2 | 3.5k / 3.5k, scoring next | ep 1: 37.5 vs 35.2; ep 2: 44.5 vs 42.5 |
| `lora_lfu20e1_q_int_ft636k_a3e-5` | H15 epoch 1 with the Abstractor unfrozen (lr 3e-5, slots frozen) | started 7 pm | — |
| `lora_w5d_q_{int,fuse}_L6ft636k` | Stage 2 + dropout 0.5 with the step-636k branch at 5% | 13k, 10k / 15k | 63.7 / 63.6% |
| `aux_ft_q_L6_lr3e-4_ep50` | 50-epoch branch, 6 layers | 718k / 750k | best 65.6% (714k) |
| `aux_ft_q_L24_lr8e-5_ep50` | 50-epoch branch, 24 layers | 600k / 750k | best 52.3% |

Every finished run is test-scored automatically by `experiments/auto/wait_5k.sh` (and `wait_ep5.sh` for the 5% LoRA epochs).

## Open decisions

| Decision | What it would answer |
|---|---|
| Epoch sweep at 20% and 5%, Abstractor frozen vs unfrozen (19 runs, designed) | Whether the gain over control falls with LoRA strength at both scales, and whether an adapting Abstractor beats averaging. Waiting on the unfrozen-Abstractor check |
| Longer LoRA-first on H15 epoch 1 (hybrid + control, ~15k steps, normal LoRA rate) | Whether the branch speeds learning or raises the final level; 3,500 steps is only 0.23 epoch at 20% |
| Seeds for H15 epoch 1 | Error bars on the headline result |
| Single-job re-score of the key models | Settles cross-job gaps under ~1 point and the H12-vs-averaging tie |
| Held-out-rule PGM split or I-RAVEN | Generalisation beyond PGM neutral |

## Known caveats

- One seed everywhere.
- H8–H12 and H14 use a branch trained on all 240k labelled puzzles (4× the 5% LoRA's data); H13, H15 and the per-epoch 5% runs are at matched data.
- Scoring jobs ran on different GPU types; bf16 rounding moves scores by ~0.5 points, so cross-job gaps under ~1 point are not settled.
- Validation puzzles were also used to choose checkpoints; quote the test numbers.
- LoRA-first lasts 3,500 steps with LoRA at 2e-5: one epoch at 5%, only 0.23 epoch at 20%.
