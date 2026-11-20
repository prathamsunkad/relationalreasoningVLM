# A3: do the interleave models use their relation tokens?

5,000 test puzzles (`test5000.tsv`, as in A2); every condition of a run scored in the same job on one GPU type.
Chance = 12.5%. Change = condition minus normal; p = exact McNemar on the same puzzles.

| Condition | What it does |
|---|---|
| zero | relation tokens set to 0, placeholders kept (= `--branch_off`) |
| drop | placeholders removed from the prompt |
| shuffle_cand | each candidate gets another candidate's tokens from the same puzzle |
| shuffle_puzzle | each puzzle gets another puzzle's tokens (same candidate slot) |
| permute | the 9 / 144 tokens of a sequence in a fixed random order |

## Summary (accuracy %)

| Exp. | Run | normal | zero | drop | shuffle_cand | shuffle_puzzle | permute |
|---|---|---|---|---|---|---|---|
| H4 | `lora_q_int_panel_lr3e-4` | **80.9** | 81.4 (+0.5) | 80.8 (-0.1) | 80.7 (-0.2) | 80.9 (+0.1) | 80.8 (-0.0) |
| H4 | `lora_q_int_pool_lr3e-4` | **80.6** | 79.9 (-0.7) | 80.3 (-0.3) | 80.4 (-0.2) | 80.0 (-0.6) |  |
| H4 | `lora_q_int_slot_control_lr3e-4` | **80.7** | 81.1 (+0.4) | 80.8 (+0.1) | 80.9 (+0.2) | 81.0 (+0.4) | 80.9 (+0.3) |
| H4 | `lora_q_int_slot_lr1e-3` | **80.2** | 80.2 (-0.0) | 80.2 (-0.1) | 80.2 (-0.0) | 80.1 (-0.1) | 79.8 (-0.4) |
| H4 | `lora_q_int_slot_lr3e-4` | **81.0** | 80.7 (-0.3) | 80.5 (-0.5) | 80.6 (-0.4) | 81.0 (-0.0) | 81.0 (-0.0) |
| H7 | `align_q_int_L6ft_lr1e-3` | **35.5** | 14.6 (-20.8) | 14.7 (-20.8) | 9.0 (-26.5) | 12.5 (-23.0) |  |
| H7 | `align_q_int_L6ft_lr3e-4` | **34.9** | 14.6 (-20.3) | 14.7 (-20.3) | 9.1 (-25.8) | 12.7 (-22.3) |  |
| H8 | `lora_w5_q_int_L6ft` | **48.8** | 31.9 (-16.9) | 31.6 (-17.2) | 24.3 (-24.5) | 25.8 (-23.0) |  |
| H9 | `lora_w5d_q_int_L6ft` | **54.9** | 52.7 (-2.2) | 52.4 (-2.5) | 46.1 (-8.8) | 47.7 (-7.2) |  |
| H10 | `align_q_int_L6ft50_lr3e-4` | **39.7** | 14.6 (-25.1) | 14.7 (-25.0) | 8.6 (-31.1) | 11.9 (-27.8) |  |
| H10 | `lora_w5d_q_int_L6ft50` | **54.6** | 49.2 (-5.4) | 48.6 (-6.0) | 33.2 (-21.4) | 39.1 (-15.5) |  |
| H11 | `lora_lf_q_int_ft50` | **54.8** | 53.2 (-1.6) | 53.5 (-1.4) | 51.9 (-2.9) | 52.2 (-2.7) |  |
| H12 | `align_q_int_L6ft276k_lr3e-4` | **45.6** | 14.6 (-30.9) | 14.7 (-30.9) | 8.1 (-37.5) | 11.7 (-33.8) |  |
| H12 | `lora_lf_q_int_ft276k` | **56.4** | 52.5 (-3.9) | 52.7 (-3.7) | 47.6 (-8.8) | 48.5 (-8.0) |  |

## Paired tests against the unablated score

| Exp. | Run | Condition | Normal | Ablated | Change | Only normal right | Only ablated right | p |
|---|---|---|---|---|---|---|---|---|
| H4 | `lora_q_int_panel_lr3e-4` | zero | 80.9 | 81.4 | **+0.5** | 85 | 111 | 0.074 |
| H4 | `lora_q_int_panel_lr3e-4` | drop | 80.9 | 80.8 | **-0.1** | 110 | 107 | 0.89 |
| H4 | `lora_q_int_panel_lr3e-4` | shuffle_cand | 80.9 | 80.7 | **-0.2** | 101 | 93 | 0.62 |
| H4 | `lora_q_int_panel_lr3e-4` | shuffle_puzzle | 80.9 | 80.9 | **+0.1** | 96 | 99 | 0.89 |
| H4 | `lora_q_int_panel_lr3e-4` | permute | 80.9 | 80.8 | **-0.0** | 104 | 103 | 1 |
| H4 | `lora_q_int_pool_lr3e-4` | zero | 80.6 | 79.9 | **-0.7** | 106 | 71 | 0.01 |
| H4 | `lora_q_int_pool_lr3e-4` | drop | 80.6 | 80.3 | **-0.3** | 106 | 89 | 0.25 |
| H4 | `lora_q_int_pool_lr3e-4` | shuffle_cand | 80.6 | 80.4 | **-0.2** | 104 | 92 | 0.43 |
| H4 | `lora_q_int_pool_lr3e-4` | shuffle_puzzle | 80.6 | 80.0 | **-0.6** | 110 | 79 | 0.029 |
| H4 | `lora_q_int_slot_control_lr3e-4` | zero | 80.7 | 81.1 | **+0.4** | 89 | 108 | 0.2 |
| H4 | `lora_q_int_slot_control_lr3e-4` | drop | 80.7 | 80.8 | **+0.1** | 109 | 113 | 0.84 |
| H4 | `lora_q_int_slot_control_lr3e-4` | shuffle_cand | 80.7 | 80.9 | **+0.2** | 88 | 100 | 0.42 |
| H4 | `lora_q_int_slot_control_lr3e-4` | shuffle_puzzle | 80.7 | 81.0 | **+0.4** | 95 | 113 | 0.24 |
| H4 | `lora_q_int_slot_control_lr3e-4` | permute | 80.7 | 80.9 | **+0.3** | 93 | 106 | 0.4 |
| H4 | `lora_q_int_slot_lr1e-3` | zero | 80.2 | 80.2 | **-0.0** | 101 | 100 | 1 |
| H4 | `lora_q_int_slot_lr1e-3` | drop | 80.2 | 80.2 | **-0.1** | 118 | 114 | 0.84 |
| H4 | `lora_q_int_slot_lr1e-3` | shuffle_cand | 80.2 | 80.2 | **-0.0** | 111 | 110 | 1 |
| H4 | `lora_q_int_slot_lr1e-3` | shuffle_puzzle | 80.2 | 80.1 | **-0.1** | 111 | 106 | 0.79 |
| H4 | `lora_q_int_slot_lr1e-3` | permute | 80.2 | 79.8 | **-0.4** | 116 | 95 | 0.17 |
| H4 | `lora_q_int_slot_lr3e-4` | zero | 81.0 | 80.7 | **-0.3** | 99 | 83 | 0.27 |
| H4 | `lora_q_int_slot_lr3e-4` | drop | 81.0 | 80.5 | **-0.5** | 112 | 86 | 0.075 |
| H4 | `lora_q_int_slot_lr3e-4` | shuffle_cand | 81.0 | 80.6 | **-0.4** | 107 | 85 | 0.13 |
| H4 | `lora_q_int_slot_lr3e-4` | shuffle_puzzle | 81.0 | 81.0 | **-0.0** | 97 | 95 | 0.94 |
| H4 | `lora_q_int_slot_lr3e-4` | permute | 81.0 | 81.0 | **-0.0** | 94 | 93 | 1 |
| H7 | `align_q_int_L6ft_lr1e-3` | zero | 35.5 | 14.6 | **-20.8** | 1430 | 388 | 7.4e-140 |
| H7 | `align_q_int_L6ft_lr1e-3` | drop | 35.5 | 14.7 | **-20.8** | 1448 | 408 | 4.2e-136 |
| H7 | `align_q_int_L6ft_lr1e-3` | shuffle_cand | 35.5 | 9.0 | **-26.5** | 1751 | 426 | 6.4e-190 |
| H7 | `align_q_int_L6ft_lr1e-3` | shuffle_puzzle | 35.5 | 12.5 | **-23.0** | 1540 | 391 | 1e-160 |
| H7 | `align_q_int_L6ft_lr3e-4` | zero | 34.9 | 14.6 | **-20.3** | 1400 | 384 | 1.9e-135 |
| H7 | `align_q_int_L6ft_lr3e-4` | drop | 34.9 | 14.7 | **-20.3** | 1417 | 403 | 6.7e-132 |
| H7 | `align_q_int_L6ft_lr3e-4` | shuffle_cand | 34.9 | 9.1 | **-25.8** | 1716 | 424 | 1.6e-183 |
| H7 | `align_q_int_L6ft_lr3e-4` | shuffle_puzzle | 34.9 | 12.7 | **-22.3** | 1520 | 406 | 6.1e-151 |
| H8 | `lora_w5_q_int_L6ft` | zero | 48.8 | 31.9 | **-16.9** | 1216 | 373 | 3.6e-104 |
| H8 | `lora_w5_q_int_L6ft` | drop | 48.8 | 31.6 | **-17.2** | 1231 | 370 | 5.9e-108 |
| H8 | `lora_w5_q_int_L6ft` | shuffle_cand | 48.8 | 24.3 | **-24.5** | 1617 | 392 | 4.3e-176 |
| H8 | `lora_w5_q_int_L6ft` | shuffle_puzzle | 48.8 | 25.8 | **-23.0** | 1516 | 367 | 2.1e-165 |
| H9 | `lora_w5d_q_int_L6ft` | zero | 54.9 | 52.7 | **-2.2** | 363 | 254 | 1.3e-05 |
| H9 | `lora_w5d_q_int_L6ft` | drop | 54.9 | 52.4 | **-2.5** | 367 | 242 | 4.6e-07 |
| H9 | `lora_w5d_q_int_L6ft` | shuffle_cand | 54.9 | 46.1 | **-8.8** | 758 | 317 | 3.1e-42 |
| H9 | `lora_w5d_q_int_L6ft` | shuffle_puzzle | 54.9 | 47.7 | **-7.2** | 654 | 296 | 9.5e-32 |
| H10 | `align_q_int_L6ft50_lr3e-4` | zero | 39.7 | 14.6 | **-25.1** | 1620 | 367 | 3.2e-187 |
| H10 | `align_q_int_L6ft50_lr3e-4` | drop | 39.7 | 14.7 | **-25.0** | 1644 | 393 | 2.8e-181 |
| H10 | `align_q_int_L6ft50_lr3e-4` | shuffle_cand | 39.7 | 8.6 | **-31.1** | 1972 | 416 | 3.3e-241 |
| H10 | `align_q_int_L6ft50_lr3e-4` | shuffle_puzzle | 39.7 | 11.9 | **-27.8** | 1736 | 346 | 5.3e-222 |
| H10 | `lora_w5d_q_int_L6ft50` | zero | 54.6 | 49.2 | **-5.4** | 557 | 286 | 6.5e-21 |
| H10 | `lora_w5d_q_int_L6ft50` | drop | 54.6 | 48.6 | **-6.0** | 590 | 291 | 3.7e-24 |
| H10 | `lora_w5d_q_int_L6ft50` | shuffle_cand | 54.6 | 33.2 | **-21.4** | 1398 | 327 | 2.1e-157 |
| H10 | `lora_w5d_q_int_L6ft50` | shuffle_puzzle | 54.6 | 39.1 | **-15.5** | 1117 | 342 | 4.9e-96 |
| H11 | `lora_lf_q_int_ft50` | zero | 54.8 | 53.2 | **-1.6** | 217 | 138 | 3.2e-05 |
| H11 | `lora_lf_q_int_ft50` | drop | 54.8 | 53.5 | **-1.4** | 248 | 180 | 0.0012 |
| H11 | `lora_lf_q_int_ft50` | shuffle_cand | 54.8 | 51.9 | **-2.9** | 333 | 188 | 2.2e-10 |
| H11 | `lora_lf_q_int_ft50` | shuffle_puzzle | 54.8 | 52.2 | **-2.7** | 331 | 198 | 8e-09 |
| H12 | `align_q_int_L6ft276k_lr3e-4` | zero | 45.6 | 14.6 | **-30.9** | 1887 | 340 | 4.5e-259 |
| H12 | `align_q_int_L6ft276k_lr3e-4` | drop | 45.6 | 14.7 | **-30.9** | 1908 | 363 | 3.1e-252 |
| H12 | `align_q_int_L6ft276k_lr3e-4` | shuffle_cand | 45.6 | 8.1 | **-37.5** | 2260 | 386 | 4.6e-321 |
| H12 | `align_q_int_L6ft276k_lr3e-4` | shuffle_puzzle | 45.6 | 11.7 | **-33.8** | 2016 | 325 | 4e-297 |
| H12 | `lora_lf_q_int_ft276k` | zero | 56.4 | 52.5 | **-3.9** | 390 | 193 | 2.5e-16 |
| H12 | `lora_lf_q_int_ft276k` | drop | 56.4 | 52.7 | **-3.7** | 385 | 201 | 2.5e-14 |
| H12 | `lora_lf_q_int_ft276k` | shuffle_cand | 56.4 | 47.6 | **-8.8** | 707 | 267 | 1.6e-46 |
| H12 | `lora_lf_q_int_ft276k` | shuffle_puzzle | 56.4 | 48.5 | **-8.0** | 670 | 272 | 2e-39 |

## Runs

| Exp. | Run | What it is | Checkpoint step |
|---|---|---|---|
| H4 | `lora_q_int_panel_lr3e-4` | full data, 9 tokens (per panel) | 59000 |
| H4 | `lora_q_int_pool_lr3e-4` | full data, LoRA from scratch, 1 token | 57000 |
| H4 | `lora_q_int_slot_control_lr3e-4` | full data, 144 tokens, non-relational control | 59000 |
| H4 | `lora_q_int_slot_lr1e-3` | full data, 144 tokens, Abstractor lr 1e-3 | 56000 |
| H4 | `lora_q_int_slot_lr3e-4` | full data, 144 tokens (per slot) | 60000 |
| H7 | `align_q_int_L6ft_lr1e-3` | alignment, frozen LLM, 10-epoch branch, lr 1e-3 | 8000 |
| H7 | `align_q_int_L6ft_lr3e-4` | alignment, frozen LLM, 10-epoch branch, lr 3e-4 | 1000 |
| H8 | `lora_w5_q_int_L6ft` | Stage 2 at 5%, no dropout | 14000 |
| H9 | `lora_w5d_q_int_L6ft` | Stage 2 at 5%, dropout 0.5 | 15000 |
| H10 | `align_q_int_L6ft50_lr3e-4` | alignment, frozen LLM, step-204k branch | 1000 |
| H10 | `lora_w5d_q_int_L6ft50` | Stage 2 at 5%, dropout 0.5, step-204k branch | 15000 |
| H11 | `lora_lf_q_int_ft50` | LoRA-first, step-204k branch | 3000 |
| H12 | `align_q_int_L6ft276k_lr3e-4` | alignment, frozen LLM, step-276k branch | 10000 |
| H12 | `lora_lf_q_int_ft276k` | LoRA-first, step-276k branch | 2000 |
