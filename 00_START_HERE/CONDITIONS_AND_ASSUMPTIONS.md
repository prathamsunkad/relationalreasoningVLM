# Conditions, hyperparameters and assumptions

The exact value of every setting for every run is in its `results/relational/run_records/<run>/args.json`
(relational) or `results/lora/run_records/<run>/args.json` (LoRA). This page collects them per recipe, then lists
the assumptions behind them and the caveats on how to read the results. Per-experiment detail: `docs/relational_deep_dives.md`.

## 1. Shared by every relational run

| Setting | Value |
|---|---|
| Model | Qwen3-VL-2B-Instruct (revision pinned in `env.sh`); InternVL3.5-1B only in H1, H2 and early branch runs |
| Seed | 0 (data order, subsets, initialisation) |
| Precision | bf16 weights, standard (sdpa) attention, no gradient checkpointing |
| Hardware | One GPU per job: A100, H100, H200 or L40S |
| Batch | 2 puzzles × 8 gradient-accumulation steps = 16 puzzles per step (each puzzle = 8 candidate sequences) |
| Optimiser | AdamW, weight decay 0.01, gradient clip 1.0 |
| Schedule | Linear warmup, hold, cosine to 10% of each group's base rate |
| Panels | 160 px; the branch runs slot attention inside each of the 9 panel boxes |
| Abstractor (unless noted) | 8 heads, feed-forward 512, dropout 0, relational cross-attention → FF → self-attention → FF per layer |
| Readout | LLM: logit(Yes) − logit(No) per candidate, 8-way cross-entropy. Branch alone: linear head on the Abstractor |
| Validation | The 1,000 shared puzzles (same as the LoRA baseline); best checkpoint kept |
| One epoch | 240k puzzles = 15,000 steps; 60k = 3,750 steps |

## 2. Recipes

| Recipe | Trains (rate) | Frozen | Length and schedule | Other |
|---|---|---|---|---|
| **LoRA baseline** (L2, L3) | LoRA r16, α32, dropout 0.05 on every LLM linear layer (1e-4) | ViT, merger, LLM base | 4 epochs; warmup 20; constant for epochs 1–3; cosine to 1e-5 in epoch 4 | 17.43M trainable |
| **Stage 0** slots (`s0_*`) | Slot attention + MLP decoder (4e-4) | Whole VLM | 100k steps; warmup 10k; cosine to 10%; 16 images per step | 16 slots, 3 iterations, 12×12 ViT grid per panel; slot dim 32 (decoder 512) or 256 (decoder 2048); per-channel standardised features; reconstruction MSE, no labels |
| **Branch alone** (Stage 1a, `aux_*`) | Abstractor + linear head (3e-4; 8e-5 for 24 layers); slots too at 3e-5 in `aux_ft*` | ViT; LLM not used | e.g. 10-epoch branch: 150k steps, warmup 500, hold to 135k, cosine; 50-epoch: 750k steps, decay from 735k | 6 layers, head dim 32 (0.80M) for the branches behind H7–H13 |
| **Frozen-LLM Stage 1** (H1–H3, H6) | Abstractor (1e-4 to 1e-3) + connector, gain / λ (3e-4 or 1e-3) | ViT, slots, LLM | H2: 15k steps, warmup 500, cosine. H6: 150k planned, warmup 500, hold to 135k | H1, H2: 24 layers, head dim 32; H3: 6 layers; H6: 6 layers, head dim 256 |
| **Joint, from scratch** (H4, H5) | Abstractor (3e-4 or 1e-3), connector + gain / λ (3e-4), LoRA (1e-4) | ViT, slots, LLM base | H4: 60k steps, warmup 500, hold to 45k. H5: 39k steps, hold to 29,250 | 6 layers, head dim 256 (28.34M), 256-dim slots, connector 4.72M; fusion λ starts at 0.5 |
| **Alignment** (`align_*`: H7, H10, H12, H13) | Connector MLP 32 → 2048 → 2048 + gain / λ (3e-4; 1e-3 in two H7 runs) | Slots, Abstractor, answer head, ViT, LLM | 10k steps, warmup 200, cosine | 4.26M trainable; fusion λ starts at 0.9; all 240k puzzles (H13: the 60k only) |
| **Stage 2** (`lora_w5*`: H8–H10, H13) | LoRA from scratch (1e-4); Abstractor + connector (3e-5) | Slots, ViT, LLM base | 15k steps (4 epochs of 60k), warmup 20, hold to 11,250, cosine | Starts from the alignment checkpoint where it levelled off; branch dropout 0 (H8) or 0.5 (H9, H10) |
| **LoRA-first** (`lora_lf*`: H11–H13) | Connector + gain / λ (1e-4) from step 0; LoRA (2e-5) from step 500 | Abstractor, slots, ViT, LLM base | 3,500 steps on the 60k; warmup 50; cosine from step 500 | LoRA starts from the LoRA-alone 5% adapter (step 14k); control = identical with `--branch_dropout 1.0` (branch never reaches the LLM) |
| **Mixture runs** (H5, B4) | as above | as above | as above | `--loss_norm chance`, `--best_on family`; mixtures in `DATA.md` |

**Parameter counts:** Abstractor 0.80M (6 layers, width 32), 3.18M (24 layers, width 32), 28.34M (6 layers, head dim
256); connector 4.26M (32 → 2048 → 2048), 4.72M (256 → 2048 → 2048), 1.08M (InternVL); LoRA 17.43M; Qwen3-VL-2B
~2.1B total.

**Rules used to pass checkpoints on:** an alignment "levels off" after at least 3 validations, once the best is not
up more than 1 point over the last two; its best checkpoint up to then is passed on. In fusion runs where two rates
tied, the one whose λ stayed closer to 0.9 was taken (H7).

## 3. How results were measured

| Item | Definition |
|---|---|
| Validation accuracy | 1,000 shared puzzles; ±3 points (95%); also used to choose checkpoints, so slightly optimistic |
| Test accuracy (hybrids) | 5,000 puzzles from the 200k test set (`test5000.tsv`), never used for training or choice; ±1.4 points; Wilson 95% interval |
| Test accuracy (baselines) | Full 200k test set |
| Paired comparison | Same puzzles, McNemar exact test on "only A right" vs "only B right" |
| Branch off | Fusion λ = 1 (pure own image); interleave gain = 0 (empty relation token) |
| Branch alone | The Abstractor's linear head, no LLM (`--readout aux`) |
| Averaging reference | Mean of LoRA alone's and the branch's log-softmax scores, weight 0.5, untuned (A1 also tried a cross-validated weight) |
| Family score (mixtures) | Mean accuracy over PGM, CLEVR, VSR, ART (4 tasks pooled), CLEVR-ART (2 tasks pooled) |
| Per attribute | A puzzle counts in every attribute (number, position, type, size, colour) it has a rule on |

## 4. Assumptions and choices made without a direct test

| # | Assumption or choice | Where it matters | Status |
|---|---|---|---|
| 1 | Slots stay frozen after Stage 0, except the branch-alone runs that fine-tune them (`aux_ft*`, 3e-5); the paper fine-tunes them | All hybrids | Fine-tuning added ~7 points to the branch (B5); never tried inside a frozen-LLM run |
| 2 | Slot size 256 / decoder 2048 from the paper's V-PROM recipe, but 16 slots and a 12×12 grid (V-PROM uses 11) | H4–H6 | Untested alternatives |
| 3 | LoRA on the LLM only, with the baseline's r16 and lr 1e-4; not re-tuned for the hybrid | H4, H5, H8–H10 | Untested |
| 4 | Joint runs use warmup 500 (baseline 20) but the baseline's decay timing | H4, H5 | Untested |
| 5 | Qwen for the main experiments; InternVL paused | All after H2 | By decision |
| 6 | H4 and H5 trained the branch from scratch, although the approved plan called for a warm start (dropped without discussion) | H4, H5 | Warm starts came later, in H7–H13 |
| 7 | 6 layers chosen from memorisation tests; the 60k sweep did not separate 6 from 24 | All later hybrids | Partly tested (B1, B3) |
| 8 | Abstractor rates 3e-4 / 1e-3 for H4 came from 32-dim tests (H3), not the 256-dim branch used | H4 | Untested at 256-dim |
| 9 | Fusion λ starts at 0.5 in H3–H5 and at 0.9 from H6 on (kept at 0.9 by decision) | Fusion runs | Other starts untested |
| 10 | Token modes (1 / 9 / 144) and both regimes went straight to LoRA, without a frozen-LLM comparison first | H4 | No difference found at full data |
| 11 | Branch dropout 0.5 is the only value tried | H9, H10 | Higher dropout for stronger branches proposed, not run |
| 12 | LoRA-first controls from H11 are reused in H12 and H13 (a branch-off control does not depend on the branch) | H12, H13 | Valid by design; scored in a different job |
| 13 | Mixture runs sized for equal compute, not equal PGM: ~187k PGM puzzles at 30% vs 960k in H4 | H5 | By design |
| 14 | H1 and H2 ran on an earlier version of the relational code (Oct 29 – Nov 1). Their `args.json` reruns with today's code, which adds only default-off options, but bit-for-bit equality is not guaranteed | H1, H2 | Not re-run |

## 5. Caveats when reading the results

- **One seed** for every run.
- **The data caveat for H8–H12:** their branch was trained on all 240k labelled puzzles (up to 18 epochs), four times what the 5% LoRA saw. H13 removes it; with the early 60k-only branch the gain disappears.
- **LoRA alone at 5% was not trained to convergence** (fixed 4-epoch schedule, validation still rising). This affects "vs LoRA alone", not "vs control".
- **GPU-type noise:** bf16 rounding differs between GPU types and flipped ~0.5% of answers when a run changed GPU type. A2 scored models in 8 jobs on 3 GPU types, so cross-job gaps under ~1 point (including the H12-vs-averaging tie) are not settled. A single-job re-score was offered and not run.
- **Validation was used for checkpoint choice;** report the test numbers.
- **The mixture family score** includes two families at 100% for every run, which compresses differences.
- **Synthetic relational tasks (ART, CLEVR-ART) saturate** under LoRA within 6k steps, so they cannot separate methods.
- **Some runs stopped early:** `lora_q_int_slot_lr1e-3` (56,425 steps), `lora_q_int_slot_control_lr3e-4` (59,000), H6 (stopped by decision at 108k–124k), `overfit2_ivl1b_fuse` (broken GPU, no result).
