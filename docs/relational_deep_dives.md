# RelVLM hybrid experiments: deep dives

As of Nov 19, 2026. Exported from the team doc; see `00_START_HERE/` for where every run, script and result lives.

## Overview

The hybrid adds a relational branch (slot attention, then an Abstractor, then a connector) to Qwen3-VL-2B. Our best result is H12: LoRA-first with the step-276k branch reaches **56.5% / 56.4%** on 5,000 PGM test puzzles at 5% data, +3.3 / +3.5 over an equal-steps control (p < 1e-11). At full data the branch adds nothing (H4, H5). A branch trained only on the 60k puzzles (early checkpoint, 26.8% alone) gave no real gain (H13).

This document collects the deep dives from our chat, one section per experiment, with every setting and result. Accuracies on 1,000 validation puzzles carry about ±3 points of noise; test accuracies on 5,000 puzzles carry about ±1.4. Chance is 12.5%.

**Terms used throughout**

| Term | Meaning |
| --- | --- |
| Interleave | The branch's output goes into the prompt as extra "relation tokens" |
| Fusion | The branch's output is blended into the LLM's own image tokens: token = λ · own + (1 − λ) · branch |
| Token gain | A learned number that sets how loud the relation tokens are (interleave) |
| Stage 0 | Slot attention trained on images only (reconstruction, no labels) |
| Stage 1a / branch alone | Abstractor trained with its own linear answer head, no LLM |
| Alignment | Branch and LLM frozen; only the connector (and gain / λ) trains |
| Stage 2 | LoRA trained from scratch together with the branch |
| LoRA-first | Start from the finished LoRA-alone adapter, attach the aligned branch, train briefly |
| Full data / 20% | The 240k PGM neutral training puzzles |
| 5% | The seed-0 60k subset of PGM neutral, the same file list for LoRA alone and the hybrids |
| LoRA alone (5%) | 53.4% val, **52.6% test** (step 14k). At full data: 81.6% val, 81.5% test |

## All experiments at a glance

| # | Dates | Experiment | Runs | LLM | Key result |
| --- | --- | --- | --- | --- | --- |
| **H1** | Oct 29–30 | Frozen-LLM memorisation, rounds 1 and 2 | `overfit_*`, `overfit2_*` (Qwen and InternVL, interleave and fusion) | Frozen | Couldn't even memorise 500 puzzles: 16–33% |
| **H2** | Oct 30 – Nov 1 | Frozen-LLM Stage 1 on full data, 1 epoch | `s1_qwen_*`, `s1_ivl1b_*` | Frozen | 16–19% val, barely above chance |
| **H3** | Nov 8 | Frozen-LLM memorisation with tuned settings (6 layers, higher rate) | `llm_ovf_q_{int,fuse}_lr{3e-4,1e-3}` | Frozen | Now it memorises: interleave 85%, fusion 100% |
| **H4** | Nov 8–11 | Full-data hybrids with LoRA, trained together from scratch; 1 / 9 / 144 tokens, fusion, and a non-relational control | `lora_q_int_{pool,panel,slot}_*`, `lora_q_fuse_*`, `lora_q_int_slot_control_lr3e-4` | LoRA | 79.5–81.4% val vs LoRA alone 81.6%: no gain, and the control ties |
| **H5** | Nov 8–11 | Mixture hybrids with LoRA (30% PGM + CLEVR, VSR, ART, CLEVR-ART) | `lora_mix30_q_int_slot`, `lora_mix30_q_fuse`, `lora_mix30_q_int_slot_control` | LoRA | Family score 87.2 / 87.2; control 87.6: no gain |
| **H6** | Nov 10–15 | Frozen-LLM Stage 1 at 10 epochs, 256-dim | `s1_q_{int_pool,fuse}_{lr3e-4,lr1e-3}_ep10` | Frozen | Best 22.6–27.2% val; stopped at \~100k steps |
| **H7** | Nov 14 | Alignment from the 10-epoch branch (connector only) | `align_q_{int,fuse}_L6ft_lr{3e-4,1e-3}` | Frozen | 35–37.5% val = the branch alone (37.1%): handoff works |
| **H8** | Nov 14–15 | Stage 2 at 5% data, no dropout, plus the branch-off check | `lora_w5_q_{int,fuse}_L6ft`, `diag_branchoff` | LoRA | Shortcut: 13–16% with the branch off; test 49.1% / 38.8%, below LoRA alone |
| **H9** | Nov 15 | Stage 2 at 5% with branch dropout 0.5 | `lora_w5d_q_{int,fuse}_L6ft` | LoRA | Test 55.2% / 54.1% vs 52.6%: dropout fixes the shortcut |
| **H10** | Nov 15–16 | Stronger branch (step 204k): its alignment plus Stage 2 with dropout | `align_q_*_L6ft50_lr3e-4`, `lora_w5d_q_{int,fuse}_L6ft50` | Frozen, then LoRA | Test 55.1% / 55.3%; branch off 49.5% / 49.3% (LoRA leaned on the branch more) |
| **H11** | Nov 15–16 | LoRA-first with the step-204k branch, plus branch-off controls | `lora_lf_q_{int,fuse}_ft50`, `_ctrl` | LoRA (pretrained) | Test 55.1% / 54.9% vs controls 53.2% / 52.9% (+1.9 / +2.0) |
| **H12** | Nov 16–17 | LoRA-first with the step-276k branch, including its alignment | `align_q_*_L6ft276k_lr3e-4`, `lora_lf_q_{int,fuse}_ft276k` | Frozen, then LoRA (pretrained) | **Test 56.5% / 56.4%** (+3.3 / +3.5 over the controls); ties averaging |
| **A1** | Nov 15 | Overlap and averaging check (`diag_overlap`, 1,000 val puzzles) | — | — | At 5% the branch gets 96–131 puzzles right that LoRA misses; averaging +2.2 to +3.5. At full data, nothing to add |
| **A2** | Nov 16–17 | 5,000-puzzle test scoring (`diag_5k`, `diag_5kw`) | 25 models | — | Source of every H8–H12 test number; branch on and off, averaging, per attribute |
| **H13** | Nov 16–19 | Clean 60k pipeline: slots, branch and alignment on the 60k only. Early branch (step 43k) run through the H8, H9 and H12 recipes | `s0_qwen_pgm60k`, `aux_ft60k_*`, `align_q_*_L6ftc60e_lr3e-4`, `lora_{w5,w5d,lf}_q_*_c60e` | Frozen, then LoRA | Branch alone 26.8% test. H12 recipe 53.7% / 53.3%, branch off identical: no gain. H8 recipe below LoRA alone; H9 recipe level with it |

## H1: frozen-LLM memorisation, rounds 1 and 2 (Oct 29–30)

Our first hybrid test: could a branch learn to help a frozen VLM well enough to memorise just 500 puzzles? It couldn't. The best run got 33% of the puzzles it trained on, where LoRA alone gets 100%.

### The question

"Memorise 500 puzzles" is the easiest possible test. It doesn't ask the model to generalise, only to fit the puzzles it trains on. If the branch can't pass even that, something in the setup is blocking the learning signal. So each run was scored on its own 500 training puzzles.

### What was trained and what was frozen

| Part | Status | Size |
| --- | --- | --- |
| ViT (image encoder) | Frozen | — |
| Slot model | Frozen, from Stage 0 (`s0_qwen_pgm` / `s0_ivl1b_pgm`, 32-dim) | — |
| **Abstractor** | **Trained from scratch**: 24 layers, 8 heads, head dim 32, MLP 512 | 3.18M |
| **Connector** | **Trained from scratch**; plus token gain (interleave) or λ (fusion) | 4.26M (Qwen), 1.08M (InternVL) |
| LLM | Frozen, no LoRA | — |

The only learning signal was the LLM's own answer: Yes vs No for each of the 8 candidates (8-way cross-entropy). The Abstractor had to receive its gradient back through the whole frozen LLM.

### Settings shared by all eight runs

| Setting | Value |
| --- | --- |
| Data | 500 PGM training puzzles (seed 0), each turned into 8 candidate images |
| Batch | 2 puzzles × 8 accumulation = 16 per step |
| Panels | 160 px; 3 × 3 grid image |
| Relation tokens | 1 (pooled) for interleave |
| Other | Weight decay 0.01, gradient clip 1.0, bf16, one GPU |
| Scored on | The same 500 puzzles they trained on |

|  | Round 1 (`overfit_*`) | Round 2 (`overfit2_*`) |
| --- | --- | --- |
| Length | 625 steps = 20 epochs | 1,250 steps = 40 epochs |
| Abstractor lr | 1e-4 | 3e-4 |
| Connector lr | 3e-4 | 1e-3 |
| Schedule | Warmup 50, cosine | Warmup 50, constant |
| Fusion λ start | 0.9 | 0.5 |
| Why | First try | Round 1 was still climbing slowly: twice as long, 3× the rates, more branch weight in fusion |

### Results: accuracy on the 500 training puzzles

| Run | Model | Regime | Round | Step 125/250 | Middle | Final | Training loss (start → end) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `overfit_qwen_interleave` | Qwen | Interleave | 1 | 15.6% | 22.2% | **26.8%** | 2.08 → 1.95 |
| `overfit_qwen_fuse` | Qwen | Fusion | 1 | 18.6% | 25.0% | **29.0%** | 2.08 → 1.98 |
| `overfit_ivl1b_interleave` | InternVL | Interleave | 1 | 13.6% | 22.8% | **32.6%** | 2.09 → 1.78 |
| `overfit_ivl1b_fuse` | InternVL | Fusion | 1 | 16.8% | 15.6% | **18.6%** | 2.10 → 2.07 |
| `overfit2_qwen_interleave` | Qwen | Interleave | 2 | 16.6% | 20.8% | **22.2%** | 2.09 → 1.98 |
| `overfit2_qwen_fuse` | Qwen | Fusion | 2 | 15.2% | 11.8% | **14.2%** | 2.08 → 2.08 |
| `overfit2_ivl1b_interleave` | InternVL | Interleave | 2 | 13.6% | 24.6% | **30.6%** | 2.09 → 1.86 |
| `overfit2_ivl1b_fuse` | InternVL | Fusion | 2 | — | — | **No result** | Stopped at step 220 |

Loss for random guessing is 2.08.

| Run | Fusion λ (start → end) | Interleave token gain (start → end) |
| --- | --- | --- |
| Qwen, round 1 | 0.90 → 0.89 | 1.00 → 1.02 |
| InternVL, round 1 | 0.90 → 0.90 | — |
| Qwen, round 2 | 0.50 → 0.49 | 1.00 → 1.03 |
| InternVL, round 2 | 0.50 → 0.51 (stopped early) | — |

**Reference:** LoRA alone on the same 500 puzzles reached 100% within 275–313 steps for both models (`vlm_lora_pc_runs/overfit_pc_*`; r64, lr 2e-4).

### What we learned

| Observation | Meaning |
| --- | --- |
| Best run 32.6%; LoRA alone 100% on the same puzzles | Through a frozen LLM, the branch couldn't even memorise |
| Training loss barely fell (2.08 → 1.8–2.0) | Hardly any learning signal reached the branch |
| Round 2's higher rates and longer training did **worse** for Qwen (26.8 → 22.2, 29.0 → 14.2) | More training wasn't the fix; the setup was the problem |
| Fusion with λ starting at 0.5 collapsed to chance (14.2%) | Blending half-random branch output into the image tokens damaged what the LLM sees |
| λ and gain hardly changed | The LLM didn't lean into the branch, because the branch carried nothing useful yet |

**Why it failed (confirmed later):**

- **The Abstractor was 24 layers deep.** In H3 and the branch-alone tests, 6 layers learned much faster: 94.6% vs 56% on the same 500 puzzles.
- **Its rates were too low.**
- **The signal came only through the frozen LLM.** The random-start Abstractor had to learn PGM rules from a weak gradient sent back through a 28-layer LLM that couldn't adapt.

That led to the current design: train the branch on its own first (Stage 1a), then only align the connector.

**Data issue:** `overfit2_ivl1b_fuse` landed on a node with a broken GPU and ran on the CPU at 0.2 puzzles/s (normal is about 5). It stopped at step 220 with no result.

### In the final write-up

Appendix, as a negative result: one small table (the 7 finished runs plus the LoRA 100% reference). One sentence: "Training the branch through a frozen LLM couldn't memorise 500 puzzles, which motivated training the branch alone first."

## H2: frozen-LLM Stage 1 on full data, 1 epoch (Oct 30 – Nov 1)

The first real Stage 1 runs: train the branch through a frozen LLM on all 240k puzzles for one epoch, then score it on unseen puzzles. All four ended at 16–19% validation, barely above chance. In the same epoch, LoRA alone reached 40–57%.

### The question

H1 showed the branch couldn't memorise 500 puzzles this way. H2 asked whether much more data, one full pass of the training set, would get the branch learning. This was the Stage 1 design from the original plan, run before any tuning.

### What was trained and what was frozen

| Part | Status | Size |
| --- | --- | --- |
| ViT | Frozen | — |
| Slot model | Frozen, from Stage 0 (`s0_qwen_pgm` / `s0_ivl1b_pgm`, 32-dim) | — |
| **Abstractor** | **Trained from scratch**: 24 layers, 8 heads, head dim 32, MLP 512 | 3.18M |
| **Connector** | **Trained from scratch**; plus token gain (interleave) or λ (fusion) | 4.26M (Qwen), 1.08M (InternVL) |
| LLM | Frozen, no LoRA | — |

As in H1, the only learning signal is the LLM's own Yes/No answer for each of the 8 candidates, passed back through the frozen LLM.

### Settings (same for all four runs)

| Setting | Value |
| --- | --- |
| Data | All 240k PGM neutral training puzzles, 1 epoch |
| Steps | 15,000 × 16 puzzles (2 × 8 accumulation) |
| Abstractor lr | 1e-4 |
| Connector lr | 3e-4 |
| Schedule | Warmup 500, cosine to 10% over the whole run |
| Fusion λ start | 0.9 (learned) |
| Relation tokens | 1 (pooled) for interleave |
| Validation | The shared 1,000 validation puzzles, every 1,000 steps |
| Speed | 6–7 puzzles/s (Qwen), 9–10 (InternVL); about 7–10 GPU-hours per run |

### Results: validation accuracy (1,000 puzzles, ±3 points)

| Step | Qwen interleave | Qwen fusion | InternVL interleave | InternVL fusion |
| --- | --- | --- | --- | --- |
| 1,000 | 14.6 | 13.6 | 14.8 | 14.5 |
| 3,000 | 16.9 | 15.1 | 15.9 | 15.7 |
| 5,000 | 17.6 | 15.0 | 15.5 | 14.2 |
| 8,000 | 18.3 | 15.0 | 17.0 | 14.4 |
| 10,000 | 18.8 | 16.3 | 17.1 | 14.5 |
| 12,000 | **19.1** | 16.3 | 16.5 | 15.1 |
| 15,000 (final) | 17.3 | **18.0** | 17.7 | 16.0 |
| **Best** | **19.1%** (12k) | **18.0%** (15k) | **17.9%** (11k) | **16.4%** (13k) |

| Run | Training loss: first 1k → last 1k steps | Training accuracy, last 1k | λ or gain: start → end |
| --- | --- | --- | --- |
| Qwen interleave | 2.075 → 2.041 | 18.9% | gain 1.00 → 1.02 |
| Qwen fusion | 2.076 → 2.060 | 16.5% | λ 0.90 → 0.81 |
| InternVL interleave | 2.077 → 2.041 | 18.3% | gain 1.00 → 1.02 |
| InternVL fusion | 2.083 → 2.073 | 15.0% | λ 0.90 → 0.86 |

Loss for random guessing is 2.079.

| Approach (1 epoch of 240k) | Qwen | InternVL |
| --- | --- | --- |
| **H2: branch through the frozen LLM** | 17–19% | 16–18% |
| Branch alone, 6 layers, lr 3e-4 (`aux_full_*_L6_lr3e-4_ep4`, at step 15k) | 19.0% | 19.0% |
| LoRA alone (val, end of epoch 1) | 56.7% | 40.1% |

### What we learned

| Observation | Meaning |
| --- | --- |
| Best 16–19%, against chance at 12.5% | The hybrid learned almost nothing in one epoch |
| Training loss fell only 0.01–0.04 from 2.08 | Like H1, almost no learning signal reached the branch |
| LoRA alone reached 40–57% with the same data and steps | The LLM *can* learn PGM; the branch path was stuck |
| The branch alone after 1 epoch was also only \~19% | PGM is hard for the branch too; one epoch is too short either way |
| Fusion's λ drifted down (0.90 → 0.81 for Qwen) | The LLM slowly gave the branch more weight, but with no useful signal, accuracy didn't follow |

**Why, in hindsight:**

- **The settings weren't tuned.** 24 layers at lr 1e-4. Later memorisation tests showed 6 layers at 3e-4 to 1e-3 learn far faster: 94.6% vs 27.6% on 500 puzzles for the branch alone.
- **The branch needs many epochs.** Even trained alone and tuned, it reached only 23–24% after 4 epochs and 37% after 10.
- **The signal through the frozen LLM is weak.** H6 (6 layers, 256-dim, 10 epochs through the frozen LLM) reached only 22–27%.

That's why H2 led to the current design: tune and train the branch on its own (Stage 1a), then connect it with alignment (H7). Alignment matched the branch's own accuracy within about 3,000 steps.

### In the final write-up

Appendix, next to H1 and H6, as the case against training the branch through a frozen LLM:

| Way of training the branch | Validation accuracy |
| --- | --- |
| Through the frozen LLM, 1 epoch (H2) | 16–19% |
| Through the frozen LLM, \~7 epochs (H6) | 22–27% |
| On its own, 10 epochs, then aligned (H7) | 36–37% |

Caveat: H2 used untuned settings (24 layers, lr 1e-4), so it shows "the untuned frozen-LLM path doesn't learn in 1 epoch", not that the path can never work. H6 is the stronger evidence for that.

## H3: frozen-LLM memorisation with tuned settings (Nov 8)

The rerun of H1 with a shallower, faster-learning branch. This time the frozen-LLM path memorised 500 puzzles: fusion reached 100% and interleave 85%. But each regime worked at only one of the two learning rates tried, and each needed a different one.

### The question

H1 failed with a 24-layer Abstractor at low rates. The branch-alone memorisation tests showed 6 layers at a higher rate learn far faster. H3 asked: with those settings, can the branch learn *through the frozen LLM* at all? Is the path itself broken, or were the settings the problem?

### What was trained and what was frozen

| Part | Status | Size |
| --- | --- | --- |
| ViT | Frozen | — |
| Slot model | Frozen, from Stage 0 (`s0_qwen_pgm`, 32-dim) | — |
| **Abstractor** | **Trained from scratch**: **6 layers** (H1 used 24), 8 heads, head dim 32, MLP 512 | 0.80M |
| **Connector** | **Trained from scratch**; plus token gain (interleave) or λ (fusion) | 4.26M |
| LLM | Frozen, no LoRA | — |

Same objective as H1 and H2. Qwen only.

### Settings

| Setting | Value |
| --- | --- |
| Data | The same 500 PGM training puzzles as H1 |
| Length | 3,125 steps = 100 epochs (H1: 625 and 1,250) |
| Batch | 16 puzzles per step (2 × 8) |
| Abstractor lr | **3e-4 or 1e-3**: the only difference between the two runs per regime |
| Connector lr | 3e-4 in all four |
| Schedule | Warmup 50, then constant |
| Fusion λ start | 0.5 (learned) |
| Relation tokens | 1 (pooled) for interleave |
| Scored on | The same 500 puzzles, every 250 steps |
| Speed | 7–8 puzzles/s; about 2 GPU-hours per run |

### Results: accuracy on the 500 training puzzles

| Step | Interleave, lr 3e-4 | Interleave, lr 1e-3 | Fusion, lr 3e-4 | Fusion, lr 1e-3 |
| --- | --- | --- | --- | --- |
| 250 | 18.0 | 16.4 | 13.6 | 34.0 |
| 500 | 21.6 | 25.4 | 11.6 | 56.4 |
| 750 | 29.4 | 14.8 | 16.2 | 84.8 |
| 1,000 | 38.4 | 15.8 | 24.6 | 96.6 |
| 1,500 | 47.4 | 35.2 | 15.4 | 99.2 |
| 2,000 | 63.6 | 48.8 | 21.2 | 99.6 |
| 2,500 | 76.2 | 56.0 | 23.8 | **100** |
| 3,125 (final) | **85.0** | 57.4 | 21.4 | **100** |
| **Best** | **85.0%** | **59.4%** | **24.6%** | **100%** (from step 2,250) |

| Run | Training loss: start → end | λ or gain: start → end |
| --- | --- | --- |
| Interleave, 3e-4 | 2.08 → 0.46 | gain 1.00 → 1.00 |
| Interleave, 1e-3 | 2.08 → 1.28 | gain 1.00 → 1.06 |
| Fusion, 3e-4 | 2.08 → 2.03 | λ 0.50 → 0.49 |
| Fusion, 1e-3 | 2.08 → **0.00** | λ 0.50 → 0.47 |

**Compared with H1 and the branch alone (same 500 puzzles):**

| Setup | Interleave | Fusion |
| --- | --- | --- |
| H1: frozen LLM, 24 layers, lr 1e-4 to 3e-4, 625–1,250 steps | 22–27% | 14–29% |
| **H3: frozen LLM, 6 layers, 3,125 steps** | **85.0%** (3e-4) | **100%** (1e-3) |
| Branch alone, 6 layers, lr 3e-4 (6,250 steps) / 1e-3 (3,125 steps) | 94.6% / 82.4% (no LLM, same branch for both regimes) |  |
| LoRA alone (r64, lr 2e-4) | 100% within \~300 steps |  |

### What we learned

| Observation | Meaning |
| --- | --- |
| The best runs reach 85–100%, against H1's 33% at most | **The frozen-LLM path isn't broken.** With a 6-layer Abstractor and enough steps, the gradient through the frozen LLM can train the branch |
| Interleave did best at 3e-4; fusion did best at 1e-3 | **The two regimes need different rates.** Each failed at the other's rate |
| Fusion at 1e-3 reached 100% by step 2,250 with loss ≈ 0 | Fusion can memorise fastest: it changes every image token, so it has the most capacity |
| Interleave at 1e-3 dropped back to \~15% between steps 750 and 1,250, then recovered | That rate is unstable for interleave |
| λ and gain barely moved | All the learning happened inside the branch and connector |

**What H3 did not show:**

- **Generalisation.** Scored only on the 500 training puzzles. On full data the same path reached only 16–27% on validation (H2, H6).
- **Robustness.** Each regime worked at only one of two rates, with one seed.

**How H3 shaped later choices:**

- **6 layers became the default** for every hybrid after this.
- **The rates 3e-4 and 1e-3 were carried into H4.** Caveat: H3 tested a 32-dim branch; H4 used 256-dim.
- **The design shifted** to training the branch on its own and only aligning the connector.

### In the final write-up

Appendix, together with H1:

| Setup | Interleave | Fusion |
| --- | --- | --- |
| H1 (untuned) | 27% | 29% |
| H3 (tuned) | 85% | 100% |

One sentence: "With a shallow Abstractor and a regime-specific learning rate, the frozen-LLM path can memorise; on unseen puzzles it remains slow (H2, H6), so we train the branch alone and then align."

## H4: full-data hybrids with LoRA (Nov 8–11)

The first hybrids trained with LoRA, on all 240k puzzles, with branch and LoRA trained together from scratch. All seven ended level with LoRA alone, at 79.5–81.4% validation against its 81.6%. The non-relational control scored the same as the best hybrid, and both regimes turned the branch's influence down during training.

### The question

With LoRA letting the LLM adapt, does the relational branch beat LoRA alone (81.5% test) when all the data is available? Does the number of relation tokens or the regime matter? A same-size control without relational attention checks whether any gain comes from the relational design or just from extra parameters.

### What was trained and what was frozen

| Part | Status | Size |
| --- | --- | --- |
| ViT | Frozen | — |
| Slot model | Frozen, from Stage 0 (`s0_qwen_pgm_d256`: **256-dim** slots, 2048 decoder) | — |
| **Abstractor** | **Trained from scratch**: 6 layers, 8 heads, **head dim 256**, MLP 512 | 28.34M |
| **Connector** | **Trained from scratch**; plus token gain (interleave) or λ (fusion) | 4.72M |
| **LoRA** | **Trained from scratch**: r16, α32, dropout 0.05, every LLM linear layer | 17.43M |
| LLM base weights | Frozen | — |

All 50.5M trainable parameters started from random (LoRA's standard zero start). Nothing was warm-started.

### The seven runs

| Run | Regime | Relation tokens sent to the LLM | Abstractor lr | Initial λ / gain | Relational attention | Note |
| --- | --- | --- | --- | --- | --- | --- |
| `lora_q_int_pool_lr3e-4` | Interleave | **1** (pooled over everything) | 3e-4 | gain 1.0 | Yes |  |
| `lora_q_int_panel_lr3e-4` | Interleave | **9** (one per panel) | 3e-4 | gain 1.0 | Yes |  |
| `lora_q_int_slot_lr3e-4` | Interleave | **144** (one per slot: 9 panels × 16) | 3e-4 | gain 1.0 | Yes |  |
| `lora_q_int_slot_lr1e-3` | Interleave | 144 | **1e-3** | gain 1.0 | Yes | Stopped at step 56,425 |
| `lora_q_fuse_lr3e-4` | Fusion | none (blended into image tokens) | 3e-4 | **λ 0.5** | Yes |  |
| `lora_q_fuse_lr1e-3` | Fusion | none | **1e-3** | **λ 0.5** | Yes |  |
| `lora_q_int_slot_control_lr3e-4` | Interleave | 144 | 3e-4 | gain 1.0 | **No (control)** | Stopped at step 59,000 |

### Architecture (same in all seven)

| Piece | Settings |
| --- | --- |
| Slot model (frozen) | 16 slots per panel, slot size 256, 3 iterations, 12 × 12 ViT grid per panel, MLP decoder hidden 2048 (Stage 0 only) |
| Abstractor, 28.34M | 6 layers, width 256, 8 heads × 256 (2,048-wide projections), feed-forward 256 → 512 → 256. Each layer: relational cross-attention → feed-forward → self-attention → feed-forward, each with residual and layer norm. Panel embedding 16 × 256. Dropout 0. Answer head 256 → 1, unused here |
| Connector, 4.72M | MLP 256 → 2048 → 2048. Interleave: 1, 9 or 144 relation tokens, scaled to word-embedding size × gain. Fusion: one branch vector per image token inside a panel, from the slots covering that spot, rescaled to the token's size |
| LoRA, 17.43M | r16, α32, dropout 0.05 on every linear layer of all 28 LLM layers (q, k, v, o, gate, up, down) |

### λ and token gain

|  | Fusion λ | Interleave token gain |
| --- | --- | --- |
| Formula | image token = λ · own token + (1 − λ) · branch vector | relation token = normalised branch vector × gain |
| Applies to | Only image tokens inside a panel | The 1 / 9 / 144 relation tokens |
| Initial value | **0.5** | **1.0** |
| Learned? | Yes, stored as a logit, λ = sigmoid(logit) | Yes |
| Learning rate | 3e-4 | 3e-4 |
| Final value | 0.77 (3e-4), 0.74 (1e-3) | 0.78–0.87 |

The 0.5 start came from H3. Later runs (H7–H12) went back to 0.9.

### Learning rates and recipe

| Setting | Value |
| --- | --- |
| Model | Qwen3-VL-2B |
| Data | All 240k PGM neutral puzzles × 4 epochs = 60,000 steps × 16 puzzles |
| Rates | Abstractor 3e-4 or 1e-3; connector, λ / gain and unused head 3e-4; LoRA 1e-4; slots not trained |
| Optimiser | AdamW, weight decay 0.01, gradient clip 1.0 |
| Schedule | Warmup 500 → hold to step 45,000 → cosine to 10% by 60,000 (decay timing matches LoRA alone; its warmup was 20) |
| Objective | LLM only: logit(Yes) − logit(No) per candidate, 8-way cross-entropy |
| Precision | bf16, standard attention, no gradient checkpointing |
| Validation | Shared 1,000 puzzles every 1,000 steps; best checkpoint kept |
| Seed | 0 |
| Speed | 3–6 puzzles/s; about 45–80 GPU-hours per run |

### Results: validation accuracy (1,000 puzzles, ±3 points)

| Step | LoRA alone | 1 token | 9 tokens | 144 tokens | 144, lr 1e-3 | Fusion | Fusion, lr 1e-3 | Control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5k | 34.7 | **38.5** | 35.2 | 35.6 | 35.6 | 32.2 | 33.4 | 36.1 |
| 15k (1 epoch) | 56.7 | 55.6 | 56.8 | 56.3 | 57.1 | 52.9 | 50.7 | 57.1 |
| 30k (2 epochs) | 68.9 | 67.0 | 67.7 | 67.4 | 69.3 | 67.4 | 67.4 | 67.8 |
| 45k (3 epochs) | 73.2 | 72.6 | 72.1 | 74.0 | 74.2 | 74.2 | 72.0 | 74.0 |
| 55k | 78.8 | 78.5 | 79.2 | 80.4 | 78.6 | 80.5 | 80.0 | 79.7 |
| 60k (4 epochs) | **81.6** | 78.6 | 80.0 | **81.3** | — | **81.2** | 80.0 | 81.3 (59k) |
| **Best** | **81.6** | 79.6 | 80.6 | **81.3** | 79.5 | **81.4** | 80.5 | **81.3** |

LoRA alone here is its own run (`qwen_pc_recipe_e1`) on the same validation puzzles; on the 200k test set it scores 81.5%. **The H4 hybrids were never test-scored.**

| Run | Token gain: start → end | λ: start → end |
| --- | --- | --- |
| 1 token | 1.00 → **0.78** | — |
| 9 tokens | 1.00 → 0.86 | — |
| 144 tokens | 1.00 → 0.87 | — |
| 144 tokens, lr 1e-3 | 1.00 → 0.86 | — |
| Control | 1.00 → 0.83 | — |
| Fusion, 3e-4 | — | 0.50 → **0.77** |
| Fusion, 1e-3 | — | 0.50 → 0.74 |

Final training loss was 0.43–0.47 for all seven (training accuracy 81–83%), the same as LoRA alone's 0.43 and 83%.

### What we learned

| Observation | Meaning |
| --- | --- |
| Every hybrid ends at or below LoRA alone (best 81.4% vs 81.6%) | **No gain at full data** |
| The control ties the best hybrid (81.3% vs 81.3%) | The relational design adds nothing measurable here |
| 1, 9 and 144 tokens: 79.6 / 80.6 / 81.3 | More tokens maybe slightly better, within noise |
| Interleave vs fusion: 81.3 vs 81.4 | No difference |
| Abstractor at 1e-3 vs 3e-4 | No difference |
| The 1-token hybrid led early (38.5% vs 34.7% at 5k), gone by 15k | A small head start that LoRA overtakes |
| Token gain fell to 0.78–0.87; λ rose to 0.74–0.77 | **The LLM learned to rely on the branch less** |

**Why, as we understood it later:**

- **The branch learned far slower than LoRA.** The branch alone needs about 10 epochs to reach 37%; here it had 4 epochs from random, while LoRA reached 82%.
- **At full data there's little left for the branch to add.** A1 found averaging even a well-trained branch with full-data LoRA loses 0.5–1.3 points.

**Gaps:**

- **Validation only, one seed.**
- **Two runs stopped early** (56,425 and 59,000 steps). Their chains ended on Nov 14 without resubmitting; the logs don't say why.
- **Branch rates weren't tested at this size** (H3 used 32-dim).
- **No warm start**, though the approved plan called for one; dropped without telling you, as noted in the Nov 9 review.

### Where these choices came from

| Choice | Source | Caveat |
| --- | --- | --- |
| 6 layers | H3 and the branch-alone memorisation tests | Tested at 32-dim, not 256 |
| Abstractor 3e-4 / 1e-3 | H3 (best rates for interleave / fusion) | Tested at 32-dim |
| 256-dim slots | V-PROM recipe from the Slot Abstractor paper | Paper uses 11 slots; we kept 16 |
| λ starting at 0.5 | H3 | Other starting values never tried |
| LoRA r16, lr 1e-4 | The LoRA-alone baseline recipe | Not re-tuned for the hybrid |
| Warmup 500 (baseline 20) | Chosen for the random-start branch | Decay timing matches the baseline |

### In the final write-up

Main results: one row, "full data: hybrid 81.3–81.4% vs LoRA alone 81.6%, control 81.3%: no gain", pointing to A1 for the reason. Appendix: the seven-run table plus the gain and λ drift, as evidence that the LLM itself turned the branch down at full data.

## H5: mixture hybrids with LoRA (Nov 8–11)

Same design as H4, trained on a mix of relational tasks instead of PGM alone. The branch added nothing: all three runs finished within 0.4 points on the family score, and the non-relational control scored highest (87.6 vs 87.2). The synthetic relational tasks were solved almost immediately, and PGM ended near 50% for every run.

### The question

Mixture training is a core part of the project. If the branch learns *general* relational structure, it should help when the model trains on many relational tasks at once. The control asks whether any gain comes from the relational design or from extra parameters.

### What was trained and what was frozen

| Part | Status | Size |
| --- | --- | --- |
| ViT | Frozen | — |
| Slot model | Frozen, `s0_qwen_mix20_d256` (256-dim, trained on an image mixture) | — |
| **Abstractor** | **From scratch**: 6 layers, width 256, 8 heads × 256, feed-forward 512 | 28.34M |
| **Connector** | **From scratch**: MLP 256 → 2048 → 2048, plus token gain or λ | 4.72M |
| **LoRA** | **From scratch**: r16, α32, dropout 0.05, every LLM linear layer | 17.43M |
| LLM base | Frozen | — |

The Stage 0 slot model saw images only (no labels): PGM 20%, CLEVR 25%, COCO 17.5%, VSR 17.5%, ART id-rules 5%, CLEVR-ART RMTS 7.5%, CLEVR-ART id-rules 7.5%.

### The three runs

| Run | Regime | Relation tokens | Initial λ / gain | Relational attention |
| --- | --- | --- | --- | --- |
| `lora_mix30_q_int_slot` | Interleave | 144 (one per slot) | gain 1.0 | Yes |
| `lora_mix30_q_fuse` | Fusion | none | λ 0.5 | Yes |
| `lora_mix30_q_int_slot_control` | Interleave | 144 | gain 1.0 | **No (control)** |

### The training mixture ("S1 mix, 30% PGM")

| Task | Share | Format | Validation size |
| --- | --- | --- | --- |
| PGM neutral | 30% | 8 candidates, pick one | 1,000 (the shared set) |
| CLEVR relational yes/no | 30% | Yes / No | 1,000 |
| CLEVR-ART RMTS | 10% | Choice | 500 |
| CLEVR-ART identity rules | 10% | Choice | 500 |
| VSR (spatial relations, real photos) | 8% | Yes / No | 1,000 |
| ART identity rules | 5% | Choice | 500 |
| ART distribution-of-3 | 3% | Choice | 360 |
| ART same / different | 2% | Yes / No | 40 |
| ART RMTS | 2% | Choice | 480 |

### Settings (shared)

| Setting | Value |
| --- | --- |
| Steps | 39,000 × 16 examples |
| PGM seen | ≈ 187k puzzles (30% of 624k examples), about 0.8 of a PGM epoch |
| Rates | Abstractor 3e-4; connector, λ and gain 3e-4; LoRA 1e-4 |
| Schedule | Warmup 500, hold to step 29,250, cosine to 10% by 39,000 |
| Loss | LLM only. Each example's loss divided by its chance loss (ln 8 for PGM, ln 2 for yes/no) |
| Checkpoint choice | **Family score**: mean accuracy over 5 families (PGM, CLEVR, VSR, ART, CLEVR-ART) |
| Validation | Every 2,000 steps |
| Speed | 4–5.5 examples/s; about 35–45 GPU-hours per run |

### Results: validation accuracy by task

| Step | Run | PGM | CLEVR | VSR | ART (4 tasks) | CLEVR-ART (2 tasks) | Family score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6k | Interleave | 24.4 | 95.3 | 85.7 | 100 | 99.4–100 | 81.0 |
|  | Fusion | 21.5 | 94.4 | 81.7 | 100 | 99.8–100 | 79.5 |
|  | Control | 23.4 | 96.0 | 83.4 | 100 | 100 | 80.6 |
| 20k | Interleave | 34.9 | 96.8 | 85.8 | 100 | 100 | 83.5 |
|  | Fusion | 31.9 | 95.7 | 85.7 | 100 | 100 | 82.7 |
|  | Control | 35.2 | 97.4 | 86.4 | 100 | 100 | 83.8 |
| 30k | Interleave | 37.0 | 97.2 | 86.8 | 100 | 100 | 84.2 |
|  | Fusion | 39.4 | 97.1 | 85.2 | 100 | 99.8–100 | 84.3 |
|  | Control | 40.4 | 95.5 | 86.0 | 100 | 100 | 84.4 |
| **39k (final = best)** | **Interleave** | **50.2** | **99.0** | **87.0** | **100** | **100** | **87.2** |
|  | **Fusion** | **49.2** | **98.9** | **88.0** | **100** | **100** | **87.2** |
|  | **Control** | **50.8** | **99.3** | **88.0** | **100** | **100** | **87.6** |

Gain or λ, start → end: interleave gain 1.00 → 0.95; fusion λ 0.50 → 0.79; control gain 1.00 → 0.91. LoRA alone on PGM only reached 48–53% validation after the same \~187k puzzles, which matches the hybrids' 49–51%.

### What we learned

| Observation | Meaning |
| --- | --- |
| Family scores 87.2 / 87.2 / 87.6; control highest | **No gain from the branch on the mixture** |
| All ART and CLEVR-ART tasks at 99.4–100% by step 6k, in all three runs | These synthetic tasks are **trivially easy** for a LoRA-tuned Qwen |
| CLEVR 99%, VSR 87–88%, the same in all three | No branch effect on natural-language relational tasks |
| PGM 49–51%, equal to LoRA alone after the same PGM exposure | Mixing in other tasks neither helped nor hurt PGM |
| PGM jumped 37–40% → 49–51% in the last 9k steps | Mostly the learning-rate decay |
| λ rose 0.50 → 0.79; gain fell to 0.91–0.95 | Like H4, the LLM turned the branch down |

**Gaps:**

- **No LoRA-alone run on the mixture.** The control is the only comparison.
- **Validation only, one seed.** ART same/different has only 40 examples.
- **The family score is dominated by saturated tasks.** Two of five families sit at 100% for everyone.
- **Sized for equal compute, not equal PGM.** About 187k PGM puzzles, against 960k for H4.

### In the final write-up

Mixture section, a null result: the final per-task table for the three runs. Two sentences: the branch gives no gain on the mixture and the control ties it; ART and CLEVR-ART are saturated by LoRA within 6k steps, so they can't separate methods. A mixture that can show a difference needs harder tasks (held-out-rule PGM, I-RAVEN) or a low-data setting.

### The two kinds of control

"Control" in H4/H5 doesn't mean LoRA only. We used the word for two different things:

|  | **Same-size control** (H4, H5) | **Branch-off control** (H11, H12) |
| --- | --- | --- |
| Run names | `lora_q_int_slot_control_lr3e-4`, `lora_mix30_q_int_slot_control` | `lora_lf_q_int_ft50_ctrl`, `lora_lf_q_fuse_ft50_ctrl` |
| Has the branch? | **Yes**: slots, Abstractor (28.3M), connector, 144 tokens, all trained | Loaded, but **never feeds the LLM** |
| Has LoRA? | Yes, trained from scratch | Yes, the pretrained 5% LoRA, trained further |
| What's removed | Only the **relational bottleneck** inside the Abstractor | The **whole branch** |
| Question it answers | "Is the gain from the *relational design*, or just from a 33M-parameter branch?" | "Is the gain from the *branch*, or just from training LoRA longer?" |
| In effect | Hybrid with a non-relational branch | LoRA alone, trained 3,500 more steps |

| Experiment | Hybrid | Same-size control | LoRA only |
| --- | --- | --- | --- |
| H4 (PGM, full data) | 81.3–81.4% val | 81.3% | 81.6% (separate baseline) |
| H5 (mixture) | 87.2 family score | 87.6 | **Not run** |

Suggested names for the write-up: "same-size non-relational control" for H4/H5, and "branch-off control" for H11/H12.

### What the relational bottleneck is

Each slot carries two pieces of information:

| Piece | Code name | What it holds | Example |
| --- | --- | --- | --- |
| **Appearance** | `z` | What the object looks like: shape, colour, size | "red triangle, large" |
| **Position code** | `s` | Which panel and which slot; no appearance | "slot 3 of panel 2" |

In attention, each object decides **which other objects to look at** (queries and keys), then **collects information from them** (values).

|  | Relational (the hybrid) | Control |
| --- | --- | --- |
| Starting code for each slot | Position only | Position + appearance (z added into s) |
| Who attends to whom | Decided by appearance | Decided by position + appearance |
| What gets passed forward | Position / relation codes only | Position + appearance |
| Output describes | Which objects relate to which | The objects themselves, plus where they are |

**Example:** a red triangle in panel 1 attends to the red triangle in panel 2 because they look alike. But what it takes away is "something at slot 3 of panel 2 matches me", not "red triangle". Appearance can only steer attention; it can't pass through. A rule like "the same shape repeats across a row" should then work for any shape or colour, which is the core claim of the Abstractor and the Slot Abstractor paper.

The control has exactly the same layers and parameter count; only the wiring differs. In H4 and H5 the two tied (81.3% vs 81.3%; 87.2 vs 87.6), but the branch there had barely trained. A fairer test would compare them inside LoRA-first at 5% data, where the branch does help; that hasn't been run.

## H6: frozen-LLM Stage 1 at 10 epochs, 256-dim (Nov 10–15)

The long version of H2 with the H3-style shallow branch: 6 layers, 256-dim, trained through a frozen LLM toward 10 epochs. The runs were still climbing slowly when all four were stopped on Nov 15, at your request. Best results were 22.6–27.2% validation, after about 7 epochs. That is about level with branch-alone runs that also keep their slots frozen.

### The question

H2 (untuned, 1 epoch) learned almost nothing; H3 showed the path *can* memorise with a 6-layer branch. H6 asked: given many epochs on all the data, does the branch learn to *generalise* through a frozen LLM, and how far does it get?

### What was trained and what was frozen

In short: frozen VLM, frozen slots, trainable Abstractor and trainable connector MLP. The connector feeds the LLM; it isn't an answer head. The answer still comes from the frozen LLM.

| Part | Status | Size |
| --- | --- | --- |
| ViT | Frozen | — |
| LLM | Frozen, **no LoRA** | — |
| Slot model | Frozen, `s0_qwen_pgm_d256` (256-dim) | — |
| **Abstractor** | **From scratch**: 6 layers, width 256, 8 heads × 256, feed-forward 512 | 28.34M |
| **Connector MLP** | **From scratch**: 256 → 2048 → 2048; 1 extra token (interleave) or a blend into image tokens (fusion) | 4.72M |
| **λ** or **token gain** | **Trained**: one number | — |
| Answer | The frozen LLM's own Yes vs No for each of the 8 candidates | — |
| Linear answer head (256 → 1) | Present but **not trained or used** | — |

The setup where a trainable head *does* produce the answer is Stage 1a (the branch-alone runs, such as `aux_ft_q_L6_lr3e-4_ep10`), where the LLM isn't used at all.

### Settings

| Setting | Value |
| --- | --- |
| Data | All 240k PGM puzzles, planned 10 epochs |
| Planned length | 150,000 steps × 16 puzzles |
| Abstractor lr | 3e-4 or 1e-3 (per run) |
| Connector, λ, gain lr | 3e-4 |
| Schedule | Warmup 500, hold to step 135,000, then cosine to 10% (never reached) |
| Fusion λ start | 0.9 (learned) |
| Interleave tokens | 1 (pooled) |
| Validation | The shared 1,000 puzzles, every 2,000 steps |
| GPUs | H100 / H200; 3.4–8.3 puzzles/s |

### The four runs and results

| Run | Regime | Abstractor lr | Stopped at step | Best val | Final val |
| --- | --- | --- | --- | --- | --- |
| `s1_q_int_pool_lr3e-4_ep10` | Interleave | 3e-4 | 121,870 | **24.2%** (100k) | 23.5% |
| `s1_q_int_pool_lr1e-3_ep10` | Interleave | 1e-3 | 117,280 | 22.6% (98k) | 20.7% |
| `s1_q_fuse_lr3e-4_ep10` | Fusion | 3e-4 | 108,300 | **27.2%** (104k) | 25.4% |
| `s1_q_fuse_lr1e-3_ep10` | Fusion | 1e-3 | 123,610 | 25.6% (122k) | 25.6% |

| Step (epochs) | Int 3e-4 | Int 1e-3 | Fuse 3e-4 | Fuse 1e-3 |
| --- | --- | --- | --- | --- |
| 10k (0.7) | 16.6 | 17.0 | 18.0 | 16.7 |
| 30k (2) | 17.2 | 19.7 | 20.2 | 20.1 |
| 50k (3.3) | 22.1 | 21.1 | 20.2 | 22.8 |
| 70k (4.7) | 20.7 | 19.8 | 23.4 | 22.5 |
| 90k (6) | 22.8 | 18.9 | 24.2 | 22.4 |
| 100k (6.7) | 24.2 | 21.4 | 23.9 | 22.3 |
| 120k (8) | 23.5 | — | — | 24.5 |

| Run | Training loss: first 2k → last 2k steps | Training accuracy, last 2k | λ or gain: start → end |
| --- | --- | --- | --- |
| Int 3e-4 | 2.07 → 1.94 | 23.8% | gain 1.00 → **1.20** |
| Int 1e-3 | 2.08 → 2.00 | 21.2% | gain 1.00 → **1.38** |
| Fuse 3e-4 | 2.07 → 1.88 | 27.0% | λ 0.90 → **0.38** |
| Fuse 1e-3 | 2.07 → 1.92 | 25.6% | λ 0.90 → **0.37** |

### Compared with the branch alone (same data, same steps)

| Branch | How trained | Val at 100k | Val at 120k | Best |
| --- | --- | --- | --- | --- |
| **H6 fusion, 256-dim, frozen slots** | Through the frozen LLM | 23.9% | — | **27.2%** |
| **H6 interleave, 256-dim, frozen slots** | Through the frozen LLM | 24.2% | 23.5% | 24.2% |
| 32-dim, frozen slots, lr 3e-4 (`aux_full_qwen_L6_lr3e-4_ep10`) | Alone | 25.7% | 27.6% | 30.2% (after decay) |
| 256-dim, frozen slots, lr 1e-3 (`aux_full10_q_d256_lr1e-3`) | Alone | 19.7% | 21.1% | 23.0% |
| 32-dim, **fine-tuned** slots, lr 3e-4 (`aux_ft_q_L6_lr3e-4_ep10`) | Alone | 31.5% | 33.2% | 37.1% (after decay) |

### What we learned

| Observation | Meaning |
| --- | --- |
| Best 22.6–27.2% after \~7 epochs; H2 reached 16–19% after 1 | The path does learn on unseen puzzles, slowly: about 1 point per epoch |
| Roughly level with branch-alone runs with **frozen** slots | The frozen LLM isn't much of a handicap in itself. **Fine-tuning the slots** is what lifted the branch-alone run to 37% |
| Fusion's λ fell 0.90 → 0.37; interleave's gain rose 1.0 → 1.2–1.4 | **The opposite of H4.** A frozen LLM can't learn PGM itself, so it shifts toward the branch: fusion ends 63% branch |
| Training ≈ validation accuracy (24–27%) | No overfitting; the branch was simply slow |
| Fusion ahead of interleave at 3e-4 (27.2 vs 24.2) | Within noise |
| Abstractor 1e-3 no better than 3e-4 | Same as H4 |

**Why it was stopped:** by Nov 15, training the branch alone with fine-tuned slots and then aligning (H7) had reached 36–37% in a few thousand alignment steps. H6 was at 22–27% after 100k+ steps, with 30–40 GPU-hours still to go per run.

**Gaps:**

- **Stopped before the learning-rate decay** (planned from step 135k), which usually adds 3–5 points.
- **Not a clean comparison with the two-step design.** H6 kept the slots frozen; a frozen-LLM run *with* slot fine-tuning was never tried.
- **One seed, validation only.**

### In the final write-up

| Path | Val after \~7–10 epochs |
| --- | --- |
| Branch through the frozen LLM, frozen slots (H6) | 24–27% (stopped before decay) |
| Branch alone, frozen slots | 26–30% |
| Branch alone, fine-tuned slots | 33–37% |
| Branch alone with fine-tuned slots, then aligned (H7) | 36–37%, after \~3k alignment steps |

One sentence: "Training through a frozen LLM is roughly as slow as training the branch alone with frozen slots; fine-tuning the slots and training the branch separately, then aligning, was faster and better." The λ contrast with H4 is worth a sentence too: a frozen LLM leans *into* the branch (λ 0.9 → 0.37), while a LoRA-tuned LLM turns it *down* (λ 0.5 → 0.77), an early sign of the shortcut found in H8.

## H7: alignment from the 10-epoch branch (Nov 14)

The first time a *pre-trained* branch was plugged into the frozen VLM with only the connector trained. It worked straight away: within 1,000 steps the LLM's accuracy matched the branch's own (35–37.5% vs 37.1%), and stayed there.

### The question

H1–H6 trained the branch *through* the frozen LLM, which was slow. By Nov 14 a branch trained on its own, with fine-tuned slots, reached 37.1% validation (`aux_ft_q_L6_lr3e-4_ep10`). H7 asked: with that branch fixed and only the connector trained, can the frozen LLM *read* what the branch knows, and how quickly?

### What was trained and what was frozen

| Part | Status | Source | Size |
| --- | --- | --- | --- |
| ViT | Frozen | Original Qwen | — |
| Slot model | Frozen | **Fine-tuned copy** from the branch-alone run (32-dim) | 0.05M |
| Abstractor | **Frozen** | From the branch-alone run: 6 layers, width 32, 8 heads × 32 | 0.80M |
| Answer head | Frozen, unused | From the branch-alone run | \~0 |
| **Connector** | **Trained** | Values stored in the branch-alone checkpoint, never trained there, so effectively random | 4.26M |
| **Token gain / λ** | **Trained** | Gain starts at 1.0; λ at **0.9** | 1 number |
| LLM | Frozen, no LoRA | Original Qwen | — |

**Loading checks, from the log:** "slot attention loaded … (fine-tuned there; frozen here)"; "branch initialised … new modules: none"; "Abstractor and aux head frozen"; "trainable: branch 4.26M + LoRA 0.00M". The branch's own answer head stayed at 36.4–37.1% at every validation, so the branch loaded unchanged.

### Settings

| Setting | Value |
| --- | --- |
| Data | All 240k PGM puzzles |
| Length | 10,000 steps × 16 puzzles (about 0.7 epoch) |
| Connector lr | **3e-4 or 1e-3** (the "lr" in the run name) |
| Schedule | Warmup 200, then cosine over the whole run |
| Objective | LLM Yes/No per candidate, 8-way cross-entropy |
| Interleave tokens | 1 (pooled) |
| Validation | The shared 1,000 puzzles, every 1,000 steps |
| Speed | 3.4–6 puzzles/s; about 11 GPU-hours per run |

### The four runs and results

| Run | Regime | Connector lr | Val at 1k | Best val | Final val (10k) |
| --- | --- | --- | --- | --- | --- |
| `align_q_int_L6ft_lr3e-4` | Interleave | 3e-4 | **37.5%** | **37.5%** (1k) | 36.4% |
| `align_q_int_L6ft_lr1e-3` | Interleave | 1e-3 | 36.2% | 37.5% (8k) | 36.7% |
| `align_q_fuse_L6ft_lr3e-4` | Fusion | 3e-4 | 34.9% | 36.4% (7k) | 36.3% |
| `align_q_fuse_L6ft_lr1e-3` | Fusion | 1e-3 | 35.4% | 37.4% (4k) | 36.4% |
| *Branch alone (reference)* | — | — | — | *37.1%* | — |

| Step | Int 3e-4 | Int 1e-3 | Fuse 3e-4 | Fuse 1e-3 |
| --- | --- | --- | --- | --- |
| 1,000 | 37.5 | 36.2 | 34.9 | 35.4 |
| 2,000 | 35.5 | 35.9 | 35.5 | 35.9 |
| 3,000 | 35.0 | 35.7 | 36.0 | 36.0 |
| 5,000 | 36.5 | 35.9 | 35.9 | 36.2 |
| 7,000 | 36.3 | 36.5 | 36.4 | 36.5 |
| 10,000 | 36.4 | 36.7 | 36.3 | 36.4 |

| Run | Training loss: first 200 → last 500 steps | Training accuracy, last 500 | λ or gain: start → 1k → 10k |
| --- | --- | --- | --- |
| Int 3e-4 | 1.85 → 1.63 | 37.7% | gain 1.00 → 1.02 → 1.03 |
| Int 1e-3 | 1.77 → 1.63 | 37.7% | gain 1.00 → 1.04 → 1.12 |
| Fuse 3e-4 | 1.81 → 1.64 | 37.4% | λ 0.90 → 0.89 → 0.85 |
| Fuse 1e-3 | 1.77 → 1.64 | 37.3% | λ 0.90 → 0.87 → 0.79 |

Training accuracy was already 30–32% in the first 200 steps.

### What we learned

| Observation | Meaning |
| --- | --- |
| The LLM matches the branch (35–37.5% vs 37.1%) within 1,000 steps | **The handoff works**: the frozen LLM reads the branch's answer almost losslessly |
| Flat from 1k to 10k steps; training loss stops falling by step 1k | Once the connector passes the answer through, the frozen LLM has nothing to add |
| Training ≈ validation accuracy | No overfitting |
| Both rates and both regimes give the same result | Alignment is robust: no tuning needed |
| λ drifts only to 0.79–0.85; gain to 1.03–1.12 | The LLM keeps mostly its own image view and takes a small dose of the branch |
| 36–37.5% here vs 22–27% in H6 after 100k+ steps | Training the branch alone, then aligning, is much faster |

**What it doesn't show:**

- **That the LLM adds anything.** Accuracy equals the branch's; a frozen LLM acts as a reader. Gains come only once LoRA trains as well (H9, H11, H12).
- **That the LLM gets the *same* puzzles right as the branch.** No puzzle-by-puzzle comparison was run.
- **A test score.** Validation only.

**What it fed into:** the level-off rule passed two checkpoints into H8 and H9: interleave `align_q_int_L6ft_lr3e-4` at step 1k (37.5%), and fusion `align_q_fuse_L6ft_lr3e-4` at step 3k (36.0%, λ 0.87; picked over the tied 1e-3 run because its λ was closer to 0.9).

### In the final write-up

Method section: "with the branch and LLM frozen, training only the 4.3M-parameter connector brings the LLM to the branch's own accuracy within 1,000 steps, for every regime and rate tried." The same held for every branch since:

| Branch (alone, val) | After alignment |
| --- | --- |
| 10-epoch (37.1%) | 36–37.5% |
| Step 204k (40.9%) | 40.3–40.6% |
| Step 276k (47.6%) | 47.6–47.9% |
| 60k-only, step 43k (28.9%) | 27.4–28.4% (H13) |

## H8: Stage 2 at 5% data, no dropout (Nov 14–15)

The first hybrid to start from the aligned branch and train LoRA alongside it, on the same 60k puzzles as LoRA alone. It ended *below* LoRA alone: 49.1% (interleave) and 38.8% (fusion) on test, against 52.6%. The branch-off check showed why: the LoRA inside the hybrid had learned almost nothing by itself. That's the **shortcut** problem.

### The question

H7 gave a frozen LLM that reads the branch at 37%. If LoRA now trains on top, at 5% data where LoRA alone is weak, does the combination beat LoRA alone (53.4% val, 52.6% test)?

### What was trained and what was frozen

| Part | Status | Source | Rate |
| --- | --- | --- | --- |
| ViT | Frozen | Original Qwen | — |
| Slot model | Frozen | Fine-tuned copy from the branch-alone run (via H7) | — |
| **Abstractor** | **Trained** (unfrozen again) | Branch-alone run, 10 epochs, 37.1% alone (via H7) | 3e-5 |
| **Connector + gain / λ** | **Trained** | H7: interleave step 1k (37.5%); fusion step 3k (36.0%, λ 0.87) | 3e-5 |
| **LoRA** | **Trained from scratch** | r16, α32, dropout 0.05, all LLM linear layers (17.43M) | 1e-4 |
| LLM base weights | Frozen | Original Qwen | — |

The branch has no loss of its own here; it trains only through the LLM's answer.

### Settings (matching LoRA alone wherever possible)

| Setting | H8 | LoRA alone (`qwen_pc_neutral5pct_recipe`) |
| --- | --- | --- |
| Data | Seed-0 60k subset (identical file list) | Same |
| Steps | 15,000 × 16 (4 epochs) | Same |
| LoRA | r16, α32, 0.05, lr 1e-4 | Same |
| Schedule | Warmup 20; hold to 11,250; cosine to 10% | Same |
| Branch dropout | **None** | — |
| Validation | Shared 1,000 puzzles, every 1,000 steps | Same |
| Runs | `lora_w5_q_int_L6ft`, `lora_w5_q_fuse_L6ft` | — |

### Results: validation over time

| Step | LoRA alone | H8 interleave | H8 fusion |
| --- | --- | --- | --- |
| 1,000 | 20.0 | **37.3** | **37.4** |
| 3,000 | 30.6 | 36.6 | 36.4 |
| 5,000 | 36.0 | 36.1 | 36.8 |
| 7,000 | \~39 | 37.4 | 37.3 |
| 8,000 | 43.1 | 40.0 | 38.7 |
| 10,000 | 46.7 | 43.7 | 37.4 |
| 12,000 | 49.6 | 46.4 | 36.7 |
| 14,000 | **53.4** | **49.7** | 36.4 |
| 15,000 | 52.8 | 49.3 | 36.3 |

| Steps | LoRA alone (loss / train acc.) | H8 interleave | H8 fusion |
| --- | --- | --- | --- |
| 0–500 | 2.06 / 17% | 1.67 / 36% | 1.68 / 36% |
| 4,500–5,000 | 1.68 / 37% | 1.61 / 39% | 1.61 / 39% |
| 9,500–10,000 | 1.36 / 48% | 1.50 / 43% | 1.65 / 37% |
| 14,500–15,000 | **1.03 / 61%** | 1.24 / 54% | **1.59 / 40%** |

λ stayed at 0.87 and the token gain at 1.02. The branch's own answer drifted from 37.9% to 35.6%, since it was changed with no loss of its own.

### The branch-off check (`diag_branchoff`, Nov 15)

The same checkpoints (around step 5k), scored on the 1,000 validation puzzles with the branch switched off (fusion λ = 1, interleave gain = 0):

| Run (step \~5k) | Branch on | Branch off | LoRA alone at step 5k |
| --- | --- | --- | --- |
| Interleave | 38.2% | **13.3%** | 36.0% |
| Fusion | 36.7% | **15.6%** | 36.0% |

### Test results (5,000 puzzles, best checkpoint)

| Model | Checkpoint | Test | vs LoRA alone | p |
| --- | --- | --- | --- | --- |
| LoRA alone | 14k | 52.6% | — | — |
| H8 interleave | 14k | **49.1%** | −3.5 | 3e-6 |
| H8 fusion | 8k | **38.8%** | −13.8 | 8e-68 |

### What we learned

| Observation | Meaning |
| --- | --- |
| Starts at 37% (the branch's level) and stays flat for \~7,000 steps | The hybrid begins well ahead of LoRA alone, then stalls |
| At step 5k: 13–16% with the branch off, vs LoRA alone's own 36% | **Shortcut:** the hybrid's LoRA learned almost nothing itself |
| Training loss flat at \~1.6 for 5k steps, while LoRA alone fell 2.06 → 1.68 | The branch already explains much of the answer, leaving little signal for LoRA |
| Interleave eventually broke out (37 → 49.7% from step 8k) | LoRA started learning, but late, and never caught up |
| Fusion never broke out (\~36–38%) | With the branch blended into every image token, LoRA stayed on the shortcut |
| The branch's own accuracy drifted 37.9 → 35.6% | Training the Abstractor with no loss of its own slowly erodes it |

**In plain words:** the branch gives the model a 37% answer for free from step 1. Learning to read it is much easier for LoRA than learning PGM itself, so LoRA takes the easy route and stops improving. LoRA alone has no such shortcut, so it learns the real task and ends higher.

**What this led to:** H9 (branch dropout: switch the branch off on half the batches) and H11/H12 (LoRA-first: train LoRA fully on its own, then add the branch). Both fixed the shortcut and beat LoRA alone.

**Gaps:** one seed; branch-off check on validation at step \~5k only; the branch wasn't kept frozen, which slightly weakened it.

### Data note: where each piece was trained

Before H8, every branch piece was trained on the full 240k puzzles (the "20%" set). Inside H8 itself it only saw the 60k.

| Step | Abstractor | Slots | Connector | Data |
| --- | --- | --- | --- | --- |
| Stage 0 | — | Trained (images only, no answers) | — | **240k** (20%) |
| Stage 1a (branch alone, 10 epochs) | **Trained from scratch** | Fine-tuned | — | **240k** (20%), with answers |
| Alignment (H7) | Frozen | Frozen | Trained | **240k** (20%) |
| H8 itself | Trained further (lr 3e-5) | Frozen | Trained further | **60k** (5%) |
| LoRA in H8 | — | — | — | **60k** (5%), from scratch |

So the Abstractor had learned from all 240k labelled puzzles over 10 epochs (2.4M puzzle-passes) before H8 started, while LoRA only ever saw the 60k. Our "full data" is 20% of PGM neutral (1.2M puzzles); "5%" is a quarter of that.

**This data caveat applies to every 5% hybrid (H8–H12):** the branch saw four times more labelled puzzles than LoRA. The clean 60k pipeline (H13) removes it.

### In the final write-up

Main text: the shortcut finding, one of the more useful methodological results.

| Model | Branch on | Branch off |
| --- | --- | --- |
| LoRA alone (step 5k) | — | 36.0% |
| Stage 2 without dropout (step \~5k) | 37–38% | **13–16%** |
| Final test (Stage 2 without dropout) | 49.1% / 38.8% | — |
| LoRA alone, final test | 52.6% | — |

One sentence: "Naively training LoRA with a pretrained branch lets the model rely on the branch's answer instead of learning: at step 5k its own LoRA scores near chance, and the final model ends below LoRA alone."

## H9: Stage 2 at 5% data with branch dropout (Nov 15)

H8 with one change: on half the training batches the branch is switched off. That forced LoRA to learn on its own and fixed the shortcut. Both runs beat LoRA alone on test: interleave 55.2% (+2.6, p = 7e-5) and fusion 54.1% (+1.5, p = 0.02). With the branch off they score the same as LoRA alone.

### The question

H8 failed because LoRA relied on the branch instead of learning. If the branch is sometimes missing during training, LoRA can't depend on it. Does that give a hybrid with both a fully learned LoRA *and* the branch's help?

### What dropout does

|  |  |
| --- | --- |
| Flag | `--branch_dropout 0.5`, new code added for this experiment |
| How | For each micro-batch (2 puzzles), a coin flip: with probability 0.5 the branch is off for that batch |
| "Branch off" means | Fusion: λ = 1 (pure own image). Interleave: token gain = 0 (relation token is empty) |
| Training sees | Half the batches with the branch, half without |
| Validation and test | Branch always **on**, unless scored "branch off" on purpose |

### What was trained and what was frozen

Identical to H8: same starting checkpoints, same recipe; only the dropout differs.

| Part | Status | Source | Rate |
| --- | --- | --- | --- |
| ViT, slots, LLM base | Frozen | Original Qwen; slots fine-tuned in the branch-alone run | — |
| **Abstractor** | **Trained** | 10-epoch branch-alone run (37.1% alone) | 3e-5 |
| **Connector + gain / λ** | **Trained** | H7: interleave step 1k; fusion step 3k (λ 0.87) | 3e-5 |
| **LoRA** | **Trained from scratch** (r16, α32, 0.05) | — | 1e-4 |

Data: the same seed-0 60k subset as LoRA alone; 15,000 steps × 16 (4 epochs); warmup 20; hold to 11,250; cosine to 10%. Runs: `lora_w5d_q_int_L6ft`, `lora_w5d_q_fuse_L6ft`.

### Results: validation over time (1,000 puzzles)

| Step | LoRA alone | H8 interleave (no dropout) | **H9 interleave** | **H9 fusion** |
| --- | --- | --- | --- | --- |
| 1,000 | 20.0 | 37.3 | 37.3 | 36.3 |
| 3,000 | 30.6 | 36.6 | 37.1 | 37.4 |
| 5,000 | 36.0 | 36.1 | 38.6 | 37.3 |
| 7,000 | 39.3 | 37.4 | 40.3 | 39.9 |
| 9,000 | 44.9 | 41.5 | 43.3 | 44.9 |
| 11,000 | 48.2 | 44.3 | 49.3 | 47.3 |
| 13,000 | 52.1 | 46.1 | 51.2 | 50.6 |
| 14,000 | **53.4** | **49.7** | 53.6 | 52.9 |
| 15,000 | 52.8 | 49.3 | **55.2** | **56.1** |

**Training accuracy, branch on vs off (from the training batches):**

| Steps | LoRA alone | H9 interleave: on / off | Gap | H9 fusion: on / off | Gap |
| --- | --- | --- | --- | --- | --- |
| 0–500 | 17% | 37% / 16% | +21 | 37% / 15% | +21 |
| 2,000–2,500 | \~28% | 38% / 22% | +17 | 39% / 21% | +18 |
| 4,500–5,000 | 37% | 42% / 30% | +12 | 40% / 30% | +10 |
| 9,500–10,000 | 48% | 48% / 44% | +4 | 48% / 43% | +5 |
| 14,500–15,000 | 61% | 60% / 57% | +3 | 60% / 56% | +3 |

λ stayed at 0.87 and the gain at 1.02. The branch's own answer drifted from 37.7% to 34–35%, as in H8.

### Test results (5,000 puzzles, final step 15k = best)

| Model | Test | vs LoRA alone (52.6%) | p | Only this right / only LoRA alone right |
| --- | --- | --- | --- | --- |
| **H9 interleave** | **55.2%** | **+2.6** | 7e-5 | 583 / 454 |
| **H9 fusion** | **54.1%** | **+1.5** | 0.02 | 560 / 485 |
| H9 interleave, branch off | 52.5% | −0.1 | 0.85 | — |
| H9 fusion, branch off | 51.7% | −0.9 | 0.15 | — |
| *H8 interleave (no dropout)* | *49.1%* | *−3.5* | — | — |
| *H8 fusion (no dropout)* | *38.8%* | *−13.8* | — | — |

Branch on vs off in the same model: interleave +2.7 (p = 6e-8), fusion +2.4 (p = 2e-6).

### What we learned

| Observation | Meaning |
| --- | --- |
| Branch off at the end: 52.5% / 51.7%, vs LoRA alone 52.6% | **Dropout fixed the shortcut**: the hybrid's own LoRA learned as much as LoRA alone |
| Branch on: +2.4 to +2.7 over the same model with the branch off | The branch adds a real, significant gain on top of a fully learned LoRA |
| On/off gap shrinks over training: +21 → +12 → +4 → +3 | **The branch is mostly a head start.** As LoRA learns, it covers most of what the branch knew |
| Validation tracks LoRA alone until step 14k, then pulls ahead | Within noise on 1,000 puzzles most of the way; the test confirms the final gap |
| Interleave +2.6 vs fusion +1.5 | Within noise (55.2 vs 54.1) |
| LoRA learned a little slower inside the hybrid (branch-off training 57% vs 61%) | Losing half its batches costs LoRA a little training, but not on test |

**Gaps:** one seed; the branch saw 240k puzzles, LoRA 60k; LoRA alone wasn't trained to convergence; the hybrid's best is its final step (still rising), LoRA alone's is step 14k; the branch wasn't frozen and drifted a few points; only dropout 0.5 was tried.

### In the final write-up

Main results, as fix #1 for the shortcut:

| Recipe | Test, branch on | Test, branch off |
| --- | --- | --- |
| LoRA alone | 52.6% | — |
| Stage 2, no dropout (H8) | 49.1% / 38.8% | 13–16% (val, step 5k) |
| **Stage 2, dropout 0.5 (H9)** | **55.2% / 54.1%** | 52.5% / 51.7% |

The shrinking on/off gap explains why the gain is modest: most of the branch's knowledge is something LoRA can learn by itself. The lasting part is concentrated in counting ("number" puzzles), where the branch alone beats LoRA alone.

### What token gain is (interleave)

The token gain is one learned number that sets how loud the relation tokens are compared with the text tokens around them. It is interleave's equivalent of λ in fusion.

1. The Abstractor produces its relation vector.
2. The connector (MLP) turns it into a vector of the LLM's size (2048 numbers).
3. That vector is **normalised** to the size of a typical Qwen word embedding.
4. It's **multiplied by the token gain**.
5. The result goes into the prompt as a "relation token" that the LLM reads like any other token.

In the code: relation token = (vector ÷ its size) × (typical word-embedding size) × gain. Normalising stops the token from swamping the prompt or being ignored; the gain is then one simple knob.

| Gain | Effect |
| --- | --- |
| **1.0** (the starting value) | Same size as a normal word |
| **> 1** | Louder: pulls more of the LLM's attention |
| **< 1** | Quieter: the LLM gives it less weight |
| **0** | Empty token: the branch is off |

| Experiment | LLM | Gain: start → end | Reading |
| --- | --- | --- | --- |
| H4 (full data, branch from scratch + LoRA) | LoRA | 1.0 → **0.78–0.87** | Turned the branch down |
| H6 (frozen LLM, 10 epochs) | Frozen | 1.0 → **1.20–1.38** | Turned the branch up: a frozen LLM leans on the branch |
| H7 (alignment) | Frozen | 1.0 → 1.03–1.12 | Small increase |
| H8, H9 (Stage 2) | LoRA | 1.02 → 1.02 | Unchanged at a rate of 3e-5 |
| H11, H12 (LoRA-first) | LoRA | \~1.02 | Unchanged |

Fusion's λ moves the opposite way: λ is how much of the LLM's own image token to keep, so a *falling* λ means more branch. In H6 it fell 0.9 → 0.37 (more branch); in H4 it rose 0.5 → 0.77 (less branch). Both regimes tell the same story.

## H10: Stage 2 with a stronger branch (Nov 15–16)

H9's recipe with a stronger branch: the 50-epoch run's checkpoint at step 204k (40.9% val alone, against 37.1% for H9's branch). The total barely moved: 55.1% (interleave) and 55.3% (fusion) on test, against H9's 55.2% / 54.1%. The LoRA inside learned *less* by itself: with the branch off it scores 49.5% / 49.3%, below LoRA alone's 52.6%. A stronger branch made LoRA lean on it more, even with dropout.

### The question

H9 showed the branch adds +1.5 to +2.6 on top of LoRA. With a better branch, does the gain grow?

### Step 1: alignment with the new branch (`align_q_{int,fuse}_L6ft50_lr3e-4`)

| Setting | Value |
| --- | --- |
| Loaded | Slots + Abstractor from `snapshots/aux_ft_q_L6_lr3e-4_ep50_step204000.pt` (about 13.6 epochs into the 50-epoch run) |
| Trained | Connector + gain / λ only (4.26M); everything else frozen, no LoRA |
| Recipe | Same as H7: all 240k puzzles, connector lr 3e-4, warmup 200, cosine over 10k steps; λ starting at 0.9 |

| Step | Interleave | Fusion | Branch alone |
| --- | --- | --- | --- |
| 1,000 | **40.6%** | 39.3% | 40.9% |
| 3,000 | 39.7% | 40.1% |  |
| 4,000 | 39.4% | **40.3%** |  |
| 10,000 | 39.6% | 39.7% |  |

λ went 0.90 → 0.87 → 0.85; gain ≈ 1.01. Passed on: interleave step 1k, fusion step 3k (40.1%, λ 0.87).

### Step 2: Stage 2 with dropout (`lora_w5d_q_{int,fuse}_L6ft50`)

Identical to H9 except for the starting branch.

| Part | Status | Rate |
| --- | --- | --- |
| ViT, slots, LLM base | Frozen | — |
| Abstractor | Trained (from the step-204k branch) | 3e-5 |
| Connector + gain / λ | Trained (from step 1) | 3e-5 |
| LoRA | Trained from scratch (r16, α32, 0.05) | 1e-4 |
| Branch dropout | 0.5 | — |
| Data / length | The same 60k as LoRA alone; 15k steps; warmup 20; hold to 11,250; cosine | — |

### Results: validation over time

| Step | LoRA alone | H9 interleave (10-epoch branch) | **H10 interleave** | H9 fusion | **H10 fusion** |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 20.0 | 37.3 | **39.9** | 36.3 | **40.0** |
| 3,000 | 30.6 | 37.1 | 41.6 | 37.4 | 41.3 |
| 5,000 | 36.0 | 38.6 | 42.2 | 37.3 | 42.5 |
| 7,000 | 39.3 | 40.3 | 44.3 | 39.9 | 45.5 |
| 9,000 | 44.9 | 43.3 | 47.5 | 44.9 | 48.1 |
| 11,000 | 48.2 | 49.3 | 48.6 | 47.3 | 50.4 |
| 13,000 | 52.1 | 51.2 | 50.6 | 50.6 | 53.5 |
| 14,000 | **53.4** | 53.6 | 54.0 | 52.9 | 55.2 |
| 15,000 | 52.8 | **55.2** | **54.7** | **56.1** | **55.8** |

**Training accuracy, branch on vs off:**

| Steps | LoRA alone | H9 int: on / off | **H10 int: on / off** | H9 fuse: on / off | **H10 fuse: on / off** |
| --- | --- | --- | --- | --- | --- |
| 0–500 | 17% | 37 / 16 | 42 / 16 | 37 / 15 | 42 / 15 |
| 4,500–5,000 | 37% | 42 / 30 | 45 / 28 | 40 / 30 | 46 / 28 |
| 9,500–10,000 | 48% | 48 / 44 | 51 / 40 | 48 / 43 | 51 / 43 |
| 14,500–15,000 | 61% | 60 / **57** | 60 / **54** | 60 / **56** | 61 / **55** |
| Final on–off gap | — | +3 | **+6** | +3 | **+5** |

λ stayed at 0.87 and the gain at 1.01. The branch's own answer drifted slightly (interleave 40.5 → 39.1%; fusion 41.1 → 39.9%).

### Test results (5,000 puzzles, final step 15k = best)

| Model | Branch on | Branch off | vs LoRA alone (on) | p |
| --- | --- | --- | --- | --- |
| LoRA alone | 52.6% | — | — | — |
| H9 interleave (10-epoch branch) | 55.2% | 52.5% | +2.6 | 7e-5 |
| **H10 interleave (step-204k branch)** | **55.1%** | **49.5%** | **+2.5** | 3e-4 |
| H9 fusion | 54.1% | 51.7% | +1.5 | 0.02 |
| **H10 fusion** | **55.3%** | **49.3%** | **+2.7** | 9e-5 |

| Paired comparison | Gap | p |
| --- | --- | --- |
| Stronger vs weaker branch, interleave (H10 vs H9) | −0.1 | 0.87 |
| Stronger vs weaker branch, fusion (H10 vs H9) | +1.2 | 0.05 |
| H10 interleave: branch on vs off | **+5.5** | 3e-21 |
| H10 fusion: branch on vs off | **+6.0** | 3e-23 |

Branch alone on test: 10-epoch 35.3%, step 204k 39.8%.

### What we learned

| Observation | Meaning |
| --- | --- |
| A stronger branch (+4.5 on test alone) adds −0.1 / +1.2 to the hybrid | **Stage 2 doesn't pass a stronger branch through into a better hybrid** |
| Branch off: 49.5% / 49.3%, vs H9's 52.5% / 51.7% and LoRA alone's 52.6% | **The hybrid's own LoRA learned less.** Dropout 0.5 wasn't enough with a stronger branch |
| On/off gap doubled (+2.7 → +5.5 interleave; +2.4 → +6.0 fusion) | The branch contributes more, LoRA less, so the total stays about the same |
| Stronger start (40% vs 37% at step 1k), ahead of H9 most of the way | A bigger head start, but H9 catches up by the final step |
| Branch-off training accuracy ends at 54–55%, vs 56–57% (H9) and 61% (LoRA alone) | The same effect, seen during training |
| Alignment matched the stronger branch within 1,000 steps | Alignment scales with the branch |

**In plain words:** in Stage 2, LoRA and the branch share the work. A stronger branch takes a bigger share, LoRA does less, and the total stays about the same.

**What this led to:** LoRA-first (H11, H12). LoRA is trained fully before the branch is attached, so the branch can't weaken it. H12 confirmed it: with LoRA-first, the step-276k branch gave +3.3 to +3.5 and the hybrid's own LoRA was unchanged.

**Gaps:** one seed; fusion's +1.2 is borderline (p = 0.05); only dropout 0.5 was tried (higher dropout for stronger branches was proposed, not run); the branch saw 240k puzzles, LoRA 60k.

### In the final write-up

| Recipe | Branch alone (test) | Hybrid | Hybrid's own LoRA (branch off) |
| --- | --- | --- | --- |
| Stage 2 + dropout, 10-epoch branch (H9) | 35.3% | 55.2% / 54.1% | 52.5% / 51.7% |
| Stage 2 + dropout, step-204k branch (H10) | 39.8% | 55.1% / 55.3% | **49.5% / 49.3%** |
| LoRA-first, step-204k branch (H11) | 39.8% | 55.1% / 54.9% | 53.3% / 52.8% |
| LoRA-first, step-276k branch (H12) | 45.0% | **56.5% / 56.4%** | 52.5% / 53.2% |

One sentence: "In joint training, a stronger branch is partly offset by a weaker LoRA; training LoRA first keeps it intact, so the gain grows with the branch."

## H11: LoRA-first with the step-204k branch (Nov 15–16)

H11 reverses the order: train LoRA fully on its own first, then attach the aligned branch and train briefly. That kept the LoRA intact and still gave a clean gain over an equal-steps control: interleave 55.1% vs 53.2% (+1.9) and fusion 54.9% vs 52.9% (+2.0) on test, both p < 1e-5.

### The question

H10 showed that in joint training a stronger branch weakens LoRA. If LoRA has already finished learning before the branch arrives, can the branch add to it rather than replace it? And is any gain from the branch, or just from a few thousand more training steps? The control answers the second question.

### Where each piece comes from

Every H11 run starts from the LoRA-alone adapter at step 14k.

| Piece | Source | Status in H11 |
| --- | --- | --- |
| LoRA | LoRA alone at 5% (`qwen_pc_neutral5pct_recipe/best`, step 14k: 53.4% val, 52.6% test) | Frozen for 500 steps, then trained |
| Slots + Abstractor | 50-epoch branch run at step 204k (40.9% val, 39.8% test alone) | **Frozen throughout** |
| Connector + gain / λ | H10 alignment: interleave step 1k (40.6%), fusion step 3k (40.1%, λ 0.87) | Trained |
| ViT, LLM base weights | Original Qwen | Frozen |

New code for this experiment (all opt-in):

- **`--lora_from`** loads a saved LoRA adapter, mapping its weight names and checking that r16 / α32 match.
- **`--freeze_lora_steps 500`** keeps LoRA frozen for the first 500 steps.
- **`--branch_dropout 1.0`** switches the branch off for the whole run; used for the controls, in training and validation.

### Settings

| Setting | Value |
| --- | --- |
| Data | The same seed-0 60k puzzles as LoRA alone |
| Length | 3,500 steps × 16 puzzles (just under 1 epoch) |
| Steps 0–500 | Only the connector (and gain / λ) trains, lr 1e-4 |
| Steps 500–3,500 | Connector (1e-4) + LoRA (2e-5) |
| Schedule | Warmup 50; cosine from step 500 to the end |
| Validation | Shared 1,000 puzzles every 500 steps; best kept |
| Speed | About 6.4 s/step, \~6 GPU-hours per run |

| Run | Regime | Branch |
| --- | --- | --- |
| `lora_lf_q_int_ft50` | Interleave | On |
| `lora_lf_q_int_ft50_ctrl` | Interleave (control) | **Always off** |
| `lora_lf_q_fuse_ft50` | Fusion | On |
| `lora_lf_q_fuse_ft50_ctrl` | Fusion (control) | **Always off** |

Each control loads the same pieces and trains identically, but the branch never reaches the LLM. So a control is LoRA alone trained 3,500 more steps under the same schedule.

### Validation over time (1,000 puzzles, ±3 points)

| Step | Interleave | Control (int) | Gap | Fusion | Control (fuse) | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| 500 *(LoRA still frozen)* | 54.9 | 52.5 | **+2.4** | 53.4 | 52.5 | +0.9 |
| 1,000 | 54.2 | 52.1 | +2.1 | 53.5 | 50.6 | +2.9 |
| 1,500 | 53.9 | 52.7 | +1.2 | 54.0 | 52.6 | +1.4 |
| 2,000 | 54.6 | 52.6 | +2.0 | 54.5 | 52.4 | +2.1 |
| 2,500 | 53.2 | 51.9 | +1.3 | **56.0** | **52.9** | +3.1 |
| 3,000 | **55.1** | 52.9 | +2.2 | 55.0 | 51.7 | +3.3 |
| 3,500 | 53.2 | **53.4** | −0.2 | 54.4 | 52.1 | +2.3 |

Bold = the checkpoint kept for test. Fusion's λ went 0.87 → 0.88.

### Test results (5,000 puzzles; hybrids and controls in the same scoring job, A100)

| Model | Checkpoint | Branch on | Branch off |
| --- | --- | --- | --- |
| LoRA alone | 14k | 52.6% | — |
| Control, interleave | 3,500 | — | 53.2% |
| Control, fusion | 2,500 | — | 52.9% |
| **Interleave** | 3,000 | **55.1%** | 53.3% |
| **Fusion** | 2,500 | **54.9%** | 52.8% |

| Paired question | A | B | Gap | Only A right | Only B right | p |
| --- | --- | --- | --- | --- | --- | --- |
| Does the branch help? (int) | 55.1 | 53.2 | **+1.9** | 244 | 151 | 3e-6 |
| Does the branch help? (fuse) | 54.9 | 52.9 | **+2.0** | 286 | 185 | 4e-6 |
| Do 3,500 more steps help on their own? (int control vs LoRA alone) | 53.2 | 52.6 | +0.6 | 300 | 269 | 0.21 |
| Same, fusion control | 52.9 | 52.6 | +0.3 | 288 | 273 | 0.56 |
| Was the hybrid's LoRA kept intact? (int, branch off vs control) | 53.3 | 53.2 | +0.1 | 191 | 188 | 0.92 |
| Same, fusion | 52.8 | 52.9 | −0.1 | 190 | 195 | 0.84 |
| LoRA-first vs Stage 2 + dropout (H9), interleave | 55.1 | 55.2 | −0.1 | 503 | 508 | 0.90 |

### By attribute (test, hybrid vs its control)

| Attribute | Puzzles | Control | Interleave gain | Fusion gain | Settled branch alone (43.3% overall) |
| --- | --- | --- | --- | --- | --- |
| Number | 539 | 37–39% | **+5.8** | **+8.2** | 48.8% |
| Position | 854 | 47–48% | +2.2 | +3.3 | 40.6% |
| Type | 2,207 | 68% | +2.3 | +1.4 | 59.7% |
| Size | 956 | 24–25% | +1.6 | −0.3 | 21.9% |
| Colour | 2,243 | 50–51% | +1.0 | +2.0 | 28.5% |

### Compared with the other recipes (same step-204k branch)

| Recipe | Hybrid (test) | Its own LoRA (branch off) | Training cost after LoRA alone |
| --- | --- | --- | --- |
| Averaging LoRA alone + branch, no training | 54.2% | — | none |
| Stage 2 + dropout (H10) | 55.1% / 55.3% | 49.5% / 49.3% | 15k steps from scratch |
| **LoRA-first (H11)** | **55.1% / 54.9%** | **53.3% / 52.8%** | **3.5k steps** |

### What we learned

| Observation | Meaning |
| --- | --- |
| +1.9 / +2.0 over an equal-steps control, p < 1e-5 | **The branch causes a real gain**, cleanly separated from extra training |
| Controls ≈ LoRA alone (+0.6 / +0.3, n.s.) | 3,500 more LoRA steps at this low rate don't help by themselves |
| Branch off = control (+0.1 / −0.1) | **LoRA-first keeps LoRA intact**, unlike H10 |
| Interleave already +2.4 at step 500, before LoRA trains | Most of the gain comes from just connecting the branch |
| Same total as H10 (55%) at under a quarter of the training | LoRA-first is cheaper and cleaner |
| Beats averaging with the same branch (55.1 / 54.9 vs 54.2) | Training together does slightly better than averaging the two models' answers |
| Gain largest on "number" (+5.8 / +8.2), where the branch alone beats LoRA | The branch contributes what it's good at: counting |

**Gaps:** one seed; the branch saw 240k puzzles, LoRA 60k; LoRA alone wasn't trained to convergence (affects "vs LoRA alone", not "vs control"); the interleave gap shrank during training (+2.4 at step 500 → about +1–2 later); checkpoints were chosen on the same validation puzzles, so the test numbers are the ones to report.

### In the final write-up

Main results: H11 and H12 together as the core LoRA-first result, with the branch-off controls, the branch-off scoring of each hybrid, and the per-attribute table. H11 shows the method works and the comparison is fair; H12 shows the gain grows with the branch.

## H12: LoRA-first with the step-276k branch (Nov 16–17)

H11's recipe with the strongest branch checkpoint we had, step 276k: 47.6% val and 45.0% test on its own, against 40.9% / 39.8% for H11's branch. The gain over control nearly doubled: interleave 56.5% vs 53.2% (+3.3) and fusion 56.4% vs 52.9% (+3.5) on test. That's our best result so far. The hybrid's own LoRA stayed intact, but the hybrid only *ties* simply averaging LoRA alone with the branch.

### The question

H10 showed a stronger branch doesn't help in joint training, because LoRA weakens. H11 showed LoRA-first keeps LoRA intact. So with LoRA-first, does a stronger branch give a bigger gain?

### Step 1: alignment with the step-276k branch

| Setting | Value |
| --- | --- |
| Runs | `align_q_int_L6ft276k_lr3e-4`, `align_q_fuse_L6ft276k_lr3e-4` |
| Loaded | Slots + Abstractor from `snapshots/aux_ft_q_L6_lr3e-4_ep50_step276000.pt` (about 18 epochs into the 50-epoch run; its best checkpoint at the time) |
| Trained | Connector + gain / λ only (4.26M); LLM frozen, no LoRA |
| Recipe | Same as H7 and H10: all 240k puzzles, lr 3e-4, warmup 200, cosine over 10k steps; λ starting at 0.9 |

| Step | Interleave | Fusion | Branch alone |
| --- | --- | --- | --- |
| 1,000 | 46.4% | 46.8% | 47.6% |
| 2,000 | 47.5% | 46.7% |  |
| 3,000 | **47.6%** | **47.9%** |  |
| 4,000 | 46.8% | 46.1% |  |
| 5,000 | 47.5% | 47.2% |  |
| 10,000 | 48.1% | 46.3% |  |

Fusion's λ went 0.90 → 0.89 (1k) → about 0.86 (5k); gain ≈ 1.01. The branch's own answer stayed at 47.6% throughout.

**Automatic hand-off:** `auto/auto_lf276.py` watched both alignments and launched LoRA-first once each levelled off (at least 3 validations, best not up more than 1 point over the last two). Interleave levelled at 10:03 pm on Nov 16, fusion at 11:28 pm; the step-3k checkpoints were passed on. Each alignment then dropped to tier 3 and ran on to 10k steps, unused.

### Step 2: LoRA-first

| Piece | Source | Status |
| --- | --- | --- |
| LoRA | LoRA alone at 5%, step 14k (53.4% val, 52.6% test) | Frozen for 500 steps, then trained at 2e-5 |
| Slots + Abstractor | Step-276k branch | Frozen |
| Connector + gain / λ | Step 1 above, step-3k checkpoints | Trained at 1e-4 |
| ViT, LLM base | Original Qwen | Frozen |
| Data / length | The same 60k as LoRA alone; 3,500 steps; warmup 50; cosine from step 500 | — |
| Runs | `lora_lf_q_int_ft276k`, `lora_lf_q_fuse_ft276k` | — |

**Controls:** reused from H11 (`lora_lf_q_{int,fuse}_ft50_ctrl`). A branch-off control never uses the branch, so it doesn't depend on which branch is attached. That saved two 6-hour runs.

### Validation over time (1,000 puzzles, ±3 points)

| Step | **Int, 276k** | Int, 204k (H11) | Control (int) | **Fuse, 276k** | Fuse, 204k (H11) | Control (fuse) |
| --- | --- | --- | --- | --- | --- | --- |
| 500 *(LoRA frozen)* | **56.2** | 54.9 | 52.5 | 55.0 | 53.4 | 52.5 |
| 1,000 | 55.3 | 54.2 | 52.1 | 55.5 | 53.5 | 50.6 |
| 1,500 | 56.5 | 53.9 | 52.7 | 56.1 | 54.0 | 52.6 |
| 2,000 | **56.9** | 54.6 | 52.6 | 56.4 | 54.5 | 52.4 |
| 2,500 | 55.7 | 53.2 | 51.9 | 56.9 | 56.0 | 52.9 |
| 3,000 | 56.4 | 55.1 | 52.9 | **57.8** | 55.0 | 51.7 |
| 3,500 | 56.7 | 53.2 | 53.4 | 57.0 | 54.4 | 52.1 |

Bold = checkpoint kept for test. Fusion's λ went 0.869 → 0.873. At step 500, before LoRA trains, interleave is already +3.7 over its control; later LoRA training adds about +0.7 (interleave) and +2.8 (fusion).

### Test results (5,000 puzzles; scoring job 7136162 on L40S, Nov 17 6:15–7:43 am)

| Model | Checkpoint | Branch on | Branch off |
| --- | --- | --- | --- |
| LoRA alone | 14k | 52.6% | — |
| Control, interleave | 3,500 | — | 53.2% |
| Control, fusion | 2,500 | — | 52.9% |
| **Interleave, 276k** | 2,000 | **56.5%** | 52.5% |
| **Fusion, 276k** | 3,000 | **56.4%** | 53.2% |

| Paired question | A | B | Gap | Only A right | Only B right | p |
| --- | --- | --- | --- | --- | --- | --- |
| Does the branch help? (int) | 56.5 | 53.2 | **+3.3** | 368 | 203 | 5e-12 |
| Does the branch help? (fuse) | 56.4 | 52.9 | **+3.5** | 377 | 200 | 2e-13 |
| vs LoRA alone (int) | 56.5 | 52.6 | +3.9 | 447 | 251 | 1e-13 |
| vs LoRA alone (fuse) | 56.4 | 52.6 | +3.8 | 444 | 252 | 3e-13 |
| Stronger vs weaker branch (int, H12 vs H11) | 56.5 | 55.1 | **+1.4** | 279 | 207 | 0.001 |
| Stronger vs weaker branch (fuse) | 56.4 | 54.9 | **+1.5** | 277 | 201 | 0.0006 |
| LoRA kept intact? (int, branch off vs control) | 52.5 | 53.2 | −0.7 | 195 | 230 | 0.10 |
| Same, fusion | 53.2 | 52.9 | +0.3 | 211 | 194 | 0.43 |

### By attribute (test, hybrid vs control)

| Attribute | Puzzles | Control | H11 int / fuse gain | **H12 int / fuse gain** | Settled branch alone |
| --- | --- | --- | --- | --- | --- |
| Number | 539 | 37–39% | +5.8 / +8.2 | **+14.5 / +13.4** | 48.8% |
| Position | 854 | 47–48% | +2.2 / +3.3 | +3.4 / +4.8 | 40.6% |
| Type | 2,207 | 68% | +2.3 / +1.4 | +3.4 / +4.0 | 59.7% |
| Size | 956 | 24–25% | +1.6 / −0.3 | +1.4 / +0.2 | 21.9% |
| Colour | 2,243 | 50–51% | +1.0 / +2.0 | +0.8 / +1.2 | 28.5% |

On "number" puzzles the H12 hybrids reach 50–53%, better than both LoRA alone (\~38%) and the branch alone (49%).

### Compared with averaging (no training)

| Branch | Averaging LoRA alone + branch | LoRA-first (int / fuse) | Difference |
| --- | --- | --- | --- |
| Step 204k (39.8% alone) | 54.2% | 55.1% / 54.9% | **+0.9 / +0.7** |
| Step 276k (45.0% alone) | 56.5% | 56.5% / 56.4% | **0 / −0.1** |

Averaging means averaging the two models' answer scores with equal weight, untuned. No paired test of hybrid vs averaging has been run.

### What we learned

| Observation | Meaning |
| --- | --- |
| +3.3 / +3.5 over control, p < 1e-11 | Our strongest, cleanest result |
| +1.4 / +1.5 over H11 (p ≤ 0.001) as the branch improves 39.8 → 45.0% | **With LoRA-first, a stronger branch adds to the result**, unlike Stage 2 (H10) |
| Branch off ≈ control (−0.7 / +0.3, n.s.) | LoRA stays intact; the whole gain comes through the branch |
| +3.7 already at step 500, before LoRA trains | Most of the gain comes from connecting the aligned branch to the tuned LoRA |
| "Number" gain more than doubles (+6–8 → +13–15) | The stronger branch got better mostly at counting, and the hybrid inherits that |
| Interleave ≈ fusion (56.5 vs 56.4) | The regime doesn't matter here |
| Ties averaging with the same branch | **The joint model adds nothing beyond an ensemble.** With the weaker branch it was about a point ahead |

**Gaps:**

- **One seed.**
- **Data caveat:** the branch saw 240k puzzles over \~18 epochs, LoRA 60k.
- **LoRA alone wasn't trained to convergence.** Affects "vs LoRA alone", not "vs control".
- **Controls reused from H11,** trained on Nov 15 and the hybrids on Nov 16. They were also scored in a different job: the controls on A100, H12 on L40S (see A2). That adds about ±0.5 points of noise, far smaller than the +3.3 / +3.5 gap.
- **The tie with averaging hasn't been tested statistically,** and it crosses scoring jobs.
- **Compute:** the branch behind this result took about 120 GPU-hours, roughly 26 EFLOP.

### In the final write-up

**Headline result:** "at 5% data, LoRA-first with the relational branch beats an equal-steps control by +3.3 to +3.5 points (p < 1e-11), mostly on counting; the gain grows with the branch's own strength."

Key exhibits: the main table (LoRA alone, controls, H11 and H12, branch on and off); a small chart of branch-alone accuracy against hybrid gain over control, one point per branch checkpoint; and the per-attribute table, especially "number". State the tie with averaging openly: it's the main thing a reviewer will push on, and the open gap for future work (a joint stage where the Abstractor adapts).

## A1: overlap and averaging check (Nov 15)

A1 (`relvlm_runs/diag_overlap/`) trains nothing. It takes LoRA alone and the branch alone, two separately trained models, and asks whether they get *different* puzzles right. At 5% data they do: the branch gets 96–131 puzzles right that LoRA misses, and simply averaging the two gains +2.2 to +3.5. At full data there is almost nothing for the branch to add.

### How it was done

|  |  |
| --- | --- |
| Puzzles | The shared 1,000 validation puzzles |
| Models | LoRA alone at 5% (`lora5`, step 14k) and at full data (`lora100`, step 60k); the branch alone at four checkpoints: 10-epoch, step 204k, step 210k, settled (`brset`, step 238k) |
| Per puzzle | Each model's 8 answer scores, recorded |
| Overlap | Count where both are right, only one is, or neither |
| Averaging | Average the two models' scores (equal weight, w = 0.5), take the top answer. Plus a weight picked on one half of the puzzles and scored on the other (cross-validated) |
| Significance | McNemar paired test, averaged model vs LoRA alone |

### Results

| LoRA | Branch (alone) | Both right | Only LoRA | **Only branch** | Neither | Averaged | **Gain vs LoRA** | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5% (51.9%) | 10-epoch (37.1%) | 275 | 244 | 96 | 385 | 54.1% | **+2.2** | 0.04 |
| 5% | step 204k (40.3%) | 302 | 217 | 101 | 380 | 54.7% | **+2.8** | 0.01 |
| 5% | step 210k (42.1%) | 310 | 209 | 111 | 370 | 55.3% | **+3.4** | 0.003 |
| 5% | settled (44.8%) | 317 | 202 | **131** | 350 | 55.4% | **+3.5** | 0.002 |
| Full (81.9%) | 10-epoch | 339 | 480 | 32 | 149 | 81.4% | −0.5 | 0.57 |
| Full | settled | 405 | 414 | **43** | 138 | 81.1% | −0.8 | 0.41 |

The cross-validated weight gave about the same: +2.1 to +3.6 at 5% (best weight ≈ 0.5–0.55), and −0.5 to −0.9 at full data (weight ≈ 0.85–0.95, meaning "mostly ignore the branch").

**By attribute (5% LoRA vs settled branch):**

| Attribute | Puzzles | LoRA | Branch | Averaged |
| --- | --- | --- | --- | --- |
| **Number** | 104 | 31.7% | **47.1%** | 48.1% |
| Size | 199 | 23.6% | 26.6% | 27.6% |
| Position | 157 | 47.8% | 41.4% | 52.2% |
| Type | 464 | 66.6% | 59.5% | 70.3% |
| Colour | 433 | 46.7% | 31.2% | 48.0% |

### What it showed

| Finding | Meaning |
| --- | --- |
| At 5%, the branch is right on 96–131 puzzles that LoRA gets wrong | **The branch knows things LoRA doesn't** at low data |
| Averaging them gains +2.2 to +3.5, rising with branch strength | A built-in reference: what a hybrid should *at least* match |
| At full data, only 32–43 such puzzles; averaging *hurts* slightly | **Full-data LoRA already knows what the branch knows.** This explains H4 and H5 |
| The branch beats LoRA on "number" (47% vs 32%) | **Counting** is where its extra knowledge lives, the attribute where the hybrids gained most |
| "Only branch" counts are about half what independent mistakes would give (131 vs 215) | Both models fail on the same hard puzzles, which caps any combined gain |

### How it's used

- **It explains the full-data null (H4/H5):** nothing for the branch to add.
- **It predicted the low-data gain** before the LoRA-first runs existed.
- **It sets the averaging bar** for the hybrids. A2 repeated the averaging on the 5,000 test puzzles; that's where the 56.5% "tie" for H12 comes from.

Caveat: A1 uses the validation puzzles, which were also used to pick checkpoints. A2's test-set version is the one to quote.

## A2: the 5,000-puzzle test scoring (Nov 16 onward)

A2 is where every test number for H8–H13 comes from. Each model is scored on the same 5,000 unseen test puzzles, with one record per puzzle per model, then compared with paired tests, an averaging reference and a per-attribute breakdown. Through Nov 17 it scored 25 models in five jobs on three GPU types; three more jobs since then added the H13 runs.

### Why it was needed

The 1,000 validation puzzles were **too noisy** (±3 points) and **already used to pick checkpoints**, so scores on them are slightly optimistic. A separate, larger test sample fixes both: ±1.4 points for one model, and much finer for two models compared puzzle by puzzle.

### How it was done

| Part | Detail |
| --- | --- |
| Puzzles | `diag_5k/test5000.tsv`: 5,000 drawn at random (seed 0) from the 200k neutral test set; never used for training or checkpoint choice |
| Scoring | `relvlm_ft/eval.py`, one run per model, recording all 8 answer scores for every puzzle; bf16 |
| "Branch off" versions | Fusion λ = 1, interleave gain = 0 (`--branch_off`) |
| Branch alone | Its own linear answer head (`--readout aux`); no LLM |
| LoRA alone (`lora5`) | Its adapter loaded into our model with the branch off, so it's scored by the same code as the hybrids |
| Rules | Each puzzle's rules read from its file: 3,426 have 1 rule, 1,355 have 2, 213 have 3, 6 have 4 |
| Analysis | `diag_5k/analyze5k.py` → `diag_5k/report.md` |
| Automation | `auto/wait_5k.sh` waits for each group of runs to finish, submits the scoring, logs it in the manifest, then re-runs the analysis |

| Section of the analysis | What |
| --- | --- |
| Accuracy | Each model, with a 95% interval |
| Paired comparisons | Puzzle by puzzle: how many only A got right, how many only B, and the exact McNemar p-value |
| Averaging reference | LoRA alone's and each branch's scores averaged (log-probabilities, equal weight, untuned); best answer taken |
| By attribute | Hybrid vs its control on puzzles with rules on number, position, type, size and colour; a puzzle counts in each attribute it has a rule on |

### The scoring jobs

| Job | When | GPU | Models scored |
| --- | --- | --- | --- |
| 7108389 | Nov 16, 3:35 am | A100 | LoRA alone, H11 hybrids + their branch-off versions, both controls, settled branch, rules |
| 7108710 | Nov 16, 4:00 am | A100 | H8 (no dropout), H9 (dropout) |
| 7120016 | Nov 16, 3:00 pm | H200 | H9 branch-off, H10 interleave (on / off), branch alone at 10-epoch / 204k / 276k |
| 7123810 | Nov 16, 6:25 pm | L40S | H10 fusion (on / off) |
| 7136162 | Nov 17, 6:15 am | L40S | H12 hybrids (on / off) |
| 7159316 | Nov 18, 10:47 am | L40S | H13: H12 recipe on the early 60k branch (on / off) |
| 7159327 | Nov 18, 10:48 am | A100 | H13: early 60k branch alone (`br60e`) |
| 7170306 | Nov 19, 7:25 am | L40S | H13: H8 and H9 recipes on the early 60k branch (on / off) |

**Why the GPU type matters:** in bf16, different GPU types round slightly differently; this flipped about 0.5% of answers when a run moved between GPU types. A comparison *across* jobs carries roughly ±0.5 points of extra noise that the McNemar test doesn't include.

| Comparison | Same job? | Affected? |
| --- | --- | --- |
| H11 vs its controls (+1.9 / +2.0) | Yes (A100) | No |
| H12 vs controls (+3.3 / +3.5) | No (L40S vs A100) | Slightly; too large to be explained by this |
| H12 vs averaging (tie) | No | **Yes**: a tie within ±0.5 can't be called either way |
| H10 fusion vs H9 fusion (+1.2, p = 0.05) | No (L40S vs A100) | **Yes**: borderline |
| H13 runs vs LoRA alone and controls (gaps of 0.4–1.1) | No (L40S vs A100) | **Yes**: not settled |

### All results (5,000 test puzzles)

| Group | Model | Test | Branch off |
| --- | --- | --- | --- |
| Baseline | LoRA alone (step 14k) | **52.6%** | — |
| Branch alone | 10-epoch / step 204k / settled / step 276k | 35.3 / 39.8 / 43.3 / 45.0% | — |
| H8 (Stage 2, no dropout) | interleave / fusion | 49.1 / 38.8% | — |
| H9 (Stage 2, dropout) | interleave / fusion | 55.2 / 54.1% | 52.5 / 51.7% |
| H10 (Stage 2, 204k branch) | interleave / fusion | 55.1 / 55.3% | 49.5 / 49.3% |
| H11 controls | interleave / fusion | 53.2 / 52.9% | (always off) |
| H11 (LoRA-first, 204k) | interleave / fusion | 55.1 / 54.9% | 53.3 / 52.8% |
| **H12 (LoRA-first, 276k)** | interleave / fusion | **56.5 / 56.4%** | 52.5 / 53.2% |
| H13 branch alone | early 60k branch (step 43k) | 26.8% | — |
| H13, H8 recipe | interleave / fusion | 49.3 / 50.0% | 43.7 / 46.9% |
| H13, H9 recipe | interleave / fusion | 53.2 / 51.6% | 53.3 / 50.8% |
| H13, H12 recipe | interleave / fusion | 53.7 / 53.3% | 53.7 / 53.3% |

**Averaging reference:**

| LoRA | Branch | Averaged | Gain | p |
| --- | --- | --- | --- | --- |
| LoRA alone | 10-epoch | 53.7% | +1.1 | 0.01 |
| LoRA alone | step 204k | 54.2% | +1.6 | 0.001 |
| LoRA alone | settled | 55.5% | +2.9 | 4e-9 |
| LoRA alone | step 276k | 56.5% | +3.9 | 2e-14 |
| H11 control (int) | settled | 56.5% | +3.3 | 2e-13 |
| H11 control (fuse) | settled | 56.6% | +3.7 | 8e-16 |

### What it established

| Finding | Evidence |
| --- | --- |
| Hybrid gains are real, not validation noise | Every key H9–H12 gap holds on 5,000 unseen puzzles with p < 1e-4 |
| Fair comparisons | Controls and branch-off scoring separate "the branch" from "more training" |
| Averaging gain grows with branch strength (+1.1 → +3.9) | The same pattern as the hybrids |
| The gain sits on "number" puzzles | Per-attribute breakdown, consistent across H11 and H12 |

### Caveats

- **Several jobs on three GPU types** (above). Large gaps are safe; gaps under about 1 point across jobs aren't settled.
- **5,000 of the 200k test puzzles:** ±1.4 points for single accuracies; paired comparisons are much more precise.
- **LoRA alone is scored inside our code** (branch off, its adapter loaded). On validation its own code gave 53.4% at step 14k, our trainer 52.5%, and `eval.py` 51.9%. Every A2 number comes from the same `eval.py`, so comparisons inside A2 are like for like.
- **Averaging used a fixed, untuned weight (0.5).** A1 showed a tuned weight gives about the same.

**How to remove the GPU caveat:** re-score every model in a single job on one GPU type, about 6 GPU-hours. You chose "Not now" on Nov 17.

## H13: clean 60k pipeline, early branch (Nov 16–19)

H13 removes the data caveat from H8–H12: slots, branch and alignment are all trained on the same 60k puzzles LoRA alone uses. The first results use an early checkpoint of the 60k branch (step 43k, 28.9% val, 26.8% test alone). With it, **no recipe beat LoRA alone**. In the H12 recipe the model ignores the branch completely: 53.7% / 53.3% with the branch on, and the same with it off.

### The question

Every branch in H8–H12 had learned from all 240k labelled puzzles, four times what LoRA saw. Does the hybrid still help when the branch sees only the 60k? You asked to run H8, H9 and H12 on the current 60k branch right away ("we can use the fully finished ones later as well").

### The pipeline

| Step | Run | What it does | Settings | Result |
| --- | --- | --- | --- | --- |
| 1. Slots | `s0_qwen_pgm60k` | Stage 0 on the images of the 60k puzzles only | Same as `s0_qwen_pgm`: 16 slots, 32-dim, 100k steps | Recon. error 0.326 (240k version: 0.331); all 16 slots used |
| 2. Branch alone | `aux_ft60k_q_L6_lr3e-4`, `aux_ft60k_q_L6_lr1e-3` | Stage 1a on the 60k only, from the 60k slots | 6 layers, width 32; Abstractor lr 3e-4 or 1e-3; head 3e-4; slots fine-tuned at 3e-5; 150k steps (40 epochs); cosine from step 135k | Still running (below) |
| 3. Early snapshot | `snapshots/aux_ft60k_q_L6_lr3e-4_step43000.pt` | Branch taken at step 43k for this early set | — | 28.9% val, 26.8% test alone |
| 4. Alignment | `align_q_{int,fuse}_L6ftc60e_lr3e-4` | Connector only, on the 60k only | Same as H7: lr 3e-4, warmup 200, cosine over 10k; λ starting at 0.9 | Levelled at step 2k: 28.4% / 27.4% |
| 5. Hybrids | `lora_{w5,w5d,lf}_q_{int,fuse}_c60e` | The H8, H9 and H12 recipes, unchanged | As in H8, H9 and H12; started by `auto/auto_c60.py` when each alignment levelled off | Below |

### Validation (1,000 puzzles, ±3 points)

**Alignment:**

| Step | 1k | 2k | 3k | 5k | 7k | 9k | 10k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Interleave | 28.1 | **28.4** | 28.1 | 27.2 | 28.2 | 29.1 | 28.6 |
| Fusion | 26.9 | **27.4** | 27.3 | 26.6 | 28.1 | 27.1 | 27.0 |

Bold = the checkpoint passed on. The LLM matched the branch (28.9%), as in every alignment so far.

**H12 recipe (LoRA-first, 3,500 steps):**

| Step | 500 | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 3,500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Interleave | 52.4 | 51.6 | 52.6 | 53.6 | 52.7 | **54.1** | 53.8 |
| Fusion | 52.4 | 51.4 | 52.3 | 52.4 | 51.9 | 51.8 | **52.7** |
| Control (H11) | 52.5 | 52.1 / 50.6 | 52.7 / 52.6 | 52.6 / 52.4 | 51.9 / 52.9 | 52.9 / 51.7 | 53.4 / 52.1 |

Control cells give interleave / fusion. At step 500, before LoRA trains, the hybrids sit level with the control (52.4 vs 52.5), where the 276k branch was already +3.7.

**H8 and H9 recipes (Stage 2, 15,000 steps):**

| Step | LoRA alone | H8 recipe int | H8 recipe fuse | H9 recipe int | H9 recipe fuse |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 20.0 | 27.9 | 27.9 | 29.4 | 28.5 |
| 3,000 | 30.6 | 28.6 | 27.8 | 28.7 | 28.1 |
| 5,000 | 36.0 | 26.0 | 26.9 | 34.2 | 33.5 |
| 7,000 | 39.3 | 28.6 | 33.5 | 36.6 | 36.2 |
| 9,000 | 44.9 | 35.6 | 37.9 | 41.6 | 41.1 |
| 11,000 | 48.2 | 41.2 | 43.4 | 46.0 | 43.1 |
| 13,000 | 52.1 | 47.4 | 48.6 | 50.4 | 47.4 |
| 14,000 | **53.4** | 50.6 | 50.3 | 52.3 | 49.7 |
| 15,000 | 52.8 | **51.7** | **51.5** | **54.9** | **52.7** |

The H8 recipe stalled at \~28% (the branch's level) for 7,000 steps, the same shortcut as H8.

### Test results (5,000 puzzles; hybrids scored on L40S, LoRA alone and controls on A100)

| Recipe | Regime | Checkpoint | Branch on | Branch off | vs LoRA alone (52.6%) | p | vs control | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H12 (LoRA-first) | Interleave | 3,000 | **53.7%** | 53.7% | +1.1 | 0.02 | +0.5 | 0.19 |
| H12 (LoRA-first) | Fusion | 3,500 | **53.3%** | 53.3% | +0.7 | 0.13 | +0.4 | 0.29 |
| H9 (dropout 0.5) | Interleave | 15,000 | 53.2% | 53.3% | +0.6 | 0.33 | — | — |
| H9 (dropout 0.5) | Fusion | 15,000 | 51.6% | 50.8% | −1.0 | 0.14 | — | — |
| H8 (no dropout) | Interleave | 15,000 | 49.3% | 43.7% | −3.3 | 3e-6 | — | — |
| H8 (no dropout) | Fusion | 15,000 | 50.0% | 46.9% | −2.6 | 2e-4 | — | — |
| Branch alone | — | step 43k | 26.8% | — | — | — | — | — |

| Paired question | Gap | p |
| --- | --- | --- |
| H12 recipe: did its own LoRA keep up? (branch off vs control, int / fuse) | +0.4 / +0.4 | 0.24 / 0.31 |
| H9 recipe: branch on vs off (int / fuse) | −0.1 / +0.8 | 0.92 / 0.09 |
| H8 recipe: branch on vs off (int / fuse) | **+5.6 / +3.1** | 8e-20 / 1e-7 |

**By attribute, H12 recipe interleave vs control:** number −0.9, position +1.1, type +1.4, size +1.3, colour +0.2. With the 276k branch, number was +14.5.

### What we learned

| Observation | Meaning |
| --- | --- |
| H12 recipe: branch on = branch off (53.7 = 53.7; 53.3 = 53.3) | **The LLM ignores this branch.** With the 276k branch, switching it off cost about 4 points |
| H12 recipe vs control: +0.5 / +0.4, not significant | No real gain. The +1.1 over LoRA alone comes from the extra 3,500 LoRA steps, and crosses GPU types |
| H9 recipe level with LoRA alone; on ≈ off | The branch adds nothing here either |
| H8 recipe below LoRA alone, but on beats off by 3–6 points | The shortcut again: the model leans on a weak branch, and its own LoRA ends weaker (43.7% / 46.9% off) |
| No "number" gain (−0.9) | The branch hasn't learned the counting that made the earlier branches useful |
| The branch alone is 26.8% on test, against 35.3–45.0% for every earlier branch | **Likely reason: this branch is too weak** to know anything LoRA doesn't |

**So far:** with a branch trained on only 60k puzzles for 43k steps, the hybrid gives no gain. The 5% gains in H9–H12 relied on a branch that saw four times the labelled data over many epochs. The finished 60k branches decide whether 60k-only training can work, but they are plateauing at 29–31% validation (next section), well below the 45% that made the 276k branch useful.

**Gaps:** one seed; an early branch checkpoint; hybrids and references were scored on different GPU types, so gaps under \~1 point aren't settled; the H12-recipe controls are H11's, which used the 240k slots (a branch-off control never uses the branch, so this doesn't change it).

## Still running and open decisions

Four branch runs are still training as of 10:15 am, Nov 19. When the two 60k branches finish (around 10 pm tonight), `auto/auto_c60.py` runs their alignment and the H8, H9 and H12 recipes, then scores them on test automatically.

| Run | What it is | Step | Best val (step) | Expected finish |
| --- | --- | --- | --- | --- |
| `aux_ft60k_q_L6_lr3e-4` | 60k-only branch, lr 3e-4 | 125k / 150k | 29.4% (101k); training accuracy \~42% | \~10 pm Nov 19 |
| `aux_ft60k_q_L6_lr1e-3` | 60k-only branch, lr 1e-3 | 125k / 150k | 30.8% (71k) | \~10 pm Nov 19 |
| `aux_ft_q_L6_lr3e-4_ep50` | 50-epoch branch, 6 layers (source of the 204k and 276k checkpoints) | 430k / 750k | **51.5%** (424k) | \~6 days |
| `aux_ft_q_L24_lr8e-5_ep50` | 50-epoch branch, 24 layers | 370k / 750k | 48.4% (366k) | \~9 days |

The better of the two 60k branches will be aligned and then run through the H8, H9 and H12 recipes. H12-recipe test results should arrive around midday Nov 20, and H8/H9 about a day later. The 60k branches' training accuracy keeps rising while validation is flat since about step 75k, a sign of overfitting.

**Decisions waiting for you:**

| Decision | What it would answer |
| --- | --- |
| Longer LoRA-alone baseline at 5% | Whether the gains over LoRA alone survive a fully converged baseline (its loss was still falling) |
| Single-job re-score of every model (\~6 GPU-hours) | Removes the GPU-type caveat, including the H12-vs-averaging tie; you chose "Not now" on Nov 17 |
| Relational vs same-size control inside LoRA-first | Whether the relational bottleneck matters where the branch actually helps |
| Higher branch dropout for stronger branches | Whether Stage 2 can keep LoRA intact with a strong branch (H10) |
| A joint stage where the Abstractor adapts | Whether a hybrid can beat simple averaging (H12 ties it) |
| LoRA-first from a newer 50-epoch checkpoint (now 51.5% val) | Whether the gain keeps growing with branch strength |
| Phase 2 / held-out-rule split | Whether the branch helps on rules never seen in training |
