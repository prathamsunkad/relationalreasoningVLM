# RelVLM: All Results, H1 to H15

Nov 24, 2026 (exported from the team doc). In `00_START_HERE/EXPERIMENTS.md` the per-epoch 5% runs are **H16** and the unfrozen-Abstractor check is **H17**.

## Summary

Adding a relational branch (slot attention + Abstractor) to Qwen3-VL-2B adds **+10.5 points** on PGM test puzzles over LoRA alone when the LoRA is undertrained (55.2% → 65.5–65.9%, 20% data, epoch 1), and nothing once the LoRA is fully trained (81.2% → 81.3%). The gain always tracks how many puzzles the branch solves that the LoRA misses.

The branch helps most when it was trained on **more data or for longer** than the LoRA. With everything at matched 5% data the gain is small (+0.6 points on the finished 5% LoRA; about +2 points on undertrained 5% LoRAs, not yet test-confirmed). Ablations show the LLM really reads the branch: removing it costs 8–12 points, and feeding it the wrong puzzle's features costs more.

| Result (test, 5,000 puzzles, ±1.4 points) | LoRA alone | LoRA + branch | Gain | What it shows |
| --- | --- | --- | --- | --- |
| 20% data, LoRA after 1 epoch (H15) | 55.2% | **65.5% / 65.9%** | +10.3 / +10.7 | Best result: beats both parts, p < 1e-14 |
| 5% data, branch trained on 20% (H14) | 52.8% | 63.2% / 62.9% | +10.4 / +10.1 | Big gain, but only +1 over the branch alone (62.0%) |
| 5% data, branch on 20%, older branch (H12) | 52.6% | 56.5% / 56.4% | +3.9 / +3.8 | Gain grows with branch strength |
| 5% data, everything at 5% (H13) | 52.6% | 53.8% / 53.4% | +0.6 (vs control) | Matched data, finished LoRA: no real gain |
| 20% data, LoRA after 4 epochs (H15) | 81.2% | 81.4% / 81.3% | +0.1 to +0.2 | Strong LoRA: nothing to add |

Pairs of numbers are interleave / fusion, the two ways of feeding the branch to the LLM (explained in the next section).

## How to read this document

This section explains every idea the rest of the document uses, from zero. Read it once; later sections assume it.

### The task: PGM puzzles

1. A PGM puzzle (Procedurally Generated Matrices) is a 3 × 3 grid of panels with the last panel missing. Each panel holds shapes.
2. The shapes in each row follow hidden rules on five attributes: **number** (how many shapes), **position**, **type** (shape), **size** and **colour**. For example: "the number of shapes goes up by one across each row".
3. Eight candidate answers are given. Exactly one completes the grid correctly. Guessing gives **12.5%** (1 in 8).
4. We use the "neutral" split: training, validation and test puzzles come from the same distribution.

### The model and the two ways we teach it

- **Qwen3-VL-2B** is a 2-billion-parameter vision-language model (VLM). A **ViT** (vision encoder) turns each image into tokens; an **LLM** (language model) reads those tokens with the question and answers.
- **Zero-shot**, the VLM gets 12.85%: chance. It has to be trained.
- **LoRA** is a cheap way to fine-tune the LLM: small adapter matrices (17.43M parameters) added to every linear layer while the original weights stay frozen. **"LoRA alone" is our baseline.**
- **How a puzzle is scored:** for each of the 8 candidates, the model sees the grid with that candidate filled in and is asked "is this correct?". Its score is logit(Yes) − logit(No). The candidate with the highest score is its answer. Training uses an 8-way cross-entropy over those scores.

### Our addition: the relational branch

The branch is a separate module that looks at the puzzle's image features and tries to work out the relations between panels. It has three parts:

| Part | What it does | Size |
| --- | --- | --- |
| **Slot attention** | Splits each panel into 16 "slots", one per object-like region, from the ViT features | \~0.05M (32-dim) |
| **Abstractor** | A transformer whose attention only compares slots with each other (relational cross-attention), so it carries relations, not raw appearance. 6 layers in most runs | 0.80M (6 layers, width 32) |
| **Connector** | A small MLP that maps the Abstractor's output into the LLM's token space | 4.26M |

The branch can also answer on its own, through a linear "answer head" on the Abstractor. We call that **"branch alone"**.

### The two ways to plug the branch into the LLM

| Name | How it works | Learned knob |
| --- | --- | --- |
| **Interleave** | The branch's output is added to the prompt as extra "relation tokens" (1, 9 or 144 of them) | **Token gain**: how loud the relation tokens are |
| **Fusion** | The branch's output is blended into the LLM's own image tokens: token = λ · own + (1 − λ) · branch | **λ**: how much of the original token is kept (starts at 0.9) |

### Training stages

| Stage | What trains | What is frozen |
| --- | --- | --- |
| **Stage 0** | Slot attention, by reconstructing ViT features (no labels) | Whole VLM |
| **Stage 1a / branch alone** | Abstractor + answer head (slots fine-tuned slowly in later runs) | ViT; LLM not used |
| **Frozen-LLM Stage 1** | Abstractor + connector, learning only through the LLM's answer | ViT, slots, LLM |
| **Alignment** | Connector only (and gain or λ) | Everything else, including the trained Abstractor |
| **Stage 2** | LoRA from scratch + the branch together | ViT, slots, LLM base weights |
| **LoRA-first** | Start from a finished LoRA; attach the aligned branch; train the connector, then LoRA gently, for 3,500 steps | Abstractor, slots, ViT, LLM base |

### Data sizes

| Name | Puzzles | Steps per pass (epoch) |
| --- | --- | --- |
| **20%** (also "full data") | 240,000 PGM neutral training puzzles | 15,000 |
| **5%** | A fixed 60,000-puzzle subset of the 240k (seed 0) | 3,750 |

One training step = 16 puzzles (2 puzzles × 8 gradient-accumulation steps).

### Words used for comparisons

- **Control**: the same recipe with the branch switched off for the whole run. It separates "the branch helped" from "more training helped".
- **Branch off**: a trained hybrid scored with the branch disconnected (fusion λ = 1; interleave relation tokens zeroed). Shows how much the LLM relies on it.
- **Averaging**: no training. Average LoRA alone's and the branch alone's answer scores (equal weight) and take the top answer. A simple ensemble, the bar a joint model should beat.
- **Val** = 1,000 shared validation puzzles (±3 points; also used to pick checkpoints). **Test** = 5,000 test puzzles never used for anything else (±1.4 points). **p** = exact McNemar test on the same puzzles: small p means the difference is not noise.

## How to frame the project

Frame the project as **"when does an explicit relational module help a fine-tuned VLM?"**, not "our branch beats LoRA". The data supports the first claim strongly and the second only in some regimes.

### The claim we can defend

> A slot-based relational branch carries relational information, especially counting, that a fine-tuned VLM does not learn when its fine-tuning is limited. Attached to the VLM with the right recipe, it adds up to +10 points on PGM test puzzles, the LLM demonstrably uses it, and the size of the gain is predicted by how many puzzles the branch solves that the VLM misses. Once the VLM is fully trained, that complementarity disappears and so does the gain.

Each part of that claim has its own evidence:

| Part of the claim | Evidence | Section |
| --- | --- | --- |
| The branch knows something the VLM does not | A1 / A1b overlap: the branch alone solves 874–979 test puzzles the weak LoRAs miss; "number" puzzles 70.9% vs 39.5% | A1b |
| The right recipe turns that into accuracy | H8 (shortcut) → H9 (dropout fixes it) → H11/H12 (LoRA-first) | H8–H12 |
| Up to +10 points | H14: 52.8% → 63.2%; H15 epoch 1: 55.2% → 65.5–65.9% | H14–H15 |
| The LLM really uses it | A3: removing the branch costs 8–12 points; another candidate's or puzzle's features cost 24–47 (H14, H15 epoch 1) | A3 |
| Gain tracks complementarity | Gain vs LoRA strength: +10 (weak LoRA), +2.3 averaging at epoch 2, 0 at epoch 4 | A1b, H15 |
| It vanishes for a strong VLM | H4 (full data, from scratch) and H15 epoch 4: within ±0.5 | H4, H15 |

### What we must say openly

1. **The biggest gains use a branch trained on more data than the LoRA** (the 240k puzzles vs the LoRA's 60k in H12 and H14). At matched 5% data the gain is +0.6 on the finished LoRA (H13). Present H14 as "a stronger relational module" and H13 as the matched-data control.
2. **At 5% the hybrid barely beats the branch alone** (63.2% vs 62.0%, p = 0.05). The clean "better than both parts" result is H15 epoch 1 (+3.5 to +3.9 over the branch alone, p < 1e-14).
3. **H12 only ties simple averaging** (56.5% vs 56.5%). H15 epoch 1 beats averaging (65.5–65.9% vs about 63.9%), so the joint model can add more than an ensemble.
4. **Compute:** per puzzle the branch is cheaper than LoRA (3.7× fewer FLOPs), but it needs many more epochs, so at equal total FLOPs LoRA is ahead (see Compute).
5. **One seed, one dataset (PGM neutral)**, one VLM size for the main results.

### A paper outline that fits the data

1. **Problem.** VLMs are at chance on PGM zero-shot; LoRA learns it but is data-hungry (52.6% at 5% vs 81.5% at 20%).
2. **Method.** The branch, the two injection modes, and three recipe lessons: train the branch on its own first (H1–H3, H6, H7); stop the LoRA from leaning on the branch (H8, H9); attach it to a finished LoRA (H11, H12).
3. **Main result.** Gain over LoRA alone and over the control across LoRA strengths (H12, H14, H15), with one chart of gain vs LoRA-alone accuracy.
4. **Why it works.** Overlap (A1, A1b), per-attribute gains (number), ablations (A3).
5. **Where it stops.** Strong LoRA (H4, H15 epoch 4), matched data (H13 and the per-epoch runs), compute.
6. **Limitations and future work.** Seeds, other splits (PGM held-out attributes, I-RAVEN), a joint stage where the Abstractor adapts.

### Where each experiment goes

| Experiment | Role in the paper |
| --- | --- |
| L2, L3 (LoRA baselines) | Main: the numbers everything is compared with |
| S0, B1–B7 (slots, branch alone) | Method / appendix: how the branch was built and how strong it gets |
| H1, H2, H3, H6 (frozen LLM) | Appendix, negative result: why the branch is trained alone first |
| H4, H5 (from scratch, full data and mixture) | Main, short: no gain when the LoRA is strong; the control ties |
| H7 (alignment) | Method: the hand-off from branch to LLM works |
| H8, H9, H10 (Stage 2) | Method lesson: the shortcut and branch dropout |
| H11, H12 (LoRA-first) | Main: the recipe used for every later result |
| H13 (matched 5%) | Main, as the honest control for the data advantage |
| A1, A1b (overlap) | Main: the explanation and the predictor of the gain |
| A3 (ablations) | Main: proof the LLM uses the branch |
| H14, H15 | Main: the headline results |
| Compute | Discussion |
| Smoke runs, per-epoch 5% runs (running) | Fill in the matched-data story |

## Common setup and baselines

Every relational run shares the settings below; each experiment section lists only what differs. LoRA alone reaches **52.6% test at 5% data and 81.5% at 20%**, the two numbers every hybrid is compared with.

### Settings shared by every relational run

| Setting | Value |
| --- | --- |
| Model | Qwen3-VL-2B-Instruct (InternVL3.5-1B only in H1, H2 and early branch runs) |
| Seed | 0 for data order, subsets and initialisation; one seed per run |
| Precision and hardware | bf16; one GPU per job (A100, H100, H200 or L40S) |
| Batch | 2 puzzles × 8 accumulation = 16 puzzles per step; each puzzle = 8 candidate sequences |
| Optimiser | AdamW, weight decay 0.01 (unless stated), gradient clip 1.0 |
| Schedule | Linear warmup, hold, then cosine down to 10% of each base rate |
| Panels | 160 px; slot attention runs inside each of the 9 panel boxes |
| Abstractor | 8 heads, feed-forward 512, dropout 0 (unless stated); each layer = relational cross-attention → feed-forward → self-attention → feed-forward |
| Checkpoint choice | Best on the 1,000 validation puzzles |

### How results are measured

| Measurement | Definition | Noise |
| --- | --- | --- |
| Validation | 1,000 shared puzzles; also used to choose checkpoints, so slightly optimistic | ±3 points |
| Test (hybrids) | 5,000 puzzles from the 200k test set (`test5000.tsv`), never used for training or choice | ±1.4 points |
| Test (LoRA baselines) | The full 200k test set | ±0.2 points |
| Paired test | Same puzzles; exact McNemar on "only A right" vs "only B right" | — |
| Per attribute | A puzzle counts under every attribute it has a rule on | — |

Scores can shift by about 0.5 points when the same model is scored on a different GPU type (bf16 rounding). Treat gaps under 1 point between separate scoring jobs as unsettled.

### Baselines

| # | Experiment | Exact setup | Result |
| --- | --- | --- | --- |
| Z1 | Zero-shot | Qwen and InternVL, no training, 200k test | Qwen 12.85%, InternVL 13.75%: chance |
| L1 | First LoRA attempt | r64, α128, lr 2e-4, no weight decay, 32 per step, 1 epoch, single-pass answer format | Stalled near 20% |
| L2 | **LoRA at 20%** | r16, α32, dropout 0.05 on every LLM linear layer; lr 1e-4; warmup 20; constant for epochs 1–3, cosine to 1e-5 in epoch 4; 240k × 4 epochs = 60k steps; per-candidate Yes/No | **Test (200k): 55.9 / 68.6 / 75.3 / 81.5%** after epochs 1–4 (InternVL 57.7% after 4). On `test5000`: 55.2 / 68.1 / 81.2% after epochs 1 / 2 / 4 |
| L3 | **LoRA at 5%** | Same recipe on the 60k subset: 15,000 steps | **53.4% val, 52.6% test** (52.8% when rescored on `test5000`) |
| L4 | LoRA memorising 500 puzzles | r64, lr 2e-4 | 100% within \~300 steps |

**Where the baselines fit:** they set the problem. LoRA can learn PGM, but at 5% data it stops at about 53%, 29 points below the 20% run. Every hybrid is measured against these numbers, and against a control trained for the same number of steps.

## Building the branch: Stage 0 slots and the branch alone

The branch on its own reaches **62.0% test after 50 epochs of the 240k puzzles**, but only **28.1% test when trained on the 60k puzzles alone**. That gap drives most later results.

### Stage 0: slot attention (S0)

**Question:** can slots represent PGM panels at all? **Setup:** slot attention on frozen ViT features, trained to reconstruct those features (mean squared error), no labels. 16 slots, 3 iterations, 12 × 12 ViT grid per panel, 16 images per step, lr 4e-4, warmup 10,000, 100,000 steps, cosine to 10%.

| Slot model | Images | Slot size / decoder | Reconstruction error | Used by |
| --- | --- | --- | --- | --- |
| `s0_qwen_pgm` | PGM 240k | 32 / 512 | 0.331 | All 32-dim branches (H1–H3, H7–H12, H14, H15) |
| `s0_qwen_pgm_d256` | PGM 240k | 256 / 2048 | 0.098 | H4, H6 |
| `s0_qwen_mix20_d256` | Mixture | 256 / 2048 | 0.111 | H5 |
| `s0_qwen_pgm60k` | **PGM 60k only** | 32 / 512 | 0.326 | H13, smoke runs, per-epoch 5% runs |
| `s0_ivl1b_pgm` | PGM 240k (InternVL) | 32 / 512 | 0.383 | Early InternVL runs |

**Where it fits:** method section. Low reconstruction error shows slots capture the panels, but says nothing about reasoning.

### The branch alone (B1–B7)

The branch is trained with its own linear answer head, no LLM ("Stage 1a"). 8-way cross-entropy over the candidates.

| # | Experiment | Exact setup | Result | What it showed |
| --- | --- | --- | --- | --- |
| B1 | Memorise 500 puzzles | 500 puzzles × 100 epochs = 3,125 steps; constant lr | 6 layers at lr 3e-4: 94.6% (6,250 steps); 24 layers at 1e-4: 27.6% | 6 layers learn far faster than 24 |
| B2 | First unseen-data runs | 240k × 4 epochs = 60k steps, 6 layers, lr 3e-4 | 24.4% val (Qwen), 24.7% (InternVL) | Plateau at 4 epochs |
| B3 | Sweep at 60k × 4 epochs | Slots on/off, 32 vs 256-dim, 6 vs 24 layers, lr 3e-4 vs 1e-3 | All 15.9–22.8% val | No setting separates at this length |
| B4 | Branch on a task mixture | 30% or 50% PGM plus CLEVR, VSR, ART, CLEVR-ART | PGM at chance; synthetic tasks 99.7–100% | Synthetic relational tasks are too easy to separate methods |
| B5 | 10 epochs, frozen vs fine-tuned slots | 240k × 10 epochs = 150k steps, 6 layers, lr 3e-4; slots frozen or fine-tuned at 3e-5 | Frozen 30.2%, **fine-tuned 37.1%** val | Fine-tuning the slots adds about 7 points |
| B6 | **50-epoch branch** | 240k × 50 epochs = 750k steps, 6 layers (0.80M), lr 3e-4, slots fine-tuned; decay from 735k | Snapshots: 204k 40.9% val / 39.8% test; 276k 47.6% / 45.0%; **636k 63.9% / 62.0%**; now 65.6% val at 714k | Keeps improving with training; the branch used in H14 and H15 |
| B6b | 50 epochs, 24 layers | Same, 24 layers, lr 8e-5 | 52.3% val at 600k | Deeper is worse at equal steps |
| B7 | **60k-only branch** | 60k × 40 epochs = 150k steps, slots from `s0_qwen_pgm60k`, lr 1e-3 or 3e-4 | Peak 30.8% val at 71k; **28.1% test**; early step-43k copy 26.8% test | With 4× less data the branch stalls near 30% |

**Where it fits:** method section, plus one chart of branch accuracy against training steps. The B6 vs B7 gap is the key fact for framing: the strong branch is strong because it saw more data for longer.

## H1–H6: the early experiments

Training the branch through a frozen LLM did not work (H1, H2, H6). Training it jointly with LoRA from scratch at full data gave no gain (H4, H5). Together they led to the staged recipe used from H7 on: train the branch alone, then plug it in.

### H1: frozen-LLM memorisation, rounds 1 and 2 (Oct 29–30)

- **Question:** can a branch learn through a frozen VLM well enough to memorise 500 puzzles?
- **Trained:** Abstractor from scratch (24 layers, head dim 32, 3.18M) + connector (4.26M) + token gain or λ. **Frozen:** ViT, 32-dim slots from Stage 0, LLM (no LoRA). Learning signal: only the LLM's Yes/No answers.
- **Settings:** 500 training puzzles, scored on the same 500. Round 1: 625 steps (20 epochs), Abstractor lr 1e-4, connector 3e-4, cosine, λ starts at 0.9. Round 2: 1,250 steps, lr 3e-4 / 1e-3, constant, λ starts at 0.5. Qwen and InternVL, interleave and fusion: 8 runs.
- **Result:** best 32.6% (InternVL interleave, round 1); Qwen 26.8% / 29.0% (round 1), 22.2% / 14.2% (round 2); one run lost to a broken GPU. LoRA alone memorises the same 500 puzzles 100% in \~300 steps.
- **Meaning:** almost no learning signal reached the branch through the frozen LLM. Later traced to a 24-layer Abstractor with too low a learning rate.
- **Fit:** appendix, as a negative result that motivates training the branch alone first.

### H2: frozen-LLM Stage 1 on full data, 1 epoch (Oct 30 – Nov 1)

- **Question:** does the frozen-LLM path generalise if trained on all 240k puzzles?
- **Setup:** same parts as H1 (24 layers), 15,000 steps = 1 epoch, Abstractor lr 1e-4, λ starts at 0.9, warmup 500, cosine. Qwen and InternVL, interleave and fusion.
- **Result (val):** Qwen 19.1% / 18.0%, InternVL 17.9% / 16.4%. Branch alone after the same 15k steps: 19.0%. LoRA alone after 1 epoch: 56.7%.
- **Meaning:** barely above chance; no better than the branch alone.
- **Fit:** appendix, with H1.

### H3: frozen-LLM memorisation with tuned settings (Nov 8)

- **Question:** with a 6-layer Abstractor and a higher rate, can the frozen-LLM path at least memorise?
- **Setup:** 500 puzzles × 100 epochs = 3,125 steps, 6 layers, head dim 32, Abstractor lr 3e-4 or 1e-3, constant rate, fusion λ starts at 0.5. Qwen only.
- **Result (on the 500 training puzzles):** interleave 85.0% (3e-4), 59.4% (1e-3); fusion 24.6% (3e-4), **100%** (1e-3). H1 for comparison: 14–33%.
- **Meaning:** depth and rate were H1's problem. But memorising is not generalising (see H6).
- **Fit:** appendix; justifies 6 layers.

### H4: full-data hybrids with LoRA, trained together from scratch (Nov 8–11)

- **Question:** at full data, does adding the branch to LoRA help, and does the way it is fed in matter?
- **Setup:** 240k × 4 epochs = 60,000 steps; LoRA (r16, lr 1e-4) and a 256-dim branch (6 layers, 28.34M, frozen 256-dim slots) trained together from scratch; connector 4.72M; warmup 500, hold to 45,000, cosine. Seven runs: interleave with 1, 9 or 144 relation tokens (lr 3e-4), 144 tokens at lr 1e-3, fusion at 3e-4 and 1e-3 (λ starts at 0.5), and a **control** with 144 tokens but no relational attention.
- **Result (best val):** 1 token 79.6%, 9 tokens 80.6%, 144 tokens 81.3%, 144 at 1e-3 79.5%, fusion 81.4% / 80.5%, control 81.3%. LoRA alone 81.6%. Test (A3): 80.2–81.0%.
- **Ablation (A3):** zeroing, dropping or shuffling the relation tokens changes accuracy by at most 0.7 points. The LLM ignores them.
- **Meaning:** when LoRA can learn the task fully, the branch adds nothing, and even a non-relational branch ties.
- **Fit:** main text, briefly: "no gain when the VLM is strong". Matches H15 epoch 4.

### H5: mixture hybrids with LoRA (Nov 8–11)

- **Question:** does training on many relational tasks at once help?
- **Setup:** 39,000 steps on a mixture: PGM 30%, CLEVR relational yes/no 30%, CLEVR-ART RMTS 10%, CLEVR-ART identity rules 10%, VSR 8%, ART tasks 12%. Loss scaled by each task's chance loss; checkpoint chosen by the family score (mean over PGM, CLEVR, VSR, ART, CLEVR-ART). Interleave (144 tokens), fusion, and a non-relational control.
- **Result (val):** family score 87.2 / 87.2, control **87.6**. PGM 50.2 / 49.2%, control 50.8%. ART and CLEVR-ART at 100% for every run.
- **Meaning:** no gain. The synthetic tasks saturate, so they cannot separate methods.
- **Fit:** appendix; reason to stay on PGM.

### H6: frozen-LLM Stage 1 at 10 epochs, 256-dim (Nov 10–15)

- **Question:** does the frozen-LLM path get there with much longer training?
- **Setup:** 240k × 10 epochs planned (150k steps), 6 layers, head dim 256, Abstractor lr 3e-4 or 1e-3, λ starts at 0.9. Stopped by decision at 108k–124k steps.
- **Result (best val):** interleave 24.2% / 22.6%, fusion **27.2%** / 25.6%. The branch alone at 10 epochs reached 37.1%.
- **Meaning:** learning through the frozen LLM stays \~10 points behind learning alone. Settles the design: branch alone first, then align.
- **Fit:** appendix, with H1–H3.

## H7: aligning the branch to the LLM (Nov 14)

Training only the connector lets the frozen LLM match the branch's own accuracy within 1,000 steps, for every branch we tried. The hand-off from branch to LLM works.

- **Question:** if the branch is already trained, can the frozen LLM learn to read it by training only the small connector?
- **Trained:** connector MLP 32 → 2048 → 2048 (4.26M) + token gain or λ. **Frozen:** slots, Abstractor and answer head (all from the branch-alone run), ViT, LLM (no LoRA).
- **Settings:** all 240k puzzles, 10,000 steps (\~0.7 epoch), connector lr 3e-4 or 1e-3, warmup 200, cosine; 1 pooled relation token for interleave; fusion λ starts at 0.9. A run "levels off" when its best validation stops rising by more than 1 point over two checks; that checkpoint is passed on.

| Branch used | Branch alone | Interleave | Fusion | Used next in |
| --- | --- | --- | --- | --- |
| 10-epoch (B5) | 37.1% val | 37.5% val (both rates) | 36.4% val | H8, H9 |
| 50-epoch, step 204k | 40.9% val | 39.7% test | — | H10, H11 |
| 50-epoch, step 276k | 47.6% val / 45.0% test | 45.6% test | — | H12 |
| 60k-only (B7) | 28.1% test | aligned on the 60k only | — | H13 |
| **50-epoch, step 636k** | 63.9% val / 62.0% test | **63.4% val, 62.1% test** | 62.3% val | H14, H15 |

- **Ablation (A3):** with the relation tokens zeroed, every aligned model falls to **14.6%** (chance is 12.5%). Feeding another puzzle's tokens gives 11.7–12.7%. The frozen LLM gets all of its PGM ability from the branch.
- **Meaning:** the LLM can read the branch's answer, but adds nothing of its own: alignment equals the branch alone. Any gain above the branch must come from adapting the LLM (LoRA).
- **Fit:** method section. One line: "a 4M-parameter connector transfers the branch's accuracy into the frozen LLM".

## H8–H10: training the branch and LoRA together (Stage 2)

Training a new LoRA alongside an aligned branch lets the LoRA take a **shortcut**: it leans on the branch and learns little itself (H8). Switching the branch off on half the batches fixes it and beats LoRA alone by 1.5–2.6 points (H9). A stronger branch barely moves the total, because the LoRA leans on it more (H10).

**Shared Stage 2 recipe:** 5% data (the 60k puzzles), 15,000 steps (4 epochs), starting from the H7 alignment checkpoint. LoRA from scratch at lr 1e-4 (r16, α32); Abstractor and connector at 3e-5; slots, ViT and LLM base frozen; warmup 20, hold to step 11,250, cosine. Compared with LoRA alone at 5% (52.6% test).

### H8: no dropout, plus the branch-off check (Nov 14–15)

- **Question:** does a LoRA trained together with the aligned 10-epoch branch (37.1% alone) beat LoRA alone?
- **Result (test):** interleave **49.1%**, fusion **38.8%**: both below LoRA alone (52.6%).
- **Branch-off check:** at step 5k, scored with the branch off, the hybrids got 13–16%. The LoRA inside had learned almost nothing alone.
- **Ablation (A3, interleave):** 48.8% normal → 31.9% zeroed, 24.3% with another candidate's tokens.
- **Meaning:** the shortcut. The LoRA relies on the branch, which is weaker than what LoRA could learn alone.

### H9: branch dropout 0.5 (Nov 15)

- **Change from H8:** on each batch, the branch is switched off with probability 0.5 (`--branch_dropout 0.5`).
- **Result (test):** interleave **55.2%** (+2.6 over LoRA alone, p = 7e-5), fusion **54.1%** (+1.5, p = 0.02). Branch off: 52.5% / 51.7%, the same as LoRA alone.
- **Ablation (A3, interleave):** 54.9% → 52.7% zeroed (−2.2), 46.1% with another candidate's tokens.
- **Meaning:** dropout forces the LoRA to learn on its own; the branch then adds on top. First hybrid to beat LoRA alone.

### H10: stronger branch, step 204k (Nov 15–16)

- **Change from H9:** the 50-epoch branch at step 204k (40.9% val alone vs 37.1%), with its own alignment.
- **Result (test):** interleave **55.1%**, fusion **55.3%**: no better than H9. Branch off: 49.5% / 49.3%, below LoRA alone.
- **Ablation (A3, interleave):** 54.6% → 49.2% zeroed (−5.4), 33.2% with another candidate's tokens.
- **Meaning:** a stronger branch made the LoRA lean on it more, even with dropout, so the total stayed flat.

**Where H8–H10 fit:** method section, as the lesson that led to LoRA-first. The pair H8 → H9 is a clean story: one change (dropout) turns −3.5 into +2.6.

## H11, H12 and the ablations: the LoRA-first recipe

Training LoRA alone first and attaching the branch afterwards keeps the LoRA intact and gives a clean gain over an equal-steps control: +1.9 to +2.0 points (H11) and **+3.3 to +3.5 points (H12)**. Every later result uses this recipe.

**LoRA-first recipe (exact):** start from the finished LoRA-alone adapter at 5% (step 14k) and the aligned branch. Steps 0–500: LoRA frozen, connector and gain / λ train at 1e-4. Steps 500–3,500: LoRA also trains at 2e-5. Abstractor and slots frozen throughout. 3,500 steps on the 60k puzzles, warmup 50, cosine from step 500. **Control:** identical but with `--branch_dropout 1.0`, so the branch never reaches the LLM (it measures what 3,500 more LoRA steps give on their own).

### H11: LoRA-first with the step-204k branch (Nov 15–16)

- **Branch:** 39.8% test alone.
- **Result (test):** interleave **55.1%** vs control 53.2% (+1.9); fusion **54.9%** vs 52.9% (+2.0); both p < 1e-5.
- **Ablation (A3, interleave):** 54.8% → 53.2% zeroed (−1.6), 51.9% with another candidate's tokens (−2.9).

### H12: LoRA-first with the step-276k branch (Nov 16–17)

- **Branch:** 45.0% test alone, with its own alignment.
- **Result (test):** interleave **56.5%** vs control 53.2% (+3.3); fusion **56.4%** vs 52.9% (+3.5); p < 1e-11. The best 5% result until H14.
- **Ablation (A3, interleave):** 56.4% → 52.5% zeroed (−3.9), 47.6% with another candidate's tokens (−8.8), 48.5% with another puzzle's (p < 1e-15).
- **Per attribute (gain over control, int / fuse):** number **+14.5 / +13.4**, position +3.4 / +4.8, type +3.4 / +4.0, size +1.4 / +0.2, colour +0.8 / +1.2. On number puzzles the hybrids reach 50–53%, above LoRA alone (\~38%) and the branch alone (49%).
- **vs averaging:** averaging LoRA alone with the same branch gives 56.5%: a tie. With H11's weaker branch the hybrid was about 1 point ahead of averaging (55.1% vs 54.2%).

### A2 and A3: how the hybrids were tested

**A2** scores every model on the same 5,000 test puzzles with per-puzzle records, branch on and off. **A3** scores each interleave model five ways, all in one job on one GPU type:

| Condition | What it does |
| --- | --- |
| Normal | As trained |
| Zero | Relation tokens set to 0 (= branch off) |
| Drop | Relation-token placeholders removed from the prompt |
| Shuffle candidates | Each candidate gets another candidate's tokens from the same puzzle |
| Shuffle puzzles | Each puzzle gets another puzzle's tokens |

| Exp. | Model | Normal | Zero | Drop | Shuffle cand. | Shuffle puzzle | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4 | 1 / 9 / 144-token hybrids, full data | 80.2–81.0 | ±0.7 | ±0.5 | ±0.4 | ±0.6 | Tokens ignored |
| H7 | Alignment, 10-epoch branch | 34.9–35.5 | 14.6 | 14.7 | 9.0 | 12.5 | All from the branch |
| H8 | Stage 2, no dropout | 48.8 | 31.9 | 31.6 | 24.3 | 25.8 | Shortcut |
| H9 | Stage 2, dropout 0.5 | 54.9 | 52.7 | 52.4 | 46.1 | 47.7 | Uses it, LoRA intact |
| H10 | Stage 2, 204k branch | 54.6 | 49.2 | 48.6 | 33.2 | 39.1 | Leans on it more |
| H11 | LoRA-first, 204k | 54.8 | 53.2 | 53.5 | 51.9 | 52.2 | Uses it a little |
| H12 | LoRA-first, 276k | 56.4 | 52.5 | 52.7 | 47.6 | 48.5 | Uses it clearly |
| H14 | LoRA-first, 636k, 5% LoRA | 63.4 | 51.7 | 52.2 | 32.2 | 38.9 | Relies on it heavily |
| H15 | LoRA-first, 636k, 20% LoRA epoch 1 | 65.7 | 57.7 | 57.6 | 18.8 | 30.7 | Relies on it heavily |
| H15 | LoRA-first, 636k, 20% LoRA epoch 4 | 81.3 | 80.8 | 80.8 | 79.9 | 79.3 | Barely uses it |

**Why "shuffle" falls below "zero":** with the tokens zeroed, the LLM falls back on its own LoRA. With wrong tokens, it trusts misleading evidence. A drop below the zero line proves the LLM reads the content of the tokens, not just their presence.

**Where H11, H12 and A3 fit:** main text. H11/H12 introduce the recipe and the "gain grows with branch strength" pattern; A3 is the strongest evidence that the branch is used. H14/H15 rows were scored a day apart from the others (same test puzzles).

## H13: everything at matched 5% data (Nov 16–21)

When the slots, the branch, the alignment and the LoRA all see only the same 60k puzzles, **no recipe beats LoRA alone by more than noise**: the best is +0.6 over its control. The 5% branch is simply too weak (28.1% test).

- **Question:** H8–H12 used a branch trained on all 240k puzzles, 4× the LoRA's data. Is the gain only that extra data?
- **Setup:** a clean 60k pipeline. Slots `s0_qwen_pgm60k` (Stage 0 on the 60k images), branch `aux_ft60k_q_L6_*` (6 layers, 40 epochs of the 60k, slots fine-tuned), alignment on the 60k, then the three recipes on the finished 5% LoRA. Two branch checkpoints: early (step 43k, 26.8% test) and best (step 71k, 30.8% val, **28.1% test**).

| Recipe | Branch | Interleave (test) | Fusion (test) | Compared with |
| --- | --- | --- | --- | --- |
| LoRA-first (H12 recipe) | Best (28.1%) | **53.8%** | **53.4%** | Control 53.2% / 52.9%: +0.6 / +0.5, not significant |
| LoRA-first | Early (26.8%) | 53.7% | 53.3% | Same with the branch off: ignored |
| Stage 2 + dropout (H9 recipe) | Best | 50.4% | 52.1% | LoRA alone 52.6% |
| Stage 2, no dropout (H8 recipe) | Best | 43.5% | 46.8% | Branch off 35.5% / 38.8%: shortcut again |

- **Per attribute:** the 60k branch is below the 5% LoRA on every attribute, including number (21.7% vs 39.5%). With nothing the LoRA lacks, there is nothing to add.
- **Is the learning rate to blame?** No. The 60k branch plateaus near 30% at both lr 3e-4 and 1e-3, and the 240k branch at the same settings keeps improving. The limit is the amount of data; at 40 epochs the 60k branch overfits (training 54% vs validation 27% at 150k steps).
- **Meaning:** the gains in H9–H12 depend on the branch having learned from more data than the LoRA. At matched data the branch adds no information.
- **Fit:** main text, as the honest control. It turns "the branch helps" into "the branch helps when it knows something the VLM does not", and motivates the smoke runs (a better 5% branch) and the per-epoch 5% runs (a weaker 5% LoRA).

## A1 and A1b: does the branch know something the LoRA does not?

The strong branch alone (62.0% test) solves **979 test puzzles the 5% LoRA gets wrong**, and simply averaging the two models adds +10.2 points. The extra knowledge shrinks as the LoRA trains: +8.7 at 20% epoch 1, +2.3 at epoch 2, and nothing at epoch 4.

### What the check does, step by step

1. Take two models trained separately: LoRA alone and the branch alone. Nothing is trained in this check.
2. Score both on the same test puzzles and record each one's 8 answer scores per puzzle.
3. **Overlap:** count the puzzles both get right, only LoRA gets right, only the branch gets right, and neither gets right. "Only the branch right" is what the branch could add.
4. **Averaging:** turn each model's 8 scores into log-probabilities (log-softmax), average the two with weight 0.5, and take the top answer. A cross-validated weight (picked on half the puzzles, scored on the other half) was also tried.
5. Compare averaging with LoRA alone using a paired McNemar test.

### A1 (Nov 15, 1,000 val puzzles, older branches)

At 5% data, branches of 37–45% got 96–131 puzzles right that LoRA missed, and averaging gained **+2.2 to +3.5** (p 0.002–0.04). At full data, averaging lost 0.5–0.8 points: nothing to add.

### A1b (Nov 23, 5,000 test puzzles, step-636k branch at 62.0%)

| LoRA alone | Its accuracy | Puzzles only the branch gets right | Averaged | Gain from averaging | p |
| --- | --- | --- | --- | --- | --- |
| 5% (finished) | 52.8% | **979** | 63.0% | **+10.2** | 3e-64 |
| 20%, epoch 1 | 55.2% | 874 | 63.9% | +8.7 | — |
| 20%, epoch 2 | 68.1% | 528 | 70.4% | +2.3 | — |
| 20%, epoch 4 | 81.2% | 238 (583 expected by chance) | 81.1% | −0.1 | 0.84 |

- **"Expected by chance"** is the overlap if the two models' mistakes were unrelated. At epoch 4 the branch's unique wins are far fewer than chance: its correct answers are almost a subset of the LoRA's.
- **Number puzzles:** the branch alone gets 70.9% vs 39.5% for the 5% LoRA. Counting is where the branch's extra knowledge lives.

**Meaning:** the size of the possible gain is visible before any hybrid is trained, and it shrinks as the LoRA gets stronger. **Fit:** main text, as the explanation of every hybrid result and a cheap predictor of when the branch will help.

## H14 and H15: the strongest branch (Nov 23–24)

With the step-636k branch (62.0% test), the LoRA-first hybrid gains **+10.4 points at 5% data** (52.8% → 63.2%) and **+10.3 to +10.7 on the 20% LoRA after one epoch** (55.2% → 65.5–65.9%). On the fully trained 20% LoRA it adds nothing (81.2% → 81.3–81.4%).

**Recipe (both):** align the step-636k branch (H7 recipe; both regimes levelled at step 2,000: interleave 63.4% val, fusion 62.3% val), then LoRA-first (H12 recipe) on a given LoRA, 3,500 steps, with a branch-off control for each.

|  | H14 | H15 |
| --- | --- | --- |
| Question | Does a much stronger branch extend the H11 → H12 trend at 5%? | Does the branch help when the LoRA is trained on the same 20% data? |
| LoRA it starts from | The finished 5% LoRA (52.8% test) | The 20% LoRA after epoch 1 (55.2%) and after epoch 4 (81.2%) |
| Data during LoRA-first | The 60k puzzles | All 240k puzzles (the branch's and the LoRA's own data: matched) |

### Results (test, 5,000 puzzles)

| LoRA | Mode | Hybrid | Branch off | Control | vs control | vs LoRA alone | vs branch alone (62.0%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5% | Interleave | **63.2%** | 52.4% | 53.2% | +10.0 (p 7e-64) | +10.4 (p 2e-64) | +1.2 (p 0.05) |
| 5% | Fusion | **62.9%** | 51.5% | 52.9% | +10.0 (p 7e-60) | +10.1 (p 4e-61) | +0.9 (p 0.1) |
| 20%, epoch 1 | Interleave | **65.5%** | 57.4% | 60.2% | +5.3 (p 4e-16) | +10.3 (p 2e-51) | +3.5 (p 1e-14) |
| 20%, epoch 1 | Fusion | **65.9%** | 57.3% | 60.2% | +5.7 (p 8e-19) | +10.7 (p 4e-55) | +3.9 (p 8e-17) |
| 20%, epoch 4 | Interleave | 81.4% | 80.4% | 80.5% | +0.9 (p 0.01) | +0.2 (p 0.5) | — |
| 20%, epoch 4 | Fusion | 81.3% | 81.0% | 80.9% | +0.4 (p 0.3) | +0.1 (p 0.9) | — |

The epoch-1 control (60.2%) is above LoRA alone (55.2%) because it trains the LoRA for 3,500 more steps on all 240k puzzles. The control isolates the branch's share.

### What it means

1. **At 5%, the branch does most of the work.** The hybrid is +10 over the LoRA but only +1 over the branch alone, about what averaging gives (63.0%). The LoRA and the branch are not yet combining.
2. **At 20% epoch 1, the hybrid beats both parts** by 3.5–3.9 points over the branch and 5–6 over the control, and beats averaging (about 63.9%). This is the cleanest evidence that joint use adds more than an ensemble.
3. **At 20% epoch 4 there is nothing to add**, as A1b predicted (only 238 branch-only puzzles).
4. **Interleave and fusion perform the same** in every setting (within 0.5 points).
5. **Ablation (A3)** shows the 5% and epoch-1 hybrids rely heavily on the branch (shuffled tokens: 18.8–38.9%), while the epoch-4 hybrid barely reacts (79.3–80.8%).

**Chart: the branch adds about 10 points to a weak LoRA and nothing to a strong one** (gain over LoRA alone, test points; step-636k branch, 62.0% alone)

| LoRA alone | Averaging LoRA + branch (no training) | LoRA-first hybrid (interleave) |
| --- | --- | --- |
| 5% LoRA (52.8%) | +10.2 | +10.4 |
| 20% LoRA, epoch 1 (55.2%) | +8.7 | +10.3 |
| 20% LoRA, epoch 2 (68.1%) | +2.3 | not run |
| 20% LoRA, epoch 4 (81.2%) | −0.1 | +0.2 |

The trained hybrid tracks the free averaging gain: about +10 for the 5% and epoch-1 LoRAs, about zero at epoch 4. The 5% hybrid uses a branch trained on 20% data.

**Fit:** the headline results. Report H15 epoch 1 as the main result (matched data, beats both parts), H14 as the strong-branch result with the "branch does most of the work" caveat, and H15 epoch 4 as the ceiling.

## Compute: accuracy at equal FLOPs

At equal training compute **LoRA is far ahead of the branch**: with the 5.5 × 10¹⁸ FLOPs of one 5% LoRA run, LoRA reaches 53.4% and the branch 28.7%. The branch is cheaper per puzzle, but needs 4–8× more total compute to reach the same accuracy.

### How it was counted

1. Count the forward-pass FLOPs analytically for each piece. Training ≈ 3× the forward pass for parts that are trained.
2. **ViT, per image:** 7.73 × 10¹¹ FLOPs (1,024 patches merged to 256 tokens). **LLM, per sequence** (365 tokens): 1.06 × 10¹².
3. **Branch, per puzzle:** 8 ViT passes = 6.18 × 10¹². Slots and Abstractor are negligible. The LLM is not used.
4. **LoRA, per puzzle:** 8 candidate sequences × (ViT + LLM forward and backward) = 2.31 × 10¹³, **3.74× the branch**.
5. Slot pretraining (Stage 0: 100k steps × 16 images) = 1.24 × 10¹⁸, added to the branch's total.
6. For each budget, take the best validation accuracy each run reached within it.

### Results (best val within each budget; 1 EF = 10¹⁸ FLOPs)

| Budget | 5% LoRA | 5% branch | 20% LoRA | 20% branch |
| --- | --- | --- | --- | --- |
| 5.5 EF (one 5% LoRA run) | 53.4% | 28.7% | 56.2% | 23.2% |
| 22 EF (one 20% LoRA run) | — | — | 81.0% | 40.9% |
| 65 EF | — | — | — | 63.9% |

| Measured GPU time | LoRA | Branch |
| --- | --- | --- |
| 5% data | 10 GPU-hours (15k steps) | 41 GPU-hours |
| 20% data | 41 GPU-hours (4 epochs) | \~288 GPU-hours (50-epoch run to step 636k) |
| Throughput | \~6.6 puzzles/s | \~10 puzzles/s |

- **Why the branch needs so much:** it learns slowly (B6: 41% after 13 epochs, 64% after 42), while LoRA builds on the pretrained LLM.
- **A possible saving, not measured:** with frozen slots, the slot features could be computed once (about 8 GB) and reused every epoch, which would make the branch about 5× cheaper than LoRA at 20%. The smoke runs show frozen slots cost about 6 points of accuracy at 5% (22% vs 28%).

**Fit:** discussion. Say it plainly: the branch is not a cheaper substitute for LoRA. Its value is complementary knowledge a limited LoRA lacks, and the LoRA-first step itself is cheap (3,500 steps, about 1/4 of a 5% LoRA run).

## Ongoing experiments: what is running and what is not

As of 6:40 PM Nov 24, **18 relvlm jobs are training and 1 is queued**. No result below is final; all numbers are validation (±3 points). Test scoring of the per-epoch runs and the 5% LoRA epochs starts automatically when they finish.

### Running now

| Experiment | Runs | Progress | Latest val | Expected |
| --- | --- | --- | --- | --- |
| **Smoke runs: a better 5% branch** | 9 `aux60s_*` | 40–52k of 75k steps | 21.4–28.0% (old run: \~27–28% at the same steps) | Verdict about midday Nov 25 |
| **Per-epoch 5% LoRA + 5% branch** | `lora_lf5e{1,2}_q_int_c60` and `_ctrl` | 3,000 of 3,500 steps | Epoch 1: 37.5% vs control 35.2%; epoch 2: 44.5% vs 42.5% | Test scores tonight |
| **5% LoRA rerun, saved per epoch** | `qwen_pc_neutral5pct_ep` | \~10.5k of 15k steps; epochs 1–2 saved | 43.7% at 10k | Epochs 3–4 tonight, then all four test-scored |
| **Stage 2 + dropout with the 636k branch at 5%** | `lora_w5d_q_{int,fuse}_L6ft636k` | 13k and 10k of 15k steps | Interleave 63.7%, fusion 63.6% (best 65.0 / 64.4) | Test-scored automatically when done, \~Nov 25 |
| **50-epoch branch, 6 layers** | `aux_ft_q_L6_lr3e-4_ep50` | 718k of 750k steps | Best **65.6%** at 714k | Finishes \~Nov 25–26; not yet set to be test-scored |
| **50-epoch branch, 24 layers** | `aux_ft_q_L24_lr8e-5_ep50` | 600k of 750k steps | Best 52.3% | Several more days |

### The smoke runs in detail

**Question:** can regularisation or a smaller Abstractor stop the 5% branch overfitting and push it past the old run's 30.8% peak? **Shared:** 60k puzzles, `s0_qwen_pgm60k` slots, branch alone, head dim 32, lr 1e-3, 75k steps (20 epochs), warmup 500, constant to 65k, cosine over the last 10k.

| Run | Layers | Weight decay | Dropout | Row swap | Slots | Last 3 val (avg) |
| --- | --- | --- | --- | --- | --- | --- |
| `aux60s_L2` | 2 | 0.01 | 0 | no | fine-tuned | 28.0% |
| `aux60s_L4` | 4 | 0.01 | 0 | no | fine-tuned | 28.0% |
| `aux60s_L4_d1_aug` | 4 | 0.05 | 0.1 | 50% | fine-tuned | 28.0% |
| `aux60s_L6_d3` | 6 | 0.05 | 0.3 | no | fine-tuned | 28.0% |
| `aux60s_L6_aug` | 6 | 0.01 | 0 | 50% | fine-tuned | 27.9% |
| `aux60s_L6_d1` | 6 | 0.05 | 0.1 | no | fine-tuned | 26.9% |
| `aux60s_L4_d1` | 4 | 0.05 | 0.1 | no | fine-tuned | 26.6% |
| `aux60s_L4_d1_lr3e-4_wd1` | 4 | 0.1 | 0.1 | no | fine-tuned, lr **3e-4** | 26.4% |
| `aux60s_L4_d1_frz` | 4 | 0.05 | 0.1 | no | **frozen** | 21.4% |

"Row swap" swaps the puzzle's first two rows (each row follows the rules on its own, so the answer is unchanged). So far every run except the bottom two tracks the old run within 1 point. Regularisation lowers training accuracy by 2–3 points without raising validation, which suggests the 5% branch is limited by data, not by overfitting. Frozen slots cost about 6 points.

### The per-epoch 5% runs in detail

**Question:** does the 5% branch (28.1% test) help a *weaker* 5% LoRA, the way the strong branch helps the epoch-1 20% LoRA? **Setup:** the 5% LoRA retrained with a copy saved after each epoch (step 3,750 = epoch 1, 7,500 = epoch 2); the H13 alignment of the 5% branch; the LoRA-first recipe on the 60k puzzles; one branch-off control per epoch. Interleave only. So far the branch is ahead at every checkpoint by about 2 points, inside the validation noise. The paired test on 5,000 puzzles will decide.

### Not running

| Item | Status | Why |
| --- | --- | --- |
| Fusion versions of the per-epoch 5% runs | Not queued | Interleave and fusion have matched in every setting so far |
| Per-epoch runs on 5% epochs 3 and 4 | Not queued | Epoch 4 is H13 (+0.6); epoch 3 can be added once the rerun saves it |
| Hybrid with the best smoke-run branch | Waiting | Only worth running if a smoke run clearly beats 30.8% |
| Test scoring of the 50-epoch branch's final checkpoint | Not set up | Needs the run to finish (750k) |
| Single-job re-scoring of all models | Offered, not run | Would settle sub-1-point gaps between scoring jobs |
| More seeds | Not run | Every result is one seed |
| Other splits (PGM held-out attributes, I-RAVEN) | Not run for the hybrid | Generalisation beyond PGM neutral is untested |
| InternVL hybrids | Paused by decision after H2 | Qwen only |
| H6, two H4 runs | Stopped early | H6 by decision at 108k–124k; `lora_q_int_slot_lr1e-3` at 56k, the H4 control at 59k |

## Caveats and recommended next experiments

The results are strong enough to write up now around H15 epoch 1, A1b and A3. The biggest gaps a reviewer will raise are single seeds, one dataset, and the data advantage of the strong branch.

### Caveats when quoting the numbers

- **One seed** for every run. A 2–3 point gap on validation is within noise; on test, about 1.4 points is.
- **Quote test, not validation:** validation was used to pick checkpoints.
- **The strong branch saw more data** (240k puzzles, up to 50 epochs) than the 5% LoRA. Only H13, H15 and the per-epoch 5% runs are at matched data.
- **Cross-job noise:** scores shift \~0.5 points across GPU types. H12's tie with averaging and gaps under 1 point are unsettled.
- **The 5% LoRA was not trained to convergence** (fixed 4-epoch schedule, still rising). It affects "vs LoRA alone", not "vs control".
- **Synthetic tasks (ART, CLEVR-ART) saturate at 100%**, so mixture results cannot separate methods.

### Recommended next experiments, in priority order

1. **Seeds for the headline results:** 2 more seeds of H15 epoch 1 (interleave) and its control. Cheap (3,500 steps each) and turns the main result into a mean ± spread.
2. **Single-job re-score** of the key models (LoRA alones, H12, H14, H15, averaging) on one GPU type, with a paired test of hybrid vs averaging.
3. **Finish the matched-data story:** the per-epoch 5% test results (tonight) and the smoke-run verdict (Nov 25). If a smoke branch clearly beats 30.8%, run its alignment and the per-epoch hybrids with it.
4. **H15 at epoch 2** (68.1% LoRA, +2.3 from averaging) to fill in the gain-vs-LoRA-strength curve between epoch 1 and epoch 4.
5. **Generalisation:** score the H15 epoch-1 hybrid on a PGM held-out-attribute split or I-RAVEN, where a relational module should matter most.
6. **Test-score the final 50-epoch branch** (750k) and, if it is higher than 62.0%, rerun H14 with it.
