# RelVLM experiment log

As of Nov 17, 2026 (numbered 1–23; the deep dives in `relational_deep_dives.md` use H1–H13). Exported from the team doc.

At 5% of the training data (60k PGM puzzles), adding the relational branch lifts Qwen3-VL-2B from 52.6% to 56.5% on 5,000 unseen test puzzles, +3.4 points over an equal-steps control. At full data it adds nothing (LoRA alone 81.5%). Every completed experiment is listed below in the order we ran it.

## Setup

The branch reads Qwen's own ViT features, groups them into object slots, relates the slots with an Abstractor, and hands the result to the LLM through a connector. Each stage trains a different part.

| Stage | Trains | Frozen | Objective |
| --- | --- | --- | --- |
| Stage 0: slots | Slot attention + its decoder | Whole VLM | Reconstruct ViT features (MSE), no labels |
| Stage 1a: branch alone | Abstractor + linear head; in `aux_ft_*` runs also the slots (lr 3e-5) | ViT, LLM (not used) | 8-way cross-entropy on the linear head |
| Stage 1: frozen LLM / alignment | Connector + token gain (interleave) or λ (fusion); Abstractor too in early runs, frozen in alignment runs | ViT, slots, LLM | 8-way cross-entropy on logit(Yes) − logit(No) |
| Stage 2: hybrid + LoRA | LoRA (r16, α32, dropout 0.05) + Abstractor + connector | ViT, slots, LLM base weights | Same as Stage 1 |
| LoRA-first | Connector (1e-4), then LoRA (2e-5) after 500 steps | Abstractor, slots, LLM base weights | Same as Stage 1 |

**Two injection regimes, always run separately.** Interleave adds relation tokens to the prompt. Fusion blends the branch into the visual tokens: λ·visual + (1−λ)·branch, λ starting at 0.9.

**Shared by every run:** Qwen3-VL-2B (InternVL3.5-1B only in early rounds) · seed 0 · bf16 · one GPU (A100, H100, H200 or L40S) · 160 px panels · 16 puzzles per step (2 × 8 accumulation) · AdamW, weight decay 0.01, clip 1.0 · Abstractor 8 heads · cosine decay to 10% of the base rate.

**How results were measured:** validation = the same 1,000 PGM puzzles as the LoRA baseline (±3 points). Test = 5,000 puzzles drawn from the 200k neutral test set, never used for training or checkpoint choice (±1.4 points; paired comparisons use McNemar). One epoch of 240k puzzles = 15,000 steps. Chance = 12.5%.

## 1–4. Baselines and first hybrid attempts (Oct 29 – Nov 8)

LoRA alone sets the bar. The first hybrids, with the LLM frozen, could not even memorise 500 puzzles.

| # | Experiment (runs) | Exact conditions | Result | What it showed | In the final result |
| --- | --- | --- | --- | --- | --- |
| 1 | LoRA alone, the baseline (`vlm_lora_pc`, `qwen_pc_neutral5pct_recipe`) | LoRA r16, α32, dropout 0.05 on the LLM only; lr 1e-4, warmup 20, constant for epochs 1–3, cosine to 1e-5 in epoch 4; weight decay 0.01. Full data = 240k PGM neutral × 4 epochs (60k steps); 5% = the seed-0 60k subset × 4 epochs (15k steps); 1% = 12k × 4 epochs | **Full data, 200k test:** Qwen 55.9 / 68.6 / 75.3 / **81.5%** after epochs 1–4 (82.1% with a 5th epoch); InternVL **57.7%** (60.3% after 6). **5% data:** 53.4% val (best, step 14k), **52.6% on the 5k test**. 1% data: 22.9% val. An early r64, lr 2e-4 attempt stalled near 20% | The numbers to beat. At 5% data, validation was still rising \~1.5 points per 1k steps when the schedule ended | Main baseline row. Note the 5% run follows a fixed 4-epoch schedule, not convergence |
| 2 | Stage 0 slot models (`s0_qwen_pgm`, `s0_qwen_pgm_d256`, `s0_qwen_mix20[_d256]`, `s0_ivl1b_*`) | 16 slots, 3 iterations, 12×12 grid per panel, 16 images per step, lr 4e-4, warmup 10k, 100k steps, cosine to 10%; slot dim 32 (decoder 512) or 256 (decoder 2048); PGM 240k or the image mixture | Reconstruction error: Qwen PGM 32-dim **0.331**, 256-dim 0.098; Qwen mixture 0.323 / 0.111; InternVL PGM 0.383 | Slots can represent the panels. Says nothing yet about reasoning | Method section (one line + a slot visualisation) |
| 3 | Frozen LLM, 500-puzzle memorisation (`overfit_*`, `overfit2_*`) | 500 PGM puzzles; 24 layers, head dim 32, λ 0.9. Round 1: 625 steps, lr 1e-4, connector 3e-4, cosine. Round 2: 1,250 steps, constant lr 3e-4, connector 1e-3 | Accuracy on the 500 it trained on: round 1 Qwen interleave 26.8%, fusion 29.0%; InternVL 32.8% / 18.6%. Round 2: Qwen 22.2% / 15.8%; InternVL interleave 30.6%. LoRA alone memorises the same 500 to 100% | Through a frozen LLM, the branch could not even memorise | Negative result, appendix: motivates tuning the branch on its own first |
| 4 | Frozen LLM, full Stage 1 (`s1_qwen_*`, `s1_ivl1b_*`) | 240k × 1 epoch (15k steps); 24 layers, head dim 32, lr 1e-4, connector 3e-4, λ 0.9 | Val: Qwen interleave **19.1%**, fusion 18.0%; InternVL 17.9% / 16.4% | Barely above chance (12.5%) with untuned settings | Negative result, appendix |

## 5–9. Tuning the branch on its own, without the LLM (Nov 8 – Nov 14)

The branch alone (Stage 1a, a linear head on the Abstractor) learns PGM only slowly: \~23% after 4 epochs, 37% after 10 once the slots are fine-tuned too.

| # | Experiment (runs) | Exact conditions | Result | What it showed | In the final result |
| --- | --- | --- | --- | --- | --- |
| 5 | 500-puzzle memorisation with tuned settings (`aux_overfit_*`, `aux_ovf_q_*`, `llm_ovf_q_*`) | 500 PGM puzzles, 3,125 steps, constant rate, warmup 50. Branch alone: layers 6 or 24, lr 1e-4 to 1e-3, head dim 32 or 256, with or without slots. Frozen LLM: 6 layers, head dim 32, lr 3e-4 or 1e-3 | Branch alone (Qwen): 24 layers at 1e-4 27.6%, at 3e-4 56.0%; 6 layers at 1e-4 36.4%, 3e-4 94.6% (6,250 steps), 1e-3 82.4%; 256-dim and no-slot variants 96.8–100%. InternVL 6 layers 3e-4: 83.8%. Frozen LLM, 6 layers: interleave 85.0% (3e-4), fusion 100% (1e-3) | Shallow (6 layers) and a higher rate are what the path needed; with them, even the frozen-LLM path memorises | Appendix: justifies 6 layers and the learning rates |
| 6 | Branch alone, first unseen-data runs (`aux_60k_qwen_*`, `aux_full_*_L6_lr3e-4_ep4`) | 6 layers, head dim 32, 32-dim slots. 60k × 1 epoch (3,750 steps) at lr 3e-4 and 1e-3; 240k × 4 epochs (60k steps, decay from 45k) at 3e-4 | 60k × 1: 15.3% / 15.4%. 240k × 4: Qwen **24.4%** (best), InternVL 24.7% | Generalises, but slowly | Appendix (learning curve) |
| 7 | Sweep at 60k × 4 epochs (`aux_60k4_q_*`, `aux_full10_q_d256_lr1e-3`) | 15k steps, warmup 100, decay from 11,250. Slots: none / 32-dim / 256-dim; layers 6 / 24; lr 3e-4 / 1e-3. Plus 240k × 10 epochs at 256-dim, lr 1e-3 | All between 15.9% and 22.8%: no slots 22.8%, 32-dim 6 layers 1e-3 22.1%, 256-dim 24 layers 3e-4 22.2%; every 256-dim run at 1e-3 ≤ 16.2%. 10-epoch 256-dim: 23.0% | Nothing separates within the ±3-point margin: slots, 256 dims and depth don't help at this budget. 1e-3 hurts the 256-dim models | Appendix: ablation table (honest null result on slots vs no slots) |
| 8 | Branch alone on the task mixture (`aux_mix30_q`, `aux_mix50_q`, `aux_mix30_noCART_q`, `aux_mix30_q_lr3e-4`) | 256-dim mixture slots, 6 layers, head dim 256; 30% or 50% PGM (or 30% without CLEVR-ART); 39k or 30k steps; lr 1e-3 (3e-4 in the rerun); loss normalised by chance | PGM stayed at chance in all four (11.8–16.3%). ART and CLEVR-ART tasks 99.7–100% at lr 3e-4. CLEVR and VSR \~50% (the linear head never sees the question). Best family score 64.3 (3e-4) | The synthetic relational tasks are easy for the branch; PGM is not learned in a mixture | Mixture section: branch-alone row; explain the CLEVR/VSR limitation |
| 9 | 10-epoch branch, slots frozen vs fine-tuned (`aux_full_qwen_*_ep10`, `aux_ft_q_*_ep10`) | 240k × 10 epochs (150k steps), 32-dim slots, head dim 32. 6 layers at lr 3e-4 or 24 layers at 8e-5. `aux_ft` also trains the slots at 3e-5 | Frozen slots: 6 layers **30.2%**, 24 layers 28.3%. Fine-tuned slots: 6 layers **37.1%**, 24 layers 36.9%. Training accuracy ≈ validation (no overfitting) | Fine-tuning the slots adds \~7 points; more training keeps helping. This became the branch for all later hybrids and was extended to 50 epochs | Key branch-alone result: the slot fine-tuning ablation |

## 10–12. Hybrids at full data, on the mixture, and frozen-LLM Stage 1 at 10 epochs (Nov 8 – Nov 15)

With all the data, every hybrid ends level with LoRA alone, and so does the same-size control without relational structure.

| # | Experiment (runs) | Exact conditions | Result | What it showed | In the final result |
| --- | --- | --- | --- | --- | --- |
| 10 | Full-data hybrids with LoRA (`lora_q_int_{pool,panel,slot}_*`, `lora_q_fuse_*`, `lora_q_int_slot_control_lr3e-4`) | Stage 2 from scratch (no warm start); 256-dim PGM slots, frozen; 6 layers, head dim 256; 240k × 4 epochs (60k steps), warmup 500, decay from 45k; LoRA r16 at 1e-4; branch lr 3e-4 or 1e-3. Interleave with 1, 9 (per panel) or 144 (per slot) tokens; fusion λ starts 0.5 and is learned. Control = per-slot interleave with relational attention removed (`no_relational`) | Best val (1,000 puzzles): 1 token 79.6%, 9 tokens 80.6%, 144 tokens **81.3%** (79.5% at 1e-3), fusion **81.4%** (80.5% at 1e-3; λ ended 0.77), control **81.3%**. LoRA alone on the same puzzles: 81.9%. Not test-scored | At full data the branch adds nothing; more tokens and the relational structure make no difference | Main results: the full-data row ("no gain at full data"). Motivates the low-data focus |
| 11 | Hybrids with LoRA on the task mixture (`lora_mix30_q_int_slot`, `lora_mix30_q_fuse`, `lora_mix30_q_int_slot_control`) | Stage 2 from scratch; 256-dim mixture slots; 6 layers, head dim 256; 39k steps, warmup 500, decay from 29,250; branch 3e-4, LoRA r16 at 1e-4; loss normalised by chance; checkpoint picked by family score. Mixture: PGM 30%, CLEVR yes/no 30%, CLEVR-ART RMTS 10% and id-rules 10%, VSR 8%, ART tasks 12% | Final family score: interleave 87.2, fusion 87.2, control **87.6**. PGM 50.2 / 49.2 / 50.8%; CLEVR 99.0 / 98.9 / 99.3%; VSR 87 / 88 / 88%; all ART and CLEVR-ART tasks 100% | The branch adds nothing on the mixture either; the control is as good. No LoRA-alone mixture run exists to compare with | Mixture section: the required multi-task result, reported as a null |
| 12 | Frozen LLM Stage 1, 10 epochs (`s1_q_{int_pool,fuse}_{lr3e-4,lr1e-3}_ep10`) | 256-dim PGM slots, 6 layers, head dim 256, lr 3e-4 or 1e-3, connector 3e-4, λ 0.9; 240k × 10 epochs planned; stopped by you on Nov 15 at 98k–122k steps | Best val: interleave 24.2% (3e-4) and 22.6% (1e-3); fusion **27.2%** (3e-4) and 25.6% (1e-3) | Training the branch through a frozen LLM is far slower than training it alone (37% at 10 epochs) | Appendix: supports training the branch on its own first, then aligning |

## 13–21. Hybrids at 5% data (Nov 14 – Nov 17)

At 5% data the hybrid beats LoRA alone once its LoRA is kept from leaning on the branch: best 56.5% on 5,000 test puzzles vs 52.6%. All hybrids here train on the same seed-0 60k puzzles as LoRA alone; the branch itself was trained on all 240k.

| # | Experiment (runs) | Exact conditions | Result | What it showed | In the final result |
| --- | --- | --- | --- | --- | --- |
| 13 | Alignment from the 10-epoch branch (`align_q_{int,fuse}_L6ft_lr{3e-4,1e-3}`) | Starts from `aux_ft_q_L6_lr3e-4_ep10` (37.1% alone). Frozen LLM, frozen Abstractor and slots; only the connector and token gain / λ train (4.3M parameters); lr 3e-4 or 1e-3, 10k steps, warmup 200, cosine; all 240k puzzles | Levelled by step 1k–3k at 35–37.5% val; final 36.3–36.7% in all four. Branch alone: 37.1% | The connector passes the branch's answer to the LLM almost losslessly | Method: the alignment step. One line in results |
| 14 | Stage 2 at 5%, no dropout (`lora_w5_q_{int,fuse}_L6ft`) + branch-off check (`diag_branchoff`) | From the alignment checkpoint (interleave step 1k, fusion step 3k). LoRA r16 at 1e-4, Abstractor and connector at 3e-5, slots frozen; 60k × 4 epochs (15k steps), warmup 20, decay from 11,250 | Val flat at \~37% through 5k steps. Scored with the branch off at step \~5.5k: interleave 13.3%, fusion 15.6% (vs 38.2 / 36.7% on). Final test: interleave **49.1%**, fusion **38.8%**, both below LoRA alone | Shortcut: LoRA leans on the branch and learns almost nothing itself | A key finding: the shortcut failure, with the branch-off numbers |
| 15 | Stage 2 at 5% with branch dropout (`lora_w5d_q_{int,fuse}_L6ft`) | As 14, plus `--branch_dropout 0.5`: on half the micro-batches the LLM gets nothing from the branch | Test: interleave **55.2%** (+2.6 over LoRA alone, p = 7e-5), fusion **54.1%** (+1.5, p = 0.02). Branch off: 52.5% / 51.7% (= LoRA alone). Val: 55.2% / 56.1% | Dropout fixes the shortcut; the hybrid beats LoRA alone | Main results: fix #1 for the shortcut |
| 16 | Stage 2 with a stronger branch (`align_q_*_L6ft50_lr3e-4`, `lora_w5d_q_{int,fuse}_L6ft50`) | Branch = 50-epoch run at step 204k (40.9% val alone). Alignment as 13 (levelled at 40.6% / 40.3%), then Stage 2 as 15 | Test: interleave **55.1%**, fusion **55.3%** (+2.5 / +2.7 over LoRA alone). Branch off: 49.5% / 49.3%, below LoRA alone | A stronger branch makes Stage 2's LoRA lean on it more, so the total barely moves | Results: shows why LoRA-first is needed for stronger branches |
| 17 | Settled branch (`aux_ft_q_L6_lr3e-4_settle`) | Copy of the 50-epoch run at step 231k, held to 232k, then cosine to 10% over 15k steps (to 247k) | **44.8%** val (step 238k), **43.3%** test | A properly decayed checkpoint of the branch, for the comparisons below | Branch-alone row |
| 18 | Overlap and averaging check (`diag_overlap`) | 1,000 val puzzles. LoRA alone at 5% (step 14k) and at full data (step 60k) vs four branch checkpoints; answers averaged as log-probabilities, w = 0.5, plus w picked by cross-validation | Full data: averaging changes −0.5 to −1.3 points (n.s.). 5% data: +2.2 to **+3.5** points (p = 0.002 with the settled branch). The branch is right on 131 puzzles LoRA misses; it is better on "number" (47.1% vs 31.7%) | The branch knows things LoRA at 5% doesn't, mostly counting; at full data LoRA already knows them | Results: the complementarity analysis and the per-attribute table |
| 19 | LoRA-first with the step-204k branch (`lora_lf_q_{int,fuse}_ft50` + `_ctrl`) | Start from LoRA alone at 5% (step 14k). Branch + connector from the alignment in 16 (interleave step 1k, fusion 3k). Steps 0–500: connector only (1e-4); then LoRA too (2e-5); Abstractor and slots frozen; 3,500 steps on the same 60k. Controls: identical, branch always off | Test: interleave **55.1%** vs control 53.2% (+1.9, p = 3e-6); fusion **54.9%** vs 52.9% (+2.0, p = 4e-6). Controls vs LoRA alone: +0.6 / +0.3 (n.s.). Hybrid with the branch off = its control | Fix #2: the gain is all from the branch, and the hybrid's own LoRA is unharmed | Main results: the fair, equal-steps comparison |
| 20 | Test scoring of everything (`diag_5k`, `diag_5kw`) | 5,000 test puzzles (`test5000.tsv`), scored in five jobs on three GPU types (A100, H200, L40S) as runs finished; McNemar paired tests; log-softmax averaging at w = 0.5; per-attribute breakdown | Branch alone: 10-epoch 35.3%, step 204k 39.8%, settled 43.3%, step 276k 45.0%. Averaging each with LoRA alone: +1.1 / +1.6 / +2.9 / +3.9 | The averaging gain grows with branch strength; the trained hybrids beat averaging with their own branch (+2.5 vs +1.6 at 204k) | Source of every test number in the paper |
| 21 | LoRA-first with the step-276k branch (`align_q_*_L6ft276k_lr3e-4`, `lora_lf_q_{int,fuse}_ft276k`) | As 19, with the branch at step 276k (47.6% val, 45.0% test alone). Alignment levelled at 47.6% / 47.9% val by step 3k; controls reused from 19 | Test: interleave **56.5%** (+3.3 over control, p = 5e-12; +3.9 over LoRA alone), fusion **56.4%** (+3.5, p = 2e-13). +1.4 / +1.5 over 19 (p ≈ 0.001). "Number" puzzles +14.5 / +13.4. Ties averaging (56.5%) | Best result so far; the gain grows with the branch's strength | Headline result, with the per-attribute table |

## 22–23. Still running (as of 11 am, Nov 17)

| # | Experiment (runs) | Exact conditions | Progress | Why it matters |
| --- | --- | --- | --- | --- |
| 22 | 50-epoch branch (`aux_ft_q_L6_lr3e-4_ep50`, `aux_ft_q_L24_lr8e-5_ep50`) | Continues experiment 9 to 750k steps (50 epochs of 240k); rate ramped back from 10% over 2k steps, held, cosine in the last epoch (from 735k); slots fine-tuned at 3e-5 | 6 layers: step 326k, best **47.6%** val at 276k (≈ +1 point per epoch so far, but flat at 44–47% since). 24 layers: step 284k, best 43.5%. Expected to finish Nov 21–28 | Each stronger checkpoint has made the hybrid better (experiments 19 → 21) |
| 23 | Clean 60k pipeline (`s0_qwen_pgm60k`, `aux_ft60k_q_L6_lr{3e-4,1e-3}`) | Everything trained only on the same 60k puzzles as LoRA alone. Slots: experiment 2's recipe on the 60k (100k steps). Branch: experiment 9's recipe on the 60k, 150k steps (40 epochs) | Slots done: reconstruction 0.326 (vs 0.331 on 240k). Branches at step \~24k: 24.8% (3e-4) and 26.1% best (1e-3). Remaining steps (alignment on the 60k, LoRA-first, test scoring) not set up yet | Removes the data confound: the branch currently saw 4× more labelled puzzles than LoRA |

## How to use these in the final result

Lead with the low-data result and its mechanism (counting); report the full-data and mixture nulls plainly; keep the early failures as the story of how we found the shortcut.

| Claim | Backed by | Shown as |
| --- | --- | --- |
| At 5% data the branch adds +3.3 to +3.5 points over an equal-steps control (56.5% vs 53.2%) | 19, 21 (test, 20) | Main results table: LoRA alone, control, hybrids with each branch, branch alone, averaging |
| The gain grows with the branch's own strength | 20, 21 (and 22 as it continues) | Chart: branch-alone accuracy vs hybrid gain over control, one point per branch checkpoint |
| The gain is mostly on counting ("number" +13 to +14 points) | 18, 21 | Per-attribute table: hybrid, control, branch alone |
| Naive joint training falls into a shortcut; dropout or LoRA-first fixes it | 14 (branch-off 13–16%), 15, 16, 19 | Small table: branch on / off for each recipe |
| The branch and LoRA know different things at low data, not at full data | 18 | Overlap table: "only branch right" counts and the averaging gain at 5% vs 100% |
| No gain at full data or on the mixture | 10, 11 | One row each in the main table; one sentence of discussion |
| Fine-tuning the slots matters for the branch (+7 points) | 9 | Ablation row |
| Slots vs no slots, 256 vs 32 dims, depth: no difference at 60k × 4 | 7 | Appendix ablation table |
| Why we train the branch alone first, then align | 3, 4, 5, 12, 13 | Appendix: frozen-LLM results vs branch-alone results |

**Leave out of the main text:** the InternVL runs (paused, only early stages), the first r64 LoRA attempt, the 60k × 1-epoch runs (6), and runs stopped early before a result (`aux_full_qwen`, `aux_full_ivl1b`, `overfit2_ivl1b_fuse`). They can sit in an appendix list of everything tried.

**Wording to use until the gaps below are closed:** "with a branch pretrained on the full training set, a LoRA fine-tuned on 5% of it gains 3.4 points", not "the hybrid is more data-efficient".

## Open gaps

Two gaps decide whether the headline is "same data, +3.4 points" or only "gets there faster".

| Gap a reviewer will raise | Why it is open | What closes it | Status |
| --- | --- | --- | --- |
| The branch saw 240k labelled puzzles; LoRA saw 60k | Stage 0, Stage 1a and alignment all used the full training set | Clean 60k pipeline (23), then LoRA-first and test scoring on it | Running; last two steps not set up |
| LoRA alone at 5% was not trained to convergence | Fixed 4-epoch schedule; val still rising \~1.5 points per 1k steps when the rate decayed | LoRA alone at 5% for \~12 epochs, best by val, scored on the same 5,000 test puzzles | Not queued (your call) |
| The hybrid only ties averaging (56.5% vs 56.5%) | With the step-276k branch, joint training adds nothing beyond averaging the two models | A joint stage that lets the Abstractor adapt (low rate), or a longer LoRA-first | Not started |
| One dataset, one model, one seed | Every hybrid number is Qwen3-VL-2B on PGM neutral, seed 0 | A second seed of the headline runs; held-out-rule PGM split; InternVL | Not started (your call) |
| Validation differences under \~3 points are noise | 1,000 val puzzles; GPU type alone flips \~0.5% of answers in bf16 | Quote only 5,000-puzzle test numbers; re-score all models in one job on one GPU type (\~6 GPU-hours) to remove a further \~0.5-point GPU-type effect on cross-job comparisons | Test numbers done for 13–21; single-job re-score not run |
