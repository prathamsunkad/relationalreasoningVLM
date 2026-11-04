# How the Slot Abstractor paper trained and reported PGM, and how to make our VLM numbers comparable

Source: *Slot Abstractors* (ICML 2024), the copy in the Reasoning_VLM project. Table/section numbers are from that PDF.
Items marked **(not stated)** were not found in the text I could retrieve; I do not guess them.

## 1. What the paper actually did on PGM

| Aspect | Paper | Where |
|---|---|---|
| What is trained | A task-specific model **from scratch**: CNN encoder + Slot Attention + 24-layer Abstractor. No pretrained VLM. Slot Attention is first pre-trained by reconstruction on the *neutral* training set (lowest validation MSE epoch kept) and then **frozen** for PGM | 4.3 |
| Input | 9 panels resized to **80x80**, scaled to [-1, 1]. For each of the 8 answer choices the candidate is *inserted into the 3x3 problem*, the model is run, and mean-pooled relation embeddings go through a linear layer to one scalar score. 8 forward passes per problem | 4.3, Fig. 1 |
| Loss | Softmax over the 8 scores, **cross-entropy** with the correct choice. **No auxiliary rule/meta labels** ("for a fair comparison we only compared to baselines trained without auxiliary loss") | 4.2, 4.3 |
| Optimiser | **Adam**, lr **8e-5**, batch **24 per GPU on 4 x A100-80GB** (global 96) | A.2 |
| Abstractor | H=8, L=24, d_head=32, d_MLP=512, dropout 0 (PGM) | Table 14 |
| Slot Attention (PGM) | slot dim 32, K=16 slots, T=3, batch 16, lr 4e-4, 75k warmup steps, 8 epochs | Table 12 |
| Length | **One model per regime**, epochs over the 1.2M-problem train set: neutral 73, interpolation 50, H.O.A.P. 90, H.O.T.P. 90, H.O.Triples 90, H.O.L-T 65, H.O.S-C 65, extrapolation 115 | Table 16 |
| Data per regime | 1.2M train / 20K validation / **200K test** problems | Sec. 4.1 |
| Checkpoint selection on PGM | **(not stated)**. For SVRT the sibling OCRA paper picks best-validation-accuracy; for ART/CLEVR-ART it does no selection | OCRA paper 4.3.3 |
| Metric | **Test accuracy = argmax over the 8 choices**, "for 1 trained model for each regime" (single run, no error bars on PGM; ART averages 10 networks) | Table 4 |
| Data efficiency | Neutral regime trained on **20% of the training data**: Slot Abstractor 71.25% vs STSN 53.35% (full data: 91.5 vs 98.2) | Table 5 |
| Baselines | Only models evaluated on >= 1 OOD regime (PredRNet, ARII, MRNet, STSN, ...), numbers copied from their papers | 4.2 |

Their headline numbers (Table 4), test accuracy %, chance = 12.5:

| | Neutral | Intp. | H.O.A.P. | H.O.T.P. | H.O.Triples | H.O.L-T | H.O.S-C | Extp. |
|---|---|---|---|---|---|---|---|---|
| Slot Abstractor | 91.5 | 91.6 | 63.3 | 78.3 | 20.4 | 16.7 | 14.3 | 39.3 |
| PredRNet | 97.4 | 70.5 | 63.4 | 67.8 | 23.4 | 27.3 | 13.1 | 19.7 |
| STSN | 98.2 | 78.5 | - | - | - | - | - | 20.4 |

## 2. How our VLM fine-tuning differs (be explicit about this in any write-up)

| | Paper | Ours (this repo) | Consequence |
|---|---|---|---|
| Model | task-specific, from scratch | pretrained 2B VLM + LoRA | VLM arrives with perception/language priors; parameter counts are not comparable |
| Input | 8 passes, candidate inserted into the matrix, 80x80 panels | **1 pass**: 3x3 context grid image + labelled 2x4 candidate grid image + prompt (the format of your `pgm_infer.py`, 160px panels) | the VLM must compare candidates itself; per-candidate scoring is what Option B later adds |
| Output | scalar score per candidate | the tokens `Answer: k<|im_end|>` | we expose **two readouts**, below |
| Loss | 8-way softmax CE | token CE on the answer (`--loss lm`), or 8-way CE over the digit-token logits (`--loss digit8`), or both | `digit8` is the paper's loss; `lm` is standard SFT |
| Optimiser | Adam 8e-5, global batch 96 | AdamW, LoRA lr 2e-4, global batch 32 (8x4), cosine | LoRA needs a larger lr than full training; not a like-for-like setting |
| Budget | 73-115 epochs x 1.2M problems (neutral: ~88M problem-views, ~0.9M steps at batch 96) | 1 epoch over a subset, typically 20k-240k problems | **compute is 2-3 orders of magnitude smaller** |
| Aux labels | none | none (only answer tokens are supervised) | comparable |
| Selection | not stated for PGM | best of periodic 1000-problem validation checkpoints (`best/`), and the untouched end-of-run adapter (`final/`) | report both, state which |

Because of the budget gap, **absolute numbers at <=1 epoch should not be compared with the paper's 73-epoch numbers as if equal compute**. What is comparable is *data efficiency at matched number of unique training problems*, which is exactly the paper's own Table 5 experiment (20% = 240k problems).

## 3. The two readouts we report

* **`logits`** (headline, paper-comparable): feed `prompt + "Answer: "`, take the next-token logits over exactly the 8 digit tokens, argmax. Always a valid choice, always one of 8, exactly like the paper's softmax over 8 scores. Chance = 12.5%.
* **`generate`**: free greedy decoding, parsed with your `extract_answer` (last standalone digit 1-8). Identical to how `pgm_infer.py` scored the zero-shot model. Also reports the invalid-output rate.

For a fine-tuned model they should agree to within a fraction of a point; a large gap means a formatting problem, not a reasoning one.

## 4. Recommended protocol (so the table reads like the paper's Table 4/5)

1. **Unit of comparison = one model per regime**, trained only on that regime's train split, evaluated on that regime's test split. Regimes: neutral, interpolation, extrapolation first (the paper's Slot Abstractor is strongest there: 91.6 / 39.3 / 78.3 on H.O.T.P.), then H.O.A.P., H.O.T.P., H.O.Triples, H.O.L-T, H.O.S-C.
2. **Evaluate on the full 200K test set** per regime when you report (28-shard array, resumable). For quick iteration use `--max_examples` on a few shards; every table cell prints its n and 95% binomial interval (about +-0.7 points at n=20K, +-0.2 at n=200K).
3. **Data-efficiency series on neutral**: 1%, 5%, 20% (`--train_samples 12000 / 60000 / 240000`; subsets are nested). The 20% point is the direct analogue of paper Table 5 (71.25%).
4. **No auxiliary rule labels** anywhere. Say so in the write-up; it is what makes the comparison fair per the paper's own criterion.
5. **Sanity checks to quote once per model**: blank-context and shuffled-context evaluation (`--context_mode blank|shuffle`, a few thousand problems) should fall to ~12.5%; the predicted-position histogram should be roughly flat.
6. **Seeds**: the paper has one run per regime. For the neutral regime run >= 3 seeds so you can quote seed-to-seed spread; the Wilson interval in the table is sampling error of the test subset only.
7. **State compute**: samples seen, trainable parameters, which modules carry LoRA, wall time. `args.json` and `metrics.jsonl` in each run directory record these.
8. **Compare against**: (a) zero-shot (same eval script, no `--adapter`), (b) LLM-only LoRA vs (c) LoRA + vision/projector (`--lora_vision --lora_projector`), (d) the paper rows (the table tool prints them). The later abstractor variants should be compared against (b)/(c) at matched trainable parameters.

## 5. Things that cannot be matched exactly

* Resolution and preprocessing (paper: 80x80 CNN input; VLM: 160px panels inside stitched grids, resized by each model's processor).
* Their per-candidate insertion gives the model 8 separate looks; the VLM sees all candidates once.
* Pretraining data contamination risk is unknown for a pretrained VLM; the near-chance zero-shot result suggests PGM is not memorised.
