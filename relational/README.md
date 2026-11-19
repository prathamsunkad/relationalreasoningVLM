# relvlm: a relational-bottleneck branch (Slot Attention → Abstractor) feeding a frozen VLM

Built from scratch after the "Implementation & Working Pipeline 2" diagram. Backbones: `Qwen/Qwen3-VL-2B-Instruct`
and `OpenGVLab/InternVL3_5-1B-HF` (the same two models as the `vlm_lora_pc` LoRA baseline). Runs go to
`~/scratch/relvlm_runs`. Nothing here is committed anywhere.

## What it does

```
image ──► ViT (frozen) ──► merger / projector (frozen) ──► visual tokens ─┐
            │                                                              ├─► LLM (frozen; LoRA in Stage 2) ─► logit(Yes) − logit(No)
            │ pre-merge patch tokens                       text tokens ────┤      or choice-letter logits
            ▼                                                              │
   ROI-sample every panel to P×P ─► Slot Attention (Stage 0, frozen) ─► z, m per slot
            ─► Abstractor (RCA: Q,K from z, V from s) ─► s per slot, r per sequence
            ├─ approach A  interleave: r ─► MLP ─► M relation tokens at <|rel|> placeholders ─────┘
            └─ approach B  fuse: per visual token  λ·v + (1−λ)·RMS-matched proj(Σ_k mask_k s_k)
```

* **One ViT forward serves both paths.** Everything is attached with forward hooks ([backbones.py](backbones.py)): a hook
  on the vision tower captures the pre-merge patch tokens; a hook on the merger output (Qwen) or the projector output
  (InternVL) applies the fusion; a forward pre-hook on the language model writes the relation tokens into
  `inputs_embeds`. The native HF forward (M-RoPE, DeepStack, chat template) runs unmodified. The test suite
  checks that fusion with λ = 1 reproduces the plain VLM exactly.
* **Panels.** Every image carries normalised panel boxes. For PGM these are the 9 cells of the matrix, and slot
  attention runs per cell (K = 16, as in the paper). A natural image is one panel `[0, 0, 1, 1]`. Patch features are
  bilinearly ROI-sampled to a fixed `P × P` grid (default 12), so the same code works for Qwen's 16-px patches,
  InternViT's 14-px patches and any image size.
* **Abstractor** ([abstractor.py](abstractor.py)) follows the paper: H8 / L24 / d_head 32 / d_mlp 512, D = 32, Wq = Wk at
  initialisation, and s⁰ = m + a panel-index embedding. It adds pre-LN, a final LN and a padding mask, so sequences with
  different panel counts can share a batch.
* **Approach A (interleave):** `"Relational summary: <|rel|>×M\n" + prompt`. The relation tokens are RMS-normalised to
  the scale of the LLM's word embeddings, with a learned gain.
* **Approach B (fuse):** for each merged visual token inside a panel, the relational feature is the relation embedding
  of the slots that own that location (slot mask × s_k, bilinear-sampled at the token centre). It is projected to
  d_LLM and RMS-matched to that token. Tokens outside panels (borders, gaps) are left unchanged. λ = sigmoid(learned
  scalar), initialised to 0.9 (`--fuse_lambda`, `--fixed_lambda`). DeepStack features are not fused.
* **Readout** is the LLM's own LM head, as in the LoRA baseline, so results are directly comparable:
  * `cand` tasks (PGM): 8 sequences, each with the matrix completed by candidate k and the same Yes/No prompt.
    score_k = logit(Yes) − logit(No), then softmax + cross-entropy over the 8 scores.
  * `binary` tasks: BCE on the Yes/No score.
  * `letter` tasks: cross-entropy over the choice-letter logits.
* **Aux head:** a linear head on r, the paper's own scorer. `--objective aux` trains the branch without ever running
  the LLM (cheap; this is the paper's standalone Stage 1a). `--objective both` adds it to the LLM loss.

## Files
| file | purpose |
|---|---|
| `backbones.py` | loading, the three hook points per family, Qwen merge-block → raster reorder, `<|rel|>` token |
| `slots.py` | slot attention, spatial-broadcast decoder, `SlotAutoencoder`, entropy diagnostics |
| `abstractor.py` | relational cross-attention Abstractor with panel embeddings and padding mask |
| `relvlm.py` | `RelationalVLM` (hooks, ROI sampling, interleave, fusion), readout and losses |
| `data.py` | matrix (PGM/RAVEN .npz) and JSONL sources, weighted `MixtureStream` (resumable), collator |
| `factory.py` | build / LoRA / save / restore |
| `train_stage0.py`, `train_stage1.py`, `eval.py` | Stage 0; Stage 1 and Stage 2; sharded eval + merge |
| `slurm/*.sbatch`, `tests/test_tiny.py` | jobs; CPU tests on random tiny Qwen3-VL / InternVL |

## Stages
| stage | trains | frozen | script |
|---|---|---|---|
| 0 | slot attention + decoder (reconstruct LayerNormed ViT features; no labels) | whole VLM | `train_stage0.py` |
| 1a (optional warm-start) | Abstractor + aux head, `--objective aux` (no LLM forward) | VLM, slots | `train_stage1.py` |
| 1 | Abstractor + projector(s) + λ (+ aux head) | VLM, slots | `train_stage1.py` |
| 2 | Stage 1 + LoRA on the LLM (`--lora_r 16 --init_from <stage1 best.pt>`) | ViT, merger, slots | `train_stage1.py` |

Stage 0 is **per backbone**, because Qwen's ViT and InternViT produce different features.

## Run order
```bash
cd ~/scratch/relvlm
sbatch slurm/tests.sbatch                                   # CPU, ~15 min
R=~/scratch/relvlm_runs; D=~/scratch/pgm_data
# Stage 0 (per backbone; chain 2-3 jobs; watch attn_entropy and the slots_step*.png pictures)
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=s0_qwen_pgm slurm/stage0.sbatch
# Stage 1a: Abstractor alone (aux head), cheap; gives the standalone Slot-Abstractor-on-VLM-features number
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,SLOT=$R/s0_qwen_pgm/best.pt,TAG=s1a_qwen,EXTRA="--objective aux --batch_size 8 --grad_accum 2" slurm/stage1.sbatch
# Stage 1: frozen LLM, approach A then B
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,SLOT=$R/s0_qwen_pgm/best.pt,TAG=s1_qwen_inter,EXTRA="--inject interleave --objective both --init_from $R/s1a_qwen/best.pt" slurm/stage1.sbatch
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,SLOT=$R/s0_qwen_pgm/best.pt,TAG=s1_qwen_fuse,EXTRA="--inject fuse --objective both --init_from $R/s1a_qwen/best.pt" slurm/stage1.sbatch
# Stage 2: + LoRA (the baseline's r16 / alpha32 / lr 1e-4)
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,SLOT=$R/s0_qwen_pgm/best.pt,TAG=s2_qwen_inter,EPOCHS=4,EXTRA="--inject interleave --objective both --lora_r 16 --init_from $R/s1_qwen_inter/best.pt" slurm/stage1.sbatch
# test (28 shards) + merge; blank-context sanity check
sbatch --export=ALL,TAG=s2_qwen_inter,SOURCE=neutral=matrix:$D/test_full/neutral slurm/eval_array.sbatch
python eval.py --merge_dir $R/results/s2_qwen_inter_neutral --expected_n 200000
```

## Training recipe

**Targets to beat:** the per-candidate LoRA baseline (`vlm_lora_pc`, r16, lr 1e-4 constant, global batch 16, the
240k neutral set, 1,000 shared validation problems) on the full 200k neutral test set:

| | 1 epoch | 4 epochs |
|---|---|---|
| Qwen3-VL-2B LoRA | 55.9% | **81.5%** |
| InternVL3.5-1B LoRA | 40.2% | **57.7%** |

For reference, the Slot Abstractor paper (pixel CNN, no LLM) reports 91.5% with 100% of PGM neutral and 71.25% with
20%. The 240k set is the paper's 20% data point.

**Stage 0** (defaults in `train_stage0.py`)
* K = 16 slots, D = 32, T = 3 (paper, PGM), P = 12, decoder 512×3.
* Adam lr 4e-4, warmup 10k, cosine over 100k steps, clip 1.0.
* 16 examples × 1 sequence × 9 panels = 144 panels per step. Uses `--seqs_per_example 1`, because the 8 sequences of
  a PGM problem share 8 panels.
* Expect a long plateau with attn_entropy ≈ ln 16 = 2.77. A useful run drops well below that and has
  `slots_used` > 4. If it stays flat past 25% of the steps: try `--slot_dim 16`, more steps, or a larger `--grid`.
* Pick the checkpoint by Stage-1a accuracy, not by MSE alone.

**Stage 1a** (`--objective aux`, a few hours)
* The LLM is never run; only the ViT plus the branch.
* lr 1e-4 Abstractor, batch 16, 1–2 epochs.
* This is the cheap way to check that the slots carry relational information before paying for LLM backprop. It also
  gives the number to compare with the paper's 71.25%.

**Stage 1** (frozen LLM)
* `--objective both --aux_weight 0.5 --init_from <1a best>`, lr 1e-4 (Abstractor) / 3e-4 (projectors, λ).
* Warmup 500, cosine, global batch 16 problems (`--batch_size 2 --grad_accum 8`, the baseline's micro-batch). bf16.
  Add `--grad_ckpt` if memory runs out.
* Cost per step ≈ the LoRA baseline: same sequences, and the same backward pass through the LLM activations.
* Interleave and fuse are **two separate training regimes** (`--inject interleave` or `--inject fuse`, one per run).
  Compare them over 1 epoch first: interleave M = 1 (diagram) vs M = 4, and fuse (λ₀ = 0.9 learned). Then run
  the winner for 4 epochs.
* Logs: `lambda`, `tok_gain`, `llm_acc` vs `aux_acc`. If aux_acc ≫ llm_acc, the LLM is not reading the relation
  tokens: increase M or the projector LR.

**Stage 2** (the most likely way to beat LoRA)
* Branch + LoRA, initialised from the Stage-1 best: `--lora_r 16 --lora_lr 1e-4`, the same 4 epochs as the baseline.
* Report trainable parameters. LoRA r16 is 17.4M (Qwen) or 10.1M (InternVL), from the baseline's `args.json`.
  The branch adds ≈ 3.2M (Abstractor) plus ≈ 4.3M (Qwen) or ≈ 1.1M (InternVL) per projector; `train_stage1.py`
  prints the exact numbers.
* A fair control at matched parameter count: LoRA alone with its rank raised to the same total (Qwen r ≈ 23).
* The baseline's 4-epoch run used a cosine schedule (`lr_schedule cosine`, 60k steps). Use the same schedule here.

**Comparisons to report**
* LoRA (existing) vs Rel-frozen-LLM (Stage 1) vs Rel + LoRA (Stage 2).
* Both backbones, same 240k / val1000 / 200k test, same readout.
* At least 3 seeds for the final Stage-2 vs LoRA comparison.
* Blank-context and shuffled-context checks (`--context_mode blank|shuffle`, should be ~12.5%).
* Data efficiency at 1% / 5% / 20% (`--train_samples 12000 / 60000`; nested subsets).
* PGM OOD regimes, one model per regime as in the paper.

## Data: what to train on and what to hold out

**What is actually on disk, the formats, and ready-to-paste `--train` / `--source` specs:
[datasets/README.md](datasets/README.md).** Everything below is the rationale.

Slot Abstractor paper (checked against the arXiv v2 HTML):

| dataset | content | size used in the paper | Slot Abstractor (prev. best) |
|---|---|---|---|
| ART (4 tasks) | unicode shapes, 5 train / 95 held-out objects | 40 / 480 / 360 / 8,640 train | S/D 96.4, RMTS 91.6, Dist-3 95.2, ID 96.4 (OCRA 88/85/86/93) |
| SVRT (23 tasks) | binary same/different & spatial rules | 500 or 1,000 per task | S/D 91.9, spatial 97.3 at 1k (GAMR 98.7 spatial) |
| CLEVR-ART | RMTS + identity rules, 3D; test uses unseen colours/shapes/materials | not stated | RMTS 96.3, ID 91.6 |
| PGM | 8 regimes, 1.2M / 20k / 200k each | full | neutral 91.5, interp 91.6, extrap 39.3, H.O.T.P. 78.3 |
| V-PROM | real images, neutral | 139k / 8k / 73k | 67.7 (Relation Net 51.2) |

The paper pretrained slot attention one dataset at a time (PGM and V-PROM on their neutral train sets; ART, SVRT and
CLEVR-ART reused a pretrained model from Webb et al. 2023b). It also fine-tuned slot attention on V-PROM only.

**Recommended mixtures**

*Stage 0 (images only; balance by dataset, not by size).* Compare (A) PGM only vs (B) the mix:

| source | why | share in B |
|---|---|---|
| PGM neutral panels | the target task | 30% |
| I-RAVEN / RAVEN-FAIR panels | other matrix rules, multi-object layouts (center, 2×2, 3×3, in-out) | 20% |
| CLEVR / CLEVR-ART scenes (train colours only) | 3D objects, occlusion, materials | 20% |
| SVRT + ART images | 2D shapes, same/different, containment | 15% |
| natural scenes (COCO or Visual Genome) | needed for any transfer to real images (V-PROM, VSR) | 15% |

Use a single K = 16. Natural images may want more slots; if a 2-slot-count setup is needed, keep 16 and check
`slots_used`. For any comparison against the paper's OOD regimes, Stage 0 must use only that regime's training data.

*Stage 1 / 2 (labelled; all converted to `cand` / `binary` / `letter`).* The primary task stays dominant so the PGM
comparison with LoRA remains meaningful. The rest teaches the branch relations that are not PGM-specific:

| source | kind | share |
|---|---|---|
| PGM neutral (240k) | cand ×8 | 50% |
| I-RAVEN (+ RAVEN-FAIR) | cand ×8 | 15% |
| CLEVR-ART RMTS + identity rules (train split, re-rendered) | letter or cand | 10% |
| ART 4 tasks (generated; held-out objects for test) | binary / cand | 5% |
| SVRT 23 tasks (rule stated in the prompt) | binary | 5% |
| VSR (Visual Spatial Reasoning, true/false captions) | binary | 10% |
| CLEVR relational yes/no questions | binary | 5% |

**Hold out (never trained on) to answer "does relational training transfer?"**
* CLEVR-ART OOD test: already on disk at `~/scratch/clevr_art_data/clevr-art-ood-test-seed1-v1`, with zero-shot
  Qwen/InternVL numbers. Its JSONL works as a `letter` source as is. Panels: give the candidate-scene boxes in
  `"panels"`, or leave the whole sheet as one panel.
* PGM OOD regimes, I-RAVEN OOD configurations.
* What'sUp, CV-Bench (2D relation / count), BLINK (spatial relation, relative depth, visual similarity), MMVP.
* A no-harm check on general VQA (e.g. MMStar or POPE subsets). This matters most for `fuse`, which changes visual
  tokens on every image.

Transfer protocol:
1. Train on PGM only and evaluate zero-shot on the held-out suite. Compare with zero-shot base and the PGM-LoRA model
   (does LoRA on PGM hurt or help?).
2. Train on the mixture and evaluate again.
3. The cleanest evidence for a relational effect is better held-out accuracy at equal or lower PGM accuracy than
   LoRA, plus the ablation where the branch is switched off at test time.

JSONL schema for new sources: see the [data.py](data.py) docstring. Image paths are relative to the JSONL file.

## Verified vs not verified
* Verified on CPU with random tiny Qwen3-VL / InternVL models (`tests/test_tiny.py`):
  * the merge-block → raster reorder matches the library's position ids;
  * ROI sampling and token-to-panel mapping;
  * λ = 1 fusion is bit-identical to the plain VLM;
  * gradients reach only the branch (and LoRA in Stage 2), never the frozen VLM;
  * eval is deterministic;
  * the full CLI chain (Stage 0 → 1 → resume → Stage 2 → sharded eval → merge).
* Verified with the real weights on an A100 (`slurm/smoke.sbatch`, synthetic panels, not PGM data): Stage 0 → Stage 1
  (`--inject both --objective both`) → eval runs for Qwen3-VL-2B and InternVL3.5-1B.
  * Stage 0 takes ≈ 0.3 s/step at 144 panels per step (Qwen), so 100k steps is ≈ 8–9 GPU-hours.
  * Stage 1 peak memory at 2 problems (16 sequences) per micro-batch: 16.7 GB (Qwen), 10.3 GB (InternVL).
  * The baseline LoRA runs on an H200 did 6.8 (Qwen) and 9.9 (InternVL) problems/s, ≈ 10 h and 7 h per 240k epoch.
    Expect Stage 1 to be similar; the smoke run was too short to measure steady-state speed.
* Not verified yet:
  * whether slots specialise on Qwen / InternViT features (the plateau);
  * whether the LLM uses the relation tokens.
  Run the 500-problem overfit check first: `--train_samples 500 --epochs 20 --objective llm`.
* InternVL: the branch requires one 448-px tile per image (`crop_to_patches=False`, set automatically). For the
  520-px PGM matrix the baseline also got a single tile.
