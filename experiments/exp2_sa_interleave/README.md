# Exp2: pretrained Slot Abstractor interleaved into the fine-tuned InternVL

**Question:** our Slot Abstractor (SA, `../../slot_abstractor/`) reaches 62.8% on validation after training on the
240k set; the fine-tuned InternVL3.5-1B reaches 60.31% on test. Can InternVL use what the SA has learned, if the SA's
object and relation embeddings are put into its prompt?

Unlike the rest of `gap_hybrid_probes/`, Exp2 is built on the **repo's own InternVL fine-tuning**
(`lora/train_lora.py`, `pgmvlm/`). It uses the same `.npz` data, the same 8 sequences per problem (the matrix
completed with candidate k), the same Yes/No score, the same 8-way cross-entropy and the same adapter
(`ivl1b_pc_recipe_e1/final_e6`). Its results are therefore directly comparable with the repo's 57.73% / 60.31% and
with the H-series hybrids, and it is scored on the same `test5000.tsv`.

## What the model sees

```
<image: the matrix completed with candidate k>                       ─► InternViT ─► projector ─┐
Object and relation features of the 9 panels of this completed matrix, row by row:              │
panel 1: <|sa|><|sa|>      ◄── object token, relation token (from the frozen SA)                ├─► LLM (+ fine-tuned LoRA)
...                                                                                             │     logit(Yes) − logit(No)
panel 9: <|sa|><|sa|>      ◄── panel 9 = candidate k                                            │
<the repo's PROMPT_TEXT: "... Is the bottom-right panel the correct completion? Answer Yes or No.">┘

same 9 panels (pixels from the .npz) ─► frozen SA ─► slots (9 × K × D) ──► object connector   ─► 9 tokens
                                                 └► relation embeddings ──► relation connector ─► 9 tokens
```

- **Connectors.** Each is a DeepSets MLP per panel: an MLP on each of the panel's K vectors, a mean, then another
  MLP to the LLM width (1024). The mean makes it independent of slot order. Tokens are RMS-normalised to the scale of
  the word embeddings, with a learned gain, as in `relational/relvlm.py`.
- **Injection.** A forward pre-hook on the language model writes the tokens into the `<|sa|>` positions of
  `inputs_embeds`. The native InternVL forward runs unchanged.
- **Other layouts** (`--layout`): `grouped` (9 object tokens, then 9 relation tokens), `obj` (objects only), `rel`
  (relations only). These show which connector carries the useful information.

## Steps and what each one tells you

| Step | Trains | Compared with | Tells you |
|---|---|---|---|
| `inspect` | nothing | the SA's own validation accuracy (62.8% best, `slot_abstractor/results/val_curve.csv`) | the SA, run on Exp2's inputs, behaves as in its own training |
| `complement` | nothing | SA alone, VLM alone, cross-fitted ensemble | whether the two fail on different puzzles. If the ensemble is no better than the SA, interleaving has little to gain |
| `train_a` (Exp2a) | connectors only; VLM, LoRA and SA frozen | `train_a_ctrl`: same training, each puzzle gets **another puzzle's** SA features | whether the fine-tuned LLM can read SA content as it is. The control removes the "extra trainable soft-prompt tokens" effect |
| `train_b` (Exp2b) | connectors + LoRA (from the adapter), branch dropout 0.5 | `train_b_ctrl`: same steps, prompt and LoRA training, tokens always zero | gain from SA content vs simply training the LoRA longer |
| `eval <run> <ablation>` | nothing | `none` vs `zero`, `shuffle_cand`, `shuffle_puzzle` | whether the trained model uses the tokens, and uses them per candidate |

Validation runs at step 0 too, so you see how the fine-tuned model reacts to the new prompt before any training.
It is logged both with real tokens and with zero tokens (`val_acc_none`, `val_acc_zero` in `metrics.jsonl`).

## Setup (once)

1. **Data and adapter.** The repo's `env.sh` sets `DATA` = the `.npz` folders (`$SHARED_ROOT/pgm_data`, read access via
   `00_START_HERE/CHECKPOINTS.md`). `IVL_ADAPTER` in `../env_local.sh` = `ivl1b_pc_recipe_e1/final_e6`.
   If you exported `DATA` for the contact sheets, `unset DATA` or `export NPZ_DATA=...` first.
2. **The Slot Abstractor.** `exp2/sa_builder.py` loads our run from `slot_abstractor/`. Set `SA_WEIGHTS` in
   `../env_local.sh` to the folder with `slot_ae_pgm20.pth.tar` and `slot_abstractor_pgm_pgm20_1gpu_d24_lr8e-05_run0_best.pth.tar`
   (your `slotabstractor/pgm/weights`). The SA code needs `pip install einops colorlog` in `$CONDA_ENV`.
   Set `SA_ANTIALIAS=0` if the `slotabs` env has torchvision older than 0.17; this changes how panels are resized.
   (`sa_builder_template.py` is only for connecting a different Slot Abstractor.)
3. **Check** (CPU, tiny random models, a few minutes):
   - `python exp2/test_exp2.py`: the Exp2 pipeline;
   - `python exp2/test_sa_builder.py`: the builder's scores equal the original `run_pgm_jsonl.py` scoring.

## Run (from `gap_hybrid_probes/`)

```bash
mkdir -p logs
sbatch -J e2_inspect exp2/exp2.sbatch inspect          # must give ~62.8% (±1.5 with the random slot init)
sbatch -J e2_comp    exp2/exp2.sbatch complement       # exp2_runs/complement_{val1000,test5000}/report.md
sbatch -J e2_a       exp2/exp2.sbatch train_a
sbatch -J e2_a_ctrl  exp2/exp2.sbatch train_a_ctrl
# when exp2a has a best.pt:
sbatch -J e2_b       exp2/exp2.sbatch train_b          # 1 epoch = 15k steps; resubmit after each time-out
sbatch -J e2_b_ctrl  exp2/exp2.sbatch train_b_ctrl
for a in none zero shuffle_cand shuffle_puzzle; do sbatch -J e2_ev_$a exp2/exp2.sbatch eval exp2a $a; done
```

Outputs: `exp2_runs/<run>/` contains `metrics.jsonl`, `best.pt` (connectors + settings), `best_lora/` (stage b),
`state.pt` (resume), and `test5000_<ablation>/summary.json`.

## Assumptions to check

- **The SA never saw validation or test puzzles.** Both it and its slot autoencoder were trained only on the 240k
  training puzzles (`slot_abstractor/`), the same set Exp2 trains on, so they cannot leak answers.
- **The SA was trained on panels cut from the contact sheets; Exp2 gives it the `.npz` panels.** The sheet cells are
  160×160 like the `.npz` panels, so they should be the same pixels. The builder then applies the training
  transform: [-1, 1], then resize to 80×80. If `inspect` comes out clearly below 62.8%, the pixels or the resize
  differ: try `SA_ANTIALIAS=0`, then compare one puzzle's `.npz` panel with its sheet crop.
- **The SA scores each completed matrix separately.** Its relation embeddings for candidate k are computed with k
  in the ninth panel, exactly like the VLM's 8 sequences, so every sequence gets its own tokens.
