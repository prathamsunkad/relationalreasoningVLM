# relvlm datasets

Everything lives in `~/scratch/relvlm_data` (`$RELVLM_DATA`). Scratch has a 1M **file-count** quota, and PGM alone
uses ~700k files, so every new dataset is **packed**: one `.jsonl` plus one `.zip` per split. Rows reference images
as `zip:<archive>::<member>`. CLEVR and COCO are read straight from their downloaded zips. Formats are described in
[../data.py](../data.py).

Example kinds (all read out with the VLM's own LM head):
* `cand`: one Yes/No sequence per candidate, then softmax over candidates.
* `binary`: Yes/No with BCE.
* `letter`: choice-letter logits.
* `image`: unlabelled, Stage 0 only.

Panel boxes:
* matrices / ART: one box per cell;
* BLINK multi-image sheets: one box per image;
* everything else: the whole image.

All 34 ready splits (table below) were loaded through `data.py` by `verify.py` (report in `_previews/report.json`).
Check any split with `python datasets/verify.py "<glob>"`. It writes preview sheets with panel boxes in red and the
correct candidate in green to `_previews/`.

## What is ready

| dataset | path (under `relvlm_data/` unless absolute) | kind | size | role |
|---|---|---|---|---|
| PGM neutral (existing) | `~/scratch/pgm_data/{train_full,val_full,test_full}/neutral` (`matrix:`) | cand ×8 | 240,000 train (the 20% set the LoRA baseline used), 200k test | Stage 0 + Stage 1 (primary) |
| ART same/different | `art/sd/{train,val,test}.jsonl` | binary | 40 trials ×100 renders / 40 / 10,000 | train + held-out-shape test |
| ART RMTS | `art/rmts/…` | cand ×2 | 480 ×9 / 480 / 10,000 | " |
| ART distribution-of-3 | `art/dist3/…` | cand ×4 | 360 ×12 / 360 / 10,000 | " |
| ART identity rules | `art/idrules/…` | cand ×4 | 8,640 / 500 / 10,000 | " |
| CLEVR relational yes/no | `clevr/relational_yesno_{train,val}.jsonl` | binary | 60,000 / 6,000 (balanced) | Stage 1 mix + eval |
| CLEVR scenes | `clevr/stage0_train.jsonl` | image | 20,000 | Stage 0 mix |
| COCO val2017 | `coco/stage0_val2017.jsonl` | image | 5,000 | Stage 0 mix (natural images) |
| VSR | `vsr/{random,zeroshot}_{train,dev,test}.jsonl` (images in `vsr/images.zip`) | binary | random 7,680 / 1,097 / 2,195; zero-shot 3,489 / 340 / 1,222 | Stage 1 mix + eval |
| CLEVR-ART OOD test | `clevr_art_ood/{rmts,idrules}_cand.jsonl` | cand ×2 / ×4 | 10,000 each | held-out transfer eval |
| " (original sheets + prompt) | `clevr_art_ood/{rmts,idrules}_sheet.jsonl` | letter | 10,000 each | comparable with the zero-shot runs |
| What'sUp | `whatsup/{Controlled_Images_A,Controlled_Images_B,COCO_QA_two_obj,VG_QA_two_obj}.jsonl` | cand ×4 / ×2 | 412 / 408 / 440 / 291 | held-out eval |
| CV-Bench | `cvbench/test_{2d,3d}.jsonl` | letter | 1,438 / 1,200 | held-out eval |
| BLINK (val, 14 subtasks) | `blink/val.jsonl` | letter | 1,901 | held-out eval |
| MMVP | `mmvp/test.jsonl` | letter | 300 (150 pairs) | held-out eval / no-harm check |

## Prepared, not yet run: these execute third-party code and need your OK

| dataset | script | what runs | output |
|---|---|---|---|
| SVRT (23 problems) | `datasets/make_svrt.py` via `slurm/data_generators.sbatch` | clones `fleuret.org/git/pysvrt` (GPL-3) and compiles its C++ generator | `svrt/{train,val,test}` 1,000 / 200 / 2,000 per problem, binary |
| I-RAVEN (7 configurations) | `datasets/make_iraven.py` via `slurm/data_generators.sbatch` | official Python 2 generator (SRAN repo), auto-ported with lib2to3; `--smoke` first | `iraven/{train,val,test}.zip`, 6k/2k/2k per configuration, `raven:` source |
| CLEVR-ART train | `datasets/make_clevr_art.py setup`, `slurm/clevr_art_render.sbatch`, `… pack` | Blender 2.79b + OCRA render scripts (same commits as the OOD test package), 80 CPU array tasks | `clevr_art/{rmts,idrules}_{train,val}`, 9,500 / 500 problems each, cand |

Not included:
* V-PROM: images are not public.
* PGM OOD regimes: download them from DeepMind's GCS bucket when needed; one model per regime.

## Ready-to-paste source specs
```bash
D=~/scratch/relvlm_data; P=~/scratch/pgm_data
# --- Stage 0 (images only; labelled sources work too, only their images are used)
S0_PGM="--train pgm=matrix:$P/train_full/neutral"
S0_MIX="--train pgm=matrix:$P/train_full/neutral:0.35 --train clevr=jsonl:$D/clevr/stage0_train.jsonl:0.2 \
  --train coco=jsonl:$D/coco/stage0_val2017.jsonl:0.15 --train art=jsonl:$D/art/idrules/train.jsonl:0.15 \
  --train vsr=jsonl:$D/vsr/random_train.jsonl:0.15"            # + iraven / svrt / clevr_art once generated
# --- Stage 1 / 2
S1_PGM="--train pgm=matrix:$P/train_full/neutral"
S1_MIX="--train pgm=matrix:$P/train_full/neutral:0.55 \
  --train art_sd=jsonl:$D/art/sd/train.jsonl:0.02 --train art_rmts=jsonl:$D/art/rmts/train.jsonl:0.02 \
  --train art_dist3=jsonl:$D/art/dist3/train.jsonl:0.02 --train art_id=jsonl:$D/art/idrules/train.jsonl:0.04 \
  --train clevr=jsonl:$D/clevr/relational_yesno_train.jsonl:0.15 --train vsr=jsonl:$D/vsr/random_train.jsonl:0.20"
#   once generated, add (and take the weight from pgm/clevr):  iraven=raven:$D/iraven/train.zip:0.15
#   svrt=jsonl:$D/svrt/train.jsonl:0.05  clevr_art_rmts=jsonl:$D/clevr_art/rmts_train.jsonl:0.05  (+ idrules)
# --- validation (first --val selects best.pt)
VAL="--val pgm=matrix:$P/val_full/neutral --val clevr=jsonl:$D/clevr/relational_yesno_val.jsonl --val vsr=jsonl:$D/vsr/random_dev.jsonl"
# --- held-out eval suite (eval.py --source ...)
#   neutral=matrix:$P/test_full/neutral
#   cart_rmts=jsonl:$D/clevr_art_ood/rmts_cand.jsonl      cart_id=jsonl:$D/clevr_art_ood/idrules_cand.jsonl
#   art_sd/rmts/dist3/id=jsonl:$D/art/<task>/test.jsonl   (95 shapes never seen in training)
#   vsr_zs=jsonl:$D/vsr/zeroshot_test.jsonl   whatsup_a=jsonl:$D/whatsup/Controlled_Images_A.jsonl   (+ B, two_obj)
#   cvb2d=jsonl:$D/cvbench/test_2d.jsonl   cvb3d=jsonl:$D/cvbench/test_3d.jsonl   blink=jsonl:$D/blink/val.jsonl
#   mmvp=jsonl:$D/mmvp/test.jsonl
```

## Notes and caveats
* **Held-out discipline.** Transfer claims rest on datasets never trained on: CLEVR-ART OOD, What'sUp, CV-Bench,
  BLINK, MMVP, and the ART test shapes.
  * If CLEVR-ART train renders are added to the mixture, CLEVR-ART OOD becomes in-domain. It still tests new
    colours, shapes and materials, as in the paper, but it is no longer a transfer test.
  * VSR: train on the `random` split, then test on `random_test` (in-domain) and `zeroshot_test`. Concept overlap
    between the two variants means `zeroshot_test` is only partly unseen after training on `random`.
  * COCO val2017 images in Stage 0 may include some VSR/What'sUp COCO images. Stage 0 sees no labels, so this is
    minor, but say so when reporting.
* **ART.** Layout and trials follow the official code:
  * shapes from the Slot-Abstractor repo, trials from ESBN's generators with m = 95 held out, 160 px canvas,
    ±5 px jitter, rendered at 2× (320 px);
  * RMTS and identity-rules test sets use ESBN's `subsample` mode (enumerating all RMTS trials needs > 64 GB);
  * every label is recomputed from the shape indices and checked against ESBN's `y`.
* **What'sUp.** Follows the dataset's convention that `caption_options[0]` is correct; captions are shuffled per
  example. The convention was checked by eye on sample images (e.g. "mug to the right of a knife", "yarn to the left
  of a table").
* **CLEVR-ART OOD crops.** The sheet layout is fixed per task; the scene boxes are voted from 200 sheets, then
  applied to all.
* **SVRT.** The prompt names the problem number. SVRT rules are hidden by design, so the model must learn each rule
  from that problem's examples, as in the paper's per-problem training.
* **MMVP.** Also report pair accuracy (both questions of a pair right); the rows carry a `pair` id.
