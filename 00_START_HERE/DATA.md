# Data: what we trained and tested on, and how it is loaded

The images themselves are not in git. Everything needed to rebuild the exact same splits is: the PGM file lists in
`results/splits/` and the mixture manifests in `data_manifests/`.

## PGM (all PGM-only runs, and the PGM part of the mixtures)

**Source:** DeepMind's Procedurally Generated Matrices, `neutral` regime, one `.npz` per puzzle (16 panels: 8 context +
8 candidates, and the target index). On PACE: `$DATA = ~psunkad3/scratch/pgm_data`.

**Layout and preparation:** `tools/extract_pgm.py` builds `$DATA/{train_full,val_full,test_full}/neutral/*.npz` from
the raw download (run with `--dry_run` first). `tools/make_train20pct.py` (+ `.sbatch`) trims `train_full` to the
shared 20% selection and writes the sorted listing used as the file cache. `tools/check_data.py` checks the layout.

| Split | Size | Exact list in `results/splits/` | Used for |
|---|---|---|---|
| Train, "full" / 20% | 240,000 | `train20pct_seed20261007.tsv.gz` (the shared selection: file, target, original order; seed 20261007, from the team's shared dataset), `pgm_neutral_train_files.txt.gz` (the same files in the order both codebases read them) | Full-data LoRA, H2, H4, H6, Stage 0, branch alone, alignment |
| Train, 5% | 60,000 | `pgm_neutral_train_5pct_60k.txt.gz` | LoRA alone 5%, all 5% hybrids (H8–H13), H13 slots and branch |
| Train, 1% | 12,000 | `pgm_neutral_train_1pct_12k.txt.gz` | LoRA alone 1% |
| Train, memorisation | 500 | first 500 of the same permutation (`--train_samples 500`) | H1, H3, B1 (L4 used its own 500: see `docs/experiments.md`) |
| Validation | 1,000 | `val1000_shared.tsv` (from `pgm_neutral_val_files.txt.gz`) | Every validation number; shared with the LoRA baseline, so curves are comparable |
| Test, full | 200,000 | `pgm_neutral_test_files.txt.gz` | Zero-shot and LoRA test numbers |
| Test, 5,000 | 5,000 | `test5000.tsv` (`random.Random(0).sample` of the 200k list) | Every hybrid test number (A2) |

**How subsets are drawn** (`relational/data.py: subset`, `pgmvlm/data.py`): the first N of
`numpy.random.default_rng(0).permutation(240000)`, applied to `pgm_neutral_train_files.txt`, then sorted. The subsets
are nested (12k inside 60k inside 240k), and both codebases produce the same 60k (their file lists are identical).
`--train_samples N` selects it.

**How a puzzle is shown to the model** (`pgmvlm/data.py`, `relational/data.py: MatrixSource`): for each candidate k,
one image of the full 3×3 matrix with candidate k in the missing cell (panels 160 px), and the question "Is the
bottom-right panel the correct completion of this matrix? Answer Yes or No." Score(k) = logit(Yes) − logit(No);
softmax over the 8 scores; cross-entropy with the answer. The relational branch runs slot attention inside each of
the 9 panel boxes.

## Mixture data (H5, B4, and the mixture Stage 0 slots)

Built by `relational/datasets/` into `$RELVLM_DATA` (`~psunkad3/scratch/relvlm_data`), packed as one `.jsonl` +
one `.zip` per split. **The full registry (every dataset, size, kind, role, how it was made, caveats) is
[`relational/datasets/README.md`](../relational/datasets/README.md).** The `.jsonl` manifests (prompts, answers,
image references, panel boxes) are copied, gzipped, into `data_manifests/<dataset>/`, with `verify_report.json`
(every split loaded and checked by `datasets/verify.py`).

| Dataset | Kind | Train / val | Made by |
|---|---|---|---|
| CLEVR relational yes/no | binary | 60,000 / 6,000 (balanced) | `get_downloads.py` (built from CLEVR v1.0) |
| CLEVR scenes (Stage 0 only) | image | 20,000 | same |
| COCO val2017 (Stage 0 only) | image | 5,000 | `get_downloads.py` |
| VSR, `random` split | binary | 7,680 / 1,097 | `get_downloads.py` |
| ART same/different, RMTS, distribution-of-3, identity rules | binary / cand ×2 / ×4 / ×4 | see registry | `make_art.py` (shapes from the Slot Abstractor repo, trials from ESBN, 95 shapes held out for test) |
| CLEVR-ART RMTS, identity rules | cand ×2 / ×4 | 9,500 / 500 each | `make_clevr_art.py` |
| Held-out eval (not trained on): CLEVR-ART OOD, What'sUp, CV-Bench, BLINK, MMVP, VSR zero-shot | various | test only | `make_clevr_art_ood.py`, `get_downloads.py` |

**The mixtures used** (weights are shares of examples; the sampler normalises them):

| Name | Used by | PGM | CLEVR | VSR | ART sd / rmts / dist3 / id | CLEVR-ART rmts / id | Other |
|---|---|---|---|---|---|---|---|
| S0 mix (images only) | `s0_*_mix20*` slots | 0.20 | 0.25 (scenes) | 0.175 | — / — / — / 0.05 | 0.075 / 0.075 | COCO 0.175 |
| S1 mix, 30% PGM | `lora_mix30_*` (H5), `aux_mix30_q*` | 0.30 | 0.30 | 0.08 | 0.02 / 0.02 / 0.03 / 0.05 | 0.10 / 0.10 | — |
| S1 mix, 50% PGM | `aux_mix50_q` | 0.50 | 0.20 | 0.10 | 0.02 / 0.02 / 0.02 / 0.04 | 0.05 / 0.05 | — |
| S1 mix, 30% PGM, no CLEVR-ART | `aux_mix30_noCART_q` | 0.30 | 0.30 | 0.08 | 0.02 / 0.02 / 0.03 / 0.05 | — | (renormalised) |

Mixture runs use `--loss_norm chance` (each example's loss divided by its chance loss: ln 8 for PGM, ln 2 for
yes/no) and `--best_on family` (checkpoint chosen by the mean accuracy over PGM, CLEVR, VSR, ART, CLEVR-ART).
Ready-to-paste `--train` / `--val` specs: the registry's "Ready-to-paste source specs"; the exact ones each run used
are in its `args.json` (`train`, `val`).

## Source spec syntax (relational code)

`--train name=<kind>:<path>[:weight]`, repeatable; the first is the primary (and `--train_samples` subsets it).

| Kind | Reads | Example |
|---|---|---|
| `matrix` | a folder of PGM `.npz` | `pgm=matrix:$DATA/train_full/neutral` |
| `jsonl` | a packed manifest; images as `zip:<archive>::<member>` | `clevr=jsonl:$RELVLM_DATA/clevr/relational_yesno_train.jsonl:0.30` |
| `raven` | I-RAVEN zip (prepared, not run) | `iraven=raven:$RELVLM_DATA/iraven/train.zip` |

Example kinds inside a manifest: `cand` (one Yes/No sequence per candidate, softmax over candidates), `binary`
(Yes/No, BCE), `letter` (choice-letter logits, zero-shot comparisons), `image` (unlabelled, Stage 0). Check any split
with `python datasets/verify.py "<glob>"` (writes preview sheets with the panel boxes and correct answer marked).

## Zero-shot data

PGM: the 200k neutral test set above. CLEVR-ART: the RMTS and identity-rules test sheets in `~psunkad3/scratch/clevr_art_data`
(`data_manifests/clevr_art_ood/*_sheet.jsonl.gz` packs comparable sheets with the zero-shot prompt for the relational code). Outputs:
`results/zeroshot/`.
