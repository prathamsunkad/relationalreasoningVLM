# Slot Abstractor on the PGM 20% package

Put these files in `slotabstractor-main/pgm/`:
- `pgm_jsonl_data.py`
- `run_pgm_jsonl.py`
- `pretrain_slot_ae_pgm.py`
- `jsonl_to_npz.py`
- `infer_layout.py`

None of the original files are modified. All commands below run from
`slotabstractor-main/pgm/`:

```bash
cd slotabstractor-main/pgm
DATA=../../datasets/pgm-neutral-train20pct-seed20261007-v1/pgm-neutral-train20pct-seed20261007-v1
mkdir -p weights
```

## What the original code expects vs. what you have

| | Original Slot Abstractor (`train_slot_abstractor_pgm.py`) | Your package |
|---|---|---|
| Files | One NPZ per puzzle in `--path`, chosen by `"train"`/`"val"`/`"test"` in the file name | `train.jsonl` (240k), `validation.jsonl` (1k), `test.jsonl` (200k) + one PNG sheet per puzzle |
| Image | `image`: 16 separate 160×160 grayscale panels (8 context, 8 choices) | One 700×1016 contact sheet: 3×3 matrix with "?" + choices A–H |
| Label | `target`: index 0–7 | `answer`: letter A–H |
| Model input | Each panel scaled to [-1, 1], resized to 80×80, encoded by a **frozen pretrained slot autoencoder** (16 slots) | – |
| Pretrained slot AE | `slot_attention_autoencoder_16slots_..._pgm_wholetrainset_neutral_run_1_best.pth.tar`, **not in the repo** and trained on all 1.2M puzzles | – |
| Launch | `srun` with 4 GPUs (reads Slurm variables, DDP/NCCL) | – |

The bridge between the two is cropping each sheet into its 16 panels, using
`layout.json`.

## Step 1: Verify the data (once)

```bash
(cd $DATA && python verify_dataset.py)            # or: --splits train validation
```

## Step 2: Layout and crop check (CPU, ~1 min)

```bash
python infer_layout.py --root $DATA --jsonl $DATA/validation.jsonl --n 300 --out layout.json
python pgm_jsonl_data.py --root $DATA --layout layout.json --out pgm_check.png
```

- **`layout_check.png`** should show red boxes on the 8 context panels, green on
  "?", and blue on A–H.
- **`pgm_check.png`** shows the 16 panels exactly as the model will see them (80×80).

Don't continue until both look right.

## Step 3: Slot autoencoder (choose one)

- **3a. Pretrain on the 20% only (recommended for a matched comparison).**
  - Smoke test (~5 min):
    ```bash
    python pretrain_slot_ae_pgm.py --jsonl_root $DATA --layout layout.json \
        --out weights/smoke_ae.pth.tar --n_train 200 --steps 200 --eval_every 100
    ```
  - Full run (single GPU, run in tmux; continue after a disconnect with the same command):
    ```bash
    python -u pretrain_slot_ae_pgm.py --jsonl_root $DATA --layout layout.json \
        --out weights/slot_ae_pgm20.pth.tar --steps 300000 --resume 2>&1 | tee ae.log
    ```
  - Watch `weights/slot_ae_pgm20_recon.png`: the top row is real panels, the
    bottom row their reconstructions. The bottom row should come to look like
    the top. Also watch `val_mse` in `weights/slot_ae_pgm20_log.csv`.
- **3b. Use the authors' checkpoint.** Ask them for it. Report that it was
  pretrained (without labels) on 100% of the training images.

## Step 4, Option A (recommended): train with the adapter

Reads your JSONL files and sheets directly, with no data copy.

- Smoke test and timing, 1 GPU (~10 min):
  ```bash
  python run_pgm_jsonl.py --mode train --jsonl_root $DATA --layout layout.json \
      --slot_checkpoint weights/slot_ae_pgm20.pth.tar --tag smoke \
      --n_train 2400 --n_val 200 --num_epochs 1
  ```
  Multiply the printed epoch time by 100 (240k ÷ 2.4k) for a full-epoch estimate.
- Full run with the paper's settings: 4 GPUs × batch 24, lr 8e-5, depth 24, up
  to 100 epochs, best checkpoint chosen on validation.
  ```bash
  torchrun --standalone --nproc_per_node 4 run_pgm_jsonl.py --mode train \
      --jsonl_root $DATA --layout layout.json --slot_checkpoint weights/slot_ae_pgm20.pth.tar \
      --tag pgm20 --batch_size 24 --resume 2>&1 | tee sa_train.log
  ```
  - With 1 GPU, use plain `python` and `--batch_size 96` to keep the same global batch.
  - Rerunning the same command with `--resume` continues from the last epoch.
  - The learning curve goes to `weights/*_log.csv`.
- Test on all 200k questions, with 3 seeds, because slot initialisation is random:
  ```bash
  for SEED in 0 1 2; do
    torchrun --standalone --nproc_per_node 4 run_pgm_jsonl.py --mode test \
        --jsonl_root $DATA --layout layout.json --slot_checkpoint weights/slot_ae_pgm20.pth.tar \
        --model_checkpoint weights/slot_abstractor_pgm_pgm20_d24_lr8e-05_run0_best.pth.tar --seed $SEED
  done
  ```
  This writes one exact accuracy to `*.eval.json` and per-question predictions
  to `*.test_preds.csv`.

## Step 4, Option B: the authors' scripts, unchanged

Converts your data to their NPZ format.

```bash
python jsonl_to_npz.py --root $DATA --layout layout.json --out pgm_datasets/neutral20/ \
    --splits train validation test --workers 8      # ~441k files, ~9 GB: check inode quota
ln -s weights/slot_ae_pgm20.pth.tar \
  slot_attention_autoencoder_16slots_dspritesdecoder_morewarmup_lrdecay_pgm_wholetrainset_neutral_run_1_best.pth.tar
# The link is only there because their scripts hard-code this file name; it is your 20% autoencoder.

# inside: salloc -N 1 --ntasks-per-node=4 --cpus-per-task=8 --gres=gpu:4 -t <time>
export MASTER_ADDR=$(hostname) MASTER_PORT=29500 WORLD_SIZE=4
srun python train_slot_abstractor_pgm.py --img_size=80 --batch_size=24 --depth=24 \
    --learning_rate=0.00008 --run='1' --num_epochs=100 --path='pgm_datasets/neutral20/'
srun python test_slot_abstractor_pgm.py --img_size=80 --batch_size=24 --depth=24 --run='1' \
    --path='pgm_datasets/neutral20/' \
    --model_checkpoint weights/slot_attention_pretrained_neutral_train_frozen_autoencoder_abstractor_scoring_multigpu_tcn_16slots_dspritesdecoder_lowerlr_nolrdecay_rowcolposemb_added_pgm_neutral20_run_1_best.pth.tar
```

**Caveats with Option B:**
- No resume: all 100 epochs must fit in one allocation.
- The test script prints one padded accuracy *per GPU*. Combine them yourself,
  weighted by the number of questions each GPU saw.

Both options feed the model byte-identical inputs: the converted NPZ files give
the same tensors as the adapter.

## What to report

- Which slot autoencoder you used (3a or 3b), and its pretraining settings.
- Epochs run, the best validation epoch, and the validation curve.
- Test accuracy as mean ± sd over evaluation seeds, all with the same GPU count.
- That this is your fixed 20% subset, not the paper's sample. Their sample used
  `random.seed(2023)` over `os.listdir` order and can't be reconstructed.
