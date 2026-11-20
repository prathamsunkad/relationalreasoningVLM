#!/bin/bash
# ART environment check: same/different, m=95 held-out shapes, one run, paper settings.
# Submit from slotabstractor-main/art (after ../setup_art.sh).
#SBATCH -J art_sd_check
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # ICE: add a GPU type if your account needs one, e.g. gpu:H100:1
#SBATCH -t 04:00:00               # original job used 200 min on an A100-80GB
#SBATCH -o art_sd_check_%j.log
# source activate <your env>

python -u train_slot_abstractor_sd.py --batch_size 16 --img_size 128 --num_epochs 600 \
    --m_holdout 95 --run 1
# result: test/same_diff/m95/slot_attention_random_spatial_heldout_unicodes_resizedcropped_pretrained_frozen_autoencoder_abstractor_scoring/run1.txt
