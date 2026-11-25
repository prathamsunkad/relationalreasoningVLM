#!/bin/bash
# Nov 25 (user: 'we need to test out H16 on internvl3.5 ... low data regime'): the Slot Abstractor retrained on OUR 5%
# (results/splits/pgm_neutral_train_5pct_60k, the same 60k as every 5% LoRA/hybrid), with the shared 20% model's exact
# recipe (README_slotabstractor_pgm20 + sa_train.sbatch): slot AE 300k steps x 4 puzzles (16 slots, 3 iters, hid 32,
# lr 4e-4, warmup 10k, decay 0.5 / 100k), then Slot Abstractor depth 24, lr 8e-5, batch 96 on 1 GPU, 80 epochs, best on
# the 1,000 shared val puzzles, snapshots every 5 epochs. Reads .npz panels via code/pgm_npz_data.py (antialiased 80 px).
#   sbatch -J sa5_ae    sa5_job.sbatch ae       (resumes; chain 3)
#   sbatch -J sa5_train sa5_job.sbatch train    (resumes; --max_hours 5.8; chain several)
#   sbatch -J sa5_test  sa5_job.sbatch test     (test5000)
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=96GB
#SBATCH -t 06:00:00
#SBATCH -o /home/hice1/psunkad3/scratch/relvlm_runs/sa5/logs/%x_%j.log
S=/home/hice1/psunkad3/scratch/relvlm_runs/sa5; D=/storage/ice1/4/7/psunkad3/pgm_data
cd $S/code
X() { apptainer exec --nv /usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif "$@"; }
echo "== $1 on $(hostname) $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
case $1 in
  ae)    X python -u pretrain_slot_ae_pgm.py --jsonl_root $D --layout none --train_file $S/lists/train_5pct_60k.txt \
           --val_file $S/lists/val1000.txt --out $S/weights/slot_ae_pgm5.pth.tar --steps 300000 --num_workers 14 --max_hours 5.8 --resume ;;
  train) X python -u run_pgm_jsonl.py --mode train --jsonl_root $D --layout none --train_file $S/lists/train_5pct_60k.txt \
           --val_file $S/lists/val1000.txt --slot_checkpoint $S/weights/slot_ae_pgm5.pth.tar --save_path $S/weights/ \
           --tag pgm5 --batch_size 96 --learning_rate 0.00008 --num_workers 14 --num_epochs 80 --max_hours 5.8 \
           --resume --snapshot_every 5 ;;
  test)  CK=$(ls $S/weights/slot_abstractor_pgm_pgm5*_best.pth.tar | head -1)
         X python -u run_pgm_jsonl.py --mode test --jsonl_root $D --layout none --test_file $S/lists/test5000.txt \
           --val_file $S/lists/val1000.txt --slot_checkpoint $S/weights/slot_ae_pgm5.pth.tar --model_checkpoint $CK \
           --save_path $S/weights/ --tag pgm5 --batch_size 96 --num_workers 14 --seed 0 ;;
esac
echo "== done rc=$? $(date)"
