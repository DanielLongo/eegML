#!/bin/bash

#SBATCH --job-name=seizurenet

# Define how long you job will run d-hh:mm:ss
#SBATCH --time 1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH -C GPU_SKU:V100_PCIE
#SBATCH --mem=64G
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8


cd ..

python ./main.py \
  --model_name SeizureNet \
  --num_workers 16 \
  --lr_init 3e-3 \
  --l2_wd 1e-5 \
  --num_epochs 200 \
  --train_batch_size 16 \
  --growth_rate 32 \
  --drop_rate 0.5 \
  --write_outputs \
  --eval_steps 2000 \
  --metric_avg macro \
  --max_checkpoints 1 \
  --save_dir /share/pi/rubin/siyitang/eeg/output/SeizureNet
