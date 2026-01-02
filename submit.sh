#!/bin/bash
#SBATCH -J xjn
#SBATCH -p CS272
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH --gres=gpu:1

module load anaconda3
source activate cv


echo "Step 2: Starting PoseNet training..."
# 2. Train PoseNet
python train.py \
    --model posenet \
    --init_weights pretrained_models/places-googlenet.pickle \
    --dataroot /storage/data/dengxy12025/KingsCollege/KingsCollege \
    --name posenet/KingsCollege/logloss\
    --n_epochs 4000 \
    --lr 0.0001 \
    --loss_type geo \
    --gpu_ids 0 \
    --batchSize 64 \
    --save_epoch_freq 10 \
    --seed 42 > train_posenet.log 2>&1
echo "Step 2 Finished: Training complete. Logs saved to train_posenet.log"