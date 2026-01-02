#!/bin/bash
#SBATCH -J xjn
#SBATCH -p CS272
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH --gres=gpu:1

module load anaconda3
source activate cv


echo "Step 2: Starting PoseSeparate training..."
# 2. Train PoseSeparate
python train.py \
    --model poseseparate \
    --init_weights pretrained_models/places-googlenet.pickle \
    --dataroot /storage/data/dengxy12025/7scenes/pumpkin\
    --name poseseparate/pumpkin/transformer_fc \
    --backbone inception \
    --n_epochs 4000 \
    --lr 0.0001 \
    --loss_type geo \
    --gpu_ids 0 \
    --batchSize 64 \
    --transformer_hidden_size 256 \
    --save_epoch_freq 10 \
    --seed 42 > train_poseseparate_transformer_fc_pumpkin.log 2>&1
echo "Step 2 Finished: Training complete. Logs saved to train_poseseparate_transformer_fc_pumpkin.log"