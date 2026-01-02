#!/bin/bash
#SBATCH -J xjn
#SBATCH -p CS272
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH --gres=gpu:1

module load anaconda3
source activate cv


echo "Starting PoseLSTM training..."
# 3. Train PoseLSTM
python train.py \
    --model poselstm \
    --init_weights pretrained_models/places-googlenet.pickle \
    --dataroot /storage/data/dengxy12025/KingsCollege/KingsCollege \
    --name poselstm/KingsCollege/geoloss \
    --n_epochs 4000 \
    --lr 0.0001 \
    --loss_type geo \
    --gpu_ids 0 \
    --batchSize 64 \
    --save_epoch_freq 10 \
    --seed 42 > train_poselstm.log 2>&1
echo "PoseLSTM training complete. Logs saved to train_poselstm_stmaryschurch.log"