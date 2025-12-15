#!/bin/bash


echo "Starting PoseLSTM training..."
# 3. Train PoseLSTM
python train.py \
    --model poselstm \
    --init_weights pretrained_models/places-googlenet.pickle \
    --dataroot /storage/data/dengxy12025/cambridge/cambridge/StMarysChurch \
    --name poselstm/StMarysChurch/beta250 \
    --beta 250 \
    --loss_type mse \
    --gpu_ids 0 \
    --n_epochs 1200 \
    --save_epoch_freq 10 \
    --seed 42 > train_poselstm_stmaryschurch.log 2>&1
echo "PoseLSTM training complete. Logs saved to train_poselstm_stmaryschurch.log"