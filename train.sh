#!/bin/bash

echo "Step 1: Computing image mean..."
# 1. Compute image mean and optionally resize images
python util/compute_image_mean.py --dataroot datasets/KingsCollege --height 256 --width 256 --save_resized_imgs
echo "Step 1 Finished: Image mean computed."

echo "Step 2: Starting PoseNet training..."
# 2. Train PoseNet
python train.py \
    --model posenet \
    --init_weights pretrained_models/places-googlenet.pickle \
    --dataroot ./datasets/KingsCollege \
    --name posenet/KingsCollege/beta500 \
    --beta 500 \
    --gpu_ids 4 \
    --save_epoch_freq 10 \
    --seed 42 > train_posenet.log 2>&1
echo "Step 2 Finished: Training complete. Logs saved to train_posenet.log"

echo "Step 3: Starting PoseLSTM training..."
# 3. Train PoseLSTM
python train.py \
    --model poselstm \
    --init_weights pretrained_models/places-googlenet.pickle \
    --dataroot ./datasets/KingsCollege \
    --name poselstm/KingsCollege/beta500 \
    --beta 500 \
    --gpu_ids 4 \
    --n_epochs 1200 \
    --save_epoch_freq 10 \
    --seed 42 > train_poselstm.log 2>&1
echo "Step 3 Finished: Training complete. Logs saved to train_poselstm.log"

