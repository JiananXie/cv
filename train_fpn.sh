#!/bin/bash

echo "Step 2: Starting PoseFPN training..."
python train.py \
    --model posefpn \
    --init_weights pretrained_models/places-googlenet.pickle \
    --dataroot /storage/data/dengxy12025/KingsCollege/KingsCollege \
    --name posefpn/KingsCollege/geoloss \
    --n_epochs 4000\
    --lr 0.0005 \
    --batchSize 64 \
    --loss_type geo \
    --gpu_ids 0 \
    --save_epoch_freq 10 \
    --seed 42 > train_posefpn.log 2>&1
echo "Step 2 Finished: Training complete. Logs saved to train_posefpn.log"
