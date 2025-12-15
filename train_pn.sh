!bin/bash

echo "Step 2: Starting PoseNet training..."
# 2. Train PoseNet
python train.py \
    --model posenet \
    --init_weights pretrained_models/places-googlenet.pickle \
    --dataroot datasets/KingsCollege \
    --name posenet/KingsCollege/geoloss \
    --n_epochs 4000 \
    --lr 0.0001 \
    --loss_type geo \
    --gpu_ids 4 \
    --batchSize 64 \
    --save_epoch_freq 10 \
    --seed 42 > train_posenet.log 2>&1
echo "Step 2 Finished: Training complete. Logs saved to train_posenet.log"