#!/bin/bash

python test.py \
    --model posenet\
    --dataroot /storage/data/dengxy12025/KingsCollege/KingsCollege \
    --name posenet/KingsCollege/geoloss\
    --batchSize 8 \
    --tta \
    --which_epoch 2900 \
    --gpu_ids 0