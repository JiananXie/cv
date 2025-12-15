#!/bin/bash

python test.py \
    --model poselstm\
    --dataroot /storage/data/dengxy12025/KingsCollege/KingsCollege \
    --name poselstm/KingsCollege/geoloss\
    --batchSize 64 \
    --gpu_ids 0