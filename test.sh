#!/bin/bash

python test.py \
    --model posenet  \
    --dataroot ./datasets/KingsCollege \
    --name posenet/KingsCollege/beta500 \
    --gpu 4