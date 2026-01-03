  #!/bin/bash

python test.py \
    --model posenet\
    --dataroot /storage/data/dengxy12025/7scenes/chess\
    --name posenet/chess/fc\
    --backbone inception \
    --batchSize 64 \
    --gpu_ids 3