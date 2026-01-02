  #!/bin/bash

python test.py \
    --model poseseparate\
    --dataroot /storage/data/dengxy12025/7scenes/heads \
    --name poseseparate/heads/transformer_fc\
    --transformer_hidden_size 256 \
    --backbone inception \
    --batchSize 64 \
    --gpu_ids 3