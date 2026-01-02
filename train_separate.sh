
echo "Step 2: Starting PoseSeparate training..."
# 2. Train PoseSeparate
python train.py \
    --model poseseparate \
    --init_weights pretrained_models/places-googlenet.pickle \
    --dataroot /storage/data/dengxy12025/KingsCollege/KingsCollege \
    --name poseseparate/KingsCollege/transformer_fc \
    --n_epochs 4000 \
    --lr 0.0001 \
    --loss_type geo \
    --gpu_ids 0 \
    --batchSize 64 \
    --transformer_hidden_size 256 \
    --continue_train \
    --which_epoch 4000 \
    --save_epoch_freq 10 \
    --seed 42 > train_poseseparate_transformer_fc.log 2>&1
echo "Step 2 Finished: Training complete. Logs saved to train_poseseparate_transformer_fc.log"