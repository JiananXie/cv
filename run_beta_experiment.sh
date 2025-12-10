#!/bin/bash

# 实验配置
BETAS=(7000 10000)
DATAROOT="./datasets/KingsCollege"
GPU_ID=4
N_EPOCHS=500  # 完整实验建议 500，快速测试可调小
SEED=42

# 结果文件
RESULTS_FILE="beta_experiment_results.csv"
echo "beta,pos_error,ori_error" > $RESULTS_FILE

echo "Starting Beta Experiment..."
echo "Betas to test: ${BETAS[*]}"

for beta in "${BETAS[@]}"
do
    echo "========================================================"
    echo "Running experiment for beta = $beta"
    echo "========================================================"
    
    EXP_NAME="posenet_beta${beta}"
    
    # 1. 训练
    echo "[Training] beta=$beta, name=$EXP_NAME"
    # 注意：这里假设你已经有了预训练权重 places-googlenet.pickle
    python train.py \
        --model posenet \
        --dataroot $DATAROOT \
        --name $EXP_NAME \
        --beta $beta \
        --gpu_ids $GPU_ID \
        --print_freq 100 \
        --save_epoch_freq 10 \
        --n_epochs $N_EPOCHS \
        --seed $SEED \
        --init_weights pretrained_models/places-googlenet.pickle

    # 2. 测试
    echo "[Testing] beta=$beta, name=$EXP_NAME"
    python test.py \
        --model posenet \
        --dataroot $DATAROOT \
        --name $EXP_NAME \
        --gpu_ids $GPU_ID
        
    # 3. 解析结果
    # test.py 会在 results/EXP_NAME/test_median.txt 最后写入最佳结果
    # 格式类似于:
    # -----------------
    # 490   1.23m 4.56°
    # ==================
    
    RESULT_TXT="./results/$EXP_NAME/test_median.txt"
    
    if [ -f "$RESULT_TXT" ]; then
        # 获取倒数第二行 (包含结果的那一行)
        LAST_LINE=$(tail -n 2 "$RESULT_TXT" | head -n 1)
        
        # 提取数值 (假设格式固定: epoch pos_err m ori_err °)
        # 使用 awk 提取第2和第3列，并去掉单位
        POS_ERR=$(echo $LAST_LINE | awk '{print $2}' | sed 's/m//')
        ORI_ERR=$(echo $LAST_LINE | awk '{print $3}' | sed 's/°//')
        
        echo "$beta,$POS_ERR,$ORI_ERR" >> $RESULTS_FILE
        echo "Result for beta=$beta: Pos=$POS_ERR m, Ori=$ORI_ERR deg"
    else
        echo "Error: Result file not found for beta=$beta"
    fi
    
done

echo "========================================================"
echo "Experiment finished. Results saved to $RESULTS_FILE"
echo "You can now run: python plot_beta_curve.py"
