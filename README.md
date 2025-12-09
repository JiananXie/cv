# PoseNet PyTorch 复现指南

本项目实现了基于 PyTorch 的 PoseNet 模型，用于相机位姿回归任务。

## 1. 环境依赖
含有torch和torchvision的Python环境都可以

## 2. 数据准备

### 数据集
请将数据集放置在 `datasets/` 目录下。例如 `datasets/KingsCollege`。
数据集结构应包含：
- `dataset_train.txt`: 训练集列表
- `dataset_test.txt`: 测试集列表
- 图像文件

这里提供下载KingsCollege数据集的链接：[KingsCollege Dataset](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset)。
或者用以下命令下载并解压数据集：
```bash
mkdir -p datasets/
cd datasets
wget -O KingsCollege.zip https://www.repository.cam.ac.uk/bitstreams/1cd2b04b-ada9-4841-8023-8207f1f3519b/download

unzip kingscollege_data.zip
rm kingscollege_data.zip
```

### 预训练模型
按照论文的建议，使用Places预训练的GoogLeNet模型进行初始化。请下载预训练模型并放置在 `pretrained_models/` 目录下。
```bash
mkdir -p pretrained_models/
cd pretrained_models
wget https://vision.in.tum.de/webarchive/hazirbas/poselstm-pytorch/places-googlenet.pickle
```

## 3. 训练 (Training)

使用 `train.sh` 脚本或直接运行以下命令进行训练。

**第一步：计算图像均值**
```bash
python util/compute_image_mean.py --dataroot datasets/KingsCollege --height 256 --width 256 --save_resized_imgs
```

**第二步：开始训练**
```bash
python train.py \
    --model posenet \
    --init_weights pretrained_models/places-googlenet.pickle \
    --dataroot ./datasets/KingsCollege \
    --name posenet/KingsCollege/beta500 \
    --beta 500 \
    --gpu_ids 0
```
*注意：请根据实际情况修改 `--gpu_ids` 参数。*

## 4. 测试 (Testing)

使用 `test.sh` 脚本或运行以下命令进行评估。测试脚本会自动加载不同 epoch 的模型并在测试集上计算误差中位数，以寻找最佳模型。

```bash
python test.py \
    --model posenet \
    --dataroot ./datasets/KingsCollege \
    --name posenet/KingsCollege/beta500 \
    --gpu_ids 0
```

## 5. 结果
训练日志将保存在 `checkpoints/` 目录下。
测试结果将保存在 `results/` 目录下。

## 6. 提供的模型权重 (Provided Checkpoints)

为了方便直接进行测试复现，我们单独提供了训练好的模型权重（第 490 epoch）及配置文件。

请确保以下文件存在于 `checkpoints/posenet/KingsCollege/beta500/` 目录下：

- `490_net_G.pth`: 模型权重
- `opt_train.txt`: 训练配置
- `opt_test.txt`: 测试配置

如果你下载了这些文件，请按如下结构放置：
```
checkpoints/
└── posenet/
    └── KingsCollege/
        └── beta500/
            ├── 490_net_G.pth
            ├── opt_train.txt
            └── opt_test.txt
```

然后可以直接运行测试命令（指定 epoch 为 490）：
```bash
python test.py \
    --model posenet \
    --dataroot ./datasets/KingsCollege \
    --name posenet/KingsCollege/beta500 \
    --gpu_ids 0 \
    --which_epoch 490
```
