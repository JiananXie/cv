# PoseNet PyTorch 复现指南

本项目实现了基于 PyTorch 的 PoseNet 模型，用于相机位姿回归任务。框架基于[poselstm-pytorch](https://github.com/hazirbas/poselstm-pytorch)修改，精简重写了代码。当前的实现基本按照Posenet原文的设置，替换了SGD为Adam优化器，尚未测试LSTM版本。

## 1. 环境依赖
含有torch和torchvision的Python环境都可以

## 2. 数据准备

### 数据集
请将数据集放置在 `datasets/` 目录下。例如 `datasets/cambridge/KingsCollege`。
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
数据集文件结构如下：
```
datasets/
└── posenet/
    ├── cambridge/
    │   ├── KingsCollege
    │   ├── OldHospital
    │   ├──    
    │   ...    
    │
    ├── 7scenes    
    ... ├── chess
        ├── fire   
        ... 
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
**这一步会裁剪图片并保存替换，只需要运行一次即可**

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
参数可以自己调整，train.sh是我复现的参数，可以修改。

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
## 7. 复现结果 (Reproduction Results)
| Dataset       | beta | PoseNet(Report) | PoseNet(reproduce) | PoseLSTM(Report) | PoseLSTM(reproduce) |
| ------------- |:----:|:---------------:|:------------------:| :----: | :----: |
| King’s College  | 500  |   1.92m 5.40°   |  **1.37m 2.82°**   | 0.99m 3.65° | **0.94m 2.61°**|
| Old Hospital  | 1500 |   2.31m 5.38°   |  **2.44m 4.29°**   | 0.99m 3.65° | **0.94m 2.61°**|
| Shop Fa ̧cade  | 100  |   1.46m 8.08°   |  **1.28m 8.39°**   | 0.99m 3.65° | **0.94m 2.61°**|
| St Mary’s Church  | 250  |   2.65m 8.48°   |  **1.93m 6.79°**   | 0.99m 3.65° | **0.94m 2.61°**|
| Chess  | 500  |   0.32m 8.12°   |  **0.33m 6.10°**   | 0.99m 3.65° | **0.94m 2.61°**|
| Fire | 500  |   0.47m 14.4°   |  **0.49m 9.90°**   | 0.99m 3.65° | **0.94m 2.61°**|
| Heads  | 500  |   0.29m 12.0°   |  **0.26m 13.54°**  | 0.99m 3.65° | **0.94m 2.61°**|
| Office  | 500  |   0.48m 7.68°   |  **0.58m 6.81°**   | 0.99m 3.65° | **0.94m 2.61°**|
| Pumpkin  | 500  |   0.47m 8.42°   |  **0.48m 6.92°**   | 0.99m 3.65° | **0.94m 2.61°**|
| Red Kitchen  | 500  |   0.59m 8.64°   |  **0.73m 8.94°**   | 0.99m 3.65° | **0.94m 2.61°**|
| Stairs  | 500  |   0.47m 13.8°   |  **0.38m 13.02°**  | 0.99m 3.65° | **0.94m 2.61°**|