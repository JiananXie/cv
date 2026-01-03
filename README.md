# PoseNet PyTorch Reproduction Guide

This project implements the PoseNet model based on PyTorch for camera pose regression tasks. The framework is modified based on [poselstm-pytorch](https://github.com/hazirbas/poselstm-pytorch), with simplified and rewritten code. The current implementation basically follows the settings of the original PoseNet paper, replacing SGD with the Adam optimizer. The LSTM version has not been tested yet.

## 1. Environment
Any Python environment with `torch` and `torchvision` installed should work.

## 2. Data Preparation

### Dataset
Please place the dataset in the `datasets/` directory. For example, `datasets/KingsCollege`.
The dataset structure should include:
- `dataset_train.txt`: Training set list
- `dataset_test.txt`: Test set list
- Image files

Here is the link to download the KingsCollege dataset: [KingsCollege Dataset](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset).
Or use the following commands to download and unzip the dataset:
```bash
mkdir -p datasets/
cd datasets
wget -O KingsCollege.zip https://www.repository.cam.ac.uk/bitstreams/1cd2b04b-ada9-4841-8023-8207f1f3519b/download

unzip kingscollege_data.zip
rm kingscollege_data.zip
```

### Pretrained Model
Following the paper's suggestion, initialize with the GoogLeNet model pretrained on Places. Please download the pretrained model and place it in the `pretrained_models/` directory.
```bash
mkdir -p pretrained_models/
cd pretrained_models
wget https://vision.in.tum.de/webarchive/hazirbas/poselstm-pytorch/places-googlenet.pickle
```

## 3. Training
Reproduce PoseNet training on the KingsCollege dataset.

**Step 1: Compute Image Mean**
```bash
python util/compute_image_mean.py --dataroot datasets/KingsCollege --height 256 --width 256 --save_resized_imgs
```
**This step will crop images and save/replace them. It only needs to be run once.**

**Step 2: Start Training**
```bash
python train.py \
    --model posenet \
    --init_weights pretrained_models/places-googlenet.pickle \
    --dataroot ./datasets/KingsCollege \
    --name posenet/KingsCollege/beta500 \
    --loss_type mse \
    --backbone inception \
    --beta 500 \
    --lr 0.0005 \
    --n_epochs 500 \
    --batchSize 64 \
    --save_epoch_freq 10 \
    --seed 42 \
    --gpu_ids 0
```
Parameters can be adjusted. `train.sh` contains the parameters I used for reproduction and can be modified.

### PoseSeparate Training(Our proposed method)
Standard template for training PoseSeparate:
```bash
echo "Step 2: Starting PoseSeparate training..."
# 2. Train PoseSeparate
python train.py \
    --model poseseparate \
    --init_weights pretrained_models/places-googlenet.pickle \
    --dataroot datasets/KingsCollege \
    --name poseseparate/KingsCollege/transformer_fc \
    --backbone inception \
    --n_epochs 4000 \
    --lr 0.0001 \
    --loss_type geo \
    --gpu_ids 0 \
    --batchSize 64 \
    --save_epoch_freq 10 \
    --transformer_hidden_size 256 \
    --seed 42 > train_poseseparate_transformer_fc_KingsCollege.log 2>&1
echo "Step 2 Finished: Training complete. Logs saved to train_poseseparate_transformer_fc_KingsCollege.log"
```
Note: Many hyperparameters can be explored, such as changing regression heads, backbones, or using hierarchical structures.

## 4. Testing

Use the `test.sh` script or run the following command for evaluation. The test script will automatically load models from different epochs and calculate the median error on the test set to find the best model.

```bash
python test.py \
    --model posenet \
    --dataroot ./datasets/KingsCollege \
    --name posenet/KingsCollege/beta500 \
    --gpu_ids 0
```

## 5. Results
Training logs will be saved in the `checkpoints/` directory.
Test results will be saved in the `results/` directory.

## 6. SiamPoseNet (Siamese Network + Cross Attention)

**Training Command**
```bash
python train.py \
    --model SiamPoseNet \
    --init_weights pretrained_models/places-googlenet.pickle \
    --dataroot datasets/cambridge/KingsCollege \
    --name SiamPoseNet/cambridge/KingsCollege/beta500 \
    --beta 500 \
    --gpu_ids 0
```

**Testing Command**
```bash
python test.py \
    --model SiamPoseNet \
    --dataroot datasets/cambridge/KingsCollege \
    --name SiamPoseNet/cambridge/KingsCollege/beta500 \
    --gpu_ids 0
```

If you want to use image retrieval during testing, please use:
```bash
python test.py \
    --model SiamPoseNet \
    --dataroot datasets/cambridge/KingsCollege \
    --name SiamPoseNet/cambridge/KingsCollege/beta500 \
    --gpu_ids 0 \
    --img_ret
```

## 7. Acknowledgement

The code framework of this project references [poselstm-pytorch](https://github.com/hazirbas/poselstm-pytorch). We would like to express our gratitude.

