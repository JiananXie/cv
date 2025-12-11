import glob
import os
import re

import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from scipy.spatial.transform import Rotation as R_scipy


class ScenesDataset(data.Dataset):
    def __init__(self, opt):
        super(ScenesDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.phase = "Train" if opt.phase == "train" else "Test"

        # Read dataset list file
        split_file = os.path.join(self.root, f'{self.phase}Split.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Dataset file not found: {split_file}")

        self.image_paths = []
        self.seq_list = []
        split_file_lines = None
        with open(split_file, 'r') as f:
            split_file_lines = f.readlines()
            for line in split_file_lines:
                match = re.search(r'\d+', line)
                seq_idx = int(match.group(0))
                seq_name = f"seq-{seq_idx:02}"
                self.seq_list.append(seq_name)
                seq_dir = os.path.join(self.root, seq_name)
                self.image_paths += glob.glob(seq_dir + "/*.color.png")


        # Load paths and poses
        # Format: image_path x y z qw qx qy qz
        # Add root to paths
        self.poses_paths = [img_path.replace(".color.png", ".pose.txt") for img_path in self.image_paths]
        self.poses = []
        for pose_path in self.poses_paths:
            matrix = np.loadtxt(pose_path)
            pose_t = matrix[:3, 3]
            r = matrix[:3, :3]
            # 1. 创建 SciPy Rotation 对象
            rotation = R_scipy.from_matrix(r)

            # 2. 转换为四元数 (xyzw 顺序)
            # SciPy 默认输出顺序是 [x, y, z, w]
            quaternion_xyzw = rotation.as_quat()

            # 3. 调整为 PoseNet 要求的 [w, x, y, z] 顺序 [W, P, Q, R]
            # PoseNet 格式: [W, P, Q, R] = [w, x, y, z]
            w, x, y, z = quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]
            pose_q = [w, x, y, z]
            self.poses.append(np.concatenate((pose_t, pose_q), axis=0))

        # Load mean image if needed
        self.mean_image = None
        if opt.model != "poselstm":
            mean_image_path = os.path.join(self.root, 'mean_image.npy')
            if os.path.exists(mean_image_path):
                self.mean_image = np.load(mean_image_path)
            else:
                print(f"Warning: mean_image.npy not found at {mean_image_path}, proceeding without mean subtraction.")

        self.dataset_size = len(self.image_paths)

    def __getitem__(self, index):
        index = index % self.dataset_size
        image_path = self.image_paths[index]
        pose = self.poses[index]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transforms manually to match original logic
        image_tensor = self._transform(image)

        return {'A': image_tensor, 'B': torch.from_numpy(pose).float(), 'A_paths': image_path}

    def __len__(self):
        return self.dataset_size

    def _transform(self, img):
        # 1. Resize
        img = img.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

        # 2. Subtract mean (and convert to numpy)
        if self.mean_image is None:
            arr = np.array(img).astype('float')
        else:
            arr = np.array(img).astype('float') - self.mean_image.astype('float')

        # 3. Crop
        h, w = arr.shape[0:2]
        size = self.opt.fineSize

        if self.opt.isTrain:
            if w == size and h == size:
                pass
            else:
                x = np.random.randint(0, w - size)
                y = np.random.randint(0, h - size)
                arr = arr[y:y + size, x:x + size, :]
        else:
            # Center crop
            x = int(round((w - size) / 2.))
            y = int(round((h - size) / 2.))
            arr = arr[y:y + size, x:x + size, :]

        # 4. To Tensor (HWC -> CHW)
        return torch.from_numpy(arr.transpose((2, 0, 1))).float()
