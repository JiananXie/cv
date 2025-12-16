import os
import random

import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
import torch.nn.functional as F

class PoseDataset(data.Dataset):
    def __init__(self, opt, model=None):
        super(PoseDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase

        self.model = model
        self.database = None
        
        # Read dataset list file
        split_file = os.path.join(self.root, f'dataset_{self.phase}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Dataset file not found: {split_file}")

        # Load paths and poses
        # Format: image_path x y z qw qx qy qz
        self.image_paths = np.loadtxt(split_file, dtype=str, delimiter=' ', skiprows=3, usecols=(0))
        # self.image_paths = sorted(np.loadtxt(split_file, dtype=str, delimiter=' ', skiprows=3, usecols=(0)))

        self.poses = np.loadtxt(split_file, dtype=float, delimiter=' ', skiprows=3, usecols=(1, 2, 3, 4, 5, 6, 7))

        sorted_indexs = np.argsort(self.image_paths)

        self.image_paths = self.image_paths[sorted_indexs]
        self.poses = self.poses[sorted_indexs]

        # Add root to paths
        self.image_paths = [os.path.join(self.root, path) for path in self.image_paths]
        
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
        if "Siam" in self.opt.model:
            index = index % self.dataset_size

            # 1. 确定有效范围
            max_range = self.opt.max_range
            start = max(0, index - max_range)
            end = min(self.__len__() - 1, index + max_range)
            ref_index = random.randint(start, end)

            # ref_index = random.randint(0, self.__len__()-1)

            query_image_path = self.image_paths[index]
            query_pose = self.poses[index]
            query_pose = torch.from_numpy(query_pose).float()
            ref_image_path = self.image_paths[ref_index]
            ref_pose = self.poses[ref_index]
            ref_pose = torch.from_numpy(ref_pose).float()

            # Load image
            query_image = Image.open(query_image_path).convert('RGB')
            ref_image = Image.open(ref_image_path).convert('RGB')

            # Apply transforms manually to match original logic
            query_image_tensor = self._transform(query_image)
            ref_image_tensor = self._transform(ref_image)

            if self.opt.img_ret:
                ref_image_tensor, ref_pose = self.images_retrieval(query_image_tensor)

            return {'A': [ref_image_tensor, query_image_tensor], 'B': [ref_pose, query_pose], 'A_paths': query_image_path}
        else:
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
                arr = arr[y:y+size, x:x+size, :]
        else:
            # Center crop
            x = int(round((w - size) / 2.))
            y = int(round((h - size) / 2.))
            arr = arr[y:y+size, x:x+size, :]
            
        # 4. To Tensor (HWC -> CHW)
        return torch.from_numpy(arr.transpose((2, 0, 1))).float()

    def images_retrieval(self, query_image_tensor):
        with torch.no_grad():
            self.model.backbone.eval()
            query_feat = self.model.backbone(query_image_tensor)
            f_max = F.adaptive_max_pool2d(
                input=query_feat,
                output_size=(1, 1)  # 目标输出尺寸为 1x1，实现全局池化
            )  # f_max 的维度: [B, C, 1, 1]
            # 2. 提取全局平均特征 (GAP)
            f_avg = F.adaptive_avg_pool2d(
                input=query_feat,
                output_size=(1, 1)  # 目标输出尺寸为 1x1
            )  # f_avg 的维度: [B, C, 1, 1]
            # 3. 展平并拼接 (Concatenation)
            # 在拼接之前，需要将 [B, C, 1, 1] 展平为 [B, C]
            query_feat = torch.cat([
                f_max.view(f_max.size(0), -1),  # 展平 f_max
                f_avg.view(f_avg.size(0), -1)  # 展平 f_avg
            ], dim=1)
            # 5. [关键修正] 归一化 (L2 Normalize)
            # 这样后续计算的点积 == 余弦相似度
            query_feat = F.normalize(query_feat, p=2, dim=1)
        # --- 修改部分：使用 KDTree 进行检索 ---
        # 1. 转换为 CPU Numpy 数组
        # Scipy 的 KDTree 不认识 PyTorch Tensor，更无法在 GPU 上运行
        query_np = query_feat.cpu().numpy()
        # 2. 获取 Tree 对象
        if self.model.data_base.get('tree') is None:
            raise RuntimeError("Database tree is not initialized! Run build_database first.")
        tree = self.model.data_base['tree']
        # 3. 批量查询 (Batch Query)
        # x: 输入的 query 数组 (Batch_Size, Feature_Dim)
        # k: 找几个最近邻 (k=1)
        # workers: 并行数 (-1 表示使用所有 CPU 核心加速)
        dists, indices = tree.query(x=query_np, k=1, workers=-1)
        # --- 4. 返回结果处理 ---
        # tree.query 返回的是 numpy array
        # dists: [d1, d2, ..., dB] (欧氏距离)
        # indices: [idx1, idx2, ..., idxB] (数据库中的索引)
        ref_path = self.model.data_base['paths'][indices]
        ref_pose = self.model.data_base['poses'][indices]
        ref_img = Image.open(ref_path).convert('RGB')
        ref_image_tensor = self._transform(ref_img)
        ref_pose_tensor = torch.from_numpy(ref_pose).float()
        return ref_image_tensor, ref_pose_tensor
