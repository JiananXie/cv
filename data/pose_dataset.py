import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data

class PoseDataset(data.Dataset):
    def __init__(self, opt):
        super(PoseDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase
        
        # Read dataset list file
        split_file = os.path.join(self.root, f'dataset_{self.phase}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Dataset file not found: {split_file}")

        # Load paths and poses
        # Format: image_path x y z qw qx qy qz
        self.image_paths = np.loadtxt(split_file, dtype=str, delimiter=' ', skiprows=3, usecols=(0))
        # Add root to paths
        self.image_paths = [os.path.join(self.root, path) for path in self.image_paths]
        
        self.poses = np.loadtxt(split_file, dtype=float, delimiter=' ', skiprows=3, usecols=(1, 2, 3, 4, 5, 6, 7))
        # Filter out specific frames for GreatCourt dataset
        if 'GreatCourt' in self.root:
            keep_indices = []
            for i, path in enumerate(self.image_paths):
                should_skip = False
                if self.phase == 'train':
                    # Skip seq 5/ frame00593
                    if 'seq5/frame00297' in path:
                        should_skip = True
                        print(f"[WARNING]Skipping {path} in training set")
                if not should_skip:
                    keep_indices.append(i)
            
            self.image_paths = [self.image_paths[i] for i in keep_indices]
            self.poses = self.poses[keep_indices]

        # Add root to paths
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
                arr = arr[y:y+size, x:x+size, :]
            return torch.from_numpy(arr.transpose((2, 0, 1))).float()
        else:
            if hasattr(self.opt, 'tta') and self.opt.tta:
                crops = []
                # Center
                x_c = int(round((w - size) / 2.))
                y_c = int(round((h - size) / 2.))
                crops.append(arr[y_c:y_c+size, x_c:x_c+size, :])
                # Top-Left
                crops.append(arr[0:size, 0:size, :])
                # Top-Right
                crops.append(arr[0:size, w-size:w, :])
                # Bottom-Left
                crops.append(arr[h-size:h, 0:size, :])
                # Bottom-Right
                crops.append(arr[h-size:h, w-size:w, :])
                
                crop_tensors = []
                for crop in crops:
                    crop_tensors.append(torch.from_numpy(crop.transpose((2, 0, 1))).float())
                return torch.stack(crop_tensors)
            else:
                # Center crop
                x = int(round((w - size) / 2.))
                y = int(round((h - size) / 2.))
                arr = arr[y:y+size, x:x+size, :]
                return torch.from_numpy(arr.transpose((2, 0, 1))).float()
