import math

import numpy as np
import torch
import torch.nn.functional as F
import os
from collections import OrderedDict

from torch import nn
from torch.autograd import Variable
import util.util as util
from data.data_loader import CreateDataLoader
from .base_model import BaseModel
from . import networks
import pickle
import numpy
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R_scipy

class PoseNetModel(BaseModel):
    def name(self):
        return 'PoseNetModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = torch.zeros(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = torch.zeros(opt.batchSize, opt.output_nc)
        if self.gpu_ids:
            self.input_A = self.input_A.cuda(self.gpu_ids[0])
            self.input_B = self.input_B.cuda(self.gpu_ids[0])

        # load/define networks
        googlenet_weights = None
        if self.isTrain and opt.init_weights != '' and opt.model != 'resnet50':
            try:
                googlenet_file = open(opt.init_weights, "rb")
                googlenet_weights = pickle.load(googlenet_file, encoding="bytes")
                googlenet_file.close()
                print('initializing the weights from '+ opt.init_weights)
            except Exception as e:
                print(f"Could not load weights from {opt.init_weights}: {e}")
                
        self.mean_image = np.load(os.path.join(opt.dataroot , 'mean_image.npy'))

        self.netG = networks.define_network(opt.input_nc, None, opt.model,
                                      init_from=googlenet_weights, isTest=not self.isTrain,
                                      gpu_ids = self.gpu_ids)

        # if not self.isTrain or opt.continue_train:
        #     self.load_network(self.netG, 'G', opt.which_epoch)

        if opt.img_ret:
            self.backbone = networks.Backbone(opt.input_nc, weights=googlenet_weights, isTest=not self.isTrain, gpu_ids=self.gpu_ids)
            self.backbone.to(self.gpu_ids[0])
            self.backbone.eval()

        if self.isTrain:
            self.loss_type = opt.loss_type
            self.old_lr = opt.lr
            # define loss functions
            if self.loss_type == 'mse':
                self.criterion = torch.nn.MSELoss()
            if self.loss_type == 'geo':
                self.sx = nn.Parameter(torch.tensor(0.0))
                self.sq = nn.Parameter(torch.tensor(-3.0))

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, eps=1,
                                                weight_decay=0.0625,
                                                betas=(self.opt.adambeta1, self.opt.adambeta2))
            self.optimizers.append(self.optimizer_G)
            # for optimizer in self.optimizers:
            #     self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        # networks.print_network(self.netG)
        # print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.image_paths = input['A_paths']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def forward(self):
        self.pred_B = self.netG(self.input_A)

    # no backprop gradients
    def test(self):
        self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward(self):
        self.loss_G = 0
        self.loss_pos = 0
        self.loss_ori = 0

        loss_weights = [0.3, 0.3, 1]
        loop_range = 3

        for l in range(loop_range):
            w = loss_weights[l]
            pred_pos = self.pred_B[2 * l]
            pred_ori = self.pred_B[2 * l + 1]
            target_pos = self.input_B[:, 0:3]
            target_ori = F.normalize(self.input_B[:, 3:], p=2, dim=1)

            if self.loss_type == 'mse':
                error_pos = self.criterion(pred_pos, target_pos)
                error_ori = self.criterion(pred_ori, target_ori)
            else:
                error_pos = torch.norm(pred_pos - target_pos, p=2, dim=1).mean()
                error_ori = torch.norm(pred_ori - target_ori, p=2, dim=1).mean()

            # Combine losses
            if self.loss_type == 'geo':
                loss_pos = torch.exp(-self.sx) * error_pos + self.sx
                loss_ori = torch.exp(-self.sq) * error_ori + self.sq
                total_loss = loss_pos + loss_ori
            else:
                total_loss = error_pos + error_ori * self.opt.beta

            self.loss_G += total_loss * w
            self.loss_pos += error_pos.item() * w
            self.loss_ori += error_ori.item() * w * self.opt.beta

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.opt.isTrain:
            return OrderedDict([('pos_err', self.loss_pos),
                                ('ori_err', self.loss_ori),
                                ])

        pos_err = torch.dist(self.pred_B[0], self.input_B[:, 0:3])
        ori_gt = F.normalize(self.input_B[:, 3:], p=2, dim=1)
        abs_distance = torch.abs((ori_gt.mul(self.pred_B[1])).sum())
        ori_err = 2*180/numpy.pi* torch.acos(abs_distance)
        return [pos_err.item(), ori_err.item()]

    def get_current_pose(self):
        return numpy.concatenate((self.pred_B[0].data[0].cpu().numpy(),
                                  self.pred_B[1].data[0].cpu().numpy()))

    def get_current_visuals(self):
        input_A = util.tensor2im(self.input_A.data)
        # pred_B = util.tensor2im(self.pred_B.data)
        # input_B = util.tensor2im(self.input_B.data)
        return OrderedDict([('input_A', input_A)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)


class SiamPoseNetModel(PoseNetModel):
    def name(self):
        return 'SiamPoseNetModel'

    def build_datatbase(self):
        self.backbone.eval()
        data_base_opt = self.opt
        data_base_opt.phase = "train"
        data_base_opt.img_ret = False
        data_loader = CreateDataLoader(data_base_opt)
        # 初始化用于收集数据的列表
        all_features = []
        all_poses = []
        all_paths = []
        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Building Retrieval Database")
            for i, data in enumerate(pbar):
                query_img = data['A'][1].to(self.gpu_ids[0])
                query_pose = data['B'][1].to(self.gpu_ids[0])
                query_img_paths = data['A_paths']
                f_v = self.backbone(query_img)
                # 1. 提取全局最大特征 (GMP)
                f_max = F.adaptive_max_pool2d(
                    input=f_v,
                    output_size=(1, 1)  # 目标输出尺寸为 1x1，实现全局池化
                )  # f_max 的维度: [B, C, 1, 1]
                # 2. 提取全局平均特征 (GAP)
                f_avg = F.adaptive_avg_pool2d(
                    input=f_v,
                    output_size=(1, 1)  # 目标输出尺寸为 1x1
                )  # f_avg 的维度: [B, C, 1, 1]
                # 3. 展平并拼接 (Concatenation)
                # 在拼接之前，需要将 [B, C, 1, 1] 展平为 [B, C]
                f_v = torch.cat([
                    f_max.view(f_max.size(0), -1),  # 展平 f_max
                    f_avg.view(f_avg.size(0), -1)  # 展平 f_avg
                ], dim=1)
                # 5. [关键修正] 归一化 (L2 Normalize)
                # 这样后续计算的点积 == 余弦相似度
                f_v = F.normalize(f_v, p=2, dim=1)
                all_features.append(f_v.cpu())
                all_poses.append(query_pose.cpu())
                all_paths.extend(query_img_paths)
        # --- 3. 数据合并与最终构建 ---
        # 合并所有批次的特征和位姿
        all_features = torch.cat(all_features, dim=0).numpy()
        all_poses = torch.cat(all_poses, dim=0).numpy()
        # 结构化存储数据库
        # ❗ 补全：构建 self.data_base 字典
        self.data_base = {
            'features': all_features,  # N x 2C 矩阵
            'poses': all_poses,  # N x 7 矩阵 (假设是平移+四元数)
            'paths': np.array(all_paths),
            'tree': None  # 留给构建 ANN 索引的位置
        }
        # --- 4. 可选：构建 ANN 索引 (用于快速检索) ---
        try:
            from scipy.spatial import KDTree
            # 使用 KDTree 或 FAISS/Annoy 建立索引
            self.data_base['tree'] = KDTree(self.data_base['features'])
        except Exception as e:
            print(f"Warning: Failed to build KDTree index. Retrieval will be slow. Error: {e}")

    # def images_retrieval(self, data):
    #     query_img = data['A'][1]
    #     with torch.no_grad():
    #         self.backbone.eval()
    #         query_feat = self.backbone(query_img)
    #         f_max = F.adaptive_max_pool2d(
    #             input=query_feat,
    #             output_size=(1, 1)  # 目标输出尺寸为 1x1，实现全局池化
    #         )  # f_max 的维度: [B, C, 1, 1]
    #         # 2. 提取全局平均特征 (GAP)
    #         f_avg = F.adaptive_avg_pool2d(
    #             input=query_feat,
    #             output_size=(1, 1)  # 目标输出尺寸为 1x1
    #         )  # f_avg 的维度: [B, C, 1, 1]
    #         # 3. 展平并拼接 (Concatenation)
    #         # 在拼接之前，需要将 [B, C, 1, 1] 展平为 [B, C]
    #         query_feat = torch.cat([
    #             f_max.view(f_max.size(0), -1),  # 展平 f_max
    #             f_avg.view(f_avg.size(0), -1)  # 展平 f_avg
    #         ], dim=1)
    #         # 5. [关键修正] 归一化 (L2 Normalize)
    #         # 这样后续计算的点积 == 余弦相似度
    #         query_feat = F.normalize(query_feat, p=2, dim=1)
    #     # --- 修改部分：使用 KDTree 进行检索 ---
    #     # 1. 转换为 CPU Numpy 数组
    #     # Scipy 的 KDTree 不认识 PyTorch Tensor，更无法在 GPU 上运行
    #     query_np = query_feat.cpu().numpy()
    #     # 2. 获取 Tree 对象
    #     if self.data_base.get('tree') is None:
    #         raise RuntimeError("Database tree is not initialized! Run build_database first.")
    #     tree = self.data_base['tree']
    #     # 3. 批量查询 (Batch Query)
    #     # x: 输入的 query 数组 (Batch_Size, Feature_Dim)
    #     # k: 找几个最近邻 (k=1)
    #     # workers: 并行数 (-1 表示使用所有 CPU 核心加速)
    #     dists, indices = tree.query(x=query_np, k=1, workers=-1)
    #     # --- 4. 返回结果处理 ---
    #     # tree.query 返回的是 numpy array
    #     # dists: [d1, d2, ..., dB] (欧氏距离)
    #     # indices: [idx1, idx2, ..., idxB] (数据库中的索引)
    #     data['A'][0] = self.data_base['features'][indices]
    #     return indices, dists

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.ref_pose = input_B[0].to(self.gpu_ids[0])
        self.query_pose = input_B[1].to(self.gpu_ids[0])
        self.image_paths = input['A_paths']
        self.input_A = [item.to(self.gpu_ids[0]) for item in input_A]

    def get_current_errors(self):
        if self.opt.isTrain:
            return OrderedDict([('pos_err', self.loss_pos),
                                ('ori_err', self.loss_ori),
                                ])

        pos_err = torch.dist(self.pred_B[0], self.query_pose[:, 0:3])
        ori_gt = F.normalize(self.query_pose[:, 3:], p=2, dim=1)
        abs_distance = torch.abs((ori_gt.mul(self.pred_B[1])).sum())
        ori_err = 2*180/numpy.pi* torch.acos(abs_distance)
        return [pos_err.item(), ori_err.item()]

    def forward(self):
        self.pred_B = self.netG(self.input_A)

    def compute_apr_from_pred(self, rpr, gt_apr):
        """
        根据 T_Q = T_Ref * Delta_T 公式，计算查询图像的预测绝对位姿 (APR_pred)。
        输入四元数格式假设为 [w, x, y, z]
        """

        # --- 1. 解包输入 ---
        t_rpr = rpr[0]  # Delta_t (B, 3)
        q_rpr = rpr[1]  # Delta_q (B, 4) - [w, x, y, z]
        t_gt_apr = gt_apr[:, :3]  # t_ref (B, 3)
        q_gt_apr = gt_apr[:, 3:]  # q_ref (B, 4) - [w, x, y, z]

        B = t_rpr.shape[0]
        device = t_rpr.device

        # --- 2. 核心运算：转换为 NumPy/SciPy 进行旋转矩阵转换 ---

        # 将 PyTorch 张量转移到 CPU 并转换为 NumPy 数组
        q_rpr_np = q_rpr.detach().cpu().numpy()
        q_gt_apr_np = q_gt_apr.detach().cpu().numpy()

        # ***** 关键修改: 维度重排 *****

        # 将 [w, x, y, z] 转换为 SciPy 的 [x, y, z, w] 格式
        # 原维度: (B, 4)
        # 新维度: (B, 4) - 顺序 [1, 2, 3, 0]
        q_rpr_scipy = q_rpr_np[:, [1, 2, 3, 0]]
        q_gt_apr_scipy = q_gt_apr_np[:, [1, 2, 3, 0]]

        # 使用 SciPy 进行四元数到旋转矩阵的批量转换

        # 预测的相对旋转矩阵 Delta_R
        R_rpr_scipy = R_scipy.from_quat(q_rpr_scipy)
        Delta_R_np = R_rpr_scipy.as_matrix()  # Delta_R (B, 3, 3)

        # 参考图像的绝对旋转矩阵 R_Ref
        R_ref_scipy = R_scipy.from_quat(q_gt_apr_scipy)
        R_Ref_np = R_ref_scipy.as_matrix()  # R_Ref (B, 3, 3)

        # --- 3. 旋转和平移运算 (回到 PyTorch) ---

        # 将 NumPy 矩阵转换回 PyTorch 张量并移回原设备
        R_Ref = torch.from_numpy(R_Ref_np).to(device).float()
        Delta_R = torch.from_numpy(Delta_R_np).to(device).float()

        # A. 计算绝对旋转 R_Q = R_Ref * Delta_R
        R_query = torch.bmm(R_Ref, Delta_R)  # R_Q (B, 3, 3)

        # B. 计算绝对平移 t_Q = R_Ref * Delta_t + t_Ref
        t_relative_transformed = torch.bmm(R_Ref, t_rpr.unsqueeze(-1)).squeeze(-1)
        t_query = t_relative_transformed + t_gt_apr  # t_Q (B, 3)

        # --- 4. 转换绝对旋转矩阵回四元数 ---

        # 将 R_query 转移到 CPU 并转换为 NumPy
        R_query_np = R_query.detach().cpu().numpy()

        # 使用 SciPy 转换 R_Q 回四元数
        R_query_scipy = R_scipy.from_matrix(R_query_np)
        q_query_scipy_np = R_query_scipy.as_quat()  # q_Q (B, 4) - [x, y, z, w] 格式

        # ***** 关键修改: 将 SciPy 格式 [x, y, z, w] 转回 [w, x, y, z] 格式 *****
        q_query_np = q_query_scipy_np[:, [3, 0, 1, 2]]

        # --- 5. 格式化最终输出 ---

        # 转换回 PyTorch 张量
        q_query = torch.from_numpy(q_query_np).to(device).float()

        # # 拼接最终的绝对位姿 [t_query, q_query]
        # apr_pred = torch.cat([t_query, q_query], dim=1)  # (B, 7)
        #
        # return apr_pred

        return [t_query, q_query]

    # no backprop gradients
    def test(self):
        self.forward()
        self.pred_B = self.compute_apr_from_pred(self.pred_B, self.ref_pose)

    def compute_rpr_from_apr(self, ref_apr, query_apr):
        """
        根据 T_RPR = T_Ref_Inv * T_Q 公式，计算相对位姿 (RPR)。

        参数:
        ref_apr (torch.Tensor): 参考图像的绝对位姿 P_Ref (B, 7) [t_ref, q_ref]
        query_apr (torch.Tensor): 查询图像的绝对位姿 P_Q (B, 7) [t_query, q_query]

        返回:
        torch.Tensor: 预测的相对位姿 (RPR) (B, 7) [Delta_t, Delta_q]
        """

        # --- 1. 解包和准备数据 ---

        t_ref = ref_apr[:, :3]  # t_ref (B, 3)
        q_ref = ref_apr[:, 3:]  # q_ref (B, 4) - [w, x, y, z]

        t_query = query_apr[:, :3]  # t_query (B, 3)
        q_query = query_apr[:, 3:]  # q_query (B, 4) - [w, x, y, z]

        B = t_ref.shape[0]
        device = t_ref.device

        # 将 PyTorch 张量转移到 CPU 并转换为 NumPy 数组
        q_ref_np = q_ref.detach().cpu().numpy()
        q_query_np = q_query.detach().cpu().numpy()

        # --- 2. 旋转矩阵转换与维度重排 ---

        # SciPy 格式: [x, y, z, w]。需要将我们的 [w, x, y, z] -> [x, y, z, w]
        q_ref_scipy = q_ref_np[:, [1, 2, 3, 0]]
        q_query_scipy = q_query_np[:, [1, 2, 3, 0]]

        # 获取旋转对象
        R_ref_scipy = R_scipy.from_quat(q_ref_scipy)
        R_query_scipy = R_scipy.from_quat(q_query_scipy)

        # 获取旋转矩阵 (用于平移计算)
        R_Ref_np = R_ref_scipy.as_matrix()  # R_Ref (B, 3, 3)

        # --- 3. 计算相对旋转 $\Delta R = R_{Ref}^T \cdot R_Q$ ---

        # SciPy 中可以直接对旋转对象进行运算，比矩阵乘法更简洁和稳定
        # 相对旋转对象 Delta_R = R_Ref.inv() * R_Q
        Delta_R_scipy = R_ref_scipy.inv() * R_query_scipy

        # 将 Delta_R 转换回四元数 (SciPy 默认输出 [x, y, z, w] 格式)
        q_delta_scipy_np = Delta_R_scipy.as_quat()

        # 转换回我们的 [w, x, y, z] 格式
        q_delta_np = q_delta_scipy_np[:, [3, 0, 1, 2]]

        # --- 4. 计算相对平移 $\Delta t = R_{Ref}^T (t_Q - t_{Ref})$ ---

        # A. 计算世界坐标系中的平移差 t_Q - t_Ref
        t_diff = t_query - t_ref  # (B, 3)

        # B. 获取 R_Ref 的转置 (R_Ref^T)
        R_Ref = torch.from_numpy(R_Ref_np).to(device).float()  # (B, 3, 3)
        R_Ref_T = R_Ref.transpose(1, 2)  # (B, 3, 3)

        # C. 计算 Delta_t = R_Ref^T * t_diff
        # [B, 3, 3] x [B, 3, 1] -> [B, 3, 1] -> [B, 3]
        delta_t = torch.bmm(R_Ref_T, t_diff.unsqueeze(-1)).squeeze(-1)  # (B, 3)

        # --- 5. 格式化最终输出 ---

        # 转换 Delta_q 回 PyTorch 张量
        delta_q = torch.from_numpy(q_delta_np).to(device).float()

        # 拼接最终的相对位姿 [Delta_t, Delta_q]
        rpr_gt = torch.cat([delta_t, delta_q], dim=1)  # (B, 7)

        return rpr_gt

    # def get_triplet_loss(self, F_ref, F_q):
    #     """
    #     【修改版】尺度不变的几何感知 InfoNCE Loss
    #
    #     核心改进：
    #     1. 移除 beta：通过除以 Batch 均值，将平移(m)和旋转(deg)统一为无量纲的相对距离。
    #     2. 移除 margin/thresh：使用 InfoNCE 分类范式，自动拉近相对最近的，推开相对远的。
    #     3. 显式包含角度：同时利用平移和旋转信息寻找 Ground Truth。
    #     """
    #     device = F_q.device
    #
    #     # --- 1. 特征空间处理 ---
    #     # 归一化特征
    #     F_q = F.normalize(F_q, p=2, dim=1)
    #     F_ref = F.normalize(F_ref, p=2, dim=1)
    #
    #     # 计算余弦相似度 Logits [B, B]
    #     # 每一行代表 1 个 Query 对所有 Ref 的相似度
    #     # temperature 是缩放系数，0.07-0.1 是对比学习的标准常数，不需要针对数据集调整
    #     temperature = 0.1
    #     logits = torch.matmul(F_q, F_ref.T) / temperature
    #
    #     # --- 2. 物理几何空间处理 ---
    #     # 获取位姿 GT
    #     P_q = self.query_pose  # [B, 7]
    #     P_ref = self.ref_pose  # [B, 7]
    #
    #     # 2.1 计算平移矩阵距离 [B, B]
    #     # dist_t[i, j] 是第 i 个 query 和第 j 个 ref 的米距离
    #     dist_t = torch.cdist(P_q[:, :3], P_ref[:, :3], p=2)
    #
    #     # 2.2 计算旋转矩阵距离 [B, B]
    #     # 提取并归一化四元数
    #     q_q = F.normalize(P_q[:, 3:], p=2, dim=1)
    #     q_ref = F.normalize(P_ref[:, 3:], p=2, dim=1)
    #
    #     # 向量化计算两两四元数夹角
    #     # dot matrix: [B, B]
    #     dot_product = torch.abs(torch.matmul(q_q, q_ref.T))
    #     dot_product = torch.clamp(dot_product, -1.0, 1.0)
    #
    #     # 角度距离 (弧度)
    #     dist_r = 2.0 * torch.acos(dot_product)
    #
    #     # --- 3. 动态归一化 (核心去参步骤) ---
    #     # 计算当前 Batch 的平均距离
    #     # 加上 1e-6 防止除以 0 (虽然在物理世界几乎不可能完全重合)
    #     mean_t = dist_t.mean() + 1e-6
    #     mean_r = dist_r.mean() + 1e-6
    #
    #     # 将绝对单位(米/弧度)转换为相对单位(倍数)
    #     # 例如：1.0 表示距离等于平均值，0.1 表示非常近
    #     norm_t = dist_t / mean_t
    #     norm_r = dist_r / mean_r
    #
    #     # --- 4. 生成动态标签 ---
    #     # 综合几何距离 = 归一化平移 + 归一化旋转
    #     # 此时两者量纲一致，直接相加即可，不需要 beta 权重
    #     geo_dist = norm_t + norm_r
    #
    #     # 找到每个 Query 在当前 Batch 中几何距离最近的 Reference 的索引
    #     # 这就是这一轮训练的"正确答案" (Ground Truth Class)
    #     target_labels = torch.argmin(geo_dist, dim=1)  # [B]
    #
    #     # --- 5. 计算 Cross Entropy Loss ---
    #     # 这是一个 B 分类问题：
    #     # 网络应该预测特征相似度最高的那个索引，正好也是几何距离最近的那个索引
    #     loss = F.cross_entropy(logits, target_labels)
    #
    #     return loss

    def backward(self):
        gt_rpr = self.compute_rpr_from_apr(self.ref_pose, self.query_pose)
        self.loss_G = 0
        self.loss_pos = 0
        self.loss_ori = 0

        loss_weights = [0.3, 0.3, 1]
        loop_range = 3

        for l in range(loop_range):
            w = loss_weights[l]
            pred_pos = self.pred_B[2 * l]
            pred_ori = self.pred_B[2 * l + 1]
            target_pos = gt_rpr[:, 0:3]
            target_ori = F.normalize(gt_rpr[:, 3:], p=2, dim=1)

            if self.loss_type == 'mse':
                error_pos = self.criterion(pred_pos, target_pos)
                error_ori = self.criterion(pred_ori, target_ori)
            else:
                error_pos = torch.norm(pred_pos - target_pos, p=2, dim=1).mean()
                error_ori = torch.norm(pred_ori - target_ori, p=2, dim=1).mean()

            # Combine losses
            if self.loss_type == 'geo':
                loss_pos = torch.exp(-self.sx) * error_pos + self.sx
                loss_ori = torch.exp(-self.sq) * error_ori + self.sq
                total_loss = loss_pos + loss_ori
            else:
                total_loss = error_pos + error_ori * self.opt.beta

            self.loss_G += total_loss * w
            self.loss_pos += error_pos.item() * w
            self.loss_ori += error_ori.item() * w * self.opt.beta

        self.loss_G.backward()

    # def backward(self):
    #     # feature_vec_ref, feature_vec_q = self.pred_B[-2:]
    #     # self.pred_B = self.pred_B[:-2]
    #
    #     gt_rpr = self.compute_rpr_from_apr(self.ref_pose, self.query_pose)
    #     self.loss_G = 0
    #     self.loss_pos = 0
    #     self.loss_ori = 0
    #     loss_weights = [0.3, 0.3, 1]
    #     for l, w in enumerate(loss_weights):
    #         mse_pos = self.criterion(self.pred_B[2 * l], gt_rpr[:, 0:3])
    #         ori_gt = F.normalize(gt_rpr[:, 3:], p=2, dim=1)
    #         mse_ori = self.criterion(self.pred_B[2 * l + 1], ori_gt)
    #         self.loss_G += (mse_pos + mse_ori * self.opt.beta) * w
    #         self.loss_pos += mse_pos.item() * w
    #         self.loss_ori += mse_ori.item() * w * self.opt.beta
    #
    #     #     pred_pos = self.pred_B[2 * l]  # 预测的相对平移 (Delta_t_pred)
    #     #     pred_ori = self.pred_B[2 * l + 1]  # 预测的相对四元数 (Delta_q_pred)
    #     #     gt_pos = gt_rpr[:, 0:3]  # GT 的相对平移 (Delta_t_gt)
    #     #     gt_ori = gt_rpr[:, 3:]  # GT 的相对四元数 (Delta_q_gt)
    #     #     B = gt_rpr.shape[0]
    #     #     # --- A. 旋转损失 (Angular Distance Loss) ---
    #     #     # 1. 四元数归一化
    #     #     gt_ori_norm = F.normalize(gt_ori, p=2, dim=1)
    #     #     pred_ori_norm = F.normalize(pred_ori, p=2, dim=1)
    #     #     # 2. 计算 Arccos Loss (无需额外的符号对齐，通过 torch.abs() 解决)
    #     #     # abs_distance 是 |q_pred . q_gt|
    #     #     # 这里的点积计算使用 torch.sum(..., dim=1) 实现 Batch 级的点积
    #     #     abs_distance = torch.abs(torch.sum(pred_ori_norm * gt_ori_norm, dim=1))
    #     #     # 数值稳定性：钳位到 [0, 1] 范围
    #     #     clamped_distance = torch.clamp(abs_distance, 0.0, 1.0)
    #     #     # L_Ang (弧度) (Batch)
    #     #     loss_ori_rad = 2.0 * torch.acos(clamped_distance)
    #     #     # --- B. 平移损失 (Huber Loss) ---
    #     #     # 计算 Huber 损失（reduction='none'，返回 Batch 上的每个样本损失）
    #     #     # L_Huber (Batch, 3) -> L_pos (Batch)
    #     #     loss_pos_comp = F.smooth_l1_loss(pred_pos, gt_pos, reduction='none')
    #     #     loss_pos_sample = torch.sum(loss_pos_comp, dim=1)
    #     #     # --- C. 尺度不变归一化 (核心) ---
    #     #     # 1. 计算 Batch 平均损失（作为尺度因子）
    #     #     mean_pos_loss = (loss_pos_sample.mean() + 1e-6).detach()
    #     #     mean_ori_loss = (loss_ori_rad.mean() + 1e-6).detach()
    #     #     # 2. 归一化（将损失转换为相对于 Batch 平均值的倍数）
    #     #     # L_norm,t (Batch)
    #     #     loss_pos_norm = loss_pos_sample / mean_pos_loss
    #     #     # L_norm,Ang (Batch)
    #     #     loss_ori_norm = loss_ori_rad / mean_ori_loss
    #     #     # 3. 组合最终的尺度不变回归损失
    #     #     # 这里必须使用张量，并且求平均得到标量损失 L_layer
    #     #     # 计算该层（l）的最终尺度不变损失 L_layer (标量)
    #     #     loss_layer = (loss_pos_norm.mean() + loss_ori_norm.mean()) * w
    #     #     # ✅ 修正 2a: 将具有梯度的张量累加到 self.loss_G 中
    #     #     self.loss_G += loss_layer
    #     #     # ✅ 修正 2b: 将用于日志的标量累加到 self.loss_pos/ori 中
    #     #     # 注意：这里的 self.loss_pos/ori 应该记录的是原始的、未归一化的损失，以便监控。
    #     #     # 否则，归一化后的平均值总是约等于 1.0，失去了监控意义。
    #     #     # 建议记录原始未归一化的损失的平均值 (Huber Loss & 弧度)
    #     #     self.loss_pos += loss_pos_sample.mean().item() * w
    #     #     self.loss_ori += loss_ori_rad.mean().item() * w
    #     #     # 注意：这里的 self.loss_G 的计算必须移到循环外，因为它还包含 self.loss_triplet
    #     #     # # 3. 组合最终的尺度不变回归损失
    #     #     # # 这里直接相加，因为它们已经是无量纲的相对值
    #     #     # self.loss_pos += loss_pos_norm.item() * w
    #     #     # self.loss_ori += loss_ori_norm.item() * w
    #     #     # self.loss_G += self.loss_pos + self.loss_ori
    #     # # 3. 计算三元组辅助损失 L_triplet
    #     # # 调用辅助函数计算 Triplet Loss (需要将 _get_triplet_loss 添加到类中)
    #     # self.loss_triplet = self.get_triplet_loss(feature_vec_ref, feature_vec_q)
    #     #
    #     # # 4. 将辅助损失加到总损失中
    #     # self.loss_G += self.loss_triplet
    #     self.loss_G.backward()

