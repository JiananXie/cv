import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
from .losses import ReprojectionLoss
import pickle
import numpy

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

        # Handle hidden sizes for LSTM and Transformer models
        lstm_hidden_size = opt.lstm_hidden_size 
        transformer_hidden_size = opt.transformer_hidden_size

        self.netG = networks.define_network(opt.input_nc, lstm_hidden_size, opt.model,
                                      init_from=googlenet_weights, isTest=not self.isTrain,
                                      gpu_ids = self.gpu_ids, transformer_hidden_size=transformer_hidden_size)

        # if not self.isTrain or opt.continue_train:
        #     self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.loss_type = opt.loss_type
            self.old_lr = opt.lr
            # define loss functions
            if self.loss_type == 'mse':
                self.criterion = torch.nn.MSELoss()
            if self.loss_type == 'geo':
                self.sx = nn.Parameter(torch.tensor(0.0))
                self.sq = nn.Parameter(torch.tensor(-3.0))
            
            # Initialize Reprojection Loss
            # Note: Focal length needs to be accurate. 
            # Assuming KingsCollege default ~1670 for 1920x1080, scaled to fineSize (224)
            # Scale factor = 256 / 1920 (based on loadSize) -> then crop to 224
            # Actually, PoseDataset resizes to (loadSize, loadSize) ignoring aspect ratio?
            # Let's assume a reasonable focal length approximation or pass it via opt
            # For now, using a hardcoded approximation based on typical Cambridge Landmarks setup
            # If original is 1920 width, f=1670. New width is 256.
            # f_new = 1670 * (256 / 1920) ~= 222.6
            focal_length = 1563 * (opt.loadSize / 1920.0) 
            self.reprojection_loss = ReprojectionLoss(focal_length, (opt.fineSize, opt.fineSize), 
                                                      device=self.gpu_ids[0] if self.gpu_ids else 'cpu')

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
        self.loss_reproj = 0
        
        # Check if we have separate heads (only 2 outputs: [xyz, wpqr])
        # or standard/aux outputs (6 outputs: [xyz1, wpqr1, xyz2, wpqr2, xyz3, wpqr3])
        if len(self.pred_B) == 2:
            loss_weights = [1.0]
            loop_range = 1
        else:
            loss_weights = [0.3, 0.3, 1]
            loop_range = 3
            print("[INFO] Using auxiliary losses with weights:", loss_weights)

        for l in range(loop_range):
            w = loss_weights[l]
            if len(self.pred_B) == 2:
                pred_pos = self.pred_B[0]
                pred_ori = self.pred_B[1]
            else:
                pred_pos = self.pred_B[2*l]
                pred_ori = self.pred_B[2*l+1]
            
            target_pos = self.input_B[:, 0:3]
            target_ori = F.normalize(self.input_B[:, 3:], p=2, dim=1)
            
            if self.loss_type == 'mse':
                error_pos = self.criterion(pred_pos, target_pos)
                error_ori = self.criterion(pred_ori, target_ori)
            else:
                error_pos = torch.norm(pred_pos - target_pos, p=2, dim=1).mean()
                error_ori = torch.norm(pred_ori - target_ori, p=2, dim=1).mean()
            
            # Reprojection Loss (Geometric Consistency)
            reproj_loss = self.reprojection_loss(pred_pos, pred_ori, target_pos, target_ori)
            
            # Combine losses
            if self.loss_type == 'geo':
                loss_pos = torch.exp(-self.sx) * error_pos + self.sx
                loss_ori = torch.exp(-self.sq) * error_ori + self.sq
                total_loss = loss_pos + loss_ori
            else:
                total_loss = error_pos + error_ori * self.opt.beta
            gamma = 0.1 
            # total_loss += reproj_loss * gamma
            
            self.loss_G += total_loss * w
            self.loss_pos += error_pos.item() * w
            self.loss_ori += error_ori.item() * w * self.opt.beta
            self.loss_reproj += reproj_loss.item() * w

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
                                ('reproj_err', self.loss_reproj),
                                ])

        # Batch processing for test time
        # pred_B[0] is position (B, 3), input_B[:, 0:3] is target position (B, 3)
        pos_err = torch.dist(self.pred_B[0], self.input_B[:, 0:3], p=2) # This computes scalar distance for whole batch if not careful
        # Correct way for batch:
        diff_pos = self.pred_B[0] - self.input_B[:, 0:3]
        pos_errs = torch.norm(diff_pos, p=2, dim=1) # (B,)

        ori_gt = F.normalize(self.input_B[:, 3:], p=2, dim=1)
        pred_ori = self.pred_B[1]
        
        # Dot product for orientation
        # (B, 4) * (B, 4) -> sum(dim=1) -> (B,)
        abs_distance = torch.abs((ori_gt * pred_ori).sum(dim=1))
        # Clamp to avoid numerical issues with acos
        abs_distance = torch.clamp(abs_distance, -1.0, 1.0)
        ori_errs = 2 * 180 / numpy.pi * torch.acos(abs_distance) # (B,)
        
        return [pos_errs.detach().cpu().numpy(), ori_errs.detach().cpu().numpy()]

    def get_current_pose(self):
        # Return batch of poses: (B, 7)
        return numpy.concatenate((self.pred_B[0].data.cpu().numpy(),
                                  self.pred_B[1].data.cpu().numpy()), axis=1)

    def get_current_visuals(self):
        input_A = util.tensor2im(self.input_A.data)
        # pred_B = util.tensor2im(self.pred_B.data)
        # input_B = util.tensor2im(self.input_B.data)
        return OrderedDict([('input_A', input_A)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
