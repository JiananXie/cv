import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import math

###############################################################################
# Functions
###############################################################################
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is All You Need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x):
        # x: [B, C, H, W]
        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def weight_init_googlenet(key, module, weights=None):

    if key == "LSTM":
        for name, param in module.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.xavier_normal_(param)
    elif weights is None or (key+"_1").encode() not in weights:
        init.constant_(module.bias.data, 0.0)
        if key == "XYZ":
            init.normal_(module.weight.data, 0.0, 0.5)
        elif key == "LSTM":
            init.xavier_normal_(module.weight.data)
        else:
            init.normal_(module.weight.data, 0.0, 0.01)
    else:
        # print(key, weights[(key+"_1").encode()].shape, module.bias.size())
        module.bias.data[...] = torch.from_numpy(weights[(key+"_1").encode()])
        module.weight.data[...] = torch.from_numpy(weights[(key+"_0").encode()])
    return module

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_network(input_nc, lstm_hidden_size, model, init_from=None, isTest=False, gpu_ids=[], transformer_hidden_size=None, backbone='inception'):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    if model == 'posenet':
        netG = PoseNet(input_nc, weights=init_from, isTest=isTest, gpu_ids=gpu_ids, backbone=backbone)
    elif model == 'poselstm':
        if lstm_hidden_size is None:
            lstm_hidden_size = 256
        netG = PoseNet(input_nc, weights=init_from, isTest=isTest, gpu_ids=gpu_ids, lstm_hidden_size=lstm_hidden_size, backbone=backbone)
    elif model == 'posetransformer':
        if transformer_hidden_size is None:
            transformer_hidden_size = 256
        netG = PoseNet(input_nc, weights=init_from, isTest=isTest, gpu_ids=gpu_ids, transformer_hidden_size=transformer_hidden_size, backbone=backbone)
    elif model == 'poseseparate':
        netG = PoseNet(input_nc, weights=init_from, isTest=isTest, gpu_ids=gpu_ids, use_separate_heads=True, lstm_hidden_size=lstm_hidden_size, transformer_hidden_size=transformer_hidden_size, backbone=backbone)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % model)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG

##############################################################################
# Classes
##############################################################################

# defines the regression heads for googlenet
class RegressionHead(nn.Module):
    def __init__(self, lossID, input_dim=None, head_type='fc', hidden_size=None, weights=None, output_type='all'):
        super(RegressionHead, self).__init__()
        self.head_type = head_type
        self.output_type = output_type # 'all', 'xyz', or 'wpqr'
        
        # Defaults for Inception if input_dim is None
        if input_dim is None:
            if lossID == "loss1": input_dim = 512
            elif lossID == "loss2": input_dim = 528
            elif lossID == "loss3": input_dim = 1024

        dropout_rate = 0.5 if lossID == "loss3" else 0.7
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Projection
        if head_type == 'transformer':
            self.d_model = hidden_size if hidden_size is not None else 128
            # Use 1x1 conv to project to d_model, preserving spatial dimensions
            self.projection = weight_init_googlenet(lossID+"/proj_trans", nn.Conv2d(input_dim, self.d_model, kernel_size=1), weights)
            self.cls_fc_pose = None # Not used for transformer
            self.feature_dim = self.d_model
            
            # Fixed 2D Sine/Cosine Position Embedding
            self.pos_embedding = PositionEmbeddingSine(self.d_model // 2, normalize=True)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            # Learnable position for CLS token only
            self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.trunc_normal_(self.cls_pos, std=0.02)
            
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=self.d_model, dropout=0.1, activation='gelu', batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.out_dim = self.d_model
                
        elif lossID != "loss3":
            self.projection = nn.Sequential(*[
                nn.AdaptiveAvgPool2d((4, 4)),
                weight_init_googlenet(lossID+"/conv", nn.Conv2d(input_dim, 128, kernel_size=1), weights),
                nn.ReLU(inplace=True)
            ])
            self.cls_fc_pose = nn.Sequential(*[
                weight_init_googlenet(lossID+"/fc", nn.Linear(2048, 1024), weights),
                nn.ReLU(inplace=True)
            ])
            self.feature_dim = 1024
            self.token_dim = 32 # 1024 / 32
        else:
            self.projection = nn.AdaptiveAvgPool2d((1, 1))
            self.cls_fc_pose = nn.Sequential(*[
                weight_init_googlenet("pose", nn.Linear(input_dim, 2048)),
                nn.ReLU(inplace=True)
            ])
            self.feature_dim = 2048
            self.token_dim = 64 # 2048 / 32

        # Head Specifics
        if head_type == 'lstm':
            print("[INFO] Using LSTM Head")
            self.lstm_hidden_size = hidden_size if hidden_size is not None else 256
            self.lstm_pose_lr = weight_init_googlenet("LSTM", nn.LSTM(input_size=self.token_dim, hidden_size=self.lstm_hidden_size, bidirectional=True, batch_first=True))
            self.lstm_pose_ud = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=self.lstm_hidden_size, bidirectional=True, batch_first=True))
            self.out_dim = self.lstm_hidden_size * 4
            
        elif head_type == 'transformer':
            print("[INFO] Using Transformer Head")
            pass
        else: # 'fc'
            print("[INFO] Using FC Head")
            self.out_dim = self.feature_dim

        # Final Regressors
        if self.output_type == 'all' or self.output_type == 'xyz':
            self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(self.out_dim, 3))
        if self.output_type == 'all' or self.output_type == 'wpqr':
            self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(self.out_dim, 4))

    def forward(self, input):
        output = self.projection(input)
        
        if self.head_type == 'transformer':
            # Generate 2D Positional Embeddings
            pos = self.pos_embedding(output) # (B, d_model, H, W)
            
            B, C, H, W = output.size()
            # Flatten spatial dimensions: (B, d_model, H*W) -> (B, H*W, d_model)
            output = output.flatten(2).permute(0, 2, 1) # [B, H*W, d_model]
            pos = pos.flatten(2).permute(0, 2, 1) # [B, H*W, d_model]
            
            cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, d_model]
            cls_pos = self.cls_pos.expand(B, -1, -1) # [B, 1, d_model]
            x = torch.cat((cls_tokens, output), dim=1) # (B, L+1, d_model)
            pos = torch.cat((cls_pos, pos), dim=1)     # (B, L+1, d_model)
            
            x = x + pos
            
            x = self.transformer_encoder(x) # [B, L+1, d_model]
            output = x[:, 0, :] # Take CLS token [B, d_model]
            output = self.dropout(output)
            
            if self.output_type == 'xyz':
                return self.cls_fc_xy(output)
            elif self.output_type == 'wpqr':
                output_wpqr = self.cls_fc_wpqr(output)
                return F.normalize(output_wpqr, p=2, dim=1)
            else:
                output_xy = self.cls_fc_xy(output) # [B, 3]
                output_wpqr = self.cls_fc_wpqr(output) # [B, 4]
                output_wpqr = F.normalize(output_wpqr, p=2, dim=1)
                return [output_xy, output_wpqr]

        # For FC and LSTM, we need flattened input
        output = self.cls_fc_pose(output.view(output.size(0), -1)) # [B, 1024] or [B, 2048]
        
        if self.head_type == 'lstm':
            output = output.view(output.size(0), 32, -1) # [B, 32, 32]
            _, (hidden_state_lr, _) = self.lstm_pose_lr(output.permute(0,1,2)) # [B, 32, 32] -> [B, 32, 256*2]
            _, (hidden_state_ud, _) = self.lstm_pose_ud(output.permute(0,2,1))
            output = torch.cat((hidden_state_lr[0,:,:],
                                hidden_state_lr[1,:,:],
                                hidden_state_ud[0,:,:],
                                hidden_state_ud[1,:,:]), 1) # [B, 256*4]
            
        output = self.dropout(output)
        
        if self.output_type == 'xyz':
            return self.cls_fc_xy(output)
        elif self.output_type == 'wpqr':
            output_wpqr = self.cls_fc_wpqr(output)
            return F.normalize(output_wpqr, p=2, dim=1)
        else:
            output_xy = self.cls_fc_xy(output) # [B, 3]
            output_wpqr = self.cls_fc_wpqr(output) # [B, 4]
            output_wpqr = F.normalize(output_wpqr, p=2, dim=1)
            return [output_xy, output_wpqr]

# define inception block for GoogleNet
class InceptionBlock(nn.Module):
    def __init__(self, incp, input_nc, x1_nc, x3_reduce_nc, x3_nc, x5_reduce_nc,
                 x5_nc, proj_nc, weights=None, gpu_ids=[]):
        super(InceptionBlock, self).__init__()
        self.gpu_ids = gpu_ids
        # first
        self.branch_x1 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/1x1", nn.Conv2d(input_nc, x1_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True)])

        self.branch_x3 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/3x3_reduce", nn.Conv2d(input_nc, x3_reduce_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            weight_init_googlenet("inception_"+incp+"/3x3", nn.Conv2d(x3_reduce_nc, x3_nc, kernel_size=3, padding=1), weights),
            nn.ReLU(inplace=True)])

        self.branch_x5 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/5x5_reduce", nn.Conv2d(input_nc, x5_reduce_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            weight_init_googlenet("inception_"+incp+"/5x5", nn.Conv2d(x5_reduce_nc, x5_nc, kernel_size=5, padding=2), weights),
            nn.ReLU(inplace=True)])

        self.branch_proj = nn.Sequential(*[
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            weight_init_googlenet("inception_"+incp+"/pool_proj", nn.Conv2d(input_nc, proj_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True)])

        if incp in ["3b", "4e"]:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pool = None

    def forward(self, input, return_unpooled=False):
        outputs = [self.branch_x1(input), self.branch_x3(input),
                   self.branch_x5(input), self.branch_proj(input)]
        # print([[o.size()] for o in outputs])
        output = torch.cat(outputs, 1)
        if self.pool is not None:
            if return_unpooled:
                return self.pool(output), output
            return self.pool(output)
        return output

class PoseNet(nn.Module):
    def __init__(self, input_nc, weights=None, isTest=False,  gpu_ids=[], lstm_hidden_size=None, transformer_hidden_size=None, backbone='inception', use_separate_heads=False):
        super(PoseNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        self.backbone = backbone
        self.use_separate_heads = use_separate_heads
        
        # Determine Head Type and Config
        head_type = 'fc'
        hidden_size = None
        if transformer_hidden_size is not None:
            head_type = 'transformer'
            hidden_size = transformer_hidden_size
        elif lstm_hidden_size is not None:
            head_type = 'lstm'
            hidden_size = lstm_hidden_size
        
        self.head_type = head_type

        if self.backbone == 'inception':
            print("[INFO] Using Inception Backbone")
            self.before_inception = nn.Sequential(*[
                weight_init_googlenet("conv1/7x7_s2", nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3), weights),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
                weight_init_googlenet("conv2/3x3_reduce", nn.Conv2d(64, 64, kernel_size=1), weights),
                nn.ReLU(inplace=True),
                weight_init_googlenet("conv2/3x3", nn.Conv2d(64, 192, kernel_size=3, padding=1), weights),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                ])

            self.inception_3a = InceptionBlock("3a", 192, 64, 96, 128, 16, 32, 32, weights, gpu_ids)
            self.inception_3b = InceptionBlock("3b", 256, 128, 128, 192, 32, 96, 64, weights, gpu_ids)
            self.inception_4a = InceptionBlock("4a", 480, 192, 96, 208, 16, 48, 64, weights, gpu_ids)
            self.inception_4b = InceptionBlock("4b", 512, 160, 112, 224, 24, 64, 64, weights, gpu_ids)
            self.inception_4c = InceptionBlock("4c", 512, 128, 128, 256, 24, 64, 64, weights, gpu_ids)
            self.inception_4d = InceptionBlock("4d", 512, 112, 144, 288, 32, 64, 64, weights, gpu_ids)
            self.inception_4e = InceptionBlock("4e", 528, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
            self.inception_5a = InceptionBlock("5a", 832, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
            self.inception_5b = InceptionBlock("5b", 832, 384, 192, 384, 48, 128, 128, weights, gpu_ids)

            # Initialize Auxiliary Heads (Always used for training)
            if head_type == 'transformer':
                self.cls1_fc_pos = RegressionHead("loss1", input_dim=512, head_type='transformer', hidden_size=hidden_size, weights=weights, output_type='xyz')
                self.cls1_fc_ori = RegressionHead("loss1", input_dim=512, head_type='fc', hidden_size=hidden_size, weights=weights, output_type='wpqr')
                
                self.cls2_fc_pos = RegressionHead("loss2", input_dim=528, head_type='transformer', hidden_size=hidden_size, weights=weights, output_type='xyz')
                self.cls2_fc_ori = RegressionHead("loss2", input_dim=528, head_type='fc', hidden_size=hidden_size, weights=weights, output_type='wpqr')
            else:
                self.cls1_fc = RegressionHead("loss1", input_dim=512, head_type=head_type, hidden_size=hidden_size, weights=weights)
                self.cls2_fc = RegressionHead("loss2", input_dim=528, head_type=head_type, hidden_size=hidden_size, weights=weights)

            if self.use_separate_heads:
                # Head for WPQR (Orientation) using 3b unpooled (480 channels)
                self.head_wpqr = RegressionHead("loss_wpqr", input_dim=480, head_type='fc', hidden_size=hidden_size, weights=weights, output_type='wpqr')
                # Head for XYZ (Position) using 4e unpooled (832 channels)
                self.head_xyz = RegressionHead("loss_xyz", input_dim=1024, head_type=head_type, hidden_size=hidden_size, weights=weights, output_type='xyz')
            else:
                if head_type == 'transformer':
                    self.cls3_fc_pos = RegressionHead("loss3", input_dim=1024, head_type='transformer', hidden_size=hidden_size, weights=weights, output_type='xyz')
                    self.cls3_fc_ori = RegressionHead("loss3", input_dim=1024, head_type='fc', hidden_size=hidden_size, weights=weights, output_type='wpqr')
                else:
                    self.cls3_fc = RegressionHead("loss3", input_dim=1024, head_type=head_type, hidden_size=hidden_size, weights=weights)

            layers = [self.inception_3a, self.inception_3b,
                      self.inception_4a, self.inception_4b,
                      self.inception_4c, self.inception_4d,
                      self.inception_4e, self.inception_5a,
                      self.inception_5b]

            if self.use_separate_heads:
                layers.extend([self.head_xyz, self.head_wpqr])
            else:
                layers.extend([self.cls1_fc, self.cls2_fc])

            self.model = nn.Sequential(*layers)

            if self.isTest:
                self.model.eval() # ensure Dropout is deactivated during test

        elif self.backbone == 'resnet50':
            print("[INFO] Using ResNet50 Backbone")
            resnet = torchvision.models.resnet50(pretrained=True)
            if input_nc != 3:
                resnet.conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
                
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            
            self.layer1 = resnet.layer1 # -> 256 ch, 56x56
            self.layer2 = resnet.layer2 # -> 512 ch, 28x28 (类似 Inception 3b/4a)
            self.layer3 = resnet.layer3 # -> 1024 ch, 14x14
            self.layer4 = resnet.layer4 # -> 2048 ch, 7x7  (类似 Inception 5b)
            
            # --- Separate Heads (ResNet Config) ---
            if self.use_separate_heads:
                # Ori Head: 接在 Layer 2 (512 ch)
                self.head_wpqr = RegressionHead("loss_wpqr", input_dim=512, head_type='fc', 
                                                hidden_size=hidden_size, weights=weights, output_type='wpqr')
                
                # Pos Head: 接在 Layer 4 (2048 ch)
                # 必须显式传入 input_dim=2048，否则 RegressionHead 默认会用 1024
                self.head_xyz = RegressionHead("loss_xyz", input_dim=2048, head_type=head_type, 
                                               hidden_size=hidden_size, weights=weights, output_type='xyz')
            else:
                # 标准 Aux Heads
                self.cls1_fc = RegressionHead("loss1", input_dim=512, head_type=head_type, hidden_size=hidden_size, weights=weights)
                self.cls2_fc = RegressionHead("loss2", input_dim=1024, head_type=head_type, hidden_size=hidden_size, weights=weights)
                self.cls3_fc = RegressionHead("loss3", input_dim=2048, head_type=head_type, hidden_size=hidden_size, weights=weights)
            
            # 定义 layers 用于 .cuda() 方便
            layers = [self.conv1, self.layer1, self.layer2, self.layer3, self.layer4]
            if self.use_separate_heads:
                layers.extend([self.head_wpqr, self.head_xyz])
            else:
                layers.extend([self.cls1_fc, self.cls2_fc, self.cls3_fc])
            
            self.model = nn.Sequential(*layers)
            
            if self.isTest:
                self.model.eval()


    def forward(self, input):
        if self.backbone == 'inception':
            output_bf = self.before_inception(input) # [B, 192, 28, 28]
            output_3a = self.inception_3a(output_bf) # [B, 256, 28, 28]
            
            if self.use_separate_heads:
                output_3b, feat_3b = self.inception_3b(output_3a, return_unpooled=True) # feat_3b: [B, 480, 28, 28]
            else:
                output_3b = self.inception_3b(output_3a) # [B, 480, 14, 14]
            
            output_4a = self.inception_4a(output_3b) # [B, 512, 14, 14]
            output_4b = self.inception_4b(output_4a) # [B, 512, 14, 14]
            output_4c = self.inception_4c(output_4b) # [B, 512, 14, 14]
            output_4d = self.inception_4d(output_4c) # [B, 528, 14, 14]
            
            if self.use_separate_heads:
                output_4e, feat_4e = self.inception_4e(output_4d, return_unpooled=True) # feat_4e: [B, 832, 14, 14]
            else:
                output_4e = self.inception_4e(output_4d) # [B, 832, 7, 7]
            
            output_5a = self.inception_5a(output_4e) # [B, 832, 7, 7]
            output_5b = self.inception_5b(output_5a) # [B, 1024, 7, 7]
            
            if self.use_separate_heads:
                pred_wpqr = self.head_wpqr(feat_3b)
                pred_xyz = self.head_xyz(output_5b)
                main_out = [pred_xyz, pred_wpqr]
                # return main_out
            else:
                if self.head_type == 'transformer':
                    main_out = [self.cls3_fc_pos(output_5b), self.cls3_fc_ori(output_5b)]
                else:
                    main_out = self.cls3_fc(output_5b)

            if not self.isTest:
                if self.head_type == 'transformer':
                    aux1 = [self.cls1_fc_pos(output_4a), self.cls1_fc_ori(output_4a)]
                    aux2 = [self.cls2_fc_pos(output_4d), self.cls2_fc_ori(output_4d)]
                else:
                    aux1 = self.cls1_fc(output_4a)
                    aux2 = self.cls2_fc(output_4d)
                return aux1 + aux2 + main_out
            return main_out
            
        elif self.backbone == 'resnet50':
            # ================= ResNet50 Forward =================
            x = self.conv1(input)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x) 
            
            x = self.layer1(x) 
            
            # Layer 2 -> 512 channels (对应 Inception 3b/4a, 适合做 Ori)
            feat_layer2 = self.layer2(x) 
            
            x = self.layer3(feat_layer2)
            
            # Layer 4 -> 2048 channels (对应 Inception 5b, 适合做 Pos)
            feat_layer4 = self.layer4(x) 
            
            if self.use_separate_heads:
                # 1. 预测 Ori
                pred_wpqr = self.head_wpqr(feat_layer2)
                
                # 2. 预测 Pos
                # 将预测出的 Ori 传入 Pos Head 的 Transformer 进行 Concat 融合
                pred_xyz = self.head_xyz(feat_layer4)
                
                return [pred_xyz, pred_wpqr]
            
            else:
                # 如果不使用 Separate Heads，走标准流程
                if not self.isTest:
                     return self.cls1_fc(feat_layer2) + self.cls2_fc(x) + self.cls3_fc(feat_layer4)
                return self.cls3_fc(feat_layer4)


