import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torchvision

###############################################################################
# Functions
###############################################################################
def weight_init_googlenet(key, module, weights=None):

    if key == "LSTM":
        for name, param in module.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.xavier_normal_(param)
    elif weights is None:
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


def define_network(input_nc, lstm_hidden_size, model, init_from=None, isTest=False, gpu_ids=[], transformer_hidden_size=None):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    if model == 'posenet':
        netG = PoseNet(input_nc, weights=init_from, isTest=isTest, gpu_ids=gpu_ids)
    elif model == 'poselstm':
        netG = PoseNet(input_nc, weights=init_from, isTest=isTest, gpu_ids=gpu_ids, lstm_hidden_size=lstm_hidden_size)
    elif model == 'posetransformer':
        netG = PoseNet(input_nc, weights=init_from, isTest=isTest, gpu_ids=gpu_ids, transformer_hidden_size=transformer_hidden_size)
    elif model == 'posefpn':
        netG = PoseNet(input_nc, weights=init_from, isTest=isTest, gpu_ids=gpu_ids, use_fpn=True)
    elif model == 'poseresnet50':
        netG = PoseNet(input_nc, weights=init_from, isTest=isTest, gpu_ids=gpu_ids, backbone='resnet50')
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
    def __init__(self, lossID, input_dim=None, head_type='fc', hidden_size=None, weights=None):
        super(RegressionHead, self).__init__()
        self.head_type = head_type
        
        # Defaults for Inception if input_dim is None
        if input_dim is None:
            if lossID == "loss1": input_dim = 512
            elif lossID == "loss2": input_dim = 528
            elif lossID == "loss3": input_dim = 1024

        dropout_rate = 0.5 if lossID == "loss3" else 0.7
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Projection
        if lossID != "loss3":
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
            self.lstm_hidden_size = hidden_size if hidden_size is not None else 256
            self.lstm_pose_lr = weight_init_googlenet("LSTM", nn.LSTM(input_size=self.token_dim, hidden_size=self.lstm_hidden_size, bidirectional=True, batch_first=True))
            self.lstm_pose_ud = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=self.lstm_hidden_size, bidirectional=True, batch_first=True))
            self.out_dim = self.lstm_hidden_size * 4
            
        elif head_type == 'transformer':
            self.d_model = hidden_size if hidden_size is not None else 128
            self.seq_len = 32
            self.embedding = nn.Linear(self.token_dim, self.d_model)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.pos_encoder = nn.Parameter(torch.zeros(1, self.seq_len + 1, self.d_model))
            nn.init.trunc_normal_(self.pos_encoder, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=self.d_model*4, dropout=0.5, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.out_dim = self.d_model
            
        else: # 'fc'
            self.out_dim = self.feature_dim

        # Final Regressors
        self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(self.out_dim, 3))
        self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(self.out_dim, 4))

    def forward(self, input):
        output = self.projection(input)
        output = self.cls_fc_pose(output.view(output.size(0), -1))
        
        if self.head_type == 'lstm':
            output = output.view(output.size(0), 32, -1)
            _, (hidden_state_lr, _) = self.lstm_pose_lr(output.permute(0,1,2))
            _, (hidden_state_ud, _) = self.lstm_pose_ud(output.permute(0,2,1))
            output = torch.cat((hidden_state_lr[0,:,:],
                                hidden_state_lr[1,:,:],
                                hidden_state_ud[0,:,:],
                                hidden_state_ud[1,:,:]), 1)
                                
        elif self.head_type == 'transformer':
            B = output.size(0)
            output = output.view(B, 32, -1)
            x = self.embedding(output)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_encoder
            x = self.transformer_encoder(x)
            output = x[:, 0, :]
            
        output = self.dropout(output)
        output_xy = self.cls_fc_xy(output)
        output_wpqr = self.cls_fc_wpqr(output)
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

    def forward(self, input):
        outputs = [self.branch_x1(input), self.branch_x3(input),
                   self.branch_x5(input), self.branch_proj(input)]
        # print([[o.size()] for o in outputs])
        output = torch.cat(outputs, 1)
        if self.pool is not None:
            return self.pool(output)
        return output

class PoseNet(nn.Module):
    def __init__(self, input_nc, weights=None, isTest=False,  gpu_ids=[], lstm_hidden_size=None, transformer_hidden_size=None, use_fpn=False, backbone='inception'):
        super(PoseNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        self.use_fpn = use_fpn
        self.backbone = backbone
        
        # Determine Head Type and Config
        head_type = 'fc'
        hidden_size = None
        if transformer_hidden_size is not None:
            head_type = 'transformer'
            hidden_size = transformer_hidden_size
        elif lstm_hidden_size is not None:
            head_type = 'lstm'
            hidden_size = lstm_hidden_size

        if self.backbone == 'inception':
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

            self.cls1_fc = RegressionHead("loss1", input_dim=512, head_type=head_type, hidden_size=hidden_size, weights=weights)
            self.cls2_fc = RegressionHead("loss2", input_dim=528, head_type=head_type, hidden_size=hidden_size, weights=weights)
            
            if self.use_fpn:
                # FPN Lateral Layers
                self.lat_layer1 = nn.Conv2d(512, 256, kernel_size=1)
                self.lat_layer2 = nn.Conv2d(528, 256, kernel_size=1)
                self.lat_layer3 = nn.Conv2d(1024, 256, kernel_size=1)
                
                self.smooth_layer1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
                self.smooth_layer2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
                self.smooth_layer3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
                
                self.fpn_head = nn.Sequential(
                    nn.Linear(768, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5)
                )
                self.reg_xy = nn.Linear(1024, 3)
                self.reg_wpqr = nn.Linear(1024, 4)
                
                # Init FPN weights
                for m in [self.lat_layer1, self.lat_layer2, self.lat_layer3, 
                          self.smooth_layer1, self.smooth_layer2, self.smooth_layer3,
                          self.reg_xy, self.reg_wpqr]:
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)
            else:
                self.cls3_fc = RegressionHead("loss3", input_dim=1024, head_type=head_type, hidden_size=hidden_size, weights=weights)

            self.model = nn.Sequential(*[self.inception_3a, self.inception_3b,
                                       self.inception_4a, self.inception_4b,
                                       self.inception_4c, self.inception_4d,
                                       self.inception_4e, self.inception_5a,
                                       self.inception_5b, self.cls1_fc,
                                       self.cls2_fc
                                       ])
            if not self.use_fpn:
                self.model.add_module("cls3_fc", self.cls3_fc)

            if self.isTest:
                self.model.eval() # ensure Dropout is deactivated during test

        elif self.backbone == 'resnet50':
            # Load pretrained ResNet50
            resnet = torchvision.models.resnet50(pretrained=True)
            if input_nc != 3:
                resnet.conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
                
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            
            self.cls1_fc = RegressionHead("loss1", input_dim=512, head_type=head_type, hidden_size=hidden_size, weights=weights)
            self.cls2_fc = RegressionHead("loss2", input_dim=1024, head_type=head_type, hidden_size=hidden_size, weights=weights)
            self.cls3_fc = RegressionHead("loss3", input_dim=2048, head_type=head_type, hidden_size=hidden_size, weights=weights)
            
            self.model = nn.Sequential(
                self.conv1, self.bn1, self.relu, self.maxpool,
                self.layer1, self.layer2, self.layer3, self.layer4,
                self.cls1_fc, self.cls2_fc, self.cls3_fc
            )
            
            if self.isTest:
                self.model.eval()

    def forward(self, input):
        if self.backbone == 'inception':
            output_bf = self.before_inception(input)
            output_3a = self.inception_3a(output_bf)
            output_3b = self.inception_3b(output_3a)
            output_4a = self.inception_4a(output_3b)
            output_4b = self.inception_4b(output_4a)
            output_4c = self.inception_4c(output_4b)
            output_4d = self.inception_4d(output_4c)
            output_4e = self.inception_4e(output_4d)
            output_5a = self.inception_5a(output_4e)
            output_5b = self.inception_5b(output_5a)
            
            if self.use_fpn:
                # FPN Forward
                p5 = self.lat_layer3(output_5b)
                c4_lat = self.lat_layer2(output_4d)
                p5_up = F.interpolate(p5, size=c4_lat.shape[-2:], mode='nearest')
                p4 = c4_lat + p5_up
                
                c3_lat = self.lat_layer1(output_4a)
                p4_up = p4 
                p3 = c3_lat + p4_up
                
                p5 = self.smooth_layer3(p5)
                p4 = self.smooth_layer2(p4)
                p3 = self.smooth_layer1(p3)
                
                f5 = F.adaptive_avg_pool2d(p5, (1, 1)).view(p5.size(0), -1)
                f4 = F.adaptive_avg_pool2d(p4, (1, 1)).view(p4.size(0), -1)
                f3 = F.adaptive_avg_pool2d(p3, (1, 1)).view(p3.size(0), -1)
                
                features = torch.cat([f3, f4, f5], dim=1)
                x = self.fpn_head(features)
                pred_xy = self.reg_xy(x)
                pred_wpqr = self.reg_wpqr(x)
                pred_wpqr = F.normalize(pred_wpqr, p=2, dim=1)
                
                main_out = [pred_xy, pred_wpqr]
            else:
                main_out = self.cls3_fc(output_5b)

            if not self.isTest:
                aux1 = self.cls1_fc(output_4a)
                aux2 = self.cls2_fc(output_4d)
                return aux1 + aux2 + main_out
            return main_out
            
        elif self.backbone == 'resnet50':
            x = self.conv1(input)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            out1 = x # 512 channels
            
            x = self.layer3(x)
            out2 = x # 1024 channels
            
            x = self.layer4(x)
            out3 = x # 2048 channels
            
            if not self.isTest:
                return self.cls1_fc(out1) + self.cls2_fc(out2) + self.cls3_fc(out3)
            return self.cls3_fc(out3)


