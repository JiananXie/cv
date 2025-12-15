import time
import os
import argparse
import torch
import numpy
from data.data_loader import CreateDataLoader
from models.models import create_model
from util import util
from util import geometry_utils

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Base options
    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--batchSize', type=int, default=75, help='input batch size')
    parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
    parser.add_argument('--fineSize', type=int, default=224, help='then crop to this size')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=7, help='# of output image channels')
    parser.add_argument('--lstm_hidden_size', type=int, default=256, help='hidden size of the LSTM layer in PoseLSTM')
    parser.add_argument('--transformer_hidden_size', type=int, default=256, help='hidden size of the Transformer layer in PoseTransformer')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--model', type=str, default='posenet', help='chooses which model to use. [posenet | poselstm | resnet50]')
    parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--display_winsize', type=int, default=224,  help='display window size')
    parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--seed', type=int, default=0, help='initial random seed for deterministic results')
    parser.add_argument('--beta', type=float, default=500, help='beta factor used in posenet.')

    # Test options
    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--tta', action='store_true', help='enable test time augmentation (5 crops)')

    opt = parser.parse_args()
    opt.isTrain = False
    
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
            
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
        
    args = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt_'+opt.phase+'.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
    return opt

opt = parse_args()
opt.serial_batches = True

data_loader = CreateDataLoader(opt)
dataset = data_loader

results_dir = os.path.join(opt.results_dir, opt.name)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

besterror  = [0, float('inf'), float('inf'), float('inf')] # nepoch, medX, medQ, medReproj

import glob
expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
files = glob.glob(os.path.join(expr_dir, '*_net_G.pth'))
epochs = []
for f in files:
    fname = os.path.basename(f)
    try:
        epoch_str = fname.split('_')[0]
        if epoch_str != 'latest':
            epochs.append(int(epoch_str))
    except ValueError:
        pass
epochs.sort()
if opt.model == 'posenet':
    testepochs = epochs
else:
    testepochs = epochs 
print("Testing epochs:", testepochs)

# Setup for Reprojection Error
nvm_path = 'datasets/KingsCollege/reconstruction.nvm'
focal_length = geometry_utils.get_focal_length_from_nvm(nvm_path)
if focal_length is None:
    print("Warning: Could not read focal length from reconstruction.nvm. Using default 1670.")
    focal_length = 1670.0

K = geometry_utils.get_intrinsics(focal_length, (256, 256), (opt.fineSize, opt.fineSize))
print(f"Using Intrinsic Matrix:\n{K}")

testfile = open(os.path.join(results_dir, 'test_median.txt'), 'a')
testfile.write('epoch median_Pos  median_Ori  median_Reproj\n')
testfile.write('==================\n')

model = create_model(opt)

for testepoch in testepochs:
    model.load_network(model.netG, 'G', testepoch)
    
    # test
    err = []
    print("epoch: "+ str(testepoch))
    for i, data in enumerate(dataset):
        
        if opt.tta:
            print("[INFO] Performing Test Time Augmentation (5 crops)...")
            B, N, C, H, W = data['A'].size()
            input_A = data['A'].view(B*N, C, H, W)
            input_B = data['B'].repeat_interleave(N, dim=0)
            
            data_tta = {'A': input_A, 'B': input_B, 'A_paths': []}
            for p in data['A_paths']:
                for _ in range(N):
                    data_tta['A_paths'].append(p)
            
            model.set_input(data_tta)
            model.test()
            
            poses_all = model.get_current_pose() # (B*N, 7) numpy
            poses_all = torch.from_numpy(poses_all)
            poses_all = poses_all.view(B, N, 7)
            
            pos_avg = torch.mean(poses_all[:, :, 0:3], dim=1)
            ori_avg = torch.mean(poses_all[:, :, 3:7], dim=1)
            ori_avg = torch.nn.functional.normalize(ori_avg, p=2, dim=1)
            
            poses = torch.cat([pos_avg, ori_avg], dim=1).numpy()
            
            gt_poses = data['B'].numpy()
            img_paths = data['A_paths']
            
            pos_errs = []
            ori_errs = []
            for b in range(B):
                p_err = numpy.linalg.norm(poses[b, 0:3] - gt_poses[b, 0:3])
                pos_errs.append(p_err)
                
                q1 = poses[b, 3:7]
                q2 = gt_poses[b, 3:7]
                q1 = q1 / (numpy.linalg.norm(q1) + 1e-8)
                q2 = q2 / (numpy.linalg.norm(q2) + 1e-8)
                d = abs(numpy.sum(numpy.multiply(q1, q2)))
                d = min(1.0, max(-1.0, d))
                theta = 2 * numpy.arccos(d) * 180 / numpy.pi
                ori_errs.append(theta)
            
            pos_errs = numpy.array(pos_errs)
            ori_errs = numpy.array(ori_errs)
            
        else:
            model.set_input(data)
            model.test()
            
            # Batch processing
            img_paths = model.get_image_paths()
            poses = model.get_current_pose() # (B, 7)
            pos_errs, ori_errs = model.get_current_errors() # (B,), (B,)
            
            gt_poses = model.input_B.cpu().numpy() # (B, 7)
        
        for b in range(len(img_paths)):
            img_path = img_paths[b]
            # print('\t%04d/%04d: process image... %s' % (i*opt.batchSize + b, len(dataset), img_path), end='\r')
            image_path = img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
            
            pose = poses[b]
            gt_pose = gt_poses[b]
            
            util.save_estimated_pose(image_path, pose)
            
            # Calculate Reprojection Error for single item
            reproj_err = geometry_utils.compute_reprojection_error(gt_pose, pose, K)
            
            err.append([pos_errs[b], ori_errs[b], reproj_err])

    median_pos = numpy.median(err, axis=0)
    if median_pos[0] < besterror[1]:
        besterror = [testepoch, median_pos[0], median_pos[1], median_pos[2]]
    print()
    print("\tmedian wrt pos.: {0:.2f}m {1:.2f}° {2:.2f}px".format(median_pos[0], median_pos[1], median_pos[2]))
    testfile.write("{0:<5} {1:.2f}m {2:.2f}° {3:.2f}px\n".format(testepoch,
                                                     median_pos[0],
                                                     median_pos[1],
                                                     median_pos[2]))
    testfile.flush()
print("{0:<5} {1:.2f}m {2:.2f}° {3:.2f}px\n".format(*besterror))
testfile.write('-----------------\n')
testfile.write("{0:<5} {1:.2f}m {2:.2f}° {3:.2f}px\n".format(*besterror))
testfile.write('==================\n')
testfile.close()
