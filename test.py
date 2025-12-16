import time
import os
import argparse
import torch
import numpy
from data.data_loader import CreateDataLoader
from models.models import create_model
from util import util

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
    parser.add_argument('--max_range', type=int, default=25, help='range of ref index')
    parser.add_argument('--img_ret', action='store_true', help='use image retrieval')

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
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle

data_loader = CreateDataLoader(opt)
dataset = data_loader

results_dir = os.path.join(opt.results_dir, opt.name)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

besterror  = [0, float('inf'), float('inf')] # nepoch, medX, medQ

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
testepochs = epochs[-10:] # Take last 10 epochs
# if opt.model == 'posenet':
#     testepochs = epochs[-10:] # Take last 10 epochs
# else:
#     testepochs = epochs[-50:]  # Take last 50 epochs
print("Testing epochs:", testepochs)

testfile = open(os.path.join(results_dir, 'test_median.txt'), 'a')
testfile.write('epoch medX  medQ\n')
testfile.write('==================\n')

model = create_model(opt)

for testepoch in testepochs:
    if opt.img_ret:
        model.build_datatbase()
        dataset = CreateDataLoader(opt, model)
    model.load_network(model.netG, 'G', testepoch)
    # test
    # err_pos = []
    # err_ori = []
    err = []
    print("epoch: "+ str(testepoch))
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        img_path = model.get_image_paths()[0]
        print('\t%04d/%04d: process image... %s' % (i, len(dataset), img_path), end='\r')
        image_path = img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
        pose = model.get_current_pose()
        util.save_estimated_pose(image_path, pose)
        err_p, err_o = model.get_current_errors()
        # err_pos.append(err_p)
        # err_ori.append(err_o)
        err.append([err_p, err_o])

    median_pos = numpy.median(err, axis=0)
    if median_pos[0] < besterror[1]:
        besterror = [testepoch, median_pos[0], median_pos[1]]
    print()
    # print("median position: {0:.2f}".format(numpy.median(err_pos)))
    # print("median orientat: {0:.2f}".format(numpy.median(err_ori)))
    print("\tmedian wrt pos.: {0:.2f}m {1:.2f}°".format(median_pos[0], median_pos[1]))
    testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(testepoch,
                                                     median_pos[0],
                                                     median_pos[1]))
    testfile.flush()
print("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('-----------------\n')
testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('==================\n')
testfile.close()
