import time
import argparse
import os
import torch
import numpy
import random
from data.data_loader import CreateDataLoader
from models.models import create_model
from util import util

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Base options
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--batchSize', type=int, default=75, help='input batch size')
    parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
    parser.add_argument('--fineSize', type=int, default=224, help='then crop to this size')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=7, help='# of output image channels')
    parser.add_argument('--lstm_hidden_size', type=int, default=None, help='hidden size of the LSTM layer in PoseLSTM')
    parser.add_argument('--transformer_hidden_size', type=int, default=None, help='hidden size of the Transformer layer in PoseTransformer')
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
    parser.add_argument('--seed', type=int, default=42, help='initial random seed for deterministic results')
    parser.add_argument('--beta', type=float, default=500, help='beta factor used in posenet.')
    parser.add_argument('--loss_type', type=str, default='mse', help='type of loss function to use: [mse | geo | l2]')

    # Train options
    parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=0, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--adambeta1', type=float, default=0.9, help='first momentum term of adam')
    parser.add_argument('--adambeta2', type=float, default=0.999, help='second momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for adam')
    parser.add_argument('--use_html', action='store_true', help='save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--init_weights', type=str, default='pretrained_models/places-googlenet.pickle', help='initiliaze network from, e.g., pretrained_models/places-googlenet.pickle')

    opt = parser.parse_args()
    opt.isTrain = True
    
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

def main():
    opt = parse_args()

    ## SEEDING
    torch.manual_seed(opt.seed)
    numpy.random.seed(opt.seed)
    random.seed(opt.seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    ## SEEDING

    data_loader = CreateDataLoader(opt)
    dataset = data_loader
    dataset_size = len(data_loader.dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                util.print_current_errors(epoch, epoch_iter, errors, t)

            # if total_steps % opt.save_latest_freq == 0:
            #     print('saving the latest model (epoch %d, total_steps %d)' %
            #           (epoch, total_steps))
            #     model.save('latest')

        if epoch % opt.save_epoch_freq == 0 and epoch >=1600:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

if __name__ == "__main__":
    main()
