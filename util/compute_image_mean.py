import argparse
import glob
import os
import re

import numpy as np
from os.path import join as jpath
from PIL import Image


def params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='datasets/KingsCollege', help='dataset root')
    parser.add_argument('--height', type=int, default=256, help='image height')
    parser.add_argument('--width', type=int, default=455, help='image width')
    parser.add_argument('--save_resized_imgs', action="store_true", default=False, help='save resized train/test images [height, width]')
    return parser.parse_args()

args = params()
dataroot = args.dataroot
imsize = [args.height, args.width] # (H, W)
imlist = []
if "cambridge" in dataroot or "KingsCollege" in dataroot:
    imlist = np.loadtxt(jpath(dataroot, 'dataset_train.txt'),
                        dtype=str, delimiter=' ', skiprows=3, usecols=(0))
    imlist = [jpath(dataroot, impath) for impath in imlist]
elif "7scenes" in dataroot or "chess" in dataroot:
    split_file = os.path.join(dataroot, 'TrainSplit.txt')
    with open(split_file, 'r') as f:
        split_file_lines = f.readlines()
        for line in split_file_lines:
            match = re.search(r'\d+', line)
            seq_idx = int(match.group(0))
            seq_name = f"seq-{seq_idx:02}"
            seq_dir = os.path.join(dataroot, seq_name)
            imlist += glob.glob(seq_dir + "/*.color.png")
mean_image = np.zeros((imsize[0], imsize[1], 3), dtype=float)
for i, impath in enumerate(imlist):
    print('[%d/%d]:%s' % (i+1, len(imlist), impath), end='\r')
    image = Image.open(impath).convert('RGB')
    image = image.resize((imsize[1], imsize[0]), Image.BICUBIC)
    mean_image += np.array(image).astype(float)

    # save resized training images
    if args.save_resized_imgs:
        image.save(impath)
print()
mean_image /= len(imlist)
Image.fromarray(mean_image.astype(np.uint8)).save(jpath(dataroot, 'mean_image.png'))
np.save(jpath(dataroot, 'mean_image.npy'), mean_image)

# save resized test images
if args.save_resized_imgs:
    imlist = []
    if "cambridge" in dataroot or "KingsCollege" in dataroot:
        imlist = np.loadtxt(jpath(dataroot, 'dataset_test.txt'),
                            dtype=str, delimiter=' ', skiprows=3, usecols=(0))
        imlist = [jpath(dataroot, impath) for impath in imlist]
    elif "7scenes" in dataroot or "chess" in dataroot:
        split_file = os.path.join(dataroot, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            split_file_lines = f.readlines()
            for line in split_file_lines:
                match = re.search(r'\d+', line)
                seq_idx = int(match.group(0))
                seq_name = f"seq-{seq_idx:02}"
                seq_dir = os.path.join(dataroot, seq_name)
                imlist += glob.glob(seq_dir + "/*.color.png")

    for i, impath in enumerate(imlist):
        print('[%d/%d]:%s' % (i+1, len(imlist), impath), end='\r')
        image = Image.open(impath).convert('RGB')
        image = image.resize((imsize[1], imsize[0]), Image.BICUBIC)
        image.save(impath)
    print()
