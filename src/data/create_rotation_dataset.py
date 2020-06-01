# -*- coding: utf-8 -*-
import os
import time
import random
import numpy as np
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import kornia.geometry as geometry

seed = 2020
random.seed(seed)
np.random.seed(seed)

cuda = True if torch.cuda.is_available() else False

if not cuda:
	print("WARNING: You have no CUDA devices")

device = torch.device("cuda" if cuda else "cpu")
print(f'Device found : {device}')

def gaussian_kernal(kernel_size=3, channels=1, sigma=1, device=None):

	# Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
	x_cord = torch.arange(kernel_size, device=device)
	x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
	y_grid = x_grid.t()
	xy_grid = torch.stack([x_grid, y_grid], dim=-1)

	mean = (kernel_size - 1)/2.
	variance = sigma**2.

	# Calculate the 2-dimensional gaussian kernel which is
	# the product of two gaussian distributions for two different
	# variables (in this case called x and y)
	gaussian_kernel = (1./(2.*math.pi*variance)) *\
										torch.exp(
												-torch.sum((xy_grid - mean)**2., dim=-1) /\
												(2*variance)
										)
	# Make sure sum of values in gaussian kernel equals 1.
	gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

	# Reshape to 2d depthwise convolutional weight
	gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
	gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

	gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
															kernel_size=kernel_size, groups=channels, padding=1, bias=False)

	gaussian_filter.weight.data = gaussian_kernel
	gaussian_filter.weight.requires_grad = False

	return gaussian_filter

class SyntheticTesla:

	# Create a synthetic image of 'T'

	def __init__(self, device, sz=28):
		self.device = device
		self.src = torch.zeros([1, 1, sz, sz], device=self.device)
		self.src[0, 0, 9:19, 18:20] = 1.0
		self.src[0, 0, 13:15, 10:19] = 1.0
		self.conv = gaussian_kernal(device=device)

	def next_batch(self, batch_size=64, std_pos=0.0):
		batch_sample = self.conv(self.src.repeat(batch_size,1,1,1))
		batch_sample = torch.clamp(batch_sample / batch_sample.max(), min=0.0, max=1.0)
		return batch_sample, None

def create_t_dataset(save_path, batch_size=1000, len_dataset=100000, split_ratio=0.1, dtype=''):
    raw_tesla = SyntheticTesla(device)

    def batch_sample(batch_size, y_start, y_end, is_rand_tran=True):
	    raw_batch, _ = raw_tesla.next_batch(batch_size)

	    val_rot = torch.rand(batch_size, device=device) * (y_end - y_start) + y_start
	    rot_batch = geometry.rotate(raw_batch, val_rot)
	    rot_batch = rot_batch.div(0.95).clamp(min=0.0, max=1.0)

	    if is_rand_tran:
	    	raw_tran = torch.rand(batch_size, 2, device=device).sub(0.5).mul(5.0*2.0)
	    	rot_batch = geometry.translate(rot_batch, raw_tran)

	    img_rot = rot_batch.sub(0.5).mul(2.0).view(batch_size, 1, 28, 28)

	    return img_rot, val_rot
 
    time_start = time.time()

    imgs_ls = []
    labels_ls = []

    for i in tqdm(range(int(round(len_dataset/batch_size)))):
        imgs, batch_rot = batch_sample(batch_size, 0, 360)
        '''for i, im in enumerate(imgs):
            plot_img(imgs[i].squeeze(0), batch_rot[i])'''

        imgs_ls.append(imgs.squeeze(1).cpu())
        labels_ls.append(batch_rot.cpu())

    imgs = np.concatenate(imgs_ls)
    labels = np.concatenate(labels_ls)

    indice = int(split_ratio * labels.shape[0])
    print(f'test size equal : {indice}')

    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, 'train_images.npy'), imgs[indice:])
    np.save(os.path.join(save_path, 'train_labels.npy'), labels[indice:])
    np.save(os.path.join(save_path, 'test_images.npy'), imgs[:indice])
    np.save(os.path.join(save_path, 'test_labels.npy'), labels[:indice])

def main(path):
    create_t_dataset(path, batch_size=1000, len_dataset=60000)

if __name__ == '__main__':
    PATH = '../../data/processed/rotation_data/'
    main(PATH)