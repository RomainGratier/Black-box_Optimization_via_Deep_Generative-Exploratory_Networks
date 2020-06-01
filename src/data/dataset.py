# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import os
import gzip
import struct

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler

import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import kornia.geometry as geometry

import src.config as cfg

if cfg.experiment == 'min_mnist':
    import src.config_min_mnist as cfg_data
elif cfg.experiment == 'max_mnist':
    import src.config_max_mnist as cfg_data
elif cfg.experiment == 'rotation_dataset':
    import src.config_rotation as cfg_data

class MNISTDataset(Dataset):
    """ MNIST dataset."""

    # Initialize your data, download, etc.
    def __init__(self, dataset, y_feature, data_type = 'morpho_mnist/original'):

        '''Initialise the data type:
        - data_type : original, global, thic, frac, local, plain, swel, thin
        '''

        if dataset == 'train':

            img_file, digit_file, morpho = self.__getdatasets__(dataset, data_type)
            # Get the labels
            labels, index = self.__getlabels__(dataset, morpho, digit_file, y_feature)
            print(labels.describe())
            self.maximum = np.max(labels[y_feature])
            self.minimum = np.min(labels[y_feature])

            # Read images from MNIST
            images = self.__transform__(load_idx(img_file))[index]

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels[y_feature].to_numpy())
            self.len = self.y_data.shape[0]

        if dataset == 'test_in':

            img_file_te, digit_file_te, morpho_te = self.__getdatasets__('test', data_type)

            # Get the labels
            labels_te, index_te = self.__getlabels__(dataset, morpho_te, digit_file_te, y_feature)

            print(labels_te.describe())
            self.maximum = np.max(labels_te[y_feature])
            self.minimum = np.min(labels_te[y_feature])

            # Read images from MNIST
            images = self.__transform__(load_idx(img_file_te))[index_te]

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te[y_feature].to_numpy())
            self.len = self.y_data.shape[0]
        
        if dataset == 'test_out':

            img_file_te, digit_file_te, morpho_te = self.__getdatasets__('test', data_type)

            # Get the labels
            labels_te, index_te = self.__getlabels__(dataset, morpho_te, digit_file_te, y_feature)

            print(labels_te.describe())
            self.maximum = np.max(labels_te[y_feature])
            self.minimum = np.min(labels_te[y_feature])

            # Read images from MNIST
            images = self.__transform__(load_idx(img_file_te))[index_te]

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te[y_feature].to_numpy())
            self.len = self.y_data.shape[0]
        
        if dataset == 'test':

            img_file_te, digit_file_te, morpho_te = self.__getdatasets__('test', data_type)

            # Get the labels
            labels_te, index_te = self.__getlabels__('test', morpho_te, digit_file_te, y_feature)

            print(labels_te.describe())
            self.maximum = np.max(labels_te[y_feature])
            self.minimum = np.min(labels_te[y_feature])

            # Read images from MNIST
            images = self.__transform__(load_idx(img_file_te))[index_te]

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te[y_feature].to_numpy())
            self.len = self.y_data.shape[0]

        if dataset == 'full':

            img_file_tr, digit_file_tr, morpho_tr = self.__getdatasets__('train', data_type)

            # Get the labels
            labels_te, index_te = self.__getlabels__('test', morpho_tr, digit_file_tr, y_feature)

            print(labels_te.describe())
            self.maximum = np.max(labels_te[y_feature])
            self.minimum = np.min(labels_te[y_feature])

            # Read images from MNIST
            images = self.__transform__(load_idx(img_file_tr))[index_te]

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te[y_feature].to_numpy())
            self.len = self.y_data.shape[0]

    def __getdatasets__(self, dataset, data_type):

        folder = 'Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/data/'
        folder = os.path.join(folder, data_type)

        if dataset == 'test':
            return os.path.join(folder,'t10k-images-idx3-ubyte.gz'), os.path.join(folder,'t10k-labels-idx1-ubyte.gz'), os.path.join(folder,'t10k-morpho.csv')
        if dataset == 'train':
            return os.path.join(folder,'train-images-idx3-ubyte.gz'), os.path.join(folder,'train-labels-idx1-ubyte.gz'), os.path.join(folder,'train-morpho.csv')

    def __getlabels__(self, dataset, file, digit_file, y_feature):
        # Get the labels
        labels = pd.read_csv(file)
        labels['digit'] = load_idx(digit_file)

        if (dataset == 'train') | (dataset == 'test_in'):
            if cfg.experiment == 'max_mnist':
                index = labels[labels[y_feature] < cfg_data.limit_dataset].index
            elif cfg.experiment == 'min_mnist':
                index = labels[(labels[y_feature] >= cfg_data.limit_dataset) & (labels[y_feature] < cfg_data.max_dataset)].index
            labels = labels.loc[index]

        elif dataset == 'test_out':
            if cfg.experiment == 'max_mnist':
                index = labels[(labels[y_feature] >= cfg_data.limit_dataset) & (labels[y_feature] < cfg_data.max_dataset)].index
            elif cfg.experiment == 'min_mnist':
                index = labels[labels[y_feature] < cfg_data.limit_dataset].index
            labels = labels.loc[index]

        else:
            if cfg.experiment == 'max_mnist':
                index = labels[labels[y_feature] < cfg_data.max_dataset].index
            elif cfg.experiment == 'min_mnist':
                index = labels[labels[y_feature] < cfg_data.max_dataset].index
            labels = labels.loc[index]

        print(labels.describe())
        print(f" -------------------- EDA -------------------- ")
        print(f"Check the distribution of our labels")
        check = pd.DataFrame(np.around(labels[y_feature].values), columns=[y_feature])
        print(check.groupby(y_feature)[y_feature].count())
        print(check.plot.hist(bins=20))

        return labels, index

    def __transform__(self, X):
        X = X.astype('float32')

        # Normalize between 0 - 1
        X = (X - X.min())
        X = X / (X.max() - X.min())

        # Normalize between -1 - 1
        X -= 0.5
        X /= 0.5

        return X

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class RotationDataset(Dataset):
    """ MNIST dataset."""

    # Initialize your data, download, etc.
    def __init__(self, dataset, data_path = 'sample_data'):

        '''Initialise the data type:
        - data_type : original, global, thic, frac, local, plain, swel, thin
        '''

        if dataset == 'train':

            # Get file paths
            img_file, label_file = self.__getdatasets__(dataset, data_path)

            # Get the labels
            print(label_file)
            labels, index = self.__getlabels__(dataset, label_file)
            
            # Read images
            images = load_compress_numpy(img_file)[index]
  
            print(labels.describe())
            self.maximum = np.max(labels)
            self.minimum = np.min(labels)

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels.to_numpy()).squeeze(1)
            self.len = self.y_data.shape[0]

        if dataset == 'test_in':

            img_file_te, label_file_te = self.__getdatasets__('test', data_path)

            # Get the labels
            labels_te, index_te = self.__getlabels__(dataset, label_file_te)
            
            # Read images
            images = load_compress_numpy(img_file_te)[index_te]

            print(labels_te.describe())
            self.maximum = np.max(labels_te)
            self.minimum = np.min(labels_te)

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te.to_numpy()).squeeze(1)
            self.len = self.y_data.shape[0]
        
        if dataset == 'test_out':

            img_file_te, label_file_te = self.__getdatasets__('test', data_path)

            # Get the labels
            labels_te, index_te = self.__getlabels__(dataset, label_file_te)
            
            # Read images 
            images = load_compress_numpy(img_file_te)[index_te]

            print(labels_te.describe())
            self.maximum = np.max(labels_te)
            self.minimum = np.min(labels_te)

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te.to_numpy()).squeeze(1)
            self.len = self.y_data.shape[0]
        
        if dataset == 'test':

            img_file_te, label_file_te = self.__getdatasets__(dataset, data_path)

            # Get the labels
            labels_te, index_te = self.__getlabels__(dataset, label_file_te)
            
            # Read images 
            images = load_compress_numpy(img_file_te)[index_te]

            print(labels_te.describe())
            self.maximum = np.max(labels_te)
            self.minimum = np.min(labels_te)

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te.to_numpy()).squeeze(1)
            self.len = self.y_data.shape[0]

        if dataset == 'full':

            img_file_tr, label_file_te = self.__getdatasets__('train', data_path)

            # Get the labels
            labels_te, index_te = self.__getlabels__('test', label_file_te)
            
            # Read images
            images = load_compress_numpy(img_file_tr)[index_te]
            print(f"Check image extremum : max={np.max(images)} | min={np.min(images)}")
            print(f"Check label extremum : max={np.max(labels_te)} | min={np.min(labels_te)}")

            print(labels_te.describe())
            self.maximum = np.max(labels_te)
            self.minimum = np.min(labels_te)

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te.to_numpy()).squeeze(1)
            self.len = self.y_data.shape[0]

    def __getdatasets__(self, dataset, data_path):

        folder = 'Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/data/'
        folder = os.path.join(folder, data_path)

        if dataset == 'test':
            return os.path.join(folder,'test_images_ubyte.gz'), os.path.join(folder,'test_labels_ubyte.gz')
        if dataset == 'train':
            return os.path.join(folder,'train_images_ubyte.gz'), os.path.join(folder,'train_labels_ubyte.gz')

    def __getlabels__(self, dataset, label_file):
        # Get the labels
        labels = pd.DataFrame(load_compress_numpy(label_file),columns=['label'])

        if (dataset == 'train') | (dataset == 'test_in'):
            index = labels[labels['label'] < cfg_data.limit_dataset].index
            labels = labels.loc[index]

        elif dataset == 'test_out':
            index = labels[(labels['label'] >= cfg_data.limit_dataset) & (labels['label'] < cfg_data.max_dataset)].index
            labels = labels.loc[index]

        else:
            index = labels[labels['label'] < cfg_data.max_dataset].index
            labels = labels.loc[index]

        print(labels.describe())
        print(f" -------------------- EDA -------------------- ")
        print(f"Check the distribution of our labels")
        check = pd.DataFrame(np.around(labels['label'].values), columns=['label'])
        print(check.groupby('label')['label'].count())
        print(check.plot.hist(bins=20))

        return labels, index

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def load_idx(path: str) -> np.ndarray:
    """Reads an array in IDX format from disk.
    Parameters
    ----------
    path : str
        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.
    Returns
    -------
    np.ndarray
        Output array of dtype ``uint8``.
    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)

def _load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data

def load_compress_numpy(file_name):
    with gzip.GzipFile(file_name, "r") as f:
        return np.load(f)

def getDataset(dataset_type):
    
    trainset = MNISTDataset('train', 'thickness', data_type=dataset_type)
    testset_in = MNISTDataset('test_in', 'thickness', data_type=dataset_type)
    testset_out = MNISTDataset('test_out', 'thickness', data_type=dataset_type)
    num_classes = 1
    inputs=1

    return trainset, testset_in, testset_out, inputs, num_classes


def getDataloader(trainset, testset_in, testset_out, valid_size, batch_size, num_workers):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        sampler=valid_sampler, num_workers=num_workers)
    test_loader_in = torch.utils.data.DataLoader(testset_in, batch_size=batch_size, 
        num_workers=num_workers)
    test_loader_out = torch.utils.data.DataLoader(testset_out, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader, valid_loader, test_loader_in, test_loader_out

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