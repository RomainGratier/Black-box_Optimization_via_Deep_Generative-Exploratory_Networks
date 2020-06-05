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
from torch.nn import functional as F
import kornia.geometry as geometry

from src.data.utils import load_idx, load_compress_numpy, gaussian_kernal

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
    def __init__(self, dataset, y_feature='thickness', folder='Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/data/', data_path = 'processed/original_thic_resample'):

        if dataset == 'train':

            img_file, digit_file, morpho = self.__getdatasets__(dataset, folder, data_path)
            # Get the labels
            labels, index = self.__getlabels__(dataset, morpho, digit_file, y_feature)

            self.maximum = np.max(labels[y_feature])
            self.minimum = np.min(labels[y_feature])

            # Read images from MNIST
            images = self.__transform__(load_idx(img_file))[index]

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels[y_feature].to_numpy())
            self.len = self.y_data.shape[0]

        if dataset == 'test_in':

            img_file_te, digit_file_te, morpho_te = self.__getdatasets__('test', folder, data_path)

            # Get the labels
            labels_te, index_te = self.__getlabels__(dataset, morpho_te, digit_file_te, y_feature)


            self.maximum = np.max(labels_te[y_feature])
            self.minimum = np.min(labels_te[y_feature])

            # Read images from MNIST
            images = self.__transform__(load_idx(img_file_te))[index_te]

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te[y_feature].to_numpy())
            self.len = self.y_data.shape[0]
        
        if dataset == 'test_out':

            img_file_te, digit_file_te, morpho_te = self.__getdatasets__('test', folder, data_path)

            # Get the labels
            labels_te, index_te = self.__getlabels__(dataset, morpho_te, digit_file_te, y_feature)

            self.maximum = np.max(labels_te[y_feature])
            self.minimum = np.min(labels_te[y_feature])

            # Read images from MNIST
            images = self.__transform__(load_idx(img_file_te))[index_te]

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te[y_feature].to_numpy())
            self.len = self.y_data.shape[0]
        
        if dataset == 'test':

            img_file_te, digit_file_te, morpho_te = self.__getdatasets__('test', folder, data_path)

            # Get the labels
            labels_te, index_te = self.__getlabels__('test', morpho_te, digit_file_te, y_feature)

            self.maximum = np.max(labels_te[y_feature])
            self.minimum = np.min(labels_te[y_feature])

            # Read images from MNIST
            images = self.__transform__(load_idx(img_file_te))[index_te]

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te[y_feature].to_numpy())
            self.len = self.y_data.shape[0]

        if dataset == 'full':

            img_file_tr, digit_file_tr, morpho_tr = self.__getdatasets__('train', folder, data_path)

            # Get the labels
            labels_te, index_te = self.__getlabels__('test', morpho_tr, digit_file_tr, y_feature)

            self.maximum = np.max(labels_te[y_feature])
            self.minimum = np.min(labels_te[y_feature])

            # Read images from MNIST
            images = self.__transform__(load_idx(img_file_tr))[index_te]

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te[y_feature].to_numpy())
            self.len = self.y_data.shape[0]

    def __getdatasets__(self, dataset, folder, data_path):
        folder = os.path.join(folder, data_path)

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
    """ Rotation dataset."""

    # Initialize your data, download, etc.
    def __init__(self, dataset, folder='Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/data/', data_path='processed/rotation_dataset'):

        if dataset == 'train':

            # Get file paths
            img_file, label_file = self.__getdatasets__(dataset, folder, data_path)

            # Get the labels
            labels, index = self.__getlabels__(dataset, label_file)
            
            # Read images
            images = load_compress_numpy(img_file)[index]
  
            self.maximum = np.max(labels)
            self.minimum = np.min(labels)

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels.to_numpy()).squeeze(1)
            self.len = self.y_data.shape[0]

        if dataset == 'test_in':

            img_file_te, label_file_te = self.__getdatasets__('test', folder, data_path)

            # Get the labels
            labels_te, index_te = self.__getlabels__(dataset, label_file_te)
            
            # Read images
            images = load_compress_numpy(img_file_te)[index_te]

            self.maximum = np.max(labels_te)
            self.minimum = np.min(labels_te)

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te.to_numpy()).squeeze(1)
            self.len = self.y_data.shape[0]
        
        if dataset == 'test_out':

            img_file_te, label_file_te = self.__getdatasets__('test', folder, data_path)

            # Get the labels
            labels_te, index_te = self.__getlabels__(dataset, label_file_te)
            
            # Read images 
            images = load_compress_numpy(img_file_te)[index_te]

            self.maximum = np.max(labels_te)
            self.minimum = np.min(labels_te)

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te.to_numpy()).squeeze(1)
            self.len = self.y_data.shape[0]
        
        if dataset == 'test':

            img_file_te, label_file_te = self.__getdatasets__(dataset, folder, data_path)

            # Get the labels
            labels_te, index_te = self.__getlabels__(dataset, label_file_te)
            
            # Read images 
            images = load_compress_numpy(img_file_te)[index_te]

            self.maximum = np.max(labels_te)
            self.minimum = np.min(labels_te)

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te.to_numpy()).squeeze(1)
            self.len = self.y_data.shape[0]

        if dataset == 'full':

            img_file_tr, label_file_te = self.__getdatasets__('train', folder, data_path)

            # Get the labels
            labels_te, index_te = self.__getlabels__('test', label_file_te)
            
            # Read images
            images = load_compress_numpy(img_file_tr)[index_te]
            print(f"Check image extremum : max={np.max(images)} | min={np.min(images)}")
            print(f"Check label extremum : max={np.max(labels_te)} | min={np.min(labels_te)}")

            self.maximum = np.max(labels_te)
            self.minimum = np.min(labels_te)

            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te.to_numpy()).squeeze(1)
            self.len = self.y_data.shape[0]

    def __getdatasets__(self, dataset, folder, data_path):
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

class MNISTDatasetLeNet(Dataset):
    """ MNIST dataset."""

    # Initialize your data, download, etc.
    def __init__(self, dataset, folder='Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/data/', data_path='morpho_mnist/original'):

        '''Initialise the data type:
        - data_path : original, global, thic, frac, local, plain, swel, thin
        '''
        img_file, digit_file = self.__getdatasets__(dataset, folder, data_path)
        labels = self.__getlabels__(digit_file)

        # Read images from MNIST
        images = self.__transform__(load_idx(img_file))

        # Select inputs
        self.x_data = images.float()
        self.y_data = torch.from_numpy(labels).long()
        self.len = self.y_data.shape[0]

    def __getdatasets__(self, dataset, folder, data_path):
        folder = os.path.join(folder, data_path)

        if dataset == 'test':
            return os.path.join(folder,'t10k-images-idx3-ubyte.gz'), os.path.join(folder,'t10k-labels-idx1-ubyte.gz')

        if dataset == 'train':
            return os.path.join(folder,'train-images-idx3-ubyte.gz'), os.path.join(folder,'train-labels-idx1-ubyte.gz')

    def __getlabels__(self, digit_file):
        # Get the labels
        return load_idx(digit_file)

    def __transform__(self, X):
        X = X.astype('float32')

        # Resize to 32 * 32
        X = F.interpolate(torch.from_numpy(X).unsqueeze(1), size=(32, 32), mode='bilinear')

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

class RotationDatasetLeNet(Dataset):
    """ Rotation dataset."""

    # Initialize your data, download, etc.
    def __init__(self, dataset, folder='Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/data/', data_path = 'processed/rotation_dataset'):

        # Get file paths
        img_file, label_file = self.__getdatasets__(dataset, folder, data_path)

        # Get the labels
        labels = load_compress_numpy(label_file)
        images = self.__transform__(load_compress_numpy(img_file))

        # Select inputs
        self.x_data = images.float()
        self.y_data = torch.from_numpy(labels).float()
        self.len = self.y_data.shape[0]

    def __getdatasets__(self, dataset, folder, data_path):
        folder = os.path.join(folder, data_path)

        if dataset == 'test':
            return os.path.join(folder,'test_images_ubyte.gz'), os.path.join(folder,'test_labels_ubyte.gz')
        if dataset == 'train':
            return os.path.join(folder,'train_images_ubyte.gz'), os.path.join(folder,'train_labels_ubyte.gz')

    def __transform__(self, X):
        X = X.astype('float32')

        # Resize to 32 * 32
        X = F.interpolate(torch.from_numpy(X).unsqueeze(1), size=(32, 32), mode='bilinear')

        return X

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

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

def getDataset():
    if (cfg.experiment == 'min_mnist') | (cfg.experiment == 'max_mnist'): 
        trainset = MNISTDataset('train', y_feature=cfg.feature, folder=cfg.data_folder, data_path=cfg.data_path)
        testset_in = MNISTDataset('test_in', y_feature=cfg.feature, folder=cfg.data_folder, data_path=cfg.data_path)
        testset_out = MNISTDataset('test_out', y_feature=cfg.feature, folder=cfg.data_folder, data_path=cfg.data_path)

    elif cfg.experiment == 'rotation_dataset':
        trainset = RotationDataset('train', folder=cfg.data_folder, data_path=cfg.data_path)
        testset_in = RotationDataset('test_in', folder=cfg.data_folder, data_path=cfg.data_path)
        testset_out = RotationDataset('test_out', folder=cfg.data_folder, data_path=cfg.data_path)

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