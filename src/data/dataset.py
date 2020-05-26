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

import src.config as cfg

if cfg.experiment == 'min_mnist':
    import src.config_min_mnist as cfg_data
elif cfg.experiment == 'max_mnist':
    import src.config_max_mnist as cfg_data

class MNISTDataset(Dataset):
    """ MNIST dataset."""

    # Initialize your data, download, etc.
    def __init__(self, dataset, experiment, y_feature, data_type = 'morpho_mnist/original'):

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

    def __getlabels__(self, dataset, file, digit_file, y_feature, in_bound=cfg_data.limit_dataset, out_bound=cfg_data.max_dataset):
        # Get the labels
        labels = pd.read_csv(file)
        labels['digit'] = load_idx(digit_file)

        if (dataset == 'train') | (dataset == 'test_in'):
            if cfg.experience == 'max_mnist':
                index = labels[labels[y_feature] < in_bound].index
            elif cfg.experience == 'min_mnist':
                index = labels[(labels[y_feature] > limit_bound) & (labels[y_feature] < max_bound)].index
            labels = labels.loc[index]

        elif dataset == 'test_out':
            if cfg.experience == 'max_mnist':
                index = labels[(labels[y_feature] >= in_bound) & (labels[y_feature] < out_bound)].index
            elif cfg.experience == 'min_mnist':
                index = labels[labels[y_feature] < limit_bound].index
            labels = labels.loc[index]

        else:
            if cfg.experience == 'max_mnist':
                index = labels[labels[y_feature] < out_bound].index
            elif cfg.experience == 'min_mnist':
                index = labels[labels[y_feature] < max_bound].index
            labels = labels.loc[index]

        print(labels.describe())
        print(f" -------------------- EDA -------------------- ")
        print()
        print(f"The extrem values from our labels:\nMaximum : {maximum}   Minimum : {minimum}")
        print()
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