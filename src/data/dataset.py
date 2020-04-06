# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import os
import gzip
import struct

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class MNISTDataset(Dataset):
    """ MNIST dataset."""

    # Initialize your data, download, etc.
    def __init__(self, dataset, y_feature):
      
        if dataset == 'train':

            img_file, digit_file, morpho = self.__getdatasets__(dataset)
            # Get the labels
            labels, minimum, maximum, index = self.__getlabels__(dataset, morpho, digit_file, y_feature)
            labels, scaler = self.__get_scaler__(8, labels, y_feature)
            print(labels.describe())
            self.maximum = np.max(labels['normalized_label'])
            self.minimum = np.min(labels['normalized_label'])
    
            # Read images from MNIST
            images = self.__transform__(load_idx(img_file), 153)[index]
    
            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels['normalized_label'].to_numpy())
            self.labels = labels[y_feature]
            self.scaler = scaler
    
            self.len = self.y_data.shape[0]

        else:
            img_file_tr, digit_file_tr, morpho_tr = self.__getdatasets__('train')
            img_file_te, digit_file_te, morpho_te = self.__getdatasets__('test')
    
            # Get the labels
            labels_tr, minimum_tr, maximum_tr, index_tr = self.__getlabels__('train', morpho_tr, digit_file_tr, y_feature)
            labels_te, minimum_te, maximum_te, index_te = self.__getlabels__('test', morpho_te, digit_file_te, y_feature)
    
            labels_tr, scaler_tr = self.__get_scaler__(8, labels_tr, y_feature)
            labels_te = self.__scale__(labels_te, y_feature, scaler_tr)
    
            print(labels_te.describe())
            self.maximum = np.max(labels_te['normalized_label'])
            self.minimum = np.min(labels_te['normalized_label'])
    
            # Read images from MNIST
            images = self.__transform__(load_idx(img_file_te), 153)[index_te]
    
            # Select inputs
            self.x_data = torch.from_numpy(images).unsqueeze(1)
            self.y_data = torch.from_numpy(labels_te['normalized_label'].to_numpy())
            self.labels = labels_te[y_feature]
            self.scaler = scaler_tr
    
            self.len = self.y_data.shape[0]

    def __getdatasets__(self, dataset):
        
        folder = 'Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/data/MNIST_morpho/'
        if dataset == 'test':
            return os.path.join(folder,'t10k-images-idx3-ubyte.gz'), os.path.join(folder,'t10k-labels-idx1-ubyte.gz'), os.path.join(folder,'t10k-morpho.csv')

        if dataset == 'train':
            return os.path.join(folder,'train-images-idx3-ubyte.gz'), os.path.join(folder,'train-labels-idx1-ubyte.gz'), os.path.join(folder,'train-morpho.csv')

    def __getlabels__(self, dataset, file, digit_file, y_feature, bound=6):
        # Get the labels
        labels = pd.read_csv(file)
        labels['digit'] = load_idx(digit_file)

        # discretize in int as bins
        #labels[y_feature] = labels[y_feature].astype('int')

        if dataset == 'train':
            #index = labels[(labels[y_feature] < bound) & (labels['digit'] == 1)].index
            index = labels[labels[y_feature] < 6].index
            labels = labels.loc[index]

            print(' ---------------------------------------- CHECK NANS')
            print(labels[labels[y_feature].isnull()])

        else:
            index = labels[labels[y_feature] < 9].index
            labels = labels.loc[index]
            print(' ---------------------------------------- CHECK NANS')
            print(labels[labels[y_feature].isnull()])

        print(labels.describe())
        minimum = labels[y_feature].min()
        maximum = labels[y_feature].max()
        print(f" -------------------- EDA -------------------- ")
        print()
        print(f"The extrem values from our labels:\nMaximum : {maximum}   Minimum : {minimum}")
        print()
        print(f"Check the distribution of our labels")
        check = pd.DataFrame(labels[y_feature].astype('int'), columns=[y_feature])
        print(check.groupby(y_feature)[y_feature].count())
        print(check.plot.hist(bins=20))

        return labels, minimum, maximum, index

    def __get_scaler__(self, extrem, df, feature):

        # Normalize the labels
        scaler = MinMaxScaler()
        df_cons = df.append({feature:float(extrem)}, ignore_index=True)
        scaler.fit(df_cons[feature].values.reshape(-1,1))
        df['normalized_label'] = scaler.transform(df[feature].values.reshape(-1,1))

        print(f"The extrem values from our normalized labels:\nMaximum : {df['normalized_label'].max()}   Minimum : {df['normalized_label'].min()}")

        return df, scaler

    def __scale__(self, df, feature, scaler):
        df['normalized_label'] = scaler.transform(df[feature].values.reshape(-1,1))
        return df

    def __transform__(self, X, middle):
        X = X.astype('float32')
        X -= middle
        X /= middle
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