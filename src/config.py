############### Configuration file for Bayesian ###############
import torch
import numpy as np
import random

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

latent_dim=100
label_dim_input = 1
img_size=28
channels=1
img_shape = (channels, img_size, img_size)
min_dataset = 0
max_dataset = 8
limit_data = 6
batch_size=128
seed = 42

n_epochs = 125
lr=0.0002
b1=0.5
b2=0.999
n_cpu=8
sample_interval=1000
n_row = 10