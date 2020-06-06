import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import src.config as cfg

if cfg.dcgan:
    import src.config_dcgan as cfgan
else:
    import src.config_gan as cfgan

if cfg.experiment == 'min_mnist':
    import src.config_min_mnist as cfg_data
elif cfg.experiment == 'max_mnist':
    import src.config_max_mnist as cfg_data
elif cfg.experiment == 'rotation_dataset':
    import src.config_rotation as cfg_data

def minmaxs(X):
    return (X - cfg_data.min_dataset) / (cfg_data.max_dataset - cfg_data.min_dataset)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(cfgan.latent_dim + cfgan.label_dim_input, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(cfg.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((minmaxs(labels).unsqueeze(1), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *cfg.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(cfgan.label_dim_input + int(np.prod(cfg.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), minmaxs(labels).unsqueeze(1)), -1)
        validity = self.model(d_in)
        return validity


class CondDCGenerator(nn.Module):
	# initializers
	def __init__(self, d=128):
		super(CondDCGenerator, self).__init__()

		self.deconv1 = nn.ConvTranspose2d(cfgan.latent_dim+1, d*4, 4, 1, 0)
		self.deconv1_bn = nn.BatchNorm2d(d*4)
		self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
		self.deconv2_bn = nn.BatchNorm2d(d*2)
		self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
		self.deconv3_bn = nn.BatchNorm2d(d)
		self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 3)

	# forward method
	def forward(self, ipt, label):
		x = torch.cat([ipt, minmaxs(label)], 1)
		x = F.relu(self.deconv1_bn(self.deconv1(x)))
		x = F.relu(self.deconv2_bn(self.deconv2(x)))
		x = F.relu(self.deconv3_bn(self.deconv3(x)))
		x = torch.tanh(self.deconv4(x))
		return x

class CondDCDiscriminator(nn.Module):
	# initializers
	def __init__(self, d=128):
		super(CondDCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(2, d, 4, 2, 1)
		self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
		self.layernorm2 = nn.LayerNorm([d * 2, 7, 7])
		self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
		self.layernorm3 = nn.LayerNorm([d * 4, 3, 3])
		self.conv4 = nn.Conv2d(d * 4, 1, 3, 1, 0)

	# forward method
	def forward(self, ipt, label):
		x = torch.cat([ipt, minmaxs(label)], 1)
		x = F.leaky_relu(self.conv1(x), 0.2)
		x = F.leaky_relu(self.layernorm2(self.conv2(x)), 0.2)
		x = F.leaky_relu(self.layernorm3(self.conv3(x)), 0.2)
		x = self.conv4(x)
		return x


class LeNet5Regressor(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5Regressor, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('tanh1', nn.Tanh()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('tanh3', nn.Tanh()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('tanh5', nn.Tanh())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('tanh6', nn.Tanh()),
            ('f7', nn.Linear(84, 1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

    def extract_features(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc[1](self.fc[0](output))
        return output

class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('tanh1', nn.Tanh()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('tanh3', nn.Tanh()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('tanh5', nn.Tanh())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('tanh6', nn.Tanh()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

    def extract_features(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc[1](self.fc[0](output))
        return output
    