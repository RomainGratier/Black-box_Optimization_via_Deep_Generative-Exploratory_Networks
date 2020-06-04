import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import struct
import gzip

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