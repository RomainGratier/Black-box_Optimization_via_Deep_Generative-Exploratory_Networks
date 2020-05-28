"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import torch
import multiprocessing

from src.morphomnist.measure import measure_batch

# ------------ Compare the forward model and the Measure from morphomnist ------------
def transform_inv(X):
    # Normalize between 0 - 1
    X *= 0.5
    X += 0.5

    # Transfrom 0 - 255
    X *= 255
    
    # Convert the float tensor to uint8 numpy
    if type(X) == type(torch.tensor([])):
        return X.round().byte().numpy()
    if type(X) == type(np.array([])):
        return X.round().astype(np.uint8)

def compute_thickness_ground_truth(images_generated, feature = 'thickness'):
    ''' Byte input image thickness measurment '''
    with multiprocessing.Pool() as pool:
        thickness = measure_batch(transform_inv(images_generated), pool=pool)[feature]
    return thickness

def se(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred))

def re(y_true, y_pred):
    return np.abs(np.subtract(y_true, y_pred))