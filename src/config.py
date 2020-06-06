# --------------------- PATHES ---------------------
experiment = 'max_mnist'
dcgan = True
data_folder = 'Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/data/'
main_model_path = '/content/drive/My Drive/master_thesis/'
forward_path = 'forward'

if (experiment=='min_mnist') | (experiment=='max_mnist'):
    feature = 'thickness'
    data_path = 'processed/original_thic_resample'
    model_fidkid_path = 'Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/models_fid_kid/lenet_mnist.pth'

elif experiment =='rotation_dataset':
    data_path = 'processed/rotation_dataset'
    model_fidkid_path = 'Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/models_fid_kid/lenet_rot.pth'

print(f'Data path : {data_path}')

if dcgan:
    gan_path = 'dcgenerator'
else:
    gan_path = 'generator'

# --------------------- WASSERSTEIN ---------------------
if (experiment=='min_mnist') | (experiment=='max_mnist'):
    lambda_gp = 0.1
    n_critic = 2
elif experiment =='rotation_dataset':
    lambda_gp = 1
    n_critic = 1

import os
if experiment == 'max_mnist':
    models_path = os.path.join(main_model_path, 'models_inference_max/')
elif experiment == 'min_mnist':
    models_path = os.path.join(main_model_path, 'models_inference_min/')
elif experiment == 'rotation_dataset':
    models_path = os.path.join(main_model_path, 'models_inference_rotation/')
print(f'Model path : {models_path}')

# --------------------- DATA ---------------------

img_size=28
channels=1
img_shape = (channels, img_size, img_size)
seed = 42