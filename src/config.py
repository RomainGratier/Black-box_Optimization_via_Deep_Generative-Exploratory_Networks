# --------------------- Dataset ---------------------
experiment = 'max_mnist'
dcgan = True
 
if (experiment=='min_mnist') | (experiment=='max_mnist'):
    data_path = 'processed/original_thic_resample'
    model_fidkid_path = 'Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/models_fid_kid/lenet_mnist.pth'

elif experimen =='rotation_dataset':
    data_path = 'processed/rotation_dataset'
    model_fidkid_path = 'Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/models_fid_kid/lenet_rot.pth'

print(f'Data path : {data_path}')

if dcgan:
    gan_path = 'dcgenerator'
    if (experiment=='min_mnist') | (experiment=='max_mnist'):
        lambda_gp = 0.1
    elif experimen =='rotation_dataset':
        lambda_gp = 1

    n_critic = 1
else:
    gan_path = 'generator'

forward_path = 'forward'
if experiment == 'max_mnist':
    models_path = '/content/drive/My Drive/master_thesis/models_inference_max/'
elif experiment == 'min_mnist':
    models_path = '/content/drive/My Drive/master_thesis/models_inference_min/'
elif experiment == 'rotation_dataset':
    models_path = '/content/drive/My Drive/master_thesis/models_inference_rotation/'
print(f'Model path : {models_path}')

img_size=28
channels=1
img_shape = (channels, img_size, img_size)
seed = 42