# --------------------- Dataset ---------------------
experiment = 'min_mnist'
model_fidkid_path = 'Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/models/lenet.pth'
if (experiment=='min_mnist') | (experiment=='max_mnist'):
    data_path = 'processed/original_thic_resample'
elif 'rotation_data':
    data_path = 'processed/rotation_dataset'

gan_path = 'generator'
forward_path = 'forward'
if experiment == 'max_mnist':
    models_path = '/content/drive/My Drive/master_thesis/models_inference_max/'
elif experiment == 'min_mnist':
    models_path = '/content/drive/My Drive/master_thesis/models_inference_min/'
elif experiment == 'rotation_dataset':
    models_path = '/content/drive/My Drive/master_thesis/models_inference_rotation/'

img_size=28
channels=1
img_shape = (channels, img_size, img_size)
seed = 42