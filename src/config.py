# --------------------- Dataset ---------------------
experiment = 'min_mnist'
model_fidkid_path = 'Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/models/lenet.pth'
<<<<<<< HEAD
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

=======
data_path = 'processed/original_thic_resample'
if experiment == 'max_mnist':
    models_path = '/content/drive/My Drive/master_thesis/models_inference_max/'
if experiment == 'min_mnist':
    models_path = '/content/drive/My Drive/master_thesis/models_inference_min/'
>>>>>>> e044e98f5fd49652c90847ed0ff72f2caa69f514
img_size=28
channels=1
img_shape = (channels, img_size, img_size)
seed = 42