# --------------------- Dataset ---------------------
experiment = 'min_mnist'
model_fidkid_path = 'Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/models/lenet.pth'
data_path = 'processed/original_thic_resample'
if experiment == 'max_mnist':
    models_path = '/content/drive/My Drive/master_thesis/models_inference_max/'
if experiment == 'min_mnist':
    models_path = '/content/drive/My Drive/master_thesis/models_inference_min/'
img_size=28
channels=1
img_shape = (channels, img_size, img_size)
seed = 42