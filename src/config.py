# --------------------- Dataset ---------------------
latent_dim=100
label_dim_input = 1
img_size=28
channels=1
img_shape = (channels, img_size, img_size)

min_dataset = 0
max_dataset = 8
limit_dataset = 6
batch_size=128
seed = 42

# --------------------- GAN ---------------------
n_epochs = 200
lr=0.0002
b1=0.5
b2=0.999
#n_cpu=8
sample_interval=500
n_row = 10
fid_kid_sample = 2000