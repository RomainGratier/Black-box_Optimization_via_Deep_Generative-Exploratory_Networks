# --------------------- GAN ---------------------
n_epochs = 50
lr=0.0002 #5e-5 # lr=0.0008 for rotation dataset
b1=0.5
b2=0.999
sample_interval=500
n_row = 10
fid_kid_sample = 1000
latent_dim=100
label_dim_input = 1
batch_size=128