import numpy as np
import pandas as pd
import os
import random

import multiprocessing
from morphomnist.measure import measure_batch

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

batch_size=128
latent_dim=100
img_size=28

from src.metrics import mse, compute_thickness_ground_truth
from .metrics import calculate_fid_given_paths, calculate_kid_given_paths
from src.generative_model import LeNet5

def create_labels_discriminator(batch_size, image_size, labels):
    #random_labels = np.random.rand(batch_size)
    gen_labels = np.zeros((batch_size, 1, image_size, image_size))
    for i in range(batch_size):
        gen_labels[i][0][:][:] = np.full((image_size, image_size), labels[i])
    return gen_labels

def save_numpy_arr(path, arr):
    np.save(path, arr)
    return path

def save_obj_csv(d, path):
    d.to_csv(path+'.csv', index=False)

def generate_sample(minimum, maximum, sample_size, generator):
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (sample_size, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.random.uniform(minimum, maximum, sample_size)
    labels = Variable(FloatTensor(labels))

    return generator(z, labels)

def compute_fid_kid_for_mnist(generator, n_row, img_size, dataset, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, sample_size):
    gen_img_in_distribution = generate_sample(0, dataset.maximum, sample_size, generator)
    gen_img_out_distribution = generate_sample(dataset.maximum, 1, sample_size, generator)

    random_id_in_distribution = random.sample(index_in_distribution.tolist(), sample_size)
    random_id_out_distribution = random.sample(index_out_distribution.tolist(), sample_size)
    real_imgs_in_distribution = real_dataset_in[random_id_in_distribution].numpy()
    real_imgs_out_distribution = real_dataset_out[random_id_out_distribution].numpy()

    folder = 'save_data'
    os.makedirs(folder, exist_ok=True)

    path_gen_in = save_numpy_arr(os.path.join(folder, 'gen_img_in_distribution.npy'), gen_img_in_distribution.cpu().detach().numpy())
    path_gen_out = save_numpy_arr(os.path.join(folder, 'gen_img_out_distribution.npy'), gen_img_out_distribution.cpu().detach().numpy())
    path_real_in = save_numpy_arr(os.path.join(folder, 'real_imgs_in_distribution.npy'), real_imgs_in_distribution)
    path_real_out = save_numpy_arr(os.path.join(folder, 'real_imgs_out_distribution.npy'), real_imgs_out_distribution)

    paths = [path_real_in, path_gen_in]
    fid_value_in_distribution = calculate_fid_given_paths(paths)
    kid_value_in_distribution = calculate_kid_given_paths(paths)

    paths = [path_real_out, path_gen_out]
    fid_value_out_distribution = calculate_fid_given_paths(paths)
    kid_value_out_distribution = calculate_kid_given_paths(paths)

    return fid_value_in_distribution, kid_value_in_distribution, fid_value_out_distribution, kid_value_out_distribution

def sample_image(n_row, batches_done, in_distribution_index, out_distribution_index, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator, dataset, sample_size):
    """Saves a grid of generated digits ranging from 0 to n_classes"""

    ## -------------- In distribution --------------
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in np.arange(0, 1, 1/n_row)])
    labels = Variable(FloatTensor(labels))

    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

    measure_batch = compute_thickness_ground_truth(gen_imgs.cpu().detach())
    thickness = measure_batch.values.reshape((n_row, n_row)).mean(axis=0)

    label_target = dataset.scaler.inverse_transform(np.array([num for num in np.arange(0, 1, 1/n_row)]).reshape(-1,1)).squeeze()
    mse_generator = mse(label_target, thickness)

    fid_value_in_distribution, kid_value_in_distribution, fid_value_out_distribution, kid_value_out_distribution  = compute_fid_kid_for_mnist(generator, n_row, img_size, dataset, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, sample_size)

    print()
    print(f"The thickness distribution =\n{dataset.scaler.transform(thickness.reshape(-1,1)).squeeze()}")
    print(f"Average MSE In dist = {np.mean(mse_generator[in_distribution_index])} \ Average MSE Out dist = {np.mean(mse_generator[out_distribution_index])}")
    print()
    print(f"FID score in distribution : mean = {np.around(fid_value_in_distribution[0], decimals=4)} \ std = {np.around(fid_value_in_distribution[1], decimals=4)}")
    print(f"FID score out distribution : mean = {np.around(fid_value_out_distribution[0], decimals=4)} \ std = {np.around(fid_value_out_distribution[1], decimals=4)}")
    print()
    print(f"KID score in distribution : mean = {np.around(kid_value_in_distribution[0], decimals=4)} \ std = {np.around(kid_value_in_distribution[1], decimals=4)}")
    print(f"KID score out distribution : mean = {np.around(kid_value_out_distribution[0], decimals=4)} \ std = {np.around(kid_value_out_distribution[1], decimals=4)}")

    return mse_generator, fid_value_in_distribution, kid_value_in_distribution, fid_value_out_distribution, kid_value_out_distribution

def save_model_check(dist, df_check, mean_out, best_res, df_acc_gen, path_generator, generator):
    if df_check is not None:
        if mean_out < df_check[f'mse_{dist}'].iloc[-1]:
            print(f" ---------- Better Results {dist} distribution of : {df_check[f'mse_{dist}'].iloc[-1] - mean_out} ---------- ")
            torch.save(generator, os.path.join(path_generator, f"best_generator_{dist}_distribution.pth"))
            save_obj_csv(df_acc_gen, os.path.join(path_generator, f"results_{dist}_distribution"))

            best_res = mean_out
            df_check = None

    else:
        if mean_out < best_res:
            print(f" ---------- Model Improving {dist} distribution of : {best_res - mean_out}---------- ")
            torch.save(generator, os.path.join(path_generator, f"best_generator_{dist}_distribution.pth"))
            save_obj_csv(df_acc_gen, os.path.join(path_generator, f"results_{dist}_distribution"))

            best_res = mean_out

    return df_check, best_res

def check_memory_cuda():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    #print(f'total    : {info.total}')
    print(f' --------------- MEMORY free     : {info.free} --------------- ')
    #print(f'used     : {info.used}')

def train_gan_model(dataloader, path_generator):

    mse_gan_in_distribution = []
    mse_gan_out_distribution = []
    df_acc_gen = pd.DataFrame(columns=['mse_in', 'mse_out', 'fid_in', 'fid_out'])

    if os.path.exists(path_generator):
        df_check_in_distribution = load_obj_csv(os.path.join(path_generator, 'results_in_distribution'))
        df_check_out_distribution = load_obj_csv(os.path.join(path_generator, 'results_out_distribution'))
    else:
        os.makedirs(path_generator)
        df_check_in_distribution = None
        df_check_out_distribution = None

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()


    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if os.path.isdir("images"):
        shutil.rmtree("images")
    os.makedirs("images", exist_ok=True)

    # FID needs
    df_test = pd.DataFrame(np.around(fulldataset.y_data.numpy(),2), columns=['label'])
    index_in_distribution = df_test[df_test['label']<=dataset.maximum].index
    index_out_distribution = df_test[df_test['label']>dataset.maximum].index
    real_dataset = deepcopy(fulldataset.x_data)

    arr = np.array([num for num in np.arange(0, 1, 1/n_row)])
    print(f"Checkup the plots of the displayed labels {arr}")
    in_distribution_index = np.where(arr <= dataset.maximum)
    out_distribution_index = np.where(arr > dataset.maximum)

    best_res_in = 100000
    best_res_out = 100000

    for epoch in range(n_epochs):
        d_loss_check = []
        g_loss_check = []

        if (epoch+1) % 50 == 0:
            optimizer_G.param_groups[0]['lr'] /= 10
            optimizer_D.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        for i, (imgs, labels) in enumerate(dataloader):
            ## Initialization
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(FloatTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(FloatTensor(np.random.rand(batch_size))) 

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            g_loss_check.append(g_loss.item())

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            d_loss_check.append(d_loss.item())

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:

                print(
                  "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                  % (epoch, n_epochs, i, len(dataloader), np.mean(d_loss_check), np.mean(g_loss_check))
                )

                # Delete useless data from GPU
                del valid; del fake; del real_imgs; del labels; del z; del gen_labels; del g_loss; del d_loss; del gen_imgs; del validity;
                torch.cuda.empty_cache()

                mse_gan, fid_in, kid_in, fid_out, kid_out = sample_image(n_row, batches_done, in_distribution_index, out_distribution_index, index_in_distribution, index_out_distribution, generator, dataset, real_dataset, 1000)

                mean_in_mse = np.mean(mse_gan[in_distribution_index])
                mean_out_mse = np.mean(mse_gan[out_distribution_index])

                mse_gan_in_distribution.append(mse_gan[in_distribution_index])
                mse_gan_out_distribution.append(mse_gan[out_distribution_index])

                df = pd.DataFrame([mean_in_mse], columns=['mse_in'])
                df['mse_out'] = mean_out_mse
                df['fid_in'] = fid_in[0]
                df['fid_out'] = fid_out[0]
                df['kid_in'] = kid_in[0]
                df['kid_out'] = kid_out[0]

                df_acc_gen = df_acc_gen.append(df, ignore_index=True)

                # Check if we have better results
                df_check_in_distribution, best_res_in = save_model_check('in', df_check_in_distribution, df['mse_in'].values, best_res_in, df_acc_gen, path_generator, generator)
                df_check_out_distribution, best_res_out = save_model_check('out', df_check_out_distribution, df['mse_out'].values, best_res_out, df_acc_gen, path_generator, generator)

    return mse_gan_in_distribution, mse_gan_out_distribution, df_acc_gen

def eval_forward(dist, df_check, mean_out, best_res, df_acc_gen, path_forward, model):
    if df_check is not None:
        if mean_out < df_check[f'mse_{dist}'].iloc[-1]:
            print(f" ---------- Better Results {dist} distribution of : {df_check[f'mse_{dist}'].iloc[-1] - mean_out} ---------- ")
            torch.save(model, os.path.join(path_forward, f"best_forward_{dist}_distribution.pth"))
            save_obj_csv(df_acc_gen, os.path.join(path_forward, f"results_{dist}_distribution"))

            best_res = mean_out
            df_check = None

    else:
        if mean_out < best_res:
            print(f" ---------- Model Improving {dist} distribution of : {best_res - mean_out}---------- ")
            torch.save(model, os.path.join(path_forward, f"best_forward_{dist}_distribution.pth"))
            save_obj_csv(df_acc_gen, os.path.join(path_forward, f"results_{dist}_distribution"))

            best_res = mean_out

    return df_check, best_res