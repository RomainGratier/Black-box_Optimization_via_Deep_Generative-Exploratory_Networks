import numpy as np
import pandas as pd
import os
import random
import shutil
from copy import deepcopy
import imageio
from glob import glob

import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.autograd as autograd
import torchvision
import matplotlib.pyplot as plt

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

from src.data import MNISTDataset, RotationDataset
from src.metrics import se, compute_thickness_ground_truth
from src.generative_model.metrics import calculate_fid_given_paths, calculate_kid_given_paths
from src.generative_model import Generator, Discriminator, LeNet5, CondDCDiscriminator, CondDCGenerator

import src.config as cfg

if cfg.dcgan:
    import src.config_dcgan as cfgan
else:
    import src.config_gan as cfgan

if cfg.experiment == 'min_mnist':
    import src.config_min_mnist as cfg_data
elif cfg.experiment == 'max_mnist':
    import src.config_max_mnist as cfg_data
elif cfg.experiment == 'rotation_dataset':
    import src.config_rotation as cfg_data

from src.utils import save_ckp, load_ckp_gan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_memory_cuda():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    #print(f'total    : {info.total}')
    print(f' --------------- MEMORY free     : {info.free} --------------- ')
    #print(f'used     : {info.used}')

def load_obj_csv(path):
    return pd.read_csv(path+'.csv')

def save_obj_csv(d, path):
    d.to_csv(path+'.csv', index=False)

def save_numpy_arr(path, arr):
    np.save(path, arr)
    return path

def show(img, condition, batches_done, distribution):
    image_path_folder = os.path.join(cfg.models_path, cfg.gan_path, f'images_{distribution}')
    os.makedirs(image_path_folder, exist_ok=True)

    npimg = img.detach().cpu().numpy()
    plt.figure(figsize=(10,10), dpi=100)
    title_str = ''
    for i, num in enumerate(condition):
        if i+1 == len(condition):
            title_str += str(num)
        else:
            title_str += str(num)+'|'

    if cfg.experiment == 'rotation_dataset':
        plt.title(f'{title_str}', fontsize=18)
    elif (cfg.experiment == 'min_mnist')|(cfg.experiment == 'max_mnist'):
        plt.title(f'{title_str}', fontsize=22.5)
    plt.xlabel(f"Batch number {batches_done}", fontsize=25)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig(os.path.join(image_path_folder, f'{batches_done}.png'))
    plt.close()

def create_gif_training(path_generator, frame_duration=0.5):
    image_path_folders = os.path.join(cfg.models_path, cfg.gan_path)
    for distribution in ['in', 'out']:
        image_folder = os.path.join(image_path_folders, f'images_{distribution}')
        filenames = sorted(glob(image_folder+'/*.png'), key=os.path.getctime)
        with imageio.get_writer(os.path.join(path_generator, f'gan_training_iterations_{distribution}.gif'), mode='I', duration=frame_duration) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

def generate_sample_from_z_condition(generator, z, condition):
    if cfg.dcgan:
        # Sample noise
        z = Variable(FloatTensor(z))
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        # Get labels ranging from 0 to n_classes for n rows
        condition = Variable(FloatTensor(condition))
        condition = condition.view(-1, 1, 1, 1)

        img_gen = generator(z, condition)

    else:
        # Sample noise
        z = Variable(FloatTensor(z))
        # Get labels ranging from 0 to n_classes for n rows
        condition = Variable(FloatTensor(condition))

        img_gen = generator(z, condition)

    return img_gen

def generate_sample_from_uniform_condition_gaussian_latent(minimum, maximum, sample_size, generator):
    # Sample noise
    z = np.random.normal(0, 1, (sample_size, cfgan.latent_dim))
    # Get labels ranging from 0 to n_classes for n rows
    condition = np.random.uniform(minimum, maximum, sample_size)
    return generate_sample_from_z_condition(generator, z, condition)

def compute_fid_kid_for_mnist(generator, n_row, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, sample_size):

    if cfg.experiment == 'min_mnist':
        gen_img_in_distribution = generate_sample_from_uniform_condition_gaussian_latent(cfg_data.limit_dataset, cfg_data.max_dataset, sample_size, generator)
        gen_img_out_distribution = generate_sample_from_uniform_condition_gaussian_latent(cfg_data.min_dataset, cfg_data.limit_dataset, sample_size, generator)
    elif (cfg.experiment == 'max_mnist') | (cfg.experiment == 'rotation_dataset'):
        gen_img_in_distribution = generate_sample_from_uniform_condition_gaussian_latent(cfg_data.min_dataset, cfg_data.limit_dataset, sample_size, generator)
        gen_img_out_distribution = generate_sample_from_uniform_condition_gaussian_latent(cfg_data.limit_dataset, cfg_data.max_dataset, sample_size, generator)

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

def sample_image_mnist(n_row, batches_done, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator, sample_size):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, cfgan.latent_dim))))
    
    if experiment == 'max_dataset':

        # ---------------------- In distribution sample ----------------------
        condition_in = np.array([num for _ in range(n_row) for num in np.linspace(cfg_data.min_dataset, cfg_data.limit_dataset, 10, endpoint=True)])
        gen_imgs_in = generate_sample_from_z_condition(generator, z, condition_in)

        grid_in = torchvision.utils.make_grid(gen_imgs_in.data, nrow=n_row)
        title = [round(num, 2) for num in np.linspace(cfg_data.min_dataset, cfg_data.limit_dataset, 10, endpoint=True)]
        show(grid_in, title, batches_done, 'in')

        measure_batch = compute_thickness_ground_truth(gen_imgs_in.cpu().detach().squeeze(1))
        thickness = measure_batch.values.reshape((n_row, n_row)).mean(axis=0)

        label_target = np.array([num for num in np.linspace(cfg_data.min_dataset, cfg_data.limit_dataset, 10, endpoint=True)])
        se_generator_in = se(label_target, thickness)

        # ---------------------- Out distribution sample ----------------------
        condition_out = np.array([num for _ in range(n_row) for num in np.linspace(cfg_data.limit_dataset, cfg_data.max_dataset, 10, endpoint=True)])
        gen_imgs_out = generate_sample_from_z_condition(generator, z, condition_out)

        grid_out = torchvision.utils.make_grid(gen_imgs_out.data, nrow=n_row)
        title = [round(num, 2) for num in np.linspace(cfg_data.limit_dataset, cfg_data.max_dataset, 10, endpoint=True)]
        show(grid_out, title, batches_done, 'out')

        measure_batch = compute_thickness_ground_truth(gen_imgs_out.cpu().detach().squeeze(1))
        thickness = measure_batch.values.reshape((n_row, n_row)).mean(axis=0)

        label_target = np.array([num for num in np.linspace(cfg_data.limit_dataset, cfg_data.max_dataset, 10, endpoint=True)])
        se_generator_out = se(label_target, thickness)

    elif experiment == 'min_dataset':

        # ---------------------- In distribution sample ----------------------
        condition_in = np.array([num for _ in range(n_row) for num in np.linspace(cfg_data.limit_dataset, cfg_data.max_dataset, 10, endpoint=True)])
        gen_imgs_in = generate_sample_from_z_condition(generator, z, condition_in)

        grid_in = torchvision.utils.make_grid(gen_imgs_in.data, nrow=n_row)
        title = [round(num, 2) for num in np.linspace(cfg_data.limit_dataset, cfg_data.max_dataset, 10, endpoint=True)]
        show(grid_in, title, batches_done, 'in')

        measure_batch = compute_thickness_ground_truth(gen_imgs_in.cpu().detach().squeeze(1))
        thickness = measure_batch.values.reshape((n_row, n_row)).mean(axis=0)

        label_target = np.array([num for num in np.linspace(cfg_data.limit_dataset, cfg_data.max_dataset, 10, endpoint=True)])
        se_generator_in = se(label_target, thickness)

        # ---------------------- Out distribution sample ----------------------
        condition_out = np.array([num for _ in range(n_row) for num in np.linspace(cfg_data.min_dataset, cfg_data.limit_dataset, 10, endpoint=True)])
        gen_imgs_out = generate_sample_from_z_condition(generator, z, condition_out)

        grid_out = torchvision.utils.make_grid(gen_imgs_out.data, nrow=n_row)
        title = [round(num, 2) for num in np.linspace(cfg_data.min_dataset, cfg_data.limit_dataset, 10, endpoint=True)]
        show(grid_out, title, batches_done, 'out')

        measure_batch = compute_thickness_ground_truth(gen_imgs_out.cpu().detach().squeeze(1))
        thickness = measure_batch.values.reshape((n_row, n_row)).mean(axis=0)

        label_target = np.array([num for num in np.linspace(cfg_data.min_dataset, cfg_data.limit_dataset, 10, endpoint=True)])
        se_generator_out = se(label_target, thickness)

    fid_value_in_distribution, kid_value_in_distribution, fid_value_out_distribution, kid_value_out_distribution  = compute_fid_kid_for_mnist(generator, n_row, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, sample_size)

    print()
    print(f"The thickness distribution =\n{thickness}")
    print(f"Average MSE In dist = {np.mean(se_generator_in)} \ Average MSE Out dist = {np.mean(se_generator_out)}")
    print()
    print(f"FID score in distribution : mean = {np.around(fid_value_in_distribution[0], decimals=4)} \ std = {np.around(fid_value_in_distribution[1], decimals=4)}")
    print(f"FID score out distribution : mean = {np.around(fid_value_out_distribution[0], decimals=4)} \ std = {np.around(fid_value_out_distribution[1], decimals=4)}")
    print()
    print(f"KID score in distribution : mean = {np.around(kid_value_in_distribution[0], decimals=4)} \ std = {np.around(kid_value_in_distribution[1], decimals=4)}")
    print(f"KID score out distribution : mean = {np.around(kid_value_out_distribution[0], decimals=4)} \ std = {np.around(kid_value_out_distribution[1], decimals=4)}")

    return se_generator_in, se_generator_out, fid_value_in_distribution, kid_value_in_distribution, fid_value_out_distribution, kid_value_out_distribution

def sample_image_rotation(n_row, batches_done, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator, sample_size):
    """Saves a grid of generated digits ranging from 0 to n_classes"""

    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, cfgan.latent_dim))))

    # ---------------------- In distribution sample ----------------------
    condition_in = np.array([num for _ in range(n_row) for num in np.linspace(cfg_data.min_dataset, cfg_data.limit_dataset, 10, endpoint=True)])
    gen_imgs_in = generate_sample_from_z_condition(generator, z, condition_in)

    grid_in = torchvision.utils.make_grid(gen_imgs_in.data, nrow=n_row)
    title = [round(num, 2) for num in np.linspace(cfg_data.min_dataset, cfg_data.limit_dataset, 10, endpoint=True)]
    show(grid_in, title, batches_done, 'in')

    # ---------------------- Out distribution sample ----------------------
    condition_out = np.array([num for _ in range(n_row) for num in np.linspace(cfg_data.limit_dataset, cfg_data.max_dataset, 10, endpoint=True)])
    gen_imgs_out = generate_sample_from_z_condition(generator, z, condition_out)

    grid_out = torchvision.utils.make_grid(gen_imgs_out.data, nrow=n_row)
    title = [round(num, 2) for num in np.linspace(cfg_data.limit_dataset, cfg_data.max_dataset, 10, endpoint=True)]
    show(grid_out, title, batches_done, 'out')

    fid_value_in_distribution, kid_value_in_distribution, fid_value_out_distribution, kid_value_out_distribution  = compute_fid_kid_for_mnist(generator, n_row, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, sample_size)

    print()
    print(f"FID score in distribution : mean = {np.around(fid_value_in_distribution[0], decimals=4)} \ std = {np.around(fid_value_in_distribution[1], decimals=4)}")
    print(f"FID score out distribution : mean = {np.around(fid_value_out_distribution[0], decimals=4)} \ std = {np.around(fid_value_out_distribution[1], decimals=4)}")
    print()
    print(f"KID score in distribution : mean = {np.around(kid_value_in_distribution[0], decimals=4)} \ std = {np.around(kid_value_in_distribution[1], decimals=4)}")
    print(f"KID score out distribution : mean = {np.around(kid_value_out_distribution[0], decimals=4)} \ std = {np.around(kid_value_out_distribution[1], decimals=4)}")

    return fid_value_in_distribution, kid_value_in_distribution, fid_value_out_distribution, kid_value_out_distribution

def save_model_check(dist, mean_out, best_res, path_generator, generator):
    os.makedirs(path_generator, exist_ok=True)
    if mean_out < best_res:
        print(f" ---------- Model Improving {dist} distribution of : {best_res - mean_out}---------- ")
        if cuda:
            torch.save(generator.cpu(), os.path.join(path_generator, f"best_generator_{dist}_distribution.pth"))
            generator.cuda()
        else:
            torch.save(generator, os.path.join(path_generator, f"best_generator_{dist}_distribution.pth"))
        best_res = mean_out
    return best_res

def get_main_data():
    
    path_generator = os.path.join(cfg.models_path, cfg.gan_path)

    if (cfg.experiment == 'max_mnist') | (cfg.experiment == 'min_mnist'):
        dataset = MNISTDataset('train', 
                               y_feature=cfg.feature, 
                               folder=cfg.data_folder, 
                               data_path=cfg.data_path)
        testset = MNISTDataset('full', 
                               y_feature=cfg.feature, 
                               folder=cfg.data_folder, 
                               data_path=cfg.data_path)
    elif cfg.experiment == 'rotation_dataset':
        dataset = RotationDataset('train', 
                                  folder=cfg.data_folder, 
                                  data_path=cfg.data_path)
        testset = RotationDataset('full', 
                                  folder=cfg.data_folder, 
                                  data_path=cfg.data_path)

    dataloader = DataLoader(dataset=dataset,
                              batch_size=cfgan.batch_size,
                              shuffle=True,
                              num_workers=4)

    # FID needs
    if cfg.experiment == 'min_mnist':
        # FID needs
        df_test_in = pd.DataFrame(testset.y_data, columns=['label'])
        df_test_out = pd.DataFrame(testset.y_data, columns=['label'])
        index_in_distribution = df_test_in[df_test_in['label'] > cfg_data.limit_dataset].index
        index_out_distribution = df_test_out[df_test_out['label'] <= cfg_data.limit_dataset].index
        print(f'size of in distribution data for fid/kid : {len(index_in_distribution)}')
        print(f'size of out distribution data for fid/kid : {len(index_out_distribution)}')
        real_dataset_in = deepcopy(testset.x_data)
        real_dataset_out = deepcopy(testset.x_data)
    
    elif (cfg.experiment == 'max_mnist') | (cfg.experiment == 'rotation_dataset'):
        df_test_in = pd.DataFrame(testset.y_data, columns=['label'])
        df_test_out = pd.DataFrame(testset.y_data, columns=['label'])
        index_in_distribution = df_test_in[df_test_in['label'] <= cfg_data.limit_dataset].index
        index_out_distribution = df_test_out[df_test_out['label'] > cfg_data.limit_dataset].index
        print(f'size of in distribution data for fid/kid : {len(index_in_distribution)}')
        print(f'size of out distribution data for fid/kid : {len(index_out_distribution)}')
        real_dataset_in = deepcopy(testset.x_data)
        real_dataset_out = deepcopy(testset.x_data)

    return dataset, dataloader, testset, index_in_distribution, index_out_distribution, real_dataset_in, real_dataset_out

def compute_and_store_results(df_acc_gen, epoch, lgt_dataloader, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator):
    if cfg.experiment == 'rotation_dataset':
        fid_in, kid_in, fid_out, kid_out = sample_image_rotation(cfgan.n_row, epoch, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator, cfgan.fid_kid_sample)

        df = pd.DataFrame([fid_in[0]], columns=['fid_in'])
        df['fid_out'] = fid_out[0]
        df['kid_in'] = kid_in[0]
        df['kid_out'] = kid_out[0]
        df['iteration'] = epoch*cfgan.batch_size*lgt_dataloader
        df['flag'] = False

        df_acc_gen = df_acc_gen.append(df, ignore_index=True)
    
    elif (cfg.experiment == 'max_mnist') | (cfg.experiment == 'min_mnist'):

        se_gan_in, se_gan_out, fid_in, kid_in, fid_out, kid_out = sample_image_mnist(cfgan.n_row, epoch, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator, cfgan.fid_kid_sample)
    
        df = pd.DataFrame([np.mean(se_gan_in)], columns=['mse_in'])
        df['mse_out'] = np.mean(se_gan_out)
        df['fid_in'] = fid_in[0]
        df['fid_out'] = fid_out[0]
        df['kid_in'] = kid_in[0]
        df['kid_out'] = kid_out[0]
        df['iteration'] = epoch*cfgan.batch_size*lgt_dataloader
        df['flag'] = False
    
        df_acc_gen = df_acc_gen.append(df, ignore_index=True)
        
    return df_acc_gen, df

def train_gan_model():
    
    dataset, dataloader, testset, index_in_distribution, index_out_distribution, real_dataset_in, real_dataset_out = get_main_data()
    path_generator = os.path.join(cfg.models_path, cfg.gan_path)

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    if cfg.dcgan:
        generator = CondDCGenerator()
        discriminator = CondDCDiscriminator()
    else:
        generator = Generator()
        discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfgan.lr, betas=(cfgan.b1, cfgan.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfgan.lr, betas=(cfgan.b1, cfgan.b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    
    df_acc_gen = pd.DataFrame(columns=['mse_in', 'mse_out', 'fid_in', 'fid_out', 'iteration', 'flag'])
    best_res_in = 100000
    best_res_out = 100000
    if cfg.experiment == 'rotation':
        stop_check = 'epoch'
    else:
        stop_check = 'fid'
    epoch_start = 0
    ckp_path = os.path.join(cfg.models_path,'checkpoints/gan')
    print(ckp_path)

    if os.path.isdir(ckp_path):
        print(F'Found a valid check point !')
        print("Do you want to resume the training? yes or no")
        resume = str(input())
        if resume == 'yes':
            generator, discriminator, optimizer_G, optimizer_D, epoch_start, df_acc_gen, best_res_in, best_res_out = load_ckp_gan(ckp_path, generator, discriminator, optimizer_G, optimizer_D)

    for epoch in range(epoch_start, cfgan.n_epochs):
        d_loss_check = []
        g_loss_check = []

        if (epoch+1) % 50 == 0:
            optimizer_G.param_groups[0]['lr'] /= 10
            optimizer_D.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = labels.shape[0]

            # Configure input
            if cfg.dcgan:
                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Inputs
                real_imgs = Variable(imgs.type(FloatTensor))
                labels = Variable(labels.type(FloatTensor)).view(-1,1,1,1)

                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, cfgan.latent_dim, 1, 1))))
                gen_labels = Variable(FloatTensor(np.random.rand(batch_size, 1, 1, 1)*cfg_data.max_dataset))

                gen_labels_discriminator = torch.zeros(batch_size, cfg.channels, cfg.img_size, cfg.img_size)
                gen_labels_discriminator[:,:,:,:] = gen_labels[:,:,:,:]
                labels_discriminator =  torch.zeros_like(gen_labels_discriminator)
                labels_discriminator[:,:,:,:] = labels[:,:,:,:]

                gen_labels_discriminator = Variable(gen_labels_discriminator.type(FloatTensor))
                labels_discriminator = Variable(labels_discriminator.type(FloatTensor))

            else:
                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Inputs
                real_imgs = Variable(imgs.type(FloatTensor))
                labels = Variable(labels.type(FloatTensor))

                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, cfgan.latent_dim))))
                gen_labels = Variable(FloatTensor(np.random.rand(batch_size)*cfg_data.max_dataset)) 
                gen_labels_discriminator = gen_labels
                labels_discriminator = labels

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()
            
            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels_discriminator)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            g_loss_check.append(g_loss.item())

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels_discriminator)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels_discriminator)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            d_loss_check.append(d_loss.item())

        if epoch == 0:
            pass

        elif epoch % 1 == 0:

            print(
              "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
              % (epoch, cfgan.n_epochs, i, len(dataloader), np.mean(d_loss_check), np.mean(g_loss_check))
            )

            df_acc_gen, df = compute_and_store_results(df_acc_gen, epoch, len(dataloader), real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator)

            # Check if we have better results
            test_in = best_res_in
            test_out = best_res_out

            if stop_check == 'fid':
                best_res_in = save_model_check('in', df['fid_in'].values, best_res_in, path_generator, generator)
                best_res_out = save_model_check('out', df['fid_out'].values, best_res_out, path_generator, generator)

                if (test_in!=best_res_in)|(test_out!=best_res_out):
                    df_acc_gen['flag'] = True

            elif stop_check == 'epoch':
                print(f'Model is saved')
                if cuda:
                    torch.save(generator.cpu(), os.path.join(path_generator, f"best_generator_in_distribution.pth"))
                    torch.save(generator.cpu(), os.path.join(path_generator, f"best_generator_out_distribution.pth"))
                    generator.cuda()
                else:
                    torch.save(generator, os.path.join(path_generator, f"best_generator_in_distribution.pth"))
                    torch.save(generator, os.path.join(path_generator, f"best_generator_out_distribution.pth"))

            save_obj_csv(df_acc_gen, os.path.join(path_generator, f"results_gan"))
            save_ckp({
                'epoch': epoch + 1,
                'state_dict_generator': generator.state_dict(),
                'state_dict_discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'df_acc_gen': df_acc_gen,
                'best_res_in': best_res_in,
                'best_res_out': best_res_out,
            }, ckp_path)
    # Create gifs of training
    create_gif_training(path_generator)

def compute_gradient_penalty(discriminator, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    labels = FloatTensor(labels)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates, labels).view(-1,1)
    fake = FloatTensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgan_model():

    dataset, dataloader, testset, index_in_distribution, index_out_distribution, real_dataset_in, real_dataset_out = get_main_data()
    path_generator = os.path.join(cfg.models_path, cfg.gan_path)

    # Initialize generator and discriminator
    if cfg.dcgan:
        generator = CondDCGenerator()
        discriminator = CondDCDiscriminator()
    else:
        generator = Generator()
        discriminator = Discriminator()

    lambda_gp = cfg.lambda_gp

    if cuda:
        generator.cuda()
        discriminator.cuda()
        torch.tensor(lambda_gp).cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfgan.lr, betas=(cfgan.b1, cfgan.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfgan.lr, betas=(cfgan.b1, cfgan.b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    df_acc_gen = pd.DataFrame(columns=['mse_in', 'mse_out', 'fid_in', 'fid_out', 'iteration', 'flag'])
    best_res_in = 100000
    best_res_out = 100000
    if cfg.experiment == 'rotation':
        stop_check = 'epoch'
    else:
        stop_check = 'fid'
    epoch_start = 0
    ckp_path = os.path.join(cfg.models_path,'checkpoints/gan')
    print(ckp_path)

    if os.path.isdir(ckp_path):
        print(F'Found a valid check point !')
        print("Do you want to resume the training? yes or no")
        resume = str(input())
        if resume == 'yes':
            generator, discriminator, optimizer_G, optimizer_D, epoch_start, df_acc_gen, best_res_in, best_res_out = load_ckp_gan(ckp_path, generator, discriminator, optimizer_G, optimizer_D)

    for epoch in range(epoch_start, cfgan.n_epochs):
        d_loss_check = []
        g_loss_check = []

        if (epoch+1) % 50 == 0:
            optimizer_G.param_groups[0]['lr'] /= 10
            optimizer_D.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = labels.shape[0]

            # ---------------------
            #  Configuration
            # ---------------------

            if cfg.dcgan:
                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # input
                real_imgs = Variable(imgs.type(FloatTensor))
                labels = Variable(labels.type(FloatTensor)).view(-1,1,1,1)

                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, cfgan.latent_dim, 1, 1))))

                labels_discriminator = torch.zeros(batch_size, cfg.channels, cfg.img_size, cfg.img_size)
                labels_discriminator[:,:,:,:] =  Variable(labels[:,:,:,:]).type(FloatTensor)
                labels_discriminator = Variable(labels_discriminator.type(FloatTensor))

            else:
                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # input
                real_imgs = Variable(imgs.type(FloatTensor))
                labels = Variable(labels.type(FloatTensor))

                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, cfgan.latent_dim))))
                labels_discriminator = labels

            # Generate a batch of images
            fake_imgs = generator(z, labels)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels_discriminator)

            # Loss for fake images
            validity_fake = discriminator(fake_imgs.detach(), labels_discriminator)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels_discriminator.data)

            # Adversarial loss
            d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            d_loss_check.append(d_loss.item())

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            if i % cfg.n_critic == 0:

                # Loss measures generator's ability to fool the discriminator
                fake_validity = discriminator(fake_imgs, labels_discriminator)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()

                g_loss_check.append(g_loss.item())

        if epoch == 0:
            pass

        elif epoch % 1 == 0:

            print(
              "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
              % (epoch, cfgan.n_epochs, i, len(dataloader), np.mean(d_loss_check), np.mean(g_loss_check))
            )

            df_acc_gen, df = compute_and_store_results(df_acc_gen, epoch, len(dataloader), real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator)

            # Check if we have better results
            test_in = best_res_in
            test_out = best_res_out

            if stop_check == 'fid':
                best_res_in = save_model_check('in', df['fid_in'].values, best_res_in, path_generator, generator)
                best_res_out = save_model_check('out', df['fid_out'].values, best_res_out, path_generator, generator)
                if (test_in!=best_res_in)|(test_out!=best_res_out):
                    df_acc_gen['flag'] = True

            elif stop_check == 'epoch':
                print(f'Model is saved')
                if cuda:
                    torch.save(generator.cpu(), os.path.join(path_generator, f"best_generator_in_distribution.pth"))
                    torch.save(generator.cpu(), os.path.join(path_generator, f"best_generator_out_distribution.pth"))
                    generator.cuda()
                else:
                    torch.save(generator, os.path.join(path_generator, f"best_generator_in_distribution.pth"))
                    torch.save(generator, os.path.join(path_generator, f"best_generator_out_distribution.pth"))

            save_obj_csv(df_acc_gen, os.path.join(path_generator, f"results_gan"))
            save_ckp({
                'epoch': epoch + 1,
                'state_dict_generator': generator.state_dict(),
                'state_dict_discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'df_acc_gen': df_acc_gen,
                'best_res_in': best_res_in,
                'best_res_out': best_res_out,
            }, ckp_path)
    # Create gifs of training
    create_gif_training(path_generator)