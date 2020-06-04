import numpy as np
import pandas as pd
import os
import random
import shutil
from copy import deepcopy

import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.autograd as autograd

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

from src.data import MNISTDataset, RotationDataset
from src.metrics import se, compute_thickness_ground_truth
from src.generative_model.metrics import calculate_fid_given_paths, calculate_kid_given_paths
from src.generative_model import Generator, Discriminator, LeNet5, CondDCDiscriminator, CondDCGenerator

import src.config_gan as cfgan
import src.config as cfg
if cfg.experiment == 'min_mnist':
    import src.config_min_mnist as cfg_data
elif cfg.experiment == 'max_mnist':
    import src.config_max_mnist as cfg_data
elif cfg.experiment == 'rotation_dataset':
    import src.config_rotation as cfg_data

from src.utils import save_ckp, load_ckp_gan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_numpy_arr(path, arr):
    np.save(path, arr)
    return path

def save_obj_csv(d, path):
    d.to_csv(path+'.csv', index=False)

def generate_sample(minimum, maximum, sample_size, generator):
    if cfg.dcgan:
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (sample_size, cfgan.latent_dim, 1, 1))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = Variable(FloatTensor(np.random.uniform(minimum, maximum, (sample_size, 1, 1, 1))))
        
        img_gen = generator(z, labels)
    else:
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (sample_size, cfgan.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = Variable(FloatTensor(np.random.uniform(minimum, maximum, sample_size)))
        
        img_gen = generator(z, labels)

    return img_gen

def compute_fid_kid_for_mnist(generator, n_row, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, sample_size):
    
    if cfg.experiment == 'min_mnist':
        gen_img_in_distribution = generate_sample(cfg_data.limit_dataset, cfg_data.max_dataset, sample_size, generator)
        gen_img_out_distribution = generate_sample(cfg_data.min_dataset, cfg_data.limit_dataset, sample_size, generator)
    elif (cfg.experiment == 'max_mnist') | (cfg.experiment == 'rotation_dataset'):
        gen_img_in_distribution = generate_sample(cfg_data.min_dataset, cfg_data.limit_dataset, sample_size, generator)
        gen_img_out_distribution = generate_sample(cfg_data.limit_dataset, cfg_data.max_dataset, sample_size, generator)

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

def sample_image_mnist(n_row, batches_done, in_distribution_index, out_distribution_index, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator, sample_size):
    """Saves a grid of generated digits ranging from 0 to n_classes"""

    ## -------------- In distribution --------------
    if cfg.dcgan:
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, cfgan.latent_dim, 1, 1))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = Variable(FloatTensor(np.array([num for _ in range(n_row) for num in np.linspace(cfg_data.min_dataset, cfg_data.max_dataset, 10, endpoint=True)]))).view(-1,1,1,1)
        gen_imgs = generator(z, labels)

    else:
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, cfgan.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = Variable(FloatTensor(np.array([num for _ in range(n_row) for num in np.linspace(cfg_data.min_dataset, cfg_data.max_dataset, 10, endpoint=True)])))
        gen_imgs = generator(z, labels)

    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

    measure_batch = compute_thickness_ground_truth(gen_imgs.cpu().detach().squeeze(1))
    thickness = measure_batch.values.reshape((n_row, n_row)).mean(axis=0)

    label_target = np.array([num for num in np.linspace(cfg_data.min_dataset, cfg_data.max_dataset, 10, endpoint=True)])
    se_generator = se(label_target, thickness)

    fid_value_in_distribution, kid_value_in_distribution, fid_value_out_distribution, kid_value_out_distribution  = compute_fid_kid_for_mnist(generator, n_row, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, sample_size)

    print()
    print(f"The thickness distribution =\n{thickness}")
    print(f"Average MSE In dist = {np.mean(se_generator[in_distribution_index])} \ Average MSE Out dist = {np.mean(se_generator[out_distribution_index])}")
    print()
    print(f"FID score in distribution : mean = {np.around(fid_value_in_distribution[0], decimals=4)} \ std = {np.around(fid_value_in_distribution[1], decimals=4)}")
    print(f"FID score out distribution : mean = {np.around(fid_value_out_distribution[0], decimals=4)} \ std = {np.around(fid_value_out_distribution[1], decimals=4)}")
    print()
    print(f"KID score in distribution : mean = {np.around(kid_value_in_distribution[0], decimals=4)} \ std = {np.around(kid_value_in_distribution[1], decimals=4)}")
    print(f"KID score out distribution : mean = {np.around(kid_value_out_distribution[0], decimals=4)} \ std = {np.around(kid_value_out_distribution[1], decimals=4)}")

    return se_generator, fid_value_in_distribution, kid_value_in_distribution, fid_value_out_distribution, kid_value_out_distribution

def sample_image_rotation(n_row, batches_done, in_distribution_index, out_distribution_index, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator, sample_size):
    """Saves a grid of generated digits ranging from 0 to n_classes"""

    ## -------------- In distribution --------------
    if cfg.dcgan:
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, cfgan.latent_dim, 1, 1))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = Variable(FloatTensor(np.array([num for _ in range(n_row) for num in np.linspace(cfg_data.min_dataset, cfg_data.max_dataset, 10, endpoint=True)]))).view(-1,1,1,1)
        gen_imgs = generator(z, labels)

    else:
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, cfgan.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = Variable(FloatTensor(np.array([num for _ in range(n_row) for num in np.linspace(cfg_data.min_dataset, cfg_data.max_dataset, 10, endpoint=True)])))
        gen_imgs = generator(z, labels)

    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

    fid_value_in_distribution, kid_value_in_distribution, fid_value_out_distribution, kid_value_out_distribution  = compute_fid_kid_for_mnist(generator, n_row, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, sample_size)

    print()
    print(f"FID score in distribution : mean = {np.around(fid_value_in_distribution[0], decimals=4)} \ std = {np.around(fid_value_in_distribution[1], decimals=4)}")
    print(f"FID score out distribution : mean = {np.around(fid_value_out_distribution[0], decimals=4)} \ std = {np.around(fid_value_out_distribution[1], decimals=4)}")
    print()
    print(f"KID score in distribution : mean = {np.around(kid_value_in_distribution[0], decimals=4)} \ std = {np.around(kid_value_in_distribution[1], decimals=4)}")
    print(f"KID score out distribution : mean = {np.around(kid_value_out_distribution[0], decimals=4)} \ std = {np.around(kid_value_out_distribution[1], decimals=4)}")

    return fid_value_in_distribution, kid_value_in_distribution, fid_value_out_distribution, kid_value_out_distribution

def save_model_check(dist, df_check, mean_out, best_res, df_acc_gen, path_generator, generator):
    if df_check is not None:
        if mean_out < df_check[f'fid_{dist}'].iloc[-1]:
            print(f" ---------- Better Results {dist} distribution of : {df_check[f'fid_{dist}'].iloc[-1] - mean_out} ---------- ")
            if cuda:
                torch.save(generator.cpu(), os.path.join(path_generator, f"best_generator_{dist}_distribution.pth"))
                generator.cuda()
            else:
                torch.save(generator, os.path.join(path_generator, f"best_generator_{dist}_distribution.pth"))

            best_res = mean_out
            df_check = None

    else:
        if mean_out < best_res:
            print(f" ---------- Model Improving {dist} distribution of : {best_res - mean_out}---------- ")
            if cuda:
                torch.save(generator.cpu(), os.path.join(path_generator, f"best_generator_{dist}_distribution.pth"))
                generator.cuda()
            else:
                torch.save(generator, os.path.join(path_generator, f"best_generator_{dist}_distribution.pth"))

            best_res = mean_out

    return df_check, best_res

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

def train_gan_model():
    
    path_generator = os.path.join(cfg.models_path, cfg.gan_path)

    if (cfg.experiment == 'max_mnist') | (cfg.experiment == 'min_mnist'):
        dataset = MNISTDataset('train', 'thickness', data_path=cfg.data_path)
        testset = MNISTDataset('full', 'thickness', data_path=cfg.data_path)
    elif cfg.experiment == 'rotation_dataset':
        dataset = RotationDataset('train', data_path=cfg.data_path)
        testset = RotationDataset('full', data_path=cfg.data_path)

    dataloader = DataLoader(dataset=dataset,
                              batch_size=cfgan.batch_size,
                              shuffle=True,
                              num_workers=4)
    
    se_gan_in_distribution = []
    se_gan_out_distribution = []
    df_acc_gen = pd.DataFrame(columns=['mse_in', 'mse_out', 'fid_in', 'fid_out', 'iteration', 'flag'])

    if os.path.exists(path_generator):
        df_check_in_distribution = None
        df_check_out_distribution = None
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
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfgan.lr, betas=(cfgan.b1, cfgan.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfgan.lr, betas=(cfgan.b1, cfgan.b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if os.path.isdir("images"):
        shutil.rmtree("images")
    os.makedirs("images", exist_ok=True)

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

        arr = np.around(np.array([num for num in np.linspace(cfg_data.min_dataset, cfg_data.max_dataset, 10, endpoint=True)]), decimals = 2)
        print(f"Checkup the plots of the displayed labels {arr}")
        in_distribution_index = np.where(arr > cfg_data.limit_dataset)
        out_distribution_index = np.where(arr <= cfg_data.limit_dataset)
    
    elif (cfg.experiment == 'max_mnist') | (cfg.experiment == 'rotation_dataset'):
        df_test_in = pd.DataFrame(testset.y_data, columns=['label'])
        df_test_out = pd.DataFrame(testset.y_data, columns=['label'])
        index_in_distribution = df_test_in[df_test_in['label'] <= cfg_data.limit_dataset].index
        index_out_distribution = df_test_out[df_test_out['label'] > cfg_data.limit_dataset].index
        print(f'size of in distribution data for fid/kid : {len(index_in_distribution)}')
        print(f'size of out distribution data for fid/kid : {len(index_out_distribution)}')
        real_dataset_in = deepcopy(testset.x_data)
        real_dataset_out = deepcopy(testset.x_data)

        arr = np.around(np.array([num for num in np.linspace(cfg_data.min_dataset, cfg_data.max_dataset, 10, endpoint=True)]), decimals = 2)
        print(f"Checkup the plots of the displayed labels {arr}")
        in_distribution_index = np.where(arr <= cfg_data.limit_dataset)
        out_distribution_index = np.where(arr > cfg_data.limit_dataset)

    best_res_in = 100000
    best_res_out = 100000
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

            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, cfgan.latent_dim))))
            gen_labels = Variable(FloatTensor(np.random.rand(batch_size)*cfg_data.max_dataset)) 
            
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

        if epoch == 0:
            pass

        elif epoch % 1 == 0:

            print(
              "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
              % (epoch, cfgan.n_epochs, i, len(dataloader), np.mean(d_loss_check), np.mean(g_loss_check))
            )

            # Delete useless data from GPU
            del valid; del fake; del real_imgs; del labels; del z; del gen_labels; del g_loss; del d_loss; del gen_imgs; del validity;
            torch.cuda.empty_cache()
            
            if cfg.experiment == 'rotation_dataset':
                fid_in, kid_in, fid_out, kid_out = sample_image_rotation(cfgan.n_row, batches_done, in_distribution_index, out_distribution_index, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator, cfgan.fid_kid_sample)

                df = pd.DataFrame([fid_in[0]], columns=['fid_in'])
                df['fid_out'] = fid_out[0]
                df['kid_in'] = kid_in[0]
                df['kid_out'] = kid_out[0]
                df['iteration'] = batches_done*cfgan.sample_interval
                df['flag'] = False

                df_acc_gen = df_acc_gen.append(df, ignore_index=True)
            
            elif (cfg.experiment == 'max_mnist') | (cfg.experiment == 'min_mnist'):

                se_gan, fid_in, kid_in, fid_out, kid_out = sample_image_mnist(cfgan.n_row, batches_done, in_distribution_index, out_distribution_index, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator, cfgan.fid_kid_sample)
    
                mean_in_se = np.mean(se_gan[in_distribution_index])
                mean_out_se = np.mean(se_gan[out_distribution_index])
    
                se_gan_in_distribution.append(se_gan[in_distribution_index])
                se_gan_out_distribution.append(se_gan[out_distribution_index])
    
                df = pd.DataFrame([mean_in_se], columns=['mse_in'])
                df['mse_out'] = mean_out_se
                df['fid_in'] = fid_in[0]
                df['fid_out'] = fid_out[0]
                df['kid_in'] = kid_in[0]
                df['kid_out'] = kid_out[0]
                df['iteration'] = batches_done*cfgan.sample_interval
                df['flag'] = False
    
                df_acc_gen = df_acc_gen.append(df, ignore_index=True)

            # Check if we have better results
            test_in = best_res_in
            test_out = best_res_out
            
            if stop_check == 'fid':
                df_check_in_distribution, best_res_in = save_model_check('in', df_check_in_distribution, df['fid_in'].values, best_res_in, df_acc_gen, path_generator, generator)
                df_check_out_distribution, best_res_out = save_model_check('out', df_check_out_distribution, df['fid_out'].values, best_res_out, df_acc_gen, path_generator, generator)
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
    
    path_generator = os.path.join(cfg.models_path, cfg.gan_path)

    if (cfg.experiment == 'max_mnist') | (cfg.experiment == 'min_mnist'):
        dataset = MNISTDataset('train', 'thickness', data_path=cfg.data_path)
        testset = MNISTDataset('full', 'thickness', data_path=cfg.data_path)
    elif cfg.experiment == 'rotation_dataset':
        dataset = RotationDataset('train', data_path=cfg.data_path)
        testset = RotationDataset('full', data_path=cfg.data_path)

    dataloader = DataLoader(dataset=dataset,
                              batch_size=cfgan.batch_size,
                              shuffle=True,
                              num_workers=4)
    
    se_gan_in_distribution = []
    se_gan_out_distribution = []
    df_acc_gen = pd.DataFrame(columns=['mse_in', 'mse_out', 'fid_in', 'fid_out', 'iteration', 'flag'])

    if os.path.exists(path_generator):
        df_check_in_distribution = None
        df_check_out_distribution = None
    else:
        os.makedirs(path_generator)
        df_check_in_distribution = None
        df_check_out_distribution = None

    # Initialize generator and discriminator
    generator = CondDCGenerator()
    discriminator = CondDCDiscriminator()

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

    if os.path.isdir("images"):
        shutil.rmtree("images")
    os.makedirs("images", exist_ok=True)

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

        arr = np.around(np.array([num for num in np.linspace(cfg_data.min_dataset, cfg_data.max_dataset, 10, endpoint=True)]), decimals = 2)
        print(f"Checkup the plots of the displayed labels {arr}")
        in_distribution_index = np.where(arr > cfg_data.limit_dataset)
        out_distribution_index = np.where(arr <= cfg_data.limit_dataset)
    
    elif (cfg.experiment == 'max_mnist') | (cfg.experiment == 'rotation_dataset'):
        df_test_in = pd.DataFrame(testset.y_data, columns=['label'])
        df_test_out = pd.DataFrame(testset.y_data, columns=['label'])
        index_in_distribution = df_test_in[df_test_in['label'] <= cfg_data.limit_dataset].index
        index_out_distribution = df_test_out[df_test_out['label'] > cfg_data.limit_dataset].index
        print(f'size of in distribution data for fid/kid : {len(index_in_distribution)}')
        print(f'size of out distribution data for fid/kid : {len(index_out_distribution)}')
        real_dataset_in = deepcopy(testset.x_data)
        real_dataset_out = deepcopy(testset.x_data)

        arr = np.around(np.array([num for num in np.linspace(cfg_data.min_dataset, cfg_data.max_dataset, 10, endpoint=True)]), decimals = 2)
        print(f"Checkup the plots of the displayed labels {arr}")
        in_distribution_index = np.where(arr <= cfg_data.limit_dataset)
        out_distribution_index = np.where(arr > cfg_data.limit_dataset)

    best_res_in = 100000
    best_res_out = 100000
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
            import time
            start_time = time.time()
            
            # ---------------------
            #  Configuration
            # ---------------------
            
            batch_size = labels.shape[0]
            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(FloatTensor)).view(-1,1,1,1)
            
            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, cfgan.latent_dim, 1, 1))))

            # Generate a batch of images
            fake_imgs = generator(z, labels)

            labels_discriminator = torch.zeros_like(fake_imgs)
            labels_discriminator[:,:,:,:] = labels[:,:,:,:]
            
            print("Conf time :--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            
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
            
            print("Discriminator time :--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

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
            
            print("Generator time :--- %s seconds ---" % (time.time() - start_time))

            batches_done = epoch * len(dataloader) + i

        elif epoch % 1 == 0:

            print(
              "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
              % (epoch, cfgan.n_epochs, i, len(dataloader), np.mean(d_loss_check), np.mean(g_loss_check))
            )

            # Delete useless data from GPU
            torch.cuda.empty_cache()
            
            if cfg.experiment == 'rotation_dataset':
                fid_in, kid_in, fid_out, kid_out = sample_image_rotation(cfgan.n_row, batches_done, in_distribution_index, out_distribution_index, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator, cfgan.fid_kid_sample)

                df = pd.DataFrame([fid_in[0]], columns=['fid_in'])
                df['fid_out'] = fid_out[0]
                df['kid_in'] = kid_in[0]
                df['kid_out'] = kid_out[0]
                df['iteration'] = batches_done*cfgan.sample_interval
                df['flag'] = False

                df_acc_gen = df_acc_gen.append(df, ignore_index=True)
            
            elif (cfg.experiment == 'max_mnist') | (cfg.experiment == 'min_mnist'):

                se_gan, fid_in, kid_in, fid_out, kid_out = sample_image_mnist(cfgan.n_row, batches_done, in_distribution_index, out_distribution_index, real_dataset_in, real_dataset_out, index_in_distribution, index_out_distribution, generator, cfgan.fid_kid_sample)
    
                mean_in_se = np.mean(se_gan[in_distribution_index])
                mean_out_se = np.mean(se_gan[out_distribution_index])
    
                se_gan_in_distribution.append(se_gan[in_distribution_index])
                se_gan_out_distribution.append(se_gan[out_distribution_index])
    
                df = pd.DataFrame([mean_in_se], columns=['mse_in'])
                df['mse_out'] = mean_out_se
                df['fid_in'] = fid_in[0]
                df['fid_out'] = fid_out[0]
                df['kid_in'] = kid_in[0]
                df['kid_out'] = kid_out[0]
                df['iteration'] = batches_done*cfgan.sample_interval
                df['flag'] = False
    
                df_acc_gen = df_acc_gen.append(df, ignore_index=True)

            # Check if we have better results
            test_in = best_res_in
            test_out = best_res_out
            
            if stop_check == 'fid':
                df_check_in_distribution, best_res_in = save_model_check('in', df_check_in_distribution, df['fid_in'].values, best_res_in, df_acc_gen, path_generator, generator)
                df_check_out_distribution, best_res_out = save_model_check('out', df_check_out_distribution, df['fid_out'].values, best_res_out, df_acc_gen, path_generator, generator)
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