from scipy.stats import truncnorm, norm
import random
import pandas as pd
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
import os 
from torch.nn import functional as F
from copy import deepcopy

from src.metrics import se, re, compute_thickness_ground_truth
from src.generative_model.metrics import calculate_fid_given_paths, calculate_kid_given_paths
from src.forward.uncertainty_estimation import get_uncertainty_per_batch

import src.config as cfg
import src.config_gan as cfgan
import src.config_inference as cfginf
from src.uncertainty_policy import uncertainty_selection

if cfg.experiment == 'min_mnist':
    import src.config_min_mnist as cfg_data
elif cfg.experiment == 'max_mnist':
    import src.config_max_mnist as cfg_data

import torch 
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def save_numpy_arr(path, arr):
    np.save(path, arr)
    return path

def get_truncated_normal(form, mean=0, sd=1, quant=0.8):
    upp = norm.ppf(quant, mean, sd)
    low = norm.ppf(1 - quant, mean, sd)
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(form)

def generate_sample_from_GAN(y_cond, z, generator):
    # Prepare labels
    z = Variable(FloatTensor(z))
    cond = Variable(FloatTensor(y_cond))

    if y_cond.shape[0] > 5000:
        X_gen = FloatTensor(torch.zeros((y_cond.shape[0], 1, cfg.img_size, cfg.img_size)))
        
        # Create chunk
    
    else:
        X_gen = generator(z, cond)

    return X_gen

def compute_fid_mnist(gen_img, index_distribution, real_dataset):
    random_id = random.sample(index_distribution.tolist(), gen_img.shape[0])
    real_imgs = real_dataset[random_id].numpy()

    folder = 'save_data'
    os.makedirs(folder, exist_ok=True)

    path_gen = save_numpy_arr(os.path.join(folder, 'gen_img.npy'), gen_img.cpu().detach().numpy())
    path_real = save_numpy_arr(os.path.join(folder, 'real_imgs.npy'), real_imgs)

    paths = [path_real, path_gen]
    fid_value = calculate_fid_given_paths(paths)
    kid_value = calculate_kid_given_paths(paths)

    return fid_value, kid_value

def compute_global_measures(distribution, images_generated, index_distribution, real_dataset, conditions, forward, size):

    # ------------ random sample ------------
    size = np.around(images_generated.shape[0] * cfginf.quantile_rate_uncertainty_policy , decimals=0)
    random_index = random.sample(np.arange(images_generated.shape[0]).tolist(), int(size))
    images_generated = images_generated[random_index]
    
    # ------------ Compute the forward predictions ------------
    try:
        y_pred = forward(F.interpolate(images_generated, size=32)).squeeze(1).cpu().detach().numpy()

    except:
        y_pred, epistemic, aleatoric = get_uncertainty_per_batch(forward, F.interpolate(images_generated, size=32), device)
    
    # ------------ Compute FID/KID from testset ------------
    fid_value_gen_glob, kid_value_gen_glob = compute_fid_mnist(images_generated, index_distribution, real_dataset)
    
    # ------------ Compare the forward model and the Measure from morphomnist ------------
    # Move variable to cpu
    images_generated = images_generated.squeeze(1).cpu().detach().numpy()
    thickness = compute_thickness_ground_truth(images_generated)
    
    # ------------ Compute the se between the target and the morpho measure predictions ------------
    se_measure_glob = se(thickness, y_pred)
    re_measure_glob = re(thickness, y_pred)
    
    print(f"{distribution} distribution GLOBAL results")
    print(f"The mean squared error : {np.mean(se_measure_glob)} \t The std of the squared error : {np.std(se_measure_glob)}")
    print(f"The mean relative error : {np.mean(re_measure_glob)} \t The std of the squared error : {np.std(re_measure_glob)}")
    print(f"Mean FID : {fid_value_gen_glob[0]} ± {fid_value_gen_glob[1]} \t Mean KID : {kid_value_gen_glob[0]} ± {kid_value_gen_glob[1]}")
    
    return se_measure_glob, re_measure_glob, fid_value_gen_glob, kid_value_gen_glob

def compute_policy_measures(distribution, images_generated, index_distribution, real_dataset, conditions, forward, size):
    
    # ------------ Compute the se between the target and the forward model predictions ------------
    try:
        y_pred = forward(F.interpolate(images_generated, size=32)).squeeze(1).cpu().detach().numpy()
        forward_pred = np.array(y_pred).T

    except:
        y_pred, epistemic, aleatoric = get_uncertainty_per_batch(forward, F.interpolate(images_generated, size=32), device)
 
        # ------------ Uncertainty policy ------------
        index_certain = uncertainty_selection(epistemic.squeeze())
        y_pred = y_pred[index_certain]
        epistemic = epistemic[index_certain]
        forward_pred = np.array([y_pred, epistemic.squeeze(1)]).T
        images_generated = images_generated[index_certain]
        conditions = conditions[index_certain]

    # ------------ Compute FID/KID from testset ------------
    fid_value_gen, kid_value_gen = compute_fid_mnist(images_generated, index_distribution, real_dataset)

    # ------------ Compare the forward model and the Measure from morphomnist ------------
    # Move variable to cpu
    images_generated = images_generated.squeeze(1).cpu().detach().numpy()
    thickness = compute_thickness_ground_truth(images_generated)

    # ------------ Compute the se between the target and the morpho measure predictions ------------
    se_measure = se(thickness, y_pred)
    re_measure = re(thickness, y_pred)

    print(f"{distribution} distribution POLICY results")
    print(f"The mean squared error : {np.mean(se_measure)} \t The std of the squared error : {np.std(se_measure)}")
    print(f"The mean relative error : {np.mean(re_measure)} \t The std of the squared error : {np.std(re_measure)}")
    print(f"Mean FID : {fid_value_gen[0]} ± {fid_value_gen[1]} \t Mean KID : {kid_value_gen[0]} ± {kid_value_gen[1]}")
    
    return se_measure, re_measure, fid_value_gen, kid_value_gen, forward_pred, images_generated

def plots_some_results(distribution, images_generated, conditions, forward_pred, testset, index_distribution, forward, fid_value_gen, kid_value_gen, nrow=4, ncol=8):
    
    # Gan img selection
    random_gan_index = random.sample(np.arange(images_generated.shape[0]).tolist(), ncol*nrow)
    size = images_generated.shape[0]
    images_generated = images_generated[random_gan_index]
    conditions = conditions[random_gan_index]
    forward_pred = forward_pred[random_gan_index]
    
    # Real img selection
    random_id = random.sample(index_distribution.tolist(), ncol*nrow)
    real_imgs = testset.x_data[random_id]
    labels = testset.y_data[random_id].numpy()

    try:
        real_pred = forward(F.interpolate(real_imgs.to(device), size=32)).squeeze(1).cpu().detach().numpy()
        bayesian=False
    except:
        real_pred, epistemic, aleatoric = get_uncertainty_per_batch(forward, F.interpolate(real_imgs, size=32), device)
        bayesian=True
    
    real_imgs = real_imgs.squeeze()

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(8,4), dpi=300)
    
    for i, row in enumerate(ax):
        if i < 2:
            imgs = images_generated
        else:
            imgs = real_imgs
        
        for j, col in enumerate(row):
            if i%2 == 0:
                k = 0
            else:
                k = ncol -1
            if i < 2:
                image = imgs[j + k]
                col.imshow(image)
                col.axis('off')
                if bayesian:
                    col.set_title(f"Fwd={np.round(float(forward_pred[j + k][0]),1)} / Cond={np.round(float(conditions[j + k]),1)} / epi={np.round(float(forward_pred[j + k][1]),4)}", fontsize=3) #/ true={np.round(float(morpho_pred[n_top_index[j + k]]),1)} 
                else:
                    col.set_title(f"Fwd={np.round(float(forward_pred[j + k]),1)} / Cond={np.round(float(conditions[j + k]),1)}", fontsize=3) #/ true={np.round(float(morpho_pred[n_top_index[j + k]]),1)} 
            else:
                image = imgs[j + k]
                col.imshow(image)
                col.axis('off')
                if bayesian:
                    col.set_title(f"Fwd={np.round(float(real_pred[j + k]),1)} / Label={np.round(float(labels[j + k]),1)} / epi={np.round(float(epistemic[j + k]),4)}", fontsize=3) #/ true={np.round(float(morpho_pred[n_top_index[j + k]]),1)} 
                else:
                    col.set_title(f"Fwd={np.round(float(forward_pred[j + k]),1)} / Label={np.round(float(labels[j + k]),1)}", fontsize=3) #/ true={np.round(float(morpho_pred[n_top_index[j]]),1)}
    
    plt.suptitle(f"{distribution} distribution / FID Value : {np.round(fid_value_gen[0])} ± {np.round(fid_value_gen[1]/np.sqrt(size))} \ KID Value : {np.around(kid_value_gen[0], decimals=3)}  ± {np.around(kid_value_gen[1]/np.sqrt(size), decimals=3)}", fontsize=6)
    plt.show()

def monte_carlo_inference_general(distribution, generator, forward, testset, ncol = 8, nrow =4, sample_number_fid_kid = 2000, size=2000):
    
    if distribution == 'in':
        if cfg.experiment == 'max_mnist':
            conditions = np.random.uniform(cfg_data.min_dataset, cfg_data.limit_dataset, size)
            df_test_in = pd.DataFrame(testset.y_data, columns=['label'])
            index_distribution = df_test_in[df_test_in['label'] <= cfg_data.limit_dataset].index
            print(f'size of in distribution data for fid/kid : {len(index_distribution)}')
            real_dataset = deepcopy(testset.x_data)
        if cfg.experiment == 'min_mnist':
            conditions = np.random.uniform(cfg_data.limit_dataset , cfg_data.max_dataset , size)
            df_test_in = pd.DataFrame(testset.y_data, columns=['label'])
            index_distribution = df_test_in[(df_test_in['label'] > cfg_data.limit_dataset) & df_test_in['label'] <= cfg_data.max_dataset].index
            print(f'size of in distribution data for fid/kid : {len(index_distribution)}')
            real_dataset = deepcopy(testset.x_data)

    if distribution == 'out':
        if cfg.experiment == 'max_mnist':
            conditions = np.random.uniform(cfg_data.limit_dataset, cfg_data.max_dataset, size)
            df_test_out = pd.DataFrame(testset.y_data, columns=['label'])
            index_distribution = df_test_out[df_test_out['label'] > cfg_data.limit_dataset].index
            print(f'size of out distribution data for fid/kid : {len(index_distribution)}')
            real_dataset = deepcopy(testset.x_data)
        if cfg.experiment == 'min_mnist':
            conditions = np.random.uniform(cfg_data.min_dataset , cfg_data.limit_dataset , size)
            df_test_out = pd.DataFrame(testset.y_data, columns=['label'])
            index_distribution = df_test_out[df_test_out['label'] <= cfg_data.limit_dataset].index
            print(f'size of out distribution data for fid/kid : {len(index_distribution)}')
            real_dataset = deepcopy(testset.x_data)

    # ------------ Sample z from normal gaussian distribution with a bound ------------
    z = get_truncated_normal((size, cfgan.latent_dim), quant=cfginf.quantile_rate_z_gen)

    # ------------ Generate sample from z and y target ------------
    images_generated = generate_sample_from_GAN(conditions, z, generator)

    # ------------ Compute global measures ------------
    se_measure_glob, re_measure_glob, fid_value_gen_glob, kid_value_gen_glob = compute_global_measures(distribution, images_generated, index_distribution, real_dataset, conditions, forward, size)

    # ------------ Compute policy measures ------------
    se_measure_pol, re_measure_pol, fid_value_gen_pol, kid_value_gen_pol, forward_pred_pol, images_generated_pol = compute_policy_measures(distribution, images_generated, index_distribution, real_dataset, conditions, forward, size)

    # ------------ Global results ------------
    se_ms_glob = [np.mean(se_measure_glob), np.std(se_measure_glob)]; re_ms_glob = [np.mean(re_measure_glob), np.std(re_measure_glob)];
    se_ms_pol = [np.mean(se_measure_pol), np.std(se_measure_pol)]; re_ms_pol = [np.mean(re_measure_pol), np.std(re_measure_pol)];

    plots_some_results(distribution, images_generated_pol, conditions, forward_pred_pol, testset, index_distribution, forward, fid_value_gen_pol, kid_value_gen_pol, nrow=4, ncol=8)

    return se_ms_glob, re_ms_glob, fid_value_gen_glob, kid_value_gen_glob, se_ms_pol, re_ms_pol, fid_value_gen_pol, kid_value_gen_pol

def monte_carlo_inference_mse(distribution, generator, forward, ncol = 8, nrow =4, size=2000):
    
    if distribution == 'in':
        if cfg.experiment == 'max_mnist':
            conditions = np.random.uniform(cfg_data.min_dataset, cfg_data.limit_dataset, size)
        if cfg.experiment == 'min_mnist':
            conditions = np.random.uniform(cfg_data.limit_dataset , cfg_data.max_dataset , size)

    if distribution == 'out':
        if cfg.experiment == 'max_mnist':
            conditions = np.random.uniform(cfg_data.limit_dataset, cfg_data.max_dataset, size)
        if cfg.experiment == 'min_mnist':
            conditions = np.random.uniform(cfg_data.min_dataset , cfg_data.limit_dataset , size)

    # ------------ Sample z from normal gaussian distribution with a bound ------------
    z = get_truncated_normal((size, cfgan.latent_dim), quant=cfginf.quantile_rate_z_gen)

    # ------------ Generate sample from z and y target ------------
    images_generated = generate_sample_from_GAN(conditions, z, generator)

    # ------------ random sample ------------
    size = np.around(images_generated.shape[0] * cfginf.quantile_rate_uncertainty_policy , decimals=0)
    random_index = random.sample(np.arange(images_generated.shape[0]).tolist(), int(size))
    images_generated_rand = images_generated[random_index]
    
    # ------------ Compute the forward predictions ------------
    try:
        y_pred = forward(F.interpolate(images_generated_rand, size=32)).squeeze(1).cpu().detach().numpy()

    except:
        y_pred, epistemic, aleatoric = get_uncertainty_per_batch(forward, F.interpolate(images_generated_rand, size=32), device)
    
    # ------------ Compare the forward model and the Measure from morphomnist ------------
    images_generated_rand = images_generated_rand.squeeze(1).cpu().detach().numpy()
    thickness = compute_thickness_ground_truth(images_generated_rand)
    
    # ------------ Compute the se between the target and the morpho measure predictions ------------
    se_measure_glob = se(thickness, y_pred)
    re_measure_glob = re(thickness, y_pred)
    
    print(f"{distribution} distribution RANDOM results")
    print(f"The mean squared error : {np.mean(se_measure_glob)} \t The std of the squared error : {np.std(se_measure_glob)}")
    print(f"The mean relative error : {np.mean(re_measure_glob)} \t The std of the squared error : {np.std(re_measure_glob)}")
    
    # ------------ Compute the se between the target and the forward model predictions ------------
    try:
        y_pred = forward(F.interpolate(images_generated, size=32)).squeeze(1).cpu().detach().numpy()
        
        # ------------ rondom policy ------------
        random_index = random.sample(np.arange(images_generated.shape[0]).tolist(), int(size))
        y_pred = y_pred[random_index]
        epistemic = epistemic[random_index]
        images_generated = images_generated[random_index]
        conditions = conditions[random_index]

    except:
        y_pred, epistemic, aleatoric = get_uncertainty_per_batch(forward, F.interpolate(images_generated, size=32), device)
 
        # ------------ Uncertainty policy ------------
        index_certain = uncertainty_selection(epistemic.squeeze())
        y_pred = y_pred[index_certain]
        epistemic = epistemic[index_certain]
        images_generated = images_generated[index_certain]
        conditions = conditions[index_certain]

    # ------------ Compare the forward model and the Measure from morphomnist ------------
    # Move variable to cpu
    images_generated = images_generated.squeeze(1).cpu().detach().numpy()
    thickness = compute_thickness_ground_truth(images_generated)

    # ------------ Compute the se between the target and the morpho measure predictions ------------
    se_measure = se(thickness, y_pred)
    re_measure = re(thickness, y_pred)

    print(f"{distribution} distribution POLICY results")
    print(f"The mean squared error : {np.mean(se_measure)} \t The std of the squared error : {np.std(se_measure)}")
    print(f"The mean relative error : {np.mean(re_measure)} \t The std of the squared error : {np.std(re_measure)}")
    print(f"Mean FID : {fid_value_gen[0]} ± {fid_value_gen[1]} \t Mean KID : {kid_value_gen[0]} ± {kid_value_gen[1]}")
    
    # ------------ Global results ------------
    stat_ms_glob = [np.mean(se_measure_glob), np.std(se_measure_glob)]; stat_mr_glob = [np.mean(re_measure_glob), np.std(re_measure_glob)];
    stat_ms_pol = [np.mean(se_measure_pol), np.std(se_measure_pol)]; stat_mr_pol = [np.mean(re_measure_pol), np.std(re_measure_pol)];
    
    return stat_ms_glob, stat_ms_pol, stat_mr_glob, stat_mr_pol

def monte_carlo_inference_fid_kid(distribution, generator, forward, testset, ncol = 8, nrow =4, sample_number_fid_kid = 1000, size_sample=50):
    
    size_full = int(sample_number_fid_kid * 1/cfginf.quantile_rate_uncertainty_policy)
    fid_pol = []; fid_rand = []; kid_pol = []; kid_rand = [];
    for i in range(size_sample):
        print(f'Iteration : {i}/{size_sample}')
        if distribution == 'in':
            if cfg.experiment == 'max_mnist':
                conditions = np.random.uniform(cfg_data.min_dataset, cfg_data.limit_dataset, size_full)
                df_test_in = pd.DataFrame(testset.y_data, columns=['label'])
                index_distribution = df_test_in[df_test_in['label'] <= cfg_data.limit_dataset].index
                print(f'size of in distribution data for fid/kid : {len(index_distribution)}')
                real_dataset = deepcopy(testset.x_data)
            if cfg.experiment == 'min_mnist':
                conditions = np.random.uniform(cfg_data.limit_dataset , cfg_data.max_dataset , size_full)
                df_test_in = pd.DataFrame(testset.y_data, columns=['label'])
                index_distribution = df_test_in[(df_test_in['label'] > cfg_data.limit_dataset) & df_test_in['label'] <= cfg_data.max_dataset].index
                print(f'size of in distribution data for fid/kid : {len(index_distribution)}')
                real_dataset = deepcopy(testset.x_data)
    
        if distribution == 'out':
            if cfg.experiment == 'max_mnist':
                conditions = np.random.uniform(cfg_data.limit_dataset, cfg_data.max_dataset, size_full)
                df_test_out = pd.DataFrame(testset.y_data, columns=['label'])
                index_distribution = df_test_out[df_test_out['label'] > cfg_data.limit_dataset].index
                print(f'size of out distribution data for fid/kid : {len(index_distribution)}')
                real_dataset = deepcopy(testset.x_data)
            if cfg.experiment == 'min_mnist':
                conditions = np.random.uniform(cfg_data.min_dataset , cfg_data.limit_dataset , size_full)
                df_test_out = pd.DataFrame(testset.y_data, columns=['label'])
                index_distribution = df_test_out[df_test_out['label'] <= cfg_data.limit_dataset].index
                print(f'size of out distribution data for fid/kid : {len(index_distribution)}')
                real_dataset = deepcopy(testset.x_data)
    
        # ------------ Sample z from normal gaussian distribution with a bound ------------
        z = get_truncated_normal((size_full, cfgan.latent_dim), quant=cfginf.quantile_rate_z_gen)
    
        # ------------ Generate sample from z and y target ------------
        images_generated = generate_sample_from_GAN(conditions, z, generator)
        
        # ------------ random sample ------------
        random_index = random.sample(np.arange(images_generated.shape[0]).tolist(), sample_number_fid_kid)
        images_generated_rand = images_generated[random_index]
    
        # ------------ Compute FID/KID from testset ------------
        fid_value_gen_rand, kid_value_gen_rand = compute_fid_mnist(images_generated_rand, index_distribution, real_dataset)

        # ------------ Compute policy measures ------------
        try:
            y_pred = forward(F.interpolate(images_generated, size=32)).squeeze(1).cpu().detach().numpy()
            forward_pred = np.array(y_pred).T
    
        except:
            y_pred, epistemic, aleatoric = get_uncertainty_per_batch(forward, F.interpolate(images_generated, size=32), device)
     
            # ------------ Uncertainty policy ------------
            index_certain = uncertainty_selection(epistemic.squeeze())
            y_pred = y_pred[index_certain]
            epistemic = epistemic[index_certain]
            forward_pred = np.array([y_pred, epistemic.squeeze(1)]).T
            images_generated = images_generated[index_certain]
            conditions = conditions[index_certain]
    
        # ------------ Compute FID/KID from testset ------------
        fid_value_gen_pol, kid_value_gen_pol = compute_fid_mnist(images_generated, index_distribution, real_dataset)

        # Update values
        fid_pol.append(fid_value_gen_pol[0]); fid_rand.append(fid_value_gen_rand[0]); kid_pol.append(kid_value_gen_pol[0]); kid_rand.append(kid_value_gen_rand[0]);
        
    stat_fid_rand = [np.mean(fid_rand), np.std(fid_rand)/np.sqrt(size_sample)]
    stat_fid_pol = [np.mean(fid_pol), np.std(fid_pol)/np.sqrt(size_sample)]
    stat_kid_rand = [np.mean(kid_rand), np.std(kid_rand)/np.sqrt(size_sample)]
    stat_kid_pol = [np.mean(kid_pol), np.std(kid_pol)/np.sqrt(size_sample)]
    
    return stat_fid_rand, stat_fid_pol, stat_kid_rand, stat_kid_pol