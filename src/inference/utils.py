import pandas as pd
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
import os 
import torch 
from torch.autograd import Variable
from torch.nn import functional as F
from copy import deepcopy
import itertools
from tabulate import tabulate
from scipy.stats import norm, truncnorm
import random

from src.data import MNISTDataset, RotationDataset
from src.metrics import se, re, compute_thickness_ground_truth
from src.generative_model.metrics import calculate_fid_given_paths, calculate_kid_given_paths
from src.forward.uncertainty_estimation import get_uncertainty_per_batch

import src.config as cfg

if cfg.dcgan:
    import src.config_dcgan as cfgan
else:
    import src.config_gan as cfgan

import src.config_inference as cfginf
from src.uncertainty_policy import uncertainty_selection

if cfg.experiment == 'min_mnist':
    import src.config_min_mnist as cfg_data
elif cfg.experiment == 'max_mnist':
    import src.config_max_mnist as cfg_data
elif cfg.experiment == 'rotation_dataset':
    import src.config_rotation as cfg_data

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

max_size = 2000
uncertainty_policy = 'epistemic'

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
    if (y_cond.shape[0] > 5000) & (z.shape[0] > 5000):

        y_cond = np.array_split(y_cond, len(y_cond) // max_size)
        z = np.array_split(z, len(z) // max_size)

        if len(y_cond) != len(z):
            print('WARNING PIPELINE BROKEN WHEN GENERATION')

        X_chunk = []
        for i, condition in enumerate(y_cond):
            X_chunk.append(generator(Variable(FloatTensor(z[i])), Variable(FloatTensor(condition))).detach().cpu())

        return torch.cat(X_chunk)
    
    else:
        return generator( Variable(FloatTensor(z)), Variable(FloatTensor(y_cond))).detach().cpu()
    
def predict_forward_model(forward, gen_images, bayesian=True):
        
    if bayesian:
        if gen_images.shape[0] > 5000:
            gen_images = np.array_split(gen_images.cpu().detach(), len(gen_images.cpu().detach()) // max_size)
            pred_chunk = []
            epistemic_chunk = []
            aleatoric_chunk = []
            for i, gen_images_batch in enumerate(gen_images):
                pred, epi, aleatoric = get_uncertainty_per_batch(forward, F.interpolate(gen_images_batch.to(device), size=32), device)
                pred_chunk.append(pred)
                epistemic_chunk.append(epi)
                aleatoric_chunk.append(aleatoric)
            
            return np.concatenate(pred_chunk), np.concatenate(epistemic_chunk), np.concatenate(aleatoric_chunk)
        
        else:
            y_pred, epistemic, aleatoric = get_uncertainty_per_batch(forward, F.interpolate(gen_images, size=32), device)
            return y_pred, epistemic, aleatoric
    
    else:
        if gen_images.shape[0] > 5000:
            gen_images = np.array_split(gen_images.cpu().detach(), len(gen_images.cpu().detach()) // max_size)
            pred_chunk = []
            for i, gen_images_batch in enumerate(gen_images):
                pred_chunk.append(forward(F.interpolate(gen_images_batch.to(device), size=32)).squeeze(1).cpu().detach().numpy())
            
            return np.concatenate(pred_chunk)

        else:
            y_pred = forward(F.interpolate(gen_images.to(device), size=32)).squeeze(1).cpu().detach().numpy()
            return y_pred
    

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

def get_main_data(distribution, size_full, testset):
    
    if distribution == 'in':
        if (cfg.experiment == 'max_mnist') | (cfg.experiment == 'rotation_dataset'):
            if cfg.dcgan:
                conditions = np.random.uniform(cfg_data.min_dataset, cfg_data.limit_dataset, (size_full, 1, 1, 1))
                z = get_truncated_normal((size_full, cfgan.latent_dim, 1, 1), quant=cfginf.quantile_rate_z_gen)
            else:
                conditions = np.random.uniform(cfg_data.min_dataset, cfg_data.limit_dataset, size_full)
                z = get_truncated_normal((size_full, cfgan.latent_dim), quant=cfginf.quantile_rate_z_gen)

            df_test_in = pd.DataFrame(testset.y_data, columns=['label'])
            index_distribution = df_test_in[df_test_in['label'] <= cfg_data.limit_dataset].index
            print(f'size of in distribution data for fid/kid : {len(index_distribution)}')
            real_dataset = deepcopy(testset.x_data)
        if cfg.experiment == 'min_mnist':
            if cfg.dcgan:
                conditions = np.random.uniform(cfg_data.limit_dataset, cfg_data.max_dataset, (size_full, 1, 1, 1))
                z = get_truncated_normal((size_full, cfgan.latent_dim, 1, 1), quant=cfginf.quantile_rate_z_gen)
            else:
                conditions = np.random.uniform(cfg_data.limit_dataset, cfg_data.max_dataset, size_full)
                z = get_truncated_normal((size_full, cfgan.latent_dim), quant=cfginf.quantile_rate_z_gen)

            df_test_in = pd.DataFrame(testset.y_data, columns=['label'])
            index_distribution = df_test_in[(df_test_in['label'] >= cfg_data.limit_dataset) & (df_test_in['label'] <= cfg_data.max_dataset)].index
            print(f'size of in distribution data for fid/kid : {len(index_distribution)}')
            real_dataset = deepcopy(testset.x_data)

    if distribution == 'out':
        if (cfg.experiment == 'max_mnist') | (cfg.experiment == 'rotation_dataset'):
            if cfg.dcgan:
                conditions = np.random.uniform(cfg_data.limit_dataset, cfg_data.max_dataset, (size_full, 1, 1, 1))
                z = get_truncated_normal((size_full, cfgan.latent_dim, 1, 1), quant=cfginf.quantile_rate_z_gen)
            else:
                conditions = np.random.uniform(cfg_data.limit_dataset, cfg_data.max_dataset, size_full)
                z = get_truncated_normal((size_full, cfgan.latent_dim), quant=cfginf.quantile_rate_z_gen)

            df_test_out = pd.DataFrame(testset.y_data, columns=['label'])
            index_distribution = df_test_out[df_test_out['label'] > cfg_data.limit_dataset].index
            print(f'size of out distribution data for fid/kid : {len(index_distribution)}')
            real_dataset = deepcopy(testset.x_data)
        if cfg.experiment == 'min_mnist':
            if cfg.dcgan:
                conditions = np.random.uniform(cfg_data.min_dataset , cfg_data.limit_dataset, (size_full, 1, 1, 1))
                z = get_truncated_normal((size_full, cfgan.latent_dim, 1, 1), quant=cfginf.quantile_rate_z_gen)
            else:
                conditions = np.random.uniform(cfg_data.min_dataset , cfg_data.limit_dataset, size_full)
                z = get_truncated_normal((size_full, cfgan.latent_dim), quant=cfginf.quantile_rate_z_gen)

            df_test_out = pd.DataFrame(testset.y_data, columns=['label'])
            index_distribution = df_test_out[df_test_out['label'] < cfg_data.limit_dataset].index
            print(f'size of out distribution data for fid/kid : {len(index_distribution)}')
            real_dataset = deepcopy(testset.x_data)
    
    return conditions, z, index_distribution, real_dataset

def monte_carlo_inference_mse_batch(distribution, generator, forward, testset, bayesian=True, sample_number=2000):

    size_full = int(sample_number * 1/cfginf.quantile_rate_uncertainty_policy)
    
    # ------------ Get data ------------
    conditions, z, _, _ = get_main_data(distribution, size_full, testset)
    
    # ------------ Generate sample from z and y target ------------
    images_generated = generate_sample_from_GAN(conditions, z, generator)
    
    # ------------ random sample ------------
    random_index = random.sample(np.arange(images_generated.shape[0]).tolist(), sample_number)
    images_generated_rand = images_generated[random_index]

    # ------------ Compute forward predictions ------------
    if bayesian:
        y_pred, epistemic, aleatoric = predict_forward_model(forward, images_generated_rand, bayesian=bayesian)
    else:
        y_pred = predict_forward_model(forward, images_generated_rand, bayesian=bayesian)

    # ------------ Compare the forward model and the Measure from morphomnist ------------
    images_generated_rand = images_generated_rand.squeeze(1).cpu().detach().numpy()
    thickness = compute_thickness_ground_truth(images_generated_rand)
    
    # ------------ Compute the se between the target and the morpho measure predictions ------------
    se_measure_glob = se(thickness, y_pred)
    re_measure_glob = re(thickness, y_pred)
    
    print(f"{distribution} distribution RANDOM results")
    print(f"The mean squared error : {np.mean(se_measure_glob)} \t The std of the squared error : {np.std(se_measure_glob)}")
    print(f"The mean relative error : {np.mean(re_measure_glob)} \t The std of the squared error : {np.std(re_measure_glob)}")
    
    # ------------ Global results ------------
    stat_ms_glob = [np.mean(se_measure_glob), np.std(se_measure_glob)]; stat_mr_glob = [np.mean(re_measure_glob), np.std(re_measure_glob)];

    if bayesian:
        # ------------ Compute forward predictions ------------
        y_pred, epistemic, aleatoric = predict_forward_model(forward, images_generated, bayesian=bayesian)
        # ------------ Uncertainty policy ------------
        if uncertainty_policy == 'epistemic':
            index_certain, _ = uncertainty_selection(epistemic.squeeze())
        elif uncertainty_policy == 'aleatoric':
            index_certain, _ = uncertainty_selection(aleatoric.squeeze())
        images_generated_sampled = images_generated[index_certain]
        y_pred = y_pred[index_certain]

        # ------------ Compare the forward model and the Measure from morphomnist ------------
        # Move variable to cpu
        images_generated_sampled = images_generated_sampled.squeeze(1).cpu().detach().numpy()
        thickness = compute_thickness_ground_truth(images_generated_sampled)

        # ------------ Compute the se between the target and the morpho measure predictions ------------
        se_measure_pol = se(thickness, y_pred)
        re_measure_pol = re(thickness, y_pred)

        print(f"{distribution} distribution POLICY results")
        print(f"The mean squared error : {np.mean(se_measure_pol)} \t The std of the squared error : {np.std(se_measure_pol)}")
        print(f"The mean relative error : {np.mean(re_measure_pol)} \t The std of the squared error : {np.std(re_measure_pol)}")

        # ------------ Global results ------------
        stat_ms_pol = [np.mean(se_measure_pol), np.std(se_measure_pol)]; stat_mr_pol = [np.mean(re_measure_pol), np.std(re_measure_pol)];
        
        return stat_ms_glob, stat_ms_pol, stat_mr_glob, stat_mr_pol

    else:
        return stat_ms_glob, [100.0, 100.0], stat_mr_glob, [100.0, 100.0]

def monte_carlo_inference_mse_sampling(distribution, generator, forward, testset, bayesian=True, sample_number=2000, size_sample=10):
    
    ms_rand = []; ms_pol = []; mr_rand = []; mr_pol = [];
    for i in range(size_sample):
        print(f'Computed sample {i+1}/{size_sample}')
        stat_ms_glob, stat_ms_pol, stat_mr_glob, stat_mr_pol = monte_carlo_inference_mse_batch(distribution, 
                                                                                               generator, 
                                                                                               forward,
                                                                                               testset,
                                                                                               bayesian=True, 
                                                                                               sample_number = 2000)

        # Update values
        ms_rand.append(stat_ms_glob[0]); ms_pol.append(stat_ms_pol[0]); mr_rand.append(stat_mr_glob[0]); mr_pol.append(stat_mr_pol[0]);

    stat_se_rand = [np.mean(ms_rand), np.std(ms_rand)]
    stat_se_pol = [np.mean(ms_pol), np.std(ms_pol)]
    stat_re_rand = [np.mean(mr_rand), np.std(mr_rand)]
    stat_re_pol = [np.mean(mr_pol), np.std(mr_pol)]
    
    return stat_se_rand, stat_se_pol, stat_re_rand, stat_re_pol

def monte_carlo_inference_fid_kid_batch(distribution, generator, forward, testset, bayesian=True, sample_number_fid_kid = 2000):

    size_full = int(sample_number_fid_kid * 1/cfginf.quantile_rate_uncertainty_policy)

    # ------------ Get data ------------
    conditions, z, index_distribution, real_dataset = get_main_data(distribution, size_full, testset)
    
    # ------------ Generate sample from z and y target ------------
    images_generated = generate_sample_from_GAN(conditions, z, generator)
    
    # ------------ random sample ------------
    random_index = random.sample(np.arange(images_generated.shape[0]).tolist(), sample_number_fid_kid)
    images_generated_rand = images_generated[random_index]
    
    # ------------ Compute FID/KID from testset ------------
    fid_value_gen_rand, kid_value_gen_rand = compute_fid_mnist(images_generated_rand, index_distribution, real_dataset)

    # ------------ Compute policy measures ------------
    if bayesian:
        y_pred, epistemic, aleatoric = predict_forward_model(forward, images_generated, bayesian=True)
        # ------------ Uncertainty policy ------------
        if uncertainty_policy == 'epistemic':
            index_certain, _ = uncertainty_selection(epistemic.squeeze())
        elif uncertainty_policy == 'aleatoric':
            index_certain, _ = uncertainty_selection(aleatoric.squeeze())
        
        images_generated_sampled = images_generated[index_certain]
    else:
        # ------------ random sample ------------
        random_index = random.sample(np.arange(images_generated.shape[0]).tolist(), sample_number_fid_kid)
        images_generated_sampled = images_generated[random_index]
    
    # ------------ Compute FID/KID from testset ------------
    fid_value_gen_pol, kid_value_gen_pol = compute_fid_mnist(images_generated_sampled, index_distribution, real_dataset)
    
    return fid_value_gen_rand, fid_value_gen_pol, kid_value_gen_rand, kid_value_gen_pol

def monte_carlo_inference_fid_kid_sampling(distribution, generator, forward, testset, bayesian=True, sample_number_fid_kid = 2000, size_sample=10):
    
    fid_pol = []; fid_rand = []; kid_pol = []; kid_rand = [];
    for i in range(size_sample):
        print(f'Computed sample {i+1}/{size_sample}')
        fid_value_gen_rand, fid_value_gen_pol, kid_value_gen_rand, kid_value_gen_pol = monte_carlo_inference_fid_kid_batch(distribution, 
                                                                                                                            generator, 
                                                                                                                            forward, 
                                                                                                                            testset, 
                                                                                                                            bayesian=bayesian, 
                                                                                                                            sample_number_fid_kid = sample_number_fid_kid)

        # Update values
        fid_pol.append(fid_value_gen_pol[0]); fid_rand.append(fid_value_gen_rand[0]); kid_pol.append(kid_value_gen_pol[0]); kid_rand.append(kid_value_gen_rand[0]);

    stat_fid_rand = [np.mean(fid_rand), np.std(fid_rand)]
    stat_fid_pol = [np.mean(fid_pol), np.std(fid_pol)]
    stat_kid_rand = [np.mean(kid_rand), np.std(kid_rand)]
    stat_kid_pol = [np.mean(kid_pol), np.std(kid_pol)]
    
    return stat_fid_rand, stat_fid_pol, stat_kid_rand, stat_kid_pol

def check_predictions_bayesian(distribution, y_pred, epistemic, aleatoric, conditions, images_generated):
    # Erase in distribution sample
    if distribution == 'in':
        if cfg.experiment == 'min_mnist':
            index_verified = np.argwhere(y_pred > cfg_data.limit_dataset).squeeze(1)
            print(y_pred.shape)
            y_pred = y_pred[index_verified]
            print(y_pred.shape)
            epistemic = epistemic[index_verified]
            aleatoric = aleatoric[index_verified]
            conditions = conditions[index_verified]
            images_generated = images_generated[index_verified]

        elif (cfg.experiment == 'max_mnist') & (cfg.experiment == 'rotation_dataset'):
            index_verified = np.argwhere(y_pred < cfg_data.limit_dataset).squeeze(1)

            y_pred = y_pred[index_verified]
            epistemic = epistemic[index_verified]
            aleatoric = aleatoric[index_verified]
            conditions = conditions[index_verified]
            images_generated = images_generated[index_verified]

    elif distribution == 'out':
        if cfg.experiment == 'min_mnist':
            index_verified = np.argwhere(y_pred < cfg_data.limit_dataset).squeeze(1)
            print(y_pred.shape)
            y_pred = y_pred[index_verified]
            print(y_pred.shape)
            epistemic = epistemic[index_verified]
            aleatoric = aleatoric[index_verified]
            conditions = conditions[index_verified]
            images_generated = images_generated[index_verified]

        elif (cfg.experiment == 'max_mnist') & (cfg.experiment == 'rotation_dataset'):
            index_verified = np.argwhere(y_pred > cfg_data.limit_dataset).squeeze(1)

            y_pred = y_pred[index_verified]
            epistemic = epistemic[index_verified]
            aleatoric = aleatoric[index_verified]
            conditions = conditions[index_verified]
            images_generated = images_generated[index_verified]

    return y_pred, epistemic, aleatoric, conditions, images_generated

def check_predictions_frequentist(distribution, y_pred, conditions, images_generated):
    # Erase in distribution sample
    if distribution == 'in':
        if cfg.experiment == 'min_mnist':
            index_verified = np.argwhere(y_pred > cfg_data.limit_dataset).squeeze(1)
            y_pred = y_pred[index_verified]
            conditions = conditions[index_verified]
            images_generated = images_generated[index_verified]

        elif (cfg.experiment == 'max_mnist') & (cfg.experiment == 'rotation_dataset'):
            index_verified = np.argwhere(y_pred < cfg_data.limit_dataset).squeeze(1)

            y_pred = y_pred[index_verified]
            conditions = conditions[index_verified]
            images_generated = images_generated[index_verified]

    elif distribution == 'out':
        if cfg.experiment == 'min_mnist':
            index_verified = np.argwhere(y_pred < cfg_data.limit_dataset).squeeze(1)
            y_pred = y_pred[index_verified]
            conditions = conditions[index_verified]
            images_generated = images_generated[index_verified]

        elif (cfg.experiment == 'max_mnist') & (cfg.experiment == 'rotation_dataset'):
            index_verified = np.argwhere(y_pred > cfg_data.limit_dataset).squeeze(1)

            y_pred = y_pred[index_verified]
            conditions = conditions[index_verified]
            images_generated = images_generated[index_verified]

    return y_pred, conditions, images_generated

def monte_carlo_inference_qualitative(distribution, forward_type, generator, forward, testset, bayesian=True, sample_number=2000, random_certainty=True):
    
    size_full = int(sample_number * 1/cfginf.quantile_rate_uncertainty_policy)

    # ------------ Get data ------------
    conditions, z, index_distribution, real_dataset = get_main_data(distribution, size_full, testset)
    
    # ------------ Generate sample from z and y target ------------
    images_generated = generate_sample_from_GAN(conditions, z, generator)

    # ------------ Compute policy measures ------------
    if bayesian:
        y_pred, epistemic, aleatoric = predict_forward_model(forward, images_generated, bayesian=bayesian)

        # Check predictions
        y_pred, epistemic, aleatoric, conditions, images_generated = check_predictions_bayesian(distribution, y_pred, epistemic, aleatoric, conditions, images_generated)

        # ------------ Uncertainty policy ------------
        if random_certainty:
            # ------------ Uncertainty policy ------------
            if uncertainty_policy == 'epistemic':
                index_certain, _ = uncertainty_selection(epistemic.squeeze())
            elif uncertainty_policy == 'aleatoric':
                index_certain, _ = uncertainty_selection(aleatoric.squeeze())
            
            y_pred = y_pred[index_certain]
            images_generated = images_generated[index_certain]
            conditions = conditions[index_certain]
            epistemic = epistemic[index_certain]
            forward_pred = np.array([y_pred, epistemic.squeeze(1)]).T

        else:
            index_certain = np.argsort(epistemic.squeeze())
            y_pred = y_pred[index_certain]
            images_generated = images_generated[index_certain]
            conditions = conditions[index_certain]
            epistemic = epistemic[index_certain]
            forward_pred = np.array([y_pred, epistemic.squeeze(1)]).T

    else:
        y_pred = predict_forward_model(forward, images_generated, bayesian=bayesian)
        
        # Check predictions
        y_pred, conditions, images_generated = check_predictions_frequentist(distribution, y_pred, conditions, images_generated)
        # ------------ random sample ------------
        try:
            random_index = random.sample(np.arange(images_generated.shape[0]).tolist(), sample_number)
        except:
            random_index = random.sample(np.arange(images_generated.shape[0]).tolist(), int(sample_number/1.5))

        y_pred = y_pred[random_index]
        images_generated = images_generated[random_index]
        conditions = conditions[random_index]
        forward_pred = np.array(y_pred).T

    plots_qualitative_results(distribution, forward_type, images_generated, conditions, forward_pred, testset, index_distribution, forward, bayesian, nrow=4, ncol=8)

def plots_qualitative_results(distribution, forward_type, images_generated, conditions, forward_pred, testset, index_distribution, forward, bayesian=True, nrow=4, ncol=8):

    # Gan img selection
    images_generated = images_generated[:ncol*nrow].squeeze(1).detach().cpu()
    conditions = conditions[:ncol*nrow]
    forward_pred = forward_pred[:ncol*nrow]

    # Real img selection
    random_id = random.sample(index_distribution.tolist(), ncol*nrow)
    real_imgs = testset.x_data[random_id]
    labels = testset.y_data[random_id]

    if bayesian:
        real_pred, epistemic, aleatoric = predict_forward_model(forward, F.interpolate(real_imgs.to(device), size=32), bayesian=bayesian)
    else:
        real_pred = predict_forward_model(forward, F.interpolate(real_imgs.to(device), size=32), bayesian=bayesian)
    
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
                    col.set_title(f"Fwd={np.round(float(real_pred[j + k]),1)} / Label={np.round(float(labels[j + k]),1)}", fontsize=3) #/ true={np.round(float(morpho_pred[n_top_index[j]]),1)}
    
    plt.suptitle(f"{distribution} distribution with {forward_type} forward model", fontsize=6)
    plt.show()

def monte_carlo_inference_target_images(target, forward_type, generator, forward, testset, bayesian=True, sample_number=2000):
    
    size_full = int(sample_number * 1/cfginf.quantile_rate_uncertainty_policy)
    
    # ------------ Get data ------------
    if cfg.dcgan:
        conditions = np.random.uniform(target - 0.1*cfg_data.max_dataset , target + 0.1*cfg_data.max_dataset, (size_full, 1, 1, 1))
        z = get_truncated_normal((size_full, cfgan.latent_dim, 1, 1), quant=cfginf.quantile_rate_z_gen)
    else:
        conditions = np.random.uniform(target - 0.1*cfg_data.max_dataset  , target + 0.1*cfg_data.max_dataset , size_full)
        z = get_truncated_normal((size_full, cfgan.latent_dim), quant=cfginf.quantile_rate_z_gen)

    df_test_out = pd.DataFrame(testset.y_data, columns=['label'])
    index_distribution = df_test_out[(df_test_out['label'] > target - 0.05 *cfg_data.max_dataset) & (df_test_out['label'] < target + 0.05 *cfg_data.max_dataset)].index
    print(f'size of out distribution data for fid/kid : {len(index_distribution)}')
    real_dataset = deepcopy(testset)

    # ------------ Generate sample from z and y target ------------
    images_generated = generate_sample_from_GAN(conditions, z, generator)

    # ------------ Compute policy measures ------------
    if bayesian:
        y_pred, epistemic, aleatoric = predict_forward_model(forward, images_generated, bayesian=bayesian)

        # ------------ Uncertainty policy ------------
        if uncertainty_policy == 'epistemic':
            index_certain, _ = uncertainty_selection(epistemic.squeeze())
        elif uncertainty_policy == 'aleatoric':
            index_certain, _ = uncertainty_selection(aleatoric.squeeze())
            
        y_pred = y_pred[index_certain]
        images_generated = images_generated[index_certain]
        conditions = conditions[index_certain]
        epistemic = epistemic[index_certain]
        forward_pred = np.array([y_pred, epistemic.squeeze(1)]).T

    else:
        y_pred = predict_forward_model(forward, images_generated, bayesian=bayesian)
        
        # ------------ random sample ------------
        try:
            random_index = random.sample(np.arange(images_generated.shape[0]).tolist(), sample_number)
        except:
            random_index = random.sample(np.arange(images_generated.shape[0]).tolist(), int(sample_number/1.5))

        y_pred = y_pred[random_index]
        images_generated = images_generated[random_index]
        conditions = conditions[random_index]
        forward_pred = np.array(y_pred).T
        
    se_preds = se(y_pred, target)
    
    # Selested predictions
    index_best = np.argsort(se_preds)
    
    se_preds = se_preds[index_best]
    images_generated = images_generated[index_best]
    conditions = conditions[index_best]
    forward_pred = forward_pred[index_best]

    
    plot_best_acc_pred(target, forward_type, images_generated, conditions, forward_pred, real_dataset, index_distribution, forward, se_preds, bayesian=bayesian, nrow=4, ncol=8)
    
def plot_best_acc_pred(target, forward_type, images_generated, conditions, forward_pred, testset, index_distribution, forward, se_preds, bayesian=True, nrow=4, ncol=8):

    # Gan img selection
    images_generated = images_generated[:ncol*nrow].squeeze(1).detach().cpu()
    conditions = conditions[:ncol*nrow]
    forward_pred = forward_pred[:ncol*nrow]

    # Real img selection
    random_id = random.sample(index_distribution.tolist(), ncol*nrow)
    real_imgs = testset.x_data[random_id]
    labels = testset.y_data[random_id]

    if bayesian:
        real_pred, epistemic, aleatoric = predict_forward_model(forward, F.interpolate(real_imgs.to(device), size=32), bayesian=bayesian)
    else:
        real_pred = predict_forward_model(forward, F.interpolate(real_imgs.to(device), size=32), bayesian=bayesian)
    
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
                    col.set_title(f"Fwd={np.round(float(real_pred[j + k]),1)} / Label={np.round(float(labels[j + k]),1)}", fontsize=3) #/ true={np.round(float(morpho_pred[n_top_index[j]]),1)}
    
    plt.suptitle(f"Target value : {target} | {forward_type} forward model ", fontsize=6) #| mse = {round(np.sum(se_preds[:ncol*nrow])/ncol*nrow, 2)}", fontsize=6)
    plt.show()