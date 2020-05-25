from scipy.stats import truncnorm, norm
import random
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os 
from torch.nn import functional as F
from copy import deepcopy

from src_min_mnist.metrics import se, compute_thickness_ground_truth
from src_min_mnist.generative_model.metrics import calculate_fid_given_paths, calculate_kid_given_paths
from src_min_mnist.forward.uncertainty_estimation import get_uncertainty_per_batch
import src_min_mnist.config as cfg

import torch 
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

quantile_rate_uncertainty_policy = 0.5
quantile_rate_z_gen = 0.9

def save_numpy_arr(path, arr):
    np.save(path, arr)
    return path

def get_truncated_normal(form, mean=0, sd=1, quant=0.8):
    upp = norm.ppf(quant, mean, sd)
    low = norm.ppf(1 - quant, mean, sd)
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(form)

def generate_sample_from_GAN(target, z, generator):
    # Prepare labels
    z = Variable(FloatTensor(z))
    labels = Variable(FloatTensor(get_truncated_normal(z.shape[0], mean=target, sd=1, quant=0.6)))
    images_generated = generator(z, labels)

    return images_generated, labels

def se_between_target_and_prediction(target, x, forward):

    y_labels = np.empty(x.shape[0])
    y_labels.fill(target)
    try:
        y_pred = forward(F.interpolate(x, size=32)).squeeze(1).cpu().detach().numpy()
        return se(y_pred, y_labels), y_pred

    except:
        y_pred, epistemic, aleatoric = get_uncertainty_per_batch(forward, F.interpolate(x, size=32), device)
        return se(y_pred, y_labels), y_pred, epistemic

def plots_results(target, forward_pred, forward_pred_train, morpho_pred, morpho_pred_train, se, conditions, testset, images_generated, select_img_label_index, fid_value_gen, kid_value_gen, nrow=2, ncol=4):

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(8,3), dpi=200)
    i=0
    for row in ax:
        j=0
        if i == 0:
            lst = pd.Series(se)
            n_top_index = lst.nsmallest(ncol).index.values.tolist()
        else:
            imgs = testset.x_data.numpy()[select_img_label_index].squeeze(1)
        for col in row:
            if i == 0:
                image = images_generated[n_top_index[j]]
                col.imshow(image)
                col.axis('off')
                col.set_title(f"Forward={np.round(float(forward_pred[n_top_index[j]]),1)} / true={np.round(float(morpho_pred[n_top_index[j]]),1)} / Cond={np.round(float(conditions[n_top_index[j]]),1)}", fontsize=6)
            else:
                image = imgs[j]
                col.imshow(image)
                col.axis('off')
                col.set_title(f"Forward={np.round(float(forward_pred_train[j]),1)} / true={np.round(float(morpho_pred_train[j]),1)}", fontsize=6)
            j+=1
        i+=1
    plt.suptitle(f"Target : {target} \ FID Value : {np.round(fid_value_gen[0])} ± {np.round(fid_value_gen[1])} \ KID Value : {np.around(kid_value_gen[0], decimals=3)}  ± {np.around(kid_value_gen[1], decimals=3)}", fontsize=9)
    plt.show()

def compute_fid_mnist_monte_carlo(fake, target, testset, sample_size):
    folder = 'save_data'
    os.makedirs(folder, exist_ok=True)

    path_gen = save_numpy_arr(os.path.join(folder, 'gen_img_in_distribution.npy'), fake)
    path_real = save_numpy_arr(os.path.join(folder, 'image_from_test.npy'), image_from_test)

    paths = [path_real, path_gen]
    return calculate_fid_given_paths(paths), calculate_kid_given_paths(paths)

def uncertainty_selection(uncertainty, policy_type='quantile'):
    if policy_type == 'quantile':
        quantile = np.quantile(uncertainty, quantile_rate_uncertainty_policy)
        new_index = np.argwhere(uncertainty < quantile)
    return new_index.squeeze()


def monte_carlo_inference(target, generator, forward, testset, ncol = 4, nrow =2, sample_number_fid_kid = 300, size=2000):

    # ------------ Sample z from normal gaussian distribution with a bound ------------
    z = get_truncated_normal((size, cfg.latent_dim), quant=quantile_rate_z_gen)

    # ------------ Generate sample from z and y target ------------
    images_generated, conditions = generate_sample_from_GAN(target, z, generator)

    # ------------ Compute the se between the target and the forward model predictions ------------
    try:
        se_forward, forward_pred = se_between_target_and_prediction(target, images_generated, forward)
    
    except:
        se_forward, forward_pred, uncertainty = se_between_target_and_prediction(target, images_generated, forward)
        
        # ------------ Uncertainty policy ------------
        index_certain = uncertainty_selection(uncertainty.squeeze())
        se_forward = se_forward[index_certain]
        forward_pred = forward_pred[index_certain]
        images_generated = images_generated[index_certain]
        conditions = conditions[index_certain]

    # Move variable to cpu
    images_generated = images_generated.squeeze(1).cpu().detach().numpy()
    conditions = conditions.cpu().detach().numpy()

    # ------------ Compare the forward model and the Measure from morphomnist ------------
    thickness = compute_thickness_ground_truth(images_generated)

    # ------------ Compute the se between the target and the morpho measure predictions ------------
    se_measure = se(target, thickness.values)

    # Measure on trained data
    train_img_label = pd.DataFrame(np.around(testset.labels).values.tolist(), columns=['label'])
    select_img_label_index = random.sample(train_img_label[train_img_label['label']==target].index.values.tolist() , ncol)
    image_from_test = testset.x_data.numpy()[select_img_label_index].squeeze(1)
    x_train = Variable(testset.x_data[select_img_label_index].type(FloatTensor))

    # ------------ Compute the se between testset and the forward model predictions ------------
    try:
        se_forward_train, forward_pred_train = se_between_target_and_prediction(target, x_train, forward)
    except:
        se_forward_train, forward_pred_train, uncertainty_train = se_between_target_and_prediction(target, x_train, forward)

    # ------------ EDA of the best x* generated ------------
    
    top_values = 10

    index = np.argsort(se_forward)[:top_values]
    forward_se_mean = np.mean(se(target,thickness.values[index])); forward_se_std = np.std(se(target,thickness.values[index])); global_mean = np.mean(se_measure);

    print()
    print(f" ------------ Best forward image ------------")
    print(f"MSE measure pred = {forward_se_mean} ± {forward_se_std} ")
    print(f"MSE morpho on Generated data: {global_mean}")

    # Transormf output to real value
    model_pred = forward_pred
    model_pred_train = forward_pred_train

    '''# Create true target values
    test_img_label = pd.DataFrame(np.around(testset.labels).values.tolist(), columns=['label'])
    select_img_label_index = random.sample(test_img_label[test_img_label['label']==target].index.values.tolist(), sample_number)
    image_from_test = testset.x_data.numpy()[select_img_label_index]'''

    # Compute FID values
    fid_value_gen, kid_value_gen = compute_fid_mnist_monte_carlo(np.expand_dims(images_generated, 1), target, testset, sample_number_fid_kid)

    plots_results(target, model_pred, model_pred_train, thickness.values, testset.y_data[select_img_label_index].numpy(), se_forward, conditions, testset, images_generated, select_img_label_index, fid_value_gen, kid_value_gen, nrow=2, ncol=4)

    return [forward_se_mean, forward_se_std], global_mean, fid_value_gen, kid_value_gen

def save_obj_csv(d, path):
    d.to_csv(path+'.csv', index=False)

def load_obj_csv(path):
    return pd.read_csv(path+'.csv')

def generate_sample_from_GAN__(y_cond, z, generator):
    # Prepare labels
    z = Variable(FloatTensor(z))
    cond = Variable(FloatTensor(y_cond))

    return generator(z, cond)

def compute_fid_mnist(gen_img, index_distribution, real_dataset, sample_size):
    random_id = random.sample(index_distribution.tolist(), sample_size)
    real_imgs = real_dataset[random_id].numpy()

    folder = 'save_data'
    os.makedirs(folder, exist_ok=True)

    path_gen = save_numpy_arr(os.path.join(folder, 'gen_img.npy'), gen_img.cpu().detach().numpy())
    path_real = save_numpy_arr(os.path.join(folder, 'real_imgs.npy'), real_imgs)

    paths = [path_real, path_gen]
    fid_value = calculate_fid_given_paths(paths)
    kid_value = calculate_kid_given_paths(paths)

    return fid_value, kid_value

def monte_carlo_inference_general(distribution, generator, forward, testset, ncol = 8, nrow =4, sample_number_fid_kid = 300, size=2000):
    
    if distribution == 'in':
        conditions = np.random.uniform(cfg.limit_dataset -0.5, cfg.max_dataset + 0.5, size)

        # FID needs
        df_test_in = pd.DataFrame(testset.labels.values, columns=['label'])
        index_distribution = df_test_in[df_test_in['label']<=cfg.limit_dataset].index
        print(f'size of in distribution data for fid/kid : {len(index_distribution)}')
        real_dataset = deepcopy(testset.x_data)

    if distribution == 'out':
        conditions = np.random.uniform(cfg.min_dataset + -0.5 , cfg.limit_dataset + 0.5, size)
        
        # FID needs
        df_test_out = pd.DataFrame(testset.labels.values, columns=['label'])
        index_distribution = df_test_out[df_test_out['label']>cfg.limit_dataset].index
        print(f'size of out distribution data for fid/kid : {len(index_distribution)}')
        real_dataset = deepcopy(testset.x_data)

    # ------------ Sample z from normal gaussian distribution with a bound ------------
    z = get_truncated_normal((size, cfg.latent_dim), quant=quantile_rate_z_gen)

    # ------------ Generate sample from z and y target ------------
    images_generated = generate_sample_from_GAN__(conditions, z, generator)

    # ------------ Compute the se between the target and the forward model predictions ------------
    try:
        y_pred = forward(F.interpolate(images_generated, size=32)).squeeze(1).cpu().detach().numpy()
        forward_pred = np.array(y_pred)

    except:
        y_pred, epistemic, aleatoric = get_uncertainty_per_batch(forward, F.interpolate(images_generated, size=32), device)
        
        # ------------ Uncertainty policy ------------
        index_certain = uncertainty_selection(epistemic.squeeze())
        y_pred = y_pred[index_certain]
        epistemic = epistemic[index_certain]
        forward_pred = np.array([y_pred, epistemic.squeeze(1)])
        forward_pred = np.swapaxes(forward_pred, 0, 1)
        images_generated = images_generated[index_certain]
        conditions = conditions[index_certain]

    # ------------ Compute FID/KID from testset ------------
    fid_value_gen, kid_value_gen = compute_fid_mnist(images_generated, index_distribution, real_dataset, size)
    
    # Move variable to cpu
    images_generated = images_generated.squeeze(1).cpu().detach().numpy()

    # ------------ Compare the forward model and the Measure from morphomnist ------------
    thickness = compute_thickness_ground_truth(images_generated)

    # ------------ Compute the se between the target and the morpho measure predictions ------------
    se_measure = se(thickness, y_pred)
    
    print(f"{distribution} distribution results")
    print(f"The mean squared error : {np.mean(se_measure)} \t The std of the squared error : {np.std(se_measure)}")
    print(f"Mean FID : {fid_value_gen[0]} ± {fid_value_gen[1]} \t Mean KID : {kid_value_gen[0]} ± {kid_value_gen[1]}")
    
    plots_some_results(distribution, images_generated, conditions, forward_pred, testset, index_distribution, forward, fid_value_gen, kid_value_gen, nrow=4, ncol=8)


def plots_some_results(distribution, images_generated, conditions, forward_pred, testset, index_distribution, forward, fid_value_gen, kid_value_gen, nrow=4, ncol=8):
    
    # Gan img selection
    random_gan_index = random.sample(np.arange(images_generated.shape[0]).tolist(), ncol*nrow)
    images_generated = images_generated[random_gan_index]
    conditions = conditions[random_gan_index]
    forward_pred = forward_pred[random_gan_index]
    
    # Real img selection
    random_id = random.sample(index_distribution.tolist(), ncol*nrow)
    real_imgs = testset.x_data[random_id].numpy()
    labels = testset.y_data[random_id].numpy()
    
    try:
        real_pred = forward(F.interpolate(real_imgs, size=32)).squeeze(1).cpu().detach().numpy()
        bayesian=False
    except:
        real_pred, epistemic, aleatoric = get_uncertainty_per_batch(forward, F.interpolate(real_imgs, size=32), device)
        bayesian=True

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(8,3), dpi=200)
    
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
                    col.set_title(f"Forward={np.round(float(forward_pred[j + k][0]),1)} / Cond={np.round(float(conditions[j + k]),1)} / Uncertainty={np.round(float(forward_pred[j + k][1]),4)}", fontsize=6) #/ true={np.round(float(morpho_pred[n_top_index[j + k]]),1)} 
                else:
                    col.set_title(f"Forward={np.round(float(forward_pred[j + k]),1)} / Cond={np.round(float(conditions[j + k]),1)}", fontsize=6) #/ true={np.round(float(morpho_pred[n_top_index[j + k]]),1)} 
            else:
                image = imgs[j + k]
                col.imshow(image)
                col.axis('off')
                if bayesian:
                    col.set_title(f"Forward={np.round(float(real_pred[j + k]),1)} / Label={np.round(float(labels[j + k]),1)} / Uncertainty={np.round(float(epistemic[j + k]),4)}", fontsize=6) #/ true={np.round(float(morpho_pred[n_top_index[j + k]]),1)} 
                else:
                    col.set_title(f"Forward={np.round(float(forward_pred[j + k]),1)} / Label={np.round(float(labels[j + k]),1)}", fontsize=6) #/ true={np.round(float(morpho_pred[n_top_index[j]]),1)}
    
    plt.suptitle(f"{distribution} distribution / FID Value : {np.round(fid_value_gen[0])} ± {np.round(fid_value_gen[1])} \ KID Value : {np.around(kid_value_gen[0], decimals=3)}  ± {np.around(kid_value_gen[1], decimals=3)}", fontsize=9)
    plt.show()
