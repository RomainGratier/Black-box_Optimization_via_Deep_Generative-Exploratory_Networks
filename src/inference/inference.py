from scipy.stats import truncnorm, norm
import random
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os 
from torch.nn import functional as F

from src.metrics import mse, compute_thickness_ground_truth
from src.generative_model.metrics import calculate_fid_given_paths, calculate_kid_given_paths
from src.metrics import mse, compute_thickness_ground_truth
from src.forward.uncertainty_estimation import get_uncertainty_per_batch

latent_dim = 100

import torch 
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

bayesian = True

def save_numpy_arr(path, arr):
    np.save(path, arr)
    return path

def get_truncated_normal(form, mean=0, sd=1, quant=0.8):
    upp = norm.ppf(quant, mean, sd)
    low = norm.ppf(1 - quant, mean, sd)
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(form)

def generate_sample_from_GAN(target, z, generator, scaler):
    # Prepare labels
    normalized_target = scaler.transform(np.array(target).reshape(-1,1))[0]
    z = Variable(FloatTensor(z))
    labels = Variable(FloatTensor(get_truncated_normal(z.shape[0], mean=normalized_target, sd=1, quant=0.6)))
    images_generated = generator(z, labels)

    return images_generated, labels

def se_between_target_and_prediction(target, x, forward, trainset):

    y_labels = np.empty(x.shape[0])
    y_labels.fill(target)
    if bayesian:
        # CUDA settings
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        y_pred, epistemic, aleatoric = get_uncertainty_per_batch(forward, F.interpolate(x, size=32), device)
        return mse(trainset.scaler.inverse_transform(y_pred.reshape(-1,1)).squeeze(), y_labels), y_pred, epistemic

    else:
        y_pred = forward(x.view(-1,28*28)).squeeze(1).cpu().detach().numpy()
        return mse(trainset.scaler.inverse_transform(y_pred.reshape(-1,1)).squeeze(), y_labels), y_pred

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

    # Measure on trained data
    test_img_label = pd.DataFrame(np.around(testset.labels).values.tolist(), columns=['label'])
    random.seed(1)
    select_img_label_index = random.sample(test_img_label[test_img_label['label']==target].index.values.tolist(), sample_size)
    image_from_test = testset.x_data.numpy()[select_img_label_index]

    path_gen = save_numpy_arr(os.path.join(folder, 'gen_img_in_distribution.npy'), fake)
    path_real = save_numpy_arr(os.path.join(folder, 'image_from_test.npy'), image_from_test)

    paths = [path_real, path_gen]
    return calculate_fid_given_paths(paths), calculate_kid_given_paths(paths)

def uncertainty_selection(uncertainty, policy_type='quantile'):
    if policy_type == 'quantile':
        quantile = np.quantile(uncertainty, 0.5)
        new_index = np.argwhere(uncertainty < quantile)
    return new_index.squeeze()


def monte_carlo_inference(target, generator, forward, trainset, testset, ncol = 4, nrow =2, sample_number_fid_kid = 300, size=2000):

    # ------------ Sample z from normal gaussian distribution with a bound ------------
    z = get_truncated_normal((size, latent_dim), quant=0.8)

    # ------------ Generate sample from z and y target ------------
    images_generated, conditions = generate_sample_from_GAN(target, z, generator, trainset.scaler)

    # ------------ Compute the mse between the target and the forward model predictions ------------
    if bayesian:
        se_forward, forward_pred, uncertainty = se_between_target_and_prediction(target, images_generated, forward, trainset)
        
        # ------------ Uncertainty policy ------------
        index_certain = uncertainty_selection(uncertainty.squeeze())
        se_forward = se_forward[index_certain]
        forward_pred = forward_pred[index_certain]
        images_generated = images_generated[index_certain]
        conditions = conditions[index_certain]

    else:
        se_forward, forward_pred = se_between_target_and_prediction(target, images_generated, forward, trainset)

    # Move variable to cpu
    images_generated = images_generated.squeeze(1).cpu().detach().numpy()
    conditions = conditions.cpu().detach().numpy()

    # ------------ Compare the forward model and the Measure from morphomnist ------------
    thickness = compute_thickness_ground_truth(images_generated)

    # ------------ Compute the mse between the target and the morpho measure predictions ------------
    se_measure = mse(target, thickness.values)

    # Measure on trained data
    train_img_label = pd.DataFrame(np.around(testset.labels).values.tolist(), columns=['label'])
    select_img_label_index = random.sample(train_img_label[train_img_label['label']==target].index.values.tolist() , ncol)
    image_from_test = testset.x_data.numpy()[select_img_label_index].squeeze(1)
    x_train = Variable(testset.x_data[select_img_label_index].type(FloatTensor))

    # ------------ Compute the mse between testset and the forward model predictions ------------
    if bayesian:
        se_forward_train, forward_pred_train, uncertainty_train = se_between_target_and_prediction(target, x_train, forward, trainset)
    else:
        se_forward_train, forward_pred_train = se_between_target_and_prediction(target, x_train, forward, trainset)

    thickness_train = compute_thickness_ground_truth(image_from_test)

    se_train = mse(target, thickness_train.values)

    # ------------ EDA of the best x* generated ------------
    
    top_values = 10

    index = np.argsort(se_forward)[:top_values]
    forward_mse_mean = np.mean(mse(target,thickness.values[index])); forward_mse_std = np.std(mse(target,thickness.values[index])); global_mean = np.mean(se_measure);

    print()
    print(f" ------------ Best forward image ------------")
    print(f"MSE measure pred = {forward_mse_mean} ± {forward_mse_std} ")
    print(f"MSE morpho on Generated data: {global_mean}")

    # Transormf output to real value
    model_pred = trainset.scaler.inverse_transform(forward_pred.reshape(-1, 1)).squeeze()
    model_pred_train = trainset.scaler.inverse_transform(forward_pred_train.reshape(-1, 1)).squeeze()
    conditions = trainset.scaler.inverse_transform(conditions.reshape(-1, 1)).squeeze()

    '''# Create true target values
    test_img_label = pd.DataFrame(np.around(testset.labels).values.tolist(), columns=['label'])
    select_img_label_index = random.sample(test_img_label[test_img_label['label']==target].index.values.tolist(), sample_number)
    image_from_test = testset.x_data.numpy()[select_img_label_index]'''

    # Compute FID values
    fid_value_gen, kid_value_gen = compute_fid_mnist_monte_carlo(np.expand_dims(images_generated, 1), target, testset, sample_number_fid_kid)

    plots_results(target, model_pred, model_pred_train, thickness.values, thickness_train.values, se_forward, conditions, testset, images_generated, select_img_label_index, fid_value_gen, kid_value_gen, nrow=2, ncol=4)

    return [forward_mse_mean, forward_mse_std], global_mean, fid_value_gen, kid_value_gen

def save_obj_csv(d, path):
    d.to_csv(path+'.csv', index=False)

def load_obj_csv(path):
    return pd.read_csv(path+'.csv')