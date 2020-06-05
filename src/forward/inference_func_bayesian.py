from __future__ import print_function

import numpy as np
import pandas as pd

import os
import argparse
from scipy.stats.stats import pearsonr   

import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F
from sklearn import preprocessing

import src.config_bayesian as cfg
import src.config as cfg_glob
import src.forward.utils
from src.forward.metrics import ELBO, calculate_kl, get_beta
from src.forward.bcnn_models import BBB3Conv3FC, BBBAlexNet, BBBLeNet
from src.data import getDataset, getDataloader
from src.metrics import se
from src.forward.uncertainty_estimation import get_uncertainty_per_batch
from src.uncertainty_policy import uncertainty_selection
from src.utils import save_ckp, load_ckp_forward

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == '3conv3fc'):
        return BBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'fc'):
        return BBBFC(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')

def train_model(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):

        optimizer.zero_grad()
        inputs = F.interpolate(inputs, size=32)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
        
        kl = kl / num_ens
        kl_list.append(kl.item())

        beta = get_beta(i-1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(net_out.squeeze(1), labels.float(), kl, beta)
        loss.backward()
        optimizer.step()

        training_loss += loss.cpu().data.numpy()

        # accuracy measures model's ability
        
        se_model = se(net_out.cpu().detach().numpy().squeeze(1), labels.cpu().detach().numpy())
        accs.extend(se_model)

    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.eval() #net.train()
    valid_loss = 0.0

    accs_val = []
    accs_avr = []
    accs_epistemic = []
    accs_aleatoric = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = F.interpolate(inputs, size=32)#.float()        

        #outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl

        beta = get_beta(i-1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(net_out.squeeze(1), labels.float(), kl, beta).item()

        # accuracy measures model's ability
        se_model = se(net_out.cpu().detach().numpy().squeeze(1), labels.cpu().detach().numpy())

        preds, epistemic, aleatoric = get_uncertainty_per_batch(net, inputs, device, T=15, normalized=False)

        # accuracy measures model's ability
        se_model_averaged = se(preds, labels.cpu().detach().numpy())

        accs_val.extend(se_model)
        accs_avr.extend(se_model_averaged)
        accs_epistemic.extend(epistemic)
        accs_aleatoric.extend(aleatoric)

    return valid_loss/len(validloader), np.mean(accs_val), np.mean(accs_avr), np.mean(accs_epistemic), np.mean(accs_aleatoric)


def test_model(net, criterion, testinloader, testoutloader, iteration=None, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.eval()

    df_acc_in = pd.DataFrame(columns=['iteration', 'label', 'val_pred', 'pred_w_uncertainty', 'epistemic', 'aleatoric', 'se_forward', 'se_forward_avg'])
    df_acc_out = pd.DataFrame(columns=['iteration', 'label', 'val_pred', 'pred_w_uncertainty', 'epistemic', 'aleatoric', 'se_forward', 'se_forward_avg'])

    for i, (inputs, labels) in enumerate(testinloader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = F.interpolate(inputs, size=32)        

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl

        beta = get_beta(i-1, len(testinloader), beta_type, epoch, num_epochs)

        # accuracy measures model's ability
        se_model = se(net_out.cpu().detach().numpy().squeeze(1), labels.cpu().detach().numpy())

        # get prediction and uncertainty
        preds, epistemic, aleatoric = get_uncertainty_per_batch(net, inputs, device, T=15, normalized=False)

        # accuracy measures model's ability
        se_model_averaged = se(preds, labels.cpu().detach().numpy())
        
        df = pd.DataFrame(np.full(inputs.shape[0], iteration), columns=['iteration'])
        df['label'] = labels.cpu().detach().numpy().reshape(-1,1)
        df['val_pred'] = net_out.cpu().detach().numpy().reshape(-1,1)
        df['pred_w_uncertainty'] = preds
        df['epistemic'] = epistemic
        df['aleatoric'] = aleatoric
        df['se_forward'] = se_model
        df['se_forward_avg'] = se_model_averaged
        df_acc_in = df_acc_in.append(df, ignore_index=True)
    
    for i, (inputs, labels) in enumerate(testoutloader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = F.interpolate(inputs, size=32)#.float()        

        #outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl

        beta = get_beta(i-1, len(testoutloader), beta_type, epoch, num_epochs)

        # accuracy measures model's ability
        se_model = se(net_out.cpu().detach().numpy().squeeze(1), labels.cpu().detach().numpy())

        # get prediction and uncertainty
        preds, epistemic, aleatoric = get_uncertainty_per_batch(net, inputs, device, T=15, normalized=False)

        # accuracy measures model's ability
        se_model_averaged = se(preds, labels.cpu().detach().numpy())

        df = pd.DataFrame(np.full(inputs.shape[0], iteration), columns=['iteration'])
        df['label'] = labels.cpu().detach().numpy().reshape(-1,1)
        df['val_pred'] = net_out.cpu().detach().numpy().reshape(-1,1)
        df['pred_w_uncertainty'] = preds
        df['epistemic'] = epistemic
        df['aleatoric'] = aleatoric
        df['se_forward'] = se_model
        df['se_forward_avg'] = se_model_averaged
        df_acc_out = df_acc_out.append(df, ignore_index=True)

    return df_acc_in, df_acc_out

def run_bayesian(net_type='lenet', verbose=False):
    
    ckpt_dir = os.path.join(cfg_glob.models_path, cfg_glob.forward_path)

    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    trainset, testset_in, testset_out, inputs, outputs = getDataset()
    train_loader, valid_loader, test_loader_in, test_loader_out = getDataloader(
        trainset, testset_in, testset_out, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type).to(device)

    ckpt_name = os.path.join(ckpt_dir, f'model_{net_type}_{layer_type}_{activation_type}.pth')

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_max = np.Inf
    df_acc_final_in = pd.DataFrame(columns=['iteration','label', 'val_pred', 'pred_w_uncertainty', 'epistemic', 'aleatoric', 'se_forward', 'se_forward_avg', 'uncertainty_flag', 'save_flag'])
    df_acc_final_out = pd.DataFrame(columns=['iteration', 'label', 'val_pred', 'pred_w_uncertainty', 'epistemic', 'aleatoric', 'se_forward', 'mse_forward_avg', 'uncertainty_flag', 'save_flag'])
    
    start_epoch = 0
    ckp_path = os.path.join(cfg_glob.models_path,f'checkpoints/forward_{net_type}_{layer_type}_{activation_type}')

    if os.path.isdir(ckp_path):
        print(F'Found a valid check point !')
        print("Do you want to resume the training? yes or no")
        resume = str(input())
        if resume == 'yes':
            net, optimizer, lr_sched, start_epoch, df_acc_final_in, df_acc_final_out, valid_loss_max = load_ckp_forward(ckp_path, net, optimizer, lr_sched)

    for epoch in range(start_epoch, n_epochs+1):  # loop over the dataset multiple times

        train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        valid_loss, valid_acc, valid_acc_avr, valid_epi, valid_ale = validate_model(net, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        df_acc_in, df_acc_out = test_model(net, criterion, test_loader_in, test_loader_out, iteration=epoch*trainset.len, num_ens=valid_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        
        lr_sched.step(valid_loss)

        # ------------ Uncertainty policy ------------
        index_certain_in = uncertainty_selection(df_acc_in['epistemic'].values)
        flag_vec_in = np.full(df_acc_in.shape[0], False)
        flag_vec_in[index_certain_in] = True
        df_acc_in['uncertainty_flag'] = flag_vec_in
        
        index_certain_out = uncertainty_selection(df_acc_out['epistemic'].values)
        flag_vec_out = np.full(df_acc_out.shape[0], False)
        flag_vec_out[index_certain_out] = True
        df_acc_out['uncertainty_flag'] = flag_vec_out
        
        # ------------ Validation flag ------------
        df_acc_out['save_flag'] = False
        df_acc_in['save_flag'] = False
        
        # ------------ Correlation ------------
        stand = preprocessing.StandardScaler()
        df_corr_in = pd.DataFrame(stand.fit_transform(df_acc_in[['se_forward_avg', 'epistemic']]))
        corr_in = list(pearsonr(df_corr_in.iloc[:,0], df_corr_in.iloc[:,1]))
        df_corr_out = pd.DataFrame(stand.fit_transform(df_acc_out[['se_forward_avg', 'epistemic']]))
        corr_out = list(pearsonr(df_corr_out.iloc[:,0], df_corr_out.iloc[:,1]))
        
        print('TESTING : IN dist  Forward mse: {:.4f}\tForward avg mse: {:.4f}\tepistemic mean: {:.4f}\taleatoric mean: {:.4f}\tcorrelation uncertainty {:.4f} p_val {:.4f} ||  OUT dist  Forward mse:{:.4f}\tForward avg mse: {:.4f}\tepistemic mean: {:.4f}\taleatoric mean: {:.4f} \tcorrelation uncertainty {:.4f} p_val {:.4f}'.format(
            np.mean(df_acc_in['se_forward']), np.mean(df_acc_in['se_forward_avg']), np.mean(df_acc_in['epistemic']), np.mean(df_acc_in['aleatoric']), corr_in[0], corr_in[1], np.mean(df_acc_out['se_forward']), np.mean(df_acc_out['se_forward_avg']), np.mean(df_acc_out['epistemic']), np.mean(df_acc_out['aleatoric']),  corr_out[0], corr_out[1]))
        print('Epoch: {} Training Loss: {:.4f}\tTraining Accuracy: {:.4f}\tValidation Loss: {:.4f}\tValidation Accuracy: {:.4f}\tValidation Accuracy Avr: {:.4f}\ttrain_kl_div: {:.4f}\tepistemic: {:.4f}\taleatoric: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc, valid_acc_avr, train_kl, valid_epi, valid_ale))

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            if cuda:
                torch.save(net.cpu(), ckpt_name)
                net.cuda()
            else:
                torch.save(net, ckpt_name)
            valid_loss_max = valid_loss
            df_acc_out['save_flag'] = True
            df_acc_in['save_flag'] = True

        df_acc_final_in = df_acc_final_in.append(df_acc_in, ignore_index=True)
        df_acc_final_out = df_acc_final_out.append(df_acc_out, ignore_index=True)

        # ------------ Save results ------------
        df_acc_final_in.to_csv(os.path.join(ckpt_dir,f'results_in_{net_type}_{layer_type}_{activation_type}.csv'))
        df_acc_final_out.to_csv(os.path.join(ckpt_dir,f'results_out_{net_type}_{layer_type}_{activation_type}.csv'))
        
        # ------------ Save checkpoints ------------
        save_ckp({
            'epoch': epoch + 1,
            'state_dict_forward': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sched': lr_sched.state_dict(),
            'df_acc_final_in': df_acc_final_in,
            'df_acc_final_out': df_acc_final_out,
            'valid_loss_max': valid_loss_max,
            }, ckp_path)

        if verbose:
            labels_in = np.around(df_acc_in['label'].values, decimals = 0).squeeze()
            for label in np.unique(labels_in):
                print(label)
                indexe = np.argwhere(labels_in == label)
                corr_out_selected = list(pearsonr(df_corr_in.iloc[indexe.squeeze(),0], df_corr_in.iloc[indexe.squeeze(),1]))
                print(corr_out_selected)

            labels_out = np.around(df_acc_out['label'].values, decimals = 0).squeeze()
            for label in np.unique(labels_out):
                print(label)
                indexe = np.argwhere(labels_out == label)
                corr_out_selected = list(pearsonr(df_corr_out.iloc[indexe.squeeze(),0], df_corr_out.iloc[indexe.squeeze(),1]))
                print(corr_out_selected)

            ## --------------------------------------------------------------------------------------------------------------

            print()
            print(f"---------- IN distribution epistemic min : {df_acc_in['epistemic'].min()}")
            print(f"---------- IN distribution epistemic max : {df_acc_in['epistemic'].max()}")
            print()
            print(f"---------- OUT distribution epistemic min : {df_acc_out['epistemic'].min()}")
            print(f"---------- OUT distribution epistemic max : {df_acc_out['epistemic'].max()}")

            print(np.mean(df_acc_in['se_forward_avg']))
            print(f"ACC forward avg IN dist : {np.mean(df_acc_in['se_forward_avg'].iloc[index_certain_in])}")
            print(np.mean(df_acc_out['se_forward_avg']))
            print(f"ACC forward avg OUT dist : {np.mean(df_acc_out['se_forward_avg'].iloc[index_certain_out])}")
            print()