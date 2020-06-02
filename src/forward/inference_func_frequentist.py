from __future__ import print_function

import numpy as np
import pandas as pd
import os
import argparse

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

import src.config_frequentist as cfg
import src.config as cfg_glob
from src.forward.cnn_models import AlexNet, LeNet, ThreeConvThreeFC
from src.data import getDataset, getDataloader
from src.metrics import se
import src.forward.utils

from torch.nn import functional as F

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

def getModel(net_type, inputs, outputs):
    print(inputs)
    if (net_type == 'lenet'):
        return LeNet(outputs, inputs)
    elif (net_type == 'alexnet'):
        return AlexNet(outputs, inputs)
    elif (net_type == '3conv3fc'):
        return ThreeConvThreeFC(outputs, inputs)
    elif (net_type == 'fc'):
        return FC(outputs, inputs)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')

def train_model(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    accs = []
    for i, (inputs, labels) in enumerate(trainloader, 1):

        optimizer.zero_grad()
        inputs = F.interpolate(inputs, size=32)
        inputs, labels = inputs.to(device), labels.to(device)
            
        net_out = net(inputs)
        loss = criterion(net_out.squeeze(1).double(), labels.double())
        loss.backward()
        optimizer.step()

        training_loss += loss.cpu().data.numpy()

        # accuracy measures model's ability
        se_model = se(net_out.cpu().detach().numpy().squeeze(1), labels.cpu().detach().numpy())
        accs.extend(se_model)

    return training_loss/len(trainloader), np.mean(accs)


def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.eval() #net.train()
    valid_loss = 0.0

    accs_val = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = F.interpolate(inputs, size=32)

        net_out = net(inputs)

        valid_loss += criterion(net_out.squeeze(1), labels.double()).item()

        # accuracy measures model's ability
        se_model = se(net_out.cpu().detach().numpy().squeeze(1), labels.cpu().detach().numpy())
        
        accs_val.extend(se_model)

    return valid_loss/len(validloader), np.mean(accs_val)


def test_model(net, criterion, testinloader, testoutloader, iteration=None, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.eval()

    df_acc_in = pd.DataFrame(columns=['iteration', 'label', 'val_pred', 'se_forward'])
    df_acc_out = pd.DataFrame(columns=['iteration', 'label', 'val_pred', 'se_forward'])

    accs_val = []
    accs_avr = []
    accs_epistemic = []
    accs_aleatoric = []

    for i, (inputs, labels) in enumerate(testinloader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = F.interpolate(inputs, size=32)       

        net_out = net(inputs)

        # accuracy measures model's ability
        se_model = se(net_out.cpu().detach().numpy().squeeze(1), labels.cpu().detach().numpy())
        
        df = pd.DataFrame(np.full(inputs.shape[0], iteration), columns=['iteration'])
        df['label'] = labels.cpu().detach().numpy()
        df['val_pred'] = net_out.cpu().detach().numpy()
        df['se_forward'] = se_model
        df_acc_in = df_acc_in.append(df, ignore_index=True)
    
    for i, (inputs, labels) in enumerate(testoutloader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = F.interpolate(inputs, size=32)      

        for j in range(num_ens):
            net_out = net(inputs)
            
        # accuracy measures model's ability
        se_model = se(net_out.cpu().detach().numpy().squeeze(1), labels.cpu().detach().numpy())
        
        df = pd.DataFrame(np.full(inputs.shape[0], iteration), columns=['iteration'])
        df['label'] = labels.cpu().detach().numpy()
        df['val_pred'] = net_out.cpu().detach().numpy()
        df['se_forward'] = se_model
        df_acc_out = df_acc_out.append(df, ignore_index=True)

    return df_acc_in, df_acc_out


def run_frequentist(net_type='lenet'):
    
    ckpt_dir = os.path.join(cfg_glob.models_path, cfg_glob.forward_path)

    # Hyper Parameter settings
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size

    trainset, testset_in, testset_out, inputs, outputs = getDataset()
    train_loader, valid_loader, test_loader_in, test_loader_out = getDataloader(
        trainset, testset_in, testset_out, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs).to(device)

    ckpt_name = os.path.join(ckpt_dir, f'model_{net_type}.pth')

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = Adam(net.parameters(), lr=lr)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_max = np.Inf

    df_acc_final_in = pd.DataFrame(columns=['iteration', 'label', 'val_pred', 'se_forward', 'save_flag'])
    df_acc_final_out = pd.DataFrame(columns=['iteration', 'label', 'val_pred', 'se_forward', 'save_flag'])

    for epoch in range(n_epochs+1):  # loop over the dataset multiple times

        train_loss, train_acc = train_model(net, optimizer, criterion, train_loader, epoch=epoch, num_epochs=n_epochs)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader, epoch=epoch, num_epochs=n_epochs)
        df_acc_in, df_acc_out = test_model(net, criterion, test_loader_in, test_loader_out, iteration=epoch*trainset.len, epoch=epoch, num_epochs=n_epochs)
        
        lr_sched.step(valid_loss)

        # Validation flag
        df_acc_in['save_flag'] = False
        df_acc_out['save_flag'] = False
        
        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc))
        print('TESTING : IN dist  Forward mse: {:.4f} ||  OUT dist  Forward mse:{:.4f}'.format(
            np.mean(df_acc_in['se_forward']), np.mean(df_acc_out['se_forward'])))

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
            df_acc_in['save_flag'] = True
            df_acc_out['save_flag'] = True

        df_acc_final_in = df_acc_final_in.append(df_acc_in, ignore_index=True)
        df_acc_final_out = df_acc_final_out.append(df_acc_out, ignore_index=True)

        # ------------ Save results ------------
        df_acc_final_in.to_csv(os.path.join(ckpt_dir,f'results_in_{net_type}.csv'))
        df_acc_final_out.to_csv(os.path.join(ckpt_dir,f'results_out_{net_type}.csv'))