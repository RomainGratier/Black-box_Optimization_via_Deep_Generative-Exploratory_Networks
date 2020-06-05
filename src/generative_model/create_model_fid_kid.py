import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os

from src.data import MNISTDatasetLeNet, RotationDatasetLeNet
from src.metrics import se
import src.config as cfg
from src.generative_model.models import LeNet5, LeNet5Regressor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from collections import OrderedDict

def train(epoch, net, trainloader, optimizer, criterion):
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        loss.backward()
        optimizer.step()

        return loss_list, batch_list


def test(net, testloader, criterion, test_size):
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(testloader):
        images = images.to(device)
        labels = labels.to(device)

        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= test_size
    acc = float(total_correct) / test_size
    loss_avg = avg_loss.detach().cpu().item()

    print('Test Avg. Loss: %f, Accuracy: %f' % (loss_avg, acc))

    return loss_avg, acc

def test_regressor(net, testloader, criterion, test_size):
    net.eval()
    res_se = []
    avg_loss = 0.0
    for i, (images, labels) in enumerate(testloader):
        images = images.to(device)
        labels = labels.to(device)

        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach()

        res_se.extend(se(pred.cpu().squeeze(1), labels.cpu()))

    avg_loss /= test_size
    acc = np.mean(res_se)
    loss_avg = avg_loss.detach().cpu().item()

    print('Test Avg. Loss: %f, Accuracy: %f' % (loss_avg, acc))

    return loss_avg, acc

def train_and_test(epoch, net, trainloader, testloader, optimizer, criterion, test_size):
    train_loss, train_batch = train(epoch, net, trainloader, optimizer, criterion)
    test_loss, test_acc = test(net, testloader, criterion, test_size)

    return test_loss, test_acc, net

def train_and_test_regressor(epoch, net, trainloader, testloader, optimizer, criterion, test_size):
    train_loss, train_batch = train(epoch, net, trainloader, optimizer, criterion)
    test_loss, test_acc = test_regressor(net, testloader, criterion, test_size)

    return test_loss, test_acc, net


def create_model_fid_kid():

    path_lenet = cfg.model_fidkid_path
    os.makedirs(path_lenet, exist_ok=True)

    batch_size = 128
    if (cfg.experiment == 'min_mnist')|(cfg.experiment == 'max_mnist'):
        model_name = 'lenet_mnist'
        trainloader = DataLoader(MNISTDatasetLeNet('train', 
                                                   folder=cfg.data_folder,
                                                   data_path=cfg.data_path),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8)
        testloader = DataLoader(MNISTDatasetLeNet('test', 
                                                  folder=cfg.data_folder, 
                                                  data_path=cfg.data_path),
                                
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8)
        test_size = len(MNISTDatasetLeNet('test', 
                                          folder=cfg.data_folder, 
                                          data_path=cfg.data_path))

        criterion = nn.CrossEntropyLoss().to(device)
        net = LeNet5().to(device)
        optimizer = optim.Adam(net.parameters(), lr=2e-3)
        test_fct = train_and_test

    elif cfg.experiment == 'rotation_dataset':
        model_name = 'lenet_rot'
        trainloader = DataLoader(RotationDatasetLeNet('train',
                                                      folder=cfg.data_folder,
                                                      data_path=cfg.data_path),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8)
        testloader = DataLoader(RotationDatasetLeNet('test', 
                                                     folder=cfg.data_folder, 
                                                     data_path=cfg.data_path),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8)
        test_size = len(RotationDatasetLeNet('test', 
                                             folder=cfg.data_folder, 
                                             data_path=cfg.data_path))

        criterion = nn.MSELoss().to(device)
        net = LeNet5Regressor().to(device)
        optimizer = optim.Adam(net.parameters(), lr=2e-3)
        test_fct = train_and_test_regressor

    res = pd.DataFrame(columns=['loss', 'accuracy'])
    acc_res_ent = 0
    acc_res_reg = np.inf
    for e in range(1, 400):
        test_loss, test_acc, net = test_fct(e, net, trainloader, testloader, optimizer, criterion, test_size)
        res = res.append(pd.DataFrame({'loss': [test_loss], 'accuracy' : [test_acc]}))
        res.append([test_loss, test_acc])

        if (cfg.experiment == 'min_mnist')|(cfg.experiment == 'max_mnist'):
            if test_acc > acc_res:
                print(f'New accuracy : {test_acc}')
                torch.save(net.state_dict(), os.path.join(path_lenet, f"{model_name}.pth"))
                res.to_csv(os.path.join(path_lenet, f"results_{model_name}.csv"), index=False)
                acc_res = test_acc

        if cfg.experiment == 'rotation_dataset':
            if test_acc < acc_res_reg:
                print(f'New accuracy : {test_acc}')
                torch.save(net.state_dict(), os.path.join(path_lenet, f"{model_name}.pth"))
                res.to_csv(os.path.join(path_lenet, f"results_{model_name}.csv"), index=False)
                acc_res = test_acc