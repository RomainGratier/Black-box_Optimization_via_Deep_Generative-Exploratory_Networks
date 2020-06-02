import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import os

from src.data import MNISTDatasetLeNet, RotationDatasetLeNet
import src.config as cfg
from src.generative_model import LeNet5

def train(epoch, net, trainloader):
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


def test(net, testloader):
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

    avg_loss /= len(test_dataset)
    acc = float(total_correct) / len(test_dataset)
    loss_avg = avg_loss.detach().cpu().item()

    print('Test Avg. Loss: %f, Accuracy: %f' % (loss_avg, acc))

    return loss_avg, acc

def train_and_test(epoch, net, trainloader, testloader):
    train_loss, train_batch = train(epoch, net, trainloader)
    test_loss, test_acc = test(net, testloader)

    return test_loss, test_acc, net


def create_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs('Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/models_fid_kid', exist_ok=True)
    
    batch_size = 128
    
    if (cfg.experiment == 'min_mnist')|(cfg.experiment == 'max_mnist'):
        model_name = 'lenet_mnist'
        trainloader = DataLoader(MNISTDatasetLeNet('train', data_path=cfg.data_path),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8)
        testloader = DataLoader(MNISTDatasetLeNet('test', data_path=cfg.data_path),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8)
        
    elif cfg.experiment == 'rotation_dataset':
        model_name = 'lenet_rot'
        trainloader = DataLoader(RotationDatasetLeNet('train', data_path=cfg.data_path),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8)
        testloader = DataLoader(RotationDatasetLeNet('test', data_path=cfg.data_path),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8)
    
    net = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=2e-3)

    res = pd.DataFrame(columns=['loss', 'accuracy'])
    acc_res = 0
    for e in range(1, 400):
        test_loss, test_acc, net = train_and_test(e, net, trainloader, testloader)
        res = res.append(pd.DataFrame({'loss': [test_loss], 'accuracy' : [test_acc]}))
        res.append([test_loss, test_acc])
        if test_acc > acc_res:
            print(f'New accuracy : {test_acc}')
            torch.save(net, os.path.join(path_lenet, f"{model_name}.pth"))
            res.to_csv(os.path.join(path_lenet, f"results_{model_name}.csv"), index=False)
            acc_res = test_acc