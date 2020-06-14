import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.forward.layers import FlattenLayer

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class AlexNet(nn.Module):

    def __init__(self, num_classes, inputs=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LeNet(nn.Module):
    def __init__(self, num_classes, inputs=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(inputs, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return(out)


class ThreeConvThreeFC(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """
    def __init__(self, outputs, inputs):
        super(ThreeConvThreeFC, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 32, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, 5, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            FlattenLayer(2 * 2 * 128),
            nn.Linear(2 * 2 * 128, 1000),
            nn.Softplus(),
            nn.Linear(1000, 1000),
            nn.Softplus(),
            nn.Linear(1000, outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class FC(nn.Module):
    def __init__(self, num_classes, inputs=3):
        super(FC, self).__init__()
        self.fc1   = nn.Linear(inputs, 256)
        self.fc2   = nn.Linear(256, 256)
        self.fc3   = nn.Linear(256, num_classes)

    def forward(self, x):
        out = out.view(x.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return(out)