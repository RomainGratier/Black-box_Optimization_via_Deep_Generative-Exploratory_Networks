import argparse
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import torchvision
from torch.nn import functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_uncertainty_per_batch(model, batch, device, T=15, normalized=False):
    batch_predictions = []
    net_outs = []
    batches = batch.unsqueeze(0).repeat(T, 1, 1, 1, 1)
    preds = []
    epistemics = []
    aleatorics = []
    
    for i in range(T):  # for T batches
        net_out, _ = model(batches[i].to(device))
        net_outs.append(net_out)
        batch_predictions.append(net_out)
    
    for sample in range(batch.shape[0]):
        # for each sample in a batch
        pred = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in net_outs], dim=0)
        pred = torch.mean(pred, dim=0)
        preds.append(pred)

        p_hat = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in batch_predictions], dim=0).detach().cpu().numpy()
        p_bar = np.mean(p_hat, axis=0)

        temp = p_hat - np.expand_dims(p_bar, 0)
        epistemic = np.dot(temp.T, temp)
        epistemic = np.sum(epistemic)/T
        epistemics.append(epistemic)

        aleatoric = np.var(p_hat)
        aleatorics.append(aleatoric)

    epistemic = np.vstack(epistemics)  # (batch_size, categories)
    aleatoric = np.vstack(aleatorics)  # (batch_size, categories)
    preds = torch.cat([i for i in preds]).cpu().detach().numpy()  # (batch_size, categories) #torch.cat([i.unsqueeze(0) for i in preds]).cpu().detach().numpy()  # (batch_size, categories)

    return preds, epistemic, aleatoric
