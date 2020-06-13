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
    net_outs_mu = []
    net_outs_var = []
    batches = batch.unsqueeze(0).repeat(T, 1, 1, 1, 1)
    preds = []
    epistemics = []
    aleatorics = []
    
    for i in range(T):
        net_out_mu, net_out_var, _ = model(batches[i].to(device))
        net_outs_mu.append(net_out_mu)
        net_outs_var.append(net_out_var)
    
    for sample in range(batch.shape[0]):
        # for each sample in a batch
        pred = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in net_outs_mu], dim=0)
        pred = torch.mean(pred, dim=0)
        preds.append(pred)

        p_hat = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in net_outs_mu], dim=0).detach().cpu().numpy()
        p_hat_var = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in net_outs_var], dim=0).detach().cpu().numpy()
        p_bar = np.mean(p_hat, axis=0)

        temp = np.square(p_hat - np.expand_dims(p_bar, 0))
        epistemic = np.sum(temp)/(T-1)
        epistemics.append(epistemic)

        aleatoric = np.sum(p_hat_var)/T
        aleatorics.append(aleatoric)
        
    epistemic = np.vstack(epistemics)  # (batch_size, dim)
    aleatoric = np.vstack(aleatorics)  # (batch_size, dim)
    preds = torch.cat([i for i in preds]).cpu().detach().numpy() # (batch_size, dim)

    return preds, epistemic, aleatoric
