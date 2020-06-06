# libraries and data
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set(color_codes=True)

def visualize_gan_training(acc_gen, sample_interval, batch_size):
    data_in = np.mean(mse_gan_in_distribution,axis=1)
    data_out = np.mean(mse_gan_out_distribution,axis=1)
    df=pd.DataFrame({'x': [i * sample_interval * batch_size for i in range(0,len(data_in))], 'y': data_in, 'z': data_out})

    # fist line:
    plt.subplots(figsize=(10,5), dpi=100)
    ax = sns.regplot(x="x", y="y", data=df, label = 'In distribution',
                 scatter_kws={"s": 80},
                 order=3, ci=None)
    ax = sns.regplot(x="x", y="z", data=df, label = 'Out distribution',
                 scatter_kws={"s": 80},
                 order=2, ci=None)

    plt.ylabel('Mean Squared Error')
    plt.xlabel('iterations')
    plt.legend()
    plt.title("Accuracy without reweighting strategy")

import os
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import src.config as cfg
from glob import glob

sns.set(style="darkgrid")

def plot_forward_results(df_acc, dist='out'):
    models_folder = os.path.join(cfg.models_path, cfg.forward_path)
    list_results_paths = sorted(glob(models_folder+'*.csv'), key=os.path.getctime)
    
    for path in list_results_paths:
        
        # Distribution
        if path.split('/')[-1].split('_')[1] == 'in':
            distribution = 'In distribution'
        else:
            distribution = 'Out distribution'
        
        # Non Bayesian
        if (path.split('/')[-1] == 'results_in_lenet.csv') | (path.split('/')[-1] == 'results_in_lenet.csv'):
            title = f'{distribution} results for non bayesian LeNet'
        
        # Bayesian
        elif (path.split('/')[-1] == 'results_in_lenet_lrt_softplus.csv') | (path.split('/')[-1] == 'results_out_lenet_lrt_softplus.csv'):
            title = f'{distribution} results for bayesian LeNet LRT SOFTPLUS'
            
        elif (path.split('/')[-1] == 'results_in_lenet_lrt_relu.csv') | (path.split('/')[-1] == 'results_out_lenet_lrt_relu.csv'):
            title = f'{distribution} results for bayesian LeNet LRT RELU' 
        
        elif (path.split('/')[-1] == 'results_in_lenet_bbb_relu.csv') | (path.split('/')[-1] == 'results_out_lenet_bbb_relu.csv'):
            title = f'{distribution} results for bayesian LeNet BBB RELU'

        elif (path.split('/')[-1] == 'results_in_lenet_bbb_softplus.csv') | (path.split('/')[-1] == 'results_out_lenet_bbb_softplus.csv'):
            title = f'{distribution} results for bayesian LeNet BBB SOFTPLUS'

        df_acc = pd.read_csv(path)

        epochs = df_acc.groupby('iteration')
        epochs['se_forward_avg'].mean()
        res = []
        save_flag = []
        for epoch in epochs:
            sub_df = epoch[1]
            n_ep = sub_df['iteration'].iloc[0]
            mse_forward = np.mean(sub_df['se_forward_avg'])
            corr = pearsonr(sub_df['epistemic'], sub_df['se_forward_avg'])[0]
            selected_pred = sub_df[sub_df['uncertainty_flag']]
            mse_forward_selected = np.mean(selected_pred['se_forward_avg'])

            if selected_pred['save_flag'].iloc[0]:
                save_flag.append(True)
            else:
                save_flag.append(False)

            res.append([n_ep, corr, mse_forward, mse_forward_selected])

        df_res = pd.DataFrame(res, columns=['iteration', 'corr', 'mse_forward', 'mse_forward_selected'])
        print(df_res)
        plt.figure(figsize=(10,5), dpi=200)
        if dist == 'out':
            plt.ylim(top=1.5)
            plt.ylim(bottom=-0.3)
        if dist == 'in':
            plt.ylim(top=0.6)
            plt.ylim(bottom=-0)

        sns.lineplot(x='iteration', y='value', hue='variable', 
                     data=pd.melt(df_res, ['iteration']))
        for index, flag in enumerate(save_flag):
            if flag:
                plt.plot(df_res.loc[index,'iteration'], df_res.loc[index, 'mse_forward'], 'o', color='red')
        plt.show()

import src.config as cfg
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

sns.set()

def plot_fc(df):
    plt.figure(figsize=(5,4), dpi=300)
    with sns.axes_style("whitegrid"):
        sns.lineplot(x='iteration', y='value', hue='variable', 
                    data=pd.melt(df, ['iteration']))

def plot_gan_results():
    path_res = os.path.join(cfg.models_path, cfg.gan_path)
    df = pd.read_csv(os.path.join(path_res, f'results_gan.csv'))
    print(df)
    print(df.columns)
    df = df.dropna(axis=1)
    df_ls = []
    if 'fid_in' in df.columns:
        df_fid = df[['iteration', 'fid_in', 'fid_out']]
        df_ls.append(df_fid)
    if 'kid_in' in df.columns:
        df_kid = df[['iteration', 'kid_in', 'kid_out']]
        df_ls.append(df_kid)
    if 'mse_in' in df.columns:
        df_mse = df[['iteration', 'mse_in', 'mse_out']]
        df_ls.append(df_mse)
    for df in df_ls:
        plot_fc(df)

plot_gan_results()