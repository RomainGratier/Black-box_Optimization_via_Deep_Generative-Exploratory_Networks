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
    
import src.config as cfg
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
sns.set()

size_ax = 8
def plot_fc(df):
    f, ax = plt.subplots(figsize=(5,4), dpi=300)
    ax.set(yscale="log")
    sns.lineplot(x='iteration', y='value', hue='variable', 
                data=pd.melt(df, ['iteration']))
    
    plt.legend(prop={'size': 8})
    ax.tick_params(axis='x', labelsize=size_ax)
    ax.tick_params(axis='y', labelsize=size_ax)
    plt.ylabel('', fontsize=16)

def plot_gan_results():
    path_res = os.path.join(cfg.models_path, cfg.gan_path)
    df = pd.read_csv(os.path.join(path_res, f'results_gan.csv'))
    df = df.dropna(axis=1)
    df = df[['iteration', 'fid_in', 'fid_out', 'kid_in', 'kid_out', 'mse_in']]
    df.loc[df['kid_in'] < 0, 'kid_in'] = 0.0001
    df.loc[df['kid_out'] < 0, 'kid_out'] = 0.0001
    df.columns = ['iteration', 'fid in', 'fid out', 'kid in', 'kid out', 'mse']
    plot_fc(df)

plot_gan_results()

import os
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import src.config as cfg
from glob import glob

sns.set(style="darkgrid")

def plot_bayesian(df_acc, title, dist):
    epochs = df_acc.groupby('iteration')

    res = []
    save_flag = []
    for epoch in epochs:
        sub_df = epoch[1]
        n_ep = sub_df['iteration'].iloc[0]
        mse_forward = np.mean(sub_df['se_forward_avg'])
        corr_epistemic = pearsonr(sub_df['epistemic'], sub_df['se_forward_avg'])[0]
        corr_aleatoric = pearsonr(sub_df['aleatoric'], sub_df['se_forward_avg'])[0]
        selected_pred = sub_df[sub_df['uncertainty_flag']]
        mse_forward_selected = np.mean(selected_pred['se_forward_avg'])
        mean_epistemic = np.mean(sub_df['epistemic'])
        mean_aleatoric = np.mean(sub_df['aleatoric'])

        if selected_pred['save_flag'].iloc[0]:
            save_flag.append(True)
        else:
            save_flag.append(False)

        res.append([n_ep, mse_forward, mse_forward_selected, corr_epistemic, corr_aleatoric])

    df_res = pd.DataFrame(res, columns=['iteration', 'mse forward', 'mse forward selected', 'epistemic correlation', 'aleatoric correlation'])

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
            plt.plot(df_res.loc[index,'iteration'], df_res.loc[index, 'mse forward'], 'o', color='red')
    plt.title(title)
    plt.show()

def plot_frequentist(df_acc, title, dist):
    epochs = df_acc.groupby('iteration')

    res = []
    for epoch in epochs:
        sub_df = epoch[1]
        n_ep = sub_df['iteration'].iloc[0]
        mse_forward = np.mean(sub_df['se_forward'])

        res.append([n_ep, mse_forward])

    df_res = pd.DataFrame(res, columns=['iteration', 'mse forward'])
    plt.figure(figsize=(10,5), dpi=200)
    if dist == 'out':
        plt.ylim(top=1.5)
        plt.ylim(bottom=-0.3)
    if dist == 'in':
        plt.ylim(top=0.6)
        plt.ylim(bottom=-0)

    sns.lineplot(x='iteration', y='value', hue='variable', 
                 data=pd.melt(df_res, ['iteration']))

    plt.title(title)
    plt.show()

def plot_forward_results():
    models_folder = os.path.join(cfg.models_path, cfg.forward_path)
    list_results_paths = sorted(glob(models_folder+'/*.csv'), key=os.path.getctime)
    
    for path in list_results_paths:
        print(path)
        
        # Distribution
        if path.split('/')[-1].split('_')[1] == 'in':
            distribution = 'In distribution'
            dist='in'
        else:
            distribution = 'Out distribution'
            dist='out'
        
        # Non Bayesian
        if (path.split('/')[-1] == 'results_in_lenet.csv') | (path.split('/')[-1] == 'results_out_lenet.csv') | (path.split('/')[-1] == 'results_in_fc.csv') | (path.split('/')[-1] == 'results_out_fc.csv'):
            title = f"{distribution} results for {path.split('/')[-1].split('.')[0]}"
            plot_frequentist(pd.read_csv(path), title, dist)
        
        # Bayesian
        elif (path.split('/')[-1] == 'results_in_lenet_lrt_softplus.csv') | (path.split('/')[-1] == 'results_out_lenet_lrt_softplus.csv'):
            df = pd.read_csv(path)
            title = f"{distribution} | Bayesian LeNet LRT SoftPplus | Epistemic mean={round(df['epistemic'].mean(), 3)}   Aleatoric mean={round(df['aleatoric'].mean(), 3)}"
            plot_bayesian(df, title, dist)
            
        elif (path.split('/')[-1] == 'results_in_lenet_lrt_relu.csv') | (path.split('/')[-1] == 'results_out_lenet_lrt_relu.csv'):
            df = pd.read_csv(path)
            title = f"{distribution} | Bayesian LeNet LRT RELU | Epistemic mean={round(df['epistemic'].mean(), 3)}   Aleatoric mean={round(df['aleatoric'].mean(), 3)}"
            plot_bayesian(df, title, dist)
        
        elif (path.split('/')[-1] == 'results_in_lenet_bbb_relu.csv') | (path.split('/')[-1] == 'results_out_lenet_bbb_relu.csv'):
            df = pd.read_csv(path)
            title = f"{distribution} | Bayesian LeNet BBB RELU | Epistemic mean={round(df['epistemic'].mean(), 3)}   Aleatoric mean={round(df['aleatoric'].mean(), 3)}"
            plot_bayesian(df, title, dist)

        elif (path.split('/')[-1] == 'results_in_lenet_bbb_softplus.csv') | (path.split('/')[-1] == 'results_out_lenet_bbb_softplus.csv'):
            df = pd.read_csv(path)
            title = f"{distribution} | Bayesian LeNet BBB SoftPplus | Epistemic mean={round(df['epistemic'].mean(), 3)}   Aleatoric mean={round(df['aleatoric'].mean(), 3)}"
            plot_bayesian(df, title, dist)

plot_forward_results()

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