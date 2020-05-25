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
