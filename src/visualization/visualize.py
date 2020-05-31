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
    
bayesian_model_types = ["bbb", "lrt"]
activation_types = ["relu", "softplus"]
models_path = MODELS_PATH
distributions = ['in', 'out']

import os
import torch
import numpy as np
import itertools
from tabulate import tabulate

def compute_results_inference(testset, models_path, distributions, bayesian_model_types, activation_types, sample_number_fid_kid=1000, size_sample=10, output_type='latex', decimals=2):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    modes = list(itertools.product(bayesian_model_types, activation_types))
    forward_models = ['_'.join(couple) for couple in modes]
    forward_models.append('non bayesian')

    if output_type == 'latex':
        # Each element in the table list is a row in the generated table
        inter = '$\pm$'

    results = []
    for forward in forward_models:
        for distribution in distributions:
            print(f"Computing inference with forward : {forward}")
            if forward == 'non bayesian':
                forward_path = os.path.join(models_path, 'frequentist_forward_resampled/model_lenet.pth')

            forward_path = os.path.join(models_path, f'bayesian_forward_resampled/model_lenet_{forward}.pth')
            gan_path = os.path.join(models_path, f'generative_resampled/best_generator_{distribution}_distribution.pth')
            if (os.path.isfile(forward_path)) & (os.path.isfile(forward_path)):
                forward_model = torch.load(forward_path, map_location=device).eval()
                generator_model = torch.load(gan_path, map_location=device).eval()
                fid_in_rand, fid_in_pol, kid_in_rand, kid_in_pol = monte_carlo_inference_fid_kid('in', generator_model, forward_model, testset, sample_number_fid_kid = sample_number_fid_kid, size_sample=size_sample)

                if output_type == 'latex':
                    fid_in_rand = f"{np.around(fid_in_rand[0],decimals=decimals)}{inter}{np.around(fid_in_rand[1],decimals=decimals)}"
                    kid_in_rand = f"{np.around(kid_in_rand[0],decimals=decimals)}{inter}{np.around(kid_in_rand[1],decimals=decimals)}"
                    fid_in_pol = f"{np.around(fid_in_pol[0],decimals=decimals)}{inter}{np.around(fid_in_pol[1],decimals=decimals)}"
                    kid_in_pol = f"{np.around(kid_in_pol[0],decimals=decimals)}{inter}{np.around(kid_in_pol[1],decimals=decimals)}"
                    
                results.append([forward, distribution, fid_in_rand, fid_in_pol, kid_in_rand, kid_in_pol])

            else:
                print('WARNING: no model was found')
    
    
    if output_type == 'latex':
        headers = ["Model", "Distribution", "fid$_r$", "fid$_p$", "kid$_r$", "kid$_p$"]
        return tabulate(results, headers, tablefmt='latex_raw')

    else:
        return results 
    
