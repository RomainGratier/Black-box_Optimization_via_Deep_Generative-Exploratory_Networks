import numpy as np 
import os 
import itertools
import torch 
from tabulate import tabulate

from src.data import MNISTDataset, RotationDataset
import src.config as cfg

from src.inference.utils import monte_carlo_inference_mse_batch, monte_carlo_inference_fid_kid_sampling, monte_carlo_inference_qualitative, plot_best_acc_pred

def compute_quantitative_and_qualitative_inference(target, metrics=['optimization', 'qualitative', 'fid_kid', 'l1_l2'], distributions=['in', 'out'], bayesian_model_types=["lrt", "bbb"], activation_types=["softplus", "relu"], output_type='latex', decimals=2):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'The pipeline is currently running on {device}')

    if (cfg.experiment == 'max_mnist') | (cfg.experiment == 'min_mnist'):
        testset = MNISTDataset('full', 
                               y_feature=cfg.feature,
                               folder=cfg.data_folder,
                               data_path=cfg.data_path)
    elif cfg.experiment == 'rotation_dataset':
        testset = RotationDataset('full', 
                                  folder=cfg.data_folder,
                                  data_path=cfg.data_path)
        metrics.remove('l1_l2')

    modes = list(itertools.product(bayesian_model_types, activation_types))
    forward_models = ['_'.join(couple) for couple in modes]
    forward_models.append('non bayesian')

    dict_results = {}

    print()
    print(f"The models path is : {cfg.models_path}")
    print()
    
    for metric_type in metrics:
        print(f" --------------- Computing {metric_type} results ---------------")
        print()
        if output_type == 'latex':
            # Each element in the table list is a row in the generated table
            inter = '$\pm$'
            if metric_type == 'l1_l2':
                headers = ["Model", "Distribution", "mse$_r$", "mse$_p$", "mre$_r$", "mre$_p$"]
            elif metric_type == 'fid_kid':
                headers = ["Model", "Distribution", "fid$_r$", "fid$_p$", "kid$_r$", "kid$_p$"]

        results = []
        for forward_type in forward_models:
            for distribution in distributions:
                if distribution=='in':
                    sample_number_fid_kid=2000
                elif distribution=='out':
                    sample_number_fid_kid=2000

                print(f"Computing inference with forward : {forward_type}")
                gan_path = os.path.join(cfg.models_path, cfg.gan_path, f'best_generator_{distribution}_distribution.pth')
                if forward_type == 'non bayesian':
                    bayesian = False
                    forward_path = os.path.join(cfg.models_path, 'forward/model_fc.pth')
                else:
                    bayesian = True
                    forward_path = os.path.join(cfg.models_path, f'forward/model_lenet_{forward_type}.pth')

                if (os.path.isfile(forward_path)) & (os.path.isfile(forward_path)):
                    forward_model = torch.load(forward_path, map_location=device).eval()
                    generator_model = torch.load(gan_path, map_location=device).eval()

                    if (metric_type=='l1_l2')|(metric_type=='fid_kid'):
                        if metric_type == 'l1_l2':
                            stat1_in_rand, stat1_in_pol, stat2_out_rand, stat2_out_pol = monte_carlo_inference_mse_batch(distribution, generator_model, forward_model, testset, sample_number=2000, bayesian=bayesian)
                        elif metric_type == 'fid_kid':
                            stat1_in_rand, stat1_in_pol, stat2_out_rand, stat2_out_pol = monte_carlo_inference_fid_kid_sampling(distribution, generator_model, forward_model, testset, sample_number_fid_kid=sample_number_fid_kid, bayesian=bayesian)

                        print((stat1_in_rand[0]-stat1_in_pol[0])/stat1_in_rand[0])
                        print((stat2_out_rand[0]-stat2_out_pol[0])/stat2_out_rand[0])
    
                        if output_type == 'latex':
                            stat1_in_rand = f"{np.around(stat1_in_rand[0],decimals=decimals)}{inter}{np.around(stat1_in_rand[1],decimals=decimals)}"
                            stat1_in_pol = f"{np.around(stat1_in_pol[0],decimals=decimals)}{inter}{np.around(stat1_in_pol[1],decimals=decimals)}"
                            stat2_in_rand = f"{np.around(stat2_out_rand[0],decimals=decimals)}{inter}{np.around(stat2_out_rand[1],decimals=decimals)}"
                            stat2_in_pol = f"{np.around(stat2_out_pol[0],decimals=decimals)}{inter}{np.around(stat2_out_pol[1],decimals=decimals)}"
    
                        results.append([forward_type, distribution, stat1_in_rand, stat1_in_pol, stat2_in_rand, stat2_in_pol])
    
                        if output_type == 'latex':
                            print(tabulate(results, headers, tablefmt='latex_raw'))

                    elif metric_type=='qualitative':
                        if distribution == 'in':
                            monte_carlo_inference_qualitative(distribution, forward_type, generator_model, forward_model, testset, sample_number=2000, bayesian=bayesian)
                        elif distribution == 'out':
                            monte_carlo_inference_qualitative(distribution, forward_type, generator_model, forward_model, testset, sample_number=2000, bayesian=bayesian, random_certainty=False)
                    
                    elif metric_type=='optimization':
                        plot_best_acc_pred(target, forward_type, generator_model, forward_model, testset, sample_number=2000, bayesian=bayesian, random_certainty=False)
    
                else:
                    print('WARNING: no model was found')
        
        if metric_type == 'qualitative':
            continue

        if output_type == 'latex':
            dict_results[metric_type] = tabulate(results, headers, tablefmt='latex_raw')

        else:
            dict_results[metric_type] = results

    return dict_results