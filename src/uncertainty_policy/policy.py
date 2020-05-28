import numpy as np
import src.config_inference as cfginf

def uncertainty_selection(uncertainty, policy_type='quantile'):
    if policy_type == 'quantile':
        quantile = np.quantile(uncertainty, cfginf.quantile_rate_uncertainty_policy)
        new_index = np.argwhere(uncertainty < quantile)
    return new_index.squeeze()