import os
import shutil

import numpy as np
import pandas as pd

from morphomnist import io


def interleave_dfs(dfs, indices, keys):
    df = pd.concat(dfs, axis=0, keys=keys)
    df = df.swaplevel(0, 1)
    df = df.loc[list(enumerate(indices))]
    df.index = df.index.droplevel(1)
    return df


if __name__ == '__main__':
    data_root = "../Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/data/processed/"
    dataset_names = ["thickened05", "thickened10", "thickened15", "thickened20"]
    pairings = [(0, 1, 2), (0, 3, 4)]
    for pairing in pairings[1:]:
        for subset in ["train", "t10k"]:
            labels_filename = f"{subset}-labels-idx1-ubyte.gz"
            images_filename = f"{subset}-images-idx3-ubyte.gz"
            metrics_filename = f"{subset}-morpho.csv"
            pert_filename = f"{subset}-pert-idx1-ubyte.gz"

            data_dirs = [os.path.join(data_root, dataset_names[i]) for i in pairing]
            imgs_paths = [os.path.join(data_dir, images_filename) for data_dir in data_dirs]
            metrics_paths = [os.path.join(data_dir, metrics_filename) for data_dir in data_dirs]
            all_images = np.array([io.load_idx(path) for path in imgs_paths])
            all_metrics = [pd.read_csv(path, index_col='index') for path in metrics_paths]

            num = all_images[0].shape[0]
            indices = np.random.choice(len(pairing), size=num)
            pert = np.asarray(pairing)[indices]
            inter_images = all_images[indices, np.arange(num)]
            inter_metrics = interleave_dfs(all_metrics, pert, pairing)

            inter_dir = os.path.join(data_root, '+'.join([dataset_names[i] for i in pairing]))
            print(f"Saving results to {inter_dir}/...")
            #os.makedirs(inter_dir, exist_ok=True)
            inter_pert_path = os.path.join(inter_dir, pert_filename)
            inter_images_path = os.path.join(inter_dir, images_filename)
            inter_metrics_path = os.path.join(inter_dir, metrics_filename)
            inter_labels_path = os.path.join(inter_dir, labels_filename)
            print(f"- Saving perturbation labels to {pert_filename}")
            #io.save_idx(pert, inter_pert_path)
            print(f"- Saving interleaved images to {images_filename}")
            #io.save_idx(inter_images, inter_images_path)
            print(f"- Saving interleaved metrics to {metrics_filename}")
            #inter_metrics.to_csv(inter_metrics_path)
            print(f"- Copying class labels to {labels_filename}")
            #shutil.copy(os.path.join(data_dirs[0], labels_filename), inter_dir)
            print()
