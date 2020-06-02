import numpy as np
import pandas as pd
from glob import glob
import os

import sys
from morphomnist import io

def get_data(path):
    df_train = pd.read_csv(os.path.join(INPUT_PATH, 'train-morpho.csv'))
    df_test = pd.read_csv(os.path.join(INPUT_PATH, 't10k-morpho.csv'))
    return df_train, df_test

def uniform_resampling(df, sample_size = 5300):
    pd_uniform = df[df['thickness'] < 10.5]

    # Round the value to discretize the domain
    pd_uniform['round'] = round(pd_uniform["thickness"])

    # Group the different descritzed numbers
    grouped = pd_uniform.groupby('round')

    print()
    print(f'Real distribution : ')
    print(pd_uniform.groupby('round')['round'].count())

    # Resample each groups
    grouped_uniform = grouped.apply(lambda x: x.sample(n=sample_size)).reset_index(drop=True)

    print()
    print(f'Artificial uniform distribution : ')
    print(grouped_uniform.groupby('round')['round'].count())

    return grouped_uniform

def load_manipulate_save(input_path, out_path, train_index, test_index):
    train_paths = [path for path in glob(os.path.join(input_path, 'train*'))]; test_paths = [path for path in glob(os.path.join(input_path, 't10k*'))];
    
    for path in train_paths:
        name = path.split('/')[-1]
        if name.split('.')[-1] == 'gz':
            data_new = io.load_idx(path)[train_index.values]
            io.save_idx(data_new, os.path.join(out_path, name))
        if name.split('.')[-1] == 'csv':
            data_new = pd.read_csv(path).loc[train_index.values]
            data_new.to_csv(os.path.join(out_path, name), index=False)
            print(' ------------ CHECK ------------ ')
            data_new['round'] = round(data_new["thickness"])
            print(data_new.groupby('round')['round'].count())

    for path in test_paths:
        name = path.split('/')[-1]
        if name.split('.')[-1] == 'gz':
            data_new = io.load_idx(path)[test_index.values]
            io.save_idx(data_new, os.path.join(out_path, name))
        if name.split('.')[-1] == 'csv':
            data_new = pd.read_csv(path).loc[test_index.values]
            data_new.to_csv(os.path.join(out_path, name), index=False)
            print(' ------------ CHECK ------------ ')
            data_new['round'] = round(data_new["thickness"])
            print(data_new.groupby('round')['round'].count())

def data_resampling(input_path, out_path):
    # Get data
    df_train, df_test = get_data(input_path)

    # Manipulate the thickness distribution
    train_uniform = uniform_resampling(df_train)
    test_uniform = uniform_resampling(df_test, sample_size = 697)

    # Load the rest of the data and keep only the selected indexes
    load_manipulate_save(input_path, out_path, train_uniform['index'], test_uniform['index'])


# Absolute path
FOLDER = '/data/processed/'
INPUT_NAME = 'original_thic'
OUTPUT_NAME = 'original_thic_resample'
INPUT_PATH = os.path.join(FOLDER, INPUT_NAME)
OUTPUT_PATH = os.path.join(FOLDER, OUTPUT_NAME)

# Created nes
os.makedirs(OUTPUT_PATH, exist_ok=True)

data_resampling(INPUT_PATH, OUTPUT_PATH)

