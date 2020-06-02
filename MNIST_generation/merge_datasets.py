import numpy as np
import pandas as pd
from glob import glob
import os

import sys
from morphomnist import io

def find_data(dirs):
    ''' glob the different data from the main path '''
    data = [[path for path in glob(os.path.join(path_i,'*.*'))] for path_i in dirs]
    return data

def merge_datasets(list_paths, data_type):
    if data_type == 'gz':
        # Define the name
        name = list_paths[0].split('/')[-1]

        # Load the data
        data = [io.load_idx(path) for path in list_paths]

        # Merge them
        print(data[0].shape)
        data_merged = np.concatenate(data)
        print(data_merged.shape)

        # Save them
        io.save_idx(data_merged, os.path.join(OUTPUT_PATH, name))

    if data_type == 'csv':
        # Define the name
        name = list_paths[0].split('/')[-1]

        # Load the data
        data = [pd.read_csv(path) for path in list_paths]

        # Merge them
        print(data[0].shape)
        data_merged = pd.concat(data, ignore_index=True)
        print(data_merged.shape)

        # Reindex the data
        data_merged['index'] = data_merged.index

        # Save them
        data_merged.to_csv(os.path.join(OUTPUT_PATH, name), index=False)

def merger(ls_paths):
    ''' Merge the dataset to form one main dataset '''
    list_of_data_to_merge = np.array(ls_paths).T.tolist()

    for data in list_of_data_to_merge:
        check = set([path.split('/')[-1] for path in data])
        if len(check) != 1:
            print(f'WARNING THE PIPELINE MAIGHT BE BROKEN {file} != {check}')
        else:
            if list(check)[-1].split('.')[-1] == 'gz':
                print(f'Successfuly assessed the data {check}')
                merge_datasets(data, data_type = 'gz')

            elif list(check)[-1].split('.')[-1] == 'csv':
                print(f'Successfuly assessed the data {check}')
                merge_datasets(data, data_type = 'csv')

            else:
                print(f'WARNING THE PIPELINE MAIGHT BE BROKEN : {check}')

def merge_dataset(dirs):
    ''' Select two path of data and merge them '''
    # Open the path of the data
    ls_absolute_path = [os.path.join(FOLDER, path_i) for path_i in dirs]
    ls_paths = find_data(ls_absolute_path)

    merger(ls_paths)

# Absolute path
FOLDER = '/data/processed/'
NAME = 'original_thic'
OUTPUT_PATH = os.path.join(FOLDER, NAME)

# Created nes
os.makedirs(OUTPUT_PATH, exist_ok=True)

dirs = ['thinned06', 'thickened05', 'thickened10', 'thickened15', 'thickened20', 'thickened25']
merge_dataset(dirs)

