import numpy as np
import pandas as pd
import gzip
import struct
from glob import glob
import os

def find_data(dirs):
    ''' glob the different data from the main path '''
    data = [[path for path in glob(os.path.join(path_i,'*.*'))] for path_i in dirs]
    return data

def _load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data

def load_idx(path: str) -> np.ndarray:
    """Reads an array in IDX format from disk.
    Parameters
    ----------
    path : str
        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.
    Returns
    -------
    np.ndarray
        Output array of dtype ``uint8``.
    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)

def _save_uint8(data, f):
    data = np.asarray(data, dtype=np.uint8)
    f.write(struct.pack('BBBB', 0, 0, 0x08, data.ndim))
    f.write(struct.pack('>' + 'I' * data.ndim, *data.shape))
    f.write(data.tobytes())

def save_idx(data: np.ndarray, path: str):
    """Writes an array to disk in IDX format.
    Parameters
    ----------
    data : array_like
        Input array of dtype ``uint8`` (will be coerced if different dtype).
    path : str
        Path of the output file. Will compress with `gzip` if path ends in '.gz'.
    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'wb') as f:
        _save_uint8(data, f)

def merge_datasets(list_paths, data_type):
    if data_type == 'gz':
        # Define the name
        name = list_paths[0].split('/')[-1]

        # Load the data
        data = [load_idx(path) for path in list_paths]

        # Merge them
        print(data[0].shape)
        data_merged = np.concatenate(data)
        print(data_merged.shape)

        # Save them
        save_idx(data_merged, os.path.join(OUTPUT_PATH, name))

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
FOLDER = '../data/processed/'
NAME = 'original_thic'
OUTPUT_PATH = os.path.join(FOLDER, NAME)

# Created nes
os.makedirs(OUTPUT_PATH, exist_ok=True)

dirs = ['thinned06', 'thickened05', 'thickened10', 'thickened15', 'thickened20', 'thickened25']
merge_dataset(dirs)

