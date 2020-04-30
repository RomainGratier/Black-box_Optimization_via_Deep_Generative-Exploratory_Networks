import numpy as np
import pandas as pd
import gzip
import struct
from glob import glob
import os

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
            data_new = load_idx(path)[train_index.values]
            save_idx(data_new, os.path.join(out_path, name))
        if name.split('.')[-1] == 'csv':
            data_new = pd.read_csv(path).loc[train_index.values]
            data_new.to_csv(os.path.join(out_path, name), index=False)

    for path in test_paths:
        name = path.split('/')[-1]
        if name.split('.')[-1] == 'gz':
            data_new = load_idx(path)[train_index.values]
            save_idx(data_new, os.path.join(out_path, name))
        if name.split('.')[-1] == 'csv':
            data_new = pd.read_csv(path).loc[train_index.values]
            data_new.to_csv(os.path.join(out_path, name), index=False)

def data_resampling(input_path, out_path):
    # Get data
    df_train, df_test = get_data(input_path)

    # Manipulate the thickness distribution
    train_uniform = uniform_resampling(df_train)
    test_uniform = uniform_resampling(df_test, sample_size = 697)

    # Load the rest of the data and keep only the selected indexes
    load_manipulate_save(input_path, out_path, train_uniform['index'], test_uniform['index'])


# Absolute path
FOLDER = '../data/processed/'
INPUT_NAME = 'original_thic'
OUTPUT_NAME = 'original_thic_resample'
INPUT_PATH = os.path.join(FOLDER, INPUT_NAME)
OUTPUT_PATH = os.path.join(FOLDER, OUTPUT_NAME)

# Created nes
os.makedirs(OUTPUT_PATH, exist_ok=True)

data_resampling(INPUT_PATH, OUTPUT_PATH)

