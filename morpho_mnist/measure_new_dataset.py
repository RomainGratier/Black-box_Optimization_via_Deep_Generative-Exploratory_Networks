import multiprocessing
import os

from morphomnist import io, measure


def measure_dir(data_dir, pool):
    for name in ['t10k', 'train']:
        in_path = os.path.join(data_dir, name + "-images-idx3-ubyte.gz")
        out_path = os.path.join(data_dir, name + "-morpho.csv")
        print(f"Processing MNIST data file {in_path}...")
        data = io.load_idx(in_path)
        df = measure.measure_batch(data, pool=pool, chunksize=100)
        df.to_csv(out_path, index_label='index')
        print(f"Morphometrics saved to {out_path}")


def main(data_dirs):
    with multiprocessing.Pool() as pool:
        for data_dir in data_dirs:
            measure_dir(data_dir, pool)


if __name__ == '__main__':
    ROOT = '../data/processed/'
    #dirs = ["thickened05", "thickened10", "thickened15", "thickened20",  "thickened25", "thinned06"]
    dirs = ["thinned06"]
    datadir = [os.path.join(ROOT, dir) for dir in dirs]
    datadir = [os.path.join(ROOT, dir) for dir in dirs]

    print(f"Processing the following directories :\n{datadir}")

    assert all(os.path.exists(data_dir) for data_dir in datadir)

    main(datadir)
