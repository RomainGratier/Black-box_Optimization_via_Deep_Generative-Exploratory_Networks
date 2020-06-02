import multiprocessing
import os
import shutil

import numpy as np

import sys

from morphomnist import io, perturb
from morphomnist.morpho import ImageMorphology

THRESHOLD = .5
UP_FACTOR = 4

PERTURBATIONS = [
    perturb.Thinning(amount=.4),
    perturb.Thinning(amount=.6),
    perturb.Thinning(amount=.8),
    #perturb.Thickening(0.5),
    #perturb.Thickening(1.0),
    #perturb.Thickening(1.5),
    #perturb.Thickening(2.0),
    #perturb.Thickening(2.5),
]


def process_image(args):
    i, img = args
    np.random.seed()
    morph = ImageMorphology(img, THRESHOLD, UP_FACTOR)
    out_imgs = [morph.downscale(morph.binary_image)] + \
               [morph.downscale(pert(morph)) for pert in PERTURBATIONS]
    return out_imgs

raw_dir = "/data/morpho_mnist/original"
dataset_root = "/data/processed"
#dataset_names = ["thickened05", "thickened10", "thickened15", "thickened20",  "thickened25"]
dataset_names = ["thinned04", "thinned06", "thinned08"]
pool = multiprocessing.Pool()
for subset in ["train", "t10k"]:

    # Get data pathes
    imgs_filename = f"{subset}-images-idx3-ubyte.gz"
    labels_filename = f"{subset}-labels-idx1-ubyte.gz"
    raw_imgs = io.load_idx(os.path.join(raw_dir, imgs_filename))

    # Multiprocessing for the image perturbation
    gen = pool.imap(process_image, enumerate(raw_imgs), chunksize=100)
    try:
        import tqdm
        gen = tqdm.tqdm(gen, total=len(raw_imgs), unit='img', ascii=True)
    except ImportError:
        def plain_progress(g):
            print(f"\rProcessing images: 0/{len(raw_imgs)}", end='')
            for i, res in enumerate(g):
                print(f"\rProcessing images: {i + 1}/{len(raw_imgs)}", end='')
                yield res
            print()
        gen = plain_progress(gen)

    # Save the results
    result = zip(*list(gen))
    for dataset_name, imgs in zip(dataset_names, result):
        imgs = np.array(imgs)
        dataset_dir = os.path.join(dataset_root, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        io.save_idx(imgs, os.path.join(dataset_dir, imgs_filename))
        shutil.copy(os.path.join(raw_dir, labels_filename), dataset_dir)

pool.close()
pool.join()
