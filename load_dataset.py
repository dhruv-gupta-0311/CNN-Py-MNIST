import os
import gzip
import urllib.request
import numpy as np

def load_mnist():
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    os.makedirs("local_mnist", exist_ok=True)
    for key, fname in files.items():
        path = os.path.join("local_mnist", fname)
        if not os.path.exists(path=path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(base + fname, path)

    def read_images(fname):
        with gzip.open(os.path.join("local_mnist", fname), 'rb') as f:
            f.read(16)  # skip header
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

    def read_labels(fname):
        with gzip.open(os.path.join("local_mnist", fname), 'rb') as f:
            f.read(8)   # skip header
            return np.frombuffer(f.read(), dtype=np.uint8)

    return (read_images(files["train_images"]), read_labels(files["train_labels"])),\
           (read_images(files["test_images"]),  read_labels(files["test_labels"]))