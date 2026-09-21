import os
import gzip
import struct
import urllib.request
import torch
from torch.utils.data import TensorDataset, DataLoader

from model import load_model
#only need test files for this
BASE_URL = "http://yann.lecun.com/exdb/mnist/"
FILES = {
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def download(root="data"):
    os.makedirs(root, exist_ok=True)
    for fname in FILES.values():
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(BASE_URL + fname, path)

def parse_images(path_gz):
    with gzip.open(path_gz, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic for images: {magic}")
        data = f.read(n * rows * cols)
        x = torch.frombuffer(data, dtype=torch.uint8).clone()
        x = x.view(n, 1, rows, cols).float() / 255.0
        return x

def parse_labels(path_gz):
    with gzip.open(path_gz, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic for labels: {magic}")
        data = f.read(n)
        y = torch.frombuffer(data, dtype=torch.uint8).clone().long()
        return y

@torch.no_grad()
def accuracy(model, loader, device):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

def main():
    download("data")
    x = parse_images(os.path.join("data", FILES["test_images"]))
    y = parse_labels(os.path.join("data", FILES["test_labels"]))

    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    model, device = load_model("mnist_mlp_best.pt")
    acc = accuracy(model, loader, device)
    print(f"MNIST test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()