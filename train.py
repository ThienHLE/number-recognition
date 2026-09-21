import os
import gzip
import struct
import urllib.request
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from model import MLP  
#downloading images and labels
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
#if file is missing then download, if not then move on(only download once)
def download(root="data"):
    os.makedirs(root, exist_ok=True)
    for fname in FILES.values():
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(BASE_URL + fname, path)
#reading the image file and returning a tensor
#shape: (N,1,28,28), dtype:float32, range: [0,1] after normalization
def parse_images(path_gz):
    with gzip.open(path_gz, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        #MNIST image file has magic number 2051
        if magic != 2051:
            raise ValueError(f"Bad magic for images: {magic}")
        #convert raw pixel bytes
        data = f.read(n * rows * cols)
        x = torch.frombuffer(data, dtype=torch.uint8).clone()
        #normalizatoin to [0,1]
        x = x.view(n, 1, rows, cols).float() / 255.0
        return x
#reading the image file and returning a tensor
#shape: (N), dtype: int64
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
    #eval mode turns off dropouts
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        #forward pass (feedforward)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

def main():
    #sets random seed used for weight init and data split randomness
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #hyperparameters
    batch_size = 128
    lr = 1e-3
    epochs = 20
    patience = 3
    dropout = 0.3

    download("data")
    #load and parse MNIST tensors
    x_train = parse_images(os.path.join("data", FILES["train_images"]))
    y_train = parse_labels(os.path.join("data", FILES["train_labels"]))
    x_test  = parse_images(os.path.join("data", FILES["test_images"]))
    y_test  = parse_labels(os.path.join("data", FILES["test_labels"]))

    #wrap raw tensors into pytorch dataset
    full_train = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    #split training into training/validation
    train_len = int(0.9 * len(full_train))
    val_len = len(full_train) - train_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)
    #creates model and training tools
    model = MLP(dropout=dropout).to(device)
    #optimizer updates weights using gradients from backpropagation
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    #training loop + earling stops 
    best_val = 0.0
    bad = 0

    for epoch in range(1, epochs + 1):
        #training mode turns on dropout
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            #clears old gradients from last iteration
            opt.zero_grad()
            #computes how wrong predictions are
            loss = loss_fn(model(x), y)
            #backpropagatin: computes gradients of loss w.r.t. weights
            loss.backward()
            opt.step()
        
        #after each epoch, eval on validation and test
        val_acc = accuracy(model, val_loader, device)
        test_acc = accuracy(model, test_loader, device)
        print(f"Epoch {epoch:02d} | val_acc={val_acc:.4f} | test_acc={test_acc:.4f}")
        #save best model checkpoint
        if val_acc > best_val:
            best_val = val_acc
            bad = 0
            #dictionary of all weights/bias tensors
            torch.save(model.state_dict(), "mnist_mlp_best.pt")
        else:
            bad += 1
            #if we haven't improved for 'patience' epochs, stop training early
            if bad >= patience:
                print("Early stopping.")
                break

    print(f"Best val_acc={best_val:.4f}")
    print("Saved: mnist_mlp_best.pt")

if __name__ == "__main__":
    main()