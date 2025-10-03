
########## CNN APPLICATIONS #############

### 1. Download MNIST dataset and implement a MNIST classifier using CNN PyTorch library.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

def load_data(batch_size=64, val_ratio=0.1):
    # Load MNIST without normalization first (to check mean and standard deviation).
    full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    img, label = full_train[0]
    # print(img.shape)   # Output: torch.Size([1, 28, 28])
    all_imgs = torch.stack([img for img, _ in full_train])  # stack all images
    mean, std = torch.mean(all_imgs), torch.std(all_imgs)  # compute mean and std
    # print(f"Computed Mean: {mean.item()}, Std: {std.item()}")

    # transforms.Compose: chain multiple pre-processing steps.
    # transforms.ToTensor(): converts image [0, 255] to float tensor [0, 1].
    # transforms.Normalize(mean, std): standardizes image for fast training.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Reload dataset with normalization -
    full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)    # 60,000 images
    val_size = int(val_ratio * len(full_train))   # val_ratio: fraction of training data to use for validation
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)   # 10,000 images

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

class MnistCnn(nn.Module):
    # More layers - model learns more complex features but may overfit.
    # conv_channels (output channels / no. of filters): the more it is, the more features can be learnt.
    # fc_units: Here, first layer has 128 neurons and second layer has 64 neurons.
    def __init__(self, num_conv_layers=2, conv_channels=None, fc_units=[128, 64], num_classes=10):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32 + 16*i for i in range(num_conv_layers)]   # Default: [32, 48, 64] if not specified

        # Build convolutional layers -
        layers = []
        in_channels = 1   # Grayscale image
        for i in range(num_conv_layers):
            # Conv1: I/P=28*28*1, out_channels=32, filter size=3*3, O/P: 28*28*32 (formula: ((n+2p-f)/S)+1 : n=28, f=3, p=1 (p=(f-1)/2 : same), S=1)).
            layers.append(nn.Conv2d(in_channels, conv_channels[i], kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(conv_channels[i]))   # Normalizes each channel's values - faster convergence, stabilizes training.
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))   # Down-samples feature maps by 2*2 - less computation
            in_channels = conv_channels[i]   # Update input channels for the next layer.
        self.conv = nn.Sequential(*layers)   # Packs the list of layers into a single module.

        # Compute flattened size for FC layers -
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)   # Creates a dummy tensor representing a single MNIST image.
            # self.conv: passes this dummy image via all layers defined earlier.
            # view: flattens the output into 1D vector per image.
            flatten_size = self.conv(dummy).view(1, -1).size(1)    # Needed as I/P size for the 1st FC layer.

        # Build fully connected layers -
        fc_layers = []
        input_size = flatten_size
        for units in fc_units:
            fc_layers.append(nn.Linear(input_size, units))
            fc_layers.append(nn.BatchNorm1d(units))
            fc_layers.append(nn.ReLU())
            input_size = units
        fc_layers.append(nn.Linear(input_size, num_classes))  # final output layer
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x   # No softmax here (nn.CrossEntropyLoss will take care of it).

def train_one_epoch(model, device, loader, optimizer, criterion):
    model.train()   # sets model in training mode (activates dropout, batch-norm).
    running_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)   # avg. loss per batch.

def evaluate(model, device, loader, criterion):
    model.eval()    # Disables dropout and batch-norm update.
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    return avg_loss, accuracy

def hyperparameter_tuning(train_loader, val_loader, device, max_evals=20):
    space = {
        "num_conv_layers": hp.choice("num_conv_layers", [1, 2, 3]),    # hp.choice: categorical choice
        "conv1": hp.choice("conv1", [16, 32, 64]), "conv2": hp.choice("conv2", [32, 64, 128]),
        "conv3": hp.choice("conv3", [64, 128, 256]), "fc1": hp.choice("fc1", [64, 128, 256]),
        "fc2": hp.choice("fc2", [32, 64, 128]), "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-2))   # continuous value on log scale for learning rate
    }
    criterion = nn.CrossEntropyLoss()

    def objective(params):
        num_conv = params["num_conv_layers"]   # Gets the number of convolutional layers chosen by Hyperopt.
        conv_channels = []   # Builds a list of output channels for each conv layer.
        if num_conv >= 1: conv_channels.append(params["conv1"])
        if num_conv >= 2: conv_channels.append(params["conv2"])
        if num_conv >= 3: conv_channels.append(params["conv3"])
        fc_units = [params["fc1"], params["fc2"]]    # Chooses the number of hidden neurons for the fully connected layers.

        # Instantiates the MnistCnn (class) model with the current hyperparameters.
        model = MnistCnn(num_conv_layers=num_conv, conv_channels=conv_channels, fc_units=fc_units).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])   # to update weights during training

        for _ in range(2):  # 2 epochs of training - just to estimate the performance.
            train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)   # Hyperopt uses validation loss to decide which hyperparameters are better.
        return {"loss": val_loss, "status": STATUS_OK, "accuracy": val_acc}   # STATUS_OK indicates successful run.
    trials = Trials()   # Creates an object to store results of each trial.
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best

def run_training(epochs=5, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = load_data(batch_size=batch_size)
    best_params = hyperparameter_tuning(train_loader, val_loader, device)
    print("Best Hyperparameters:", best_params)

    # Now we want to train the final model (earlier we did just a quick trial), using the best hyperparameters that Hyperopt found.
    num_conv_layers = [1, 2, 3][best_params["num_conv_layers"]]
    conv_channels = []
    if num_conv_layers >= 1: conv_channels.append([16, 32, 64][best_params["conv1"]])
    if num_conv_layers >= 2: conv_channels.append([32, 64, 128][best_params["conv2"]])
    if num_conv_layers >= 3: conv_channels.append([64, 128, 256][best_params["conv3"]])
    fc_units = [[64, 32], [128, 64], [256, 128]][best_params["fc1"] % 3]  # safe mapping
    lr = best_params["lr"]

    # Final model
    model = MnistCnn(num_conv_layers=num_conv_layers, conv_channels=conv_channels, fc_units=fc_units).to(
        device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    return model

if __name__ == "__main__":
    run_training(epochs=5, batch_size=64)
