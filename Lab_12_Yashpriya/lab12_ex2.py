
########## CNN APPLICATIONS #############

### 2. Implement CNN using PyTorch for image classification using cifar10 dataset - https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html .
### Plot train error vs increasing number of layers. After some point, the training error increases with the number of layers.

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from lab12_ex1 import MnistCnn, train_one_epoch, evaluate    # Few MNIST functions can be re-used here.

def load_cifar10(batch_size=64, val_ratio=0.1):
    # Load CIFAR10 dataset without normalization first (to check mean and standard deviation).
    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    img, label = full_train[0]
    # print(img.shape)   # Output: torch.Size([3, 32, 32])
    all_labels = [label for _, label in full_train]
    unique_labels = torch.tensor(all_labels).unique()
    # print("No. of unique labels:", len(unique_labels))   # Output: 0-9
    all_imgs = torch.stack([img for img, _ in full_train])   # a tensor of shape [N, C, H, W]
    mean_per_channel = all_imgs.mean(dim=[0, 2, 3])   # 0: average over all images (N) and 2,3: over height and width (H, W)
    std_per_channel = all_imgs.std(dim=[0, 2, 3])
    # print(f"Computed Mean: {mean_per_channel}, Std: {std_per_channel}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_per_channel, std_per_channel)])
    # Reload dataset with normalization -
    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    val_size = int(val_ratio * len(full_train))
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

class CifarCnn(MnistCnn):
    def __init__(self, num_conv_layers=2, conv_channels=None, fc_units=[128, 64], num_classes=10):
        super().__init__(num_conv_layers=num_conv_layers, conv_channels=conv_channels, fc_units=fc_units, num_classes=num_classes)
        # Override first conv layer in_channels to 3
        self.conv[0] = nn.Conv2d(3, self.conv[0].out_channels, kernel_size=self.conv[0].kernel_size, stride=1, padding="same")

def build_model_optimizer_criterion(num_conv_layers=2, lr=0.001, device=None):
    model = CifarCnn(num_conv_layers=num_conv_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

def run_cifar_training(epochs=5, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = load_cifar10(batch_size=batch_size)
    model, optimizer, criterion = build_model_optimizer_criterion(device=device)
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    return model

def plot_train_error_vs_layers(max_layers=5, epochs=5, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = load_cifar10(batch_size=batch_size)
    train_errors = []
    for num_layers in range(1, max_layers + 1):
        print(f"\nTraining model with {num_layers} conv layer(s)...")
        model, optimizer, criterion = build_model_optimizer_criterion(num_conv_layers=num_layers, device=device)
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, device, train_loader, optimizer, criterion)
        final_train_loss, _ = evaluate(model, device, train_loader, criterion)
        train_errors.append(final_train_loss)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_layers + 1), train_errors, marker='o')
    plt.xlabel("Number of Conv Layers")
    plt.ylabel("Training Loss")
    plt.title("Training Error vs Number of Layers")
    plt.grid(True)
    plt.show()

    # After a certain number of conv layers, training error may stop decreasing or even increase due to:
    # 1. Vanishing gradients – deeper layers can make backpropagation harder, or
    # 2. Overfitting – if the model becomes too complex for the dataset size, or
    # 3. Insufficient regularization – no dropout, batch-norm helps but only partially, or
    # 4. Learning rate / optimizer limits – deeper networks may need careful tuning.

if __name__ == "__main__":
    run_cifar_training(epochs=5, batch_size=64)
    plot_train_error_vs_layers(max_layers=5, epochs=5, batch_size=64)
