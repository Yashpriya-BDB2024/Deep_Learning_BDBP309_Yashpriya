### TRANSFER LEARNING
### CIFAR10 Feature Extraction + SVM Classifier

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load CIFAR10 as tensors (without normalization)
transform = transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Stack all images into a single tensor
loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
images, _ = next(iter(loader))   # all training images in one batch

# images shape = (50000, 3, 32, 32)
mean = images.mean(dim=[0,2,3])   # mean over N, H, W
std = images.std(dim=[0,2,3])     # std over N, H, W
# print("Mean:", mean)   # Mean: tensor([0.4914, 0.4822, 0.4465])
# print("Std:", std)   # Std: tensor([0.2470, 0.2435, 0.2616])

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((mean),(std))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.features = nn.Sequential(*list(backbone.children())[:-1])   # remove the last FC layer
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)   # flatten 

resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.eval()
for param in resnet.parameters():
    param.requires_grad = False
feature_extractor = FeatureExtractor(resnet).to(device)

def extract_features(dataloader):
    feats, labels = [], []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            outputs = feature_extractor(images)   # (batch, 512)
            feats.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels

print("Extracting training features...")
X_train, y_train = extract_features(trainloader)
print("Extracting test features...")
X_test, y_test = extract_features(testloader)
print("Train features:", X_train.shape, " Test features:", X_test.shape)

print("Training SVM classifier...")
svc = SVC(kernel='rbf', C=10, gamma='scale') 
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

