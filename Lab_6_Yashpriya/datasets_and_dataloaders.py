
### REFERENCE: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

# LOADING A DATASET

# torch.utils.data.Dataset - stores the samples & their corresponding labels
# torch.utils.data.DataLoader - wraps an iterable around the Dataset to enable easy access to the samples.

# FashionMNIST dataset from TorchVision -
# Consists of Zalando’s article images (60,000 training examples and 10,000 test examples).
# Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

import torch
from torch.utils.data import Dataset   # This is a base class used to create custom datasets.
from torchvision import datasets
from torchvision.transforms import ToTensor   # converts images to PyTorch tensors and scales pixel values from [0, 255] to [0.0, 1.0].
import matplotlib.pyplot as plt
import pandas as pd
import os
from torchvision.io import decode_image
from torch.utils.data import DataLoader

training_data = datasets.FashionMNIST(   # Loads the FashionMNIST dataset
    root="data",   # path where train/test data is stored; where 'data' is the directory
    train=True,   # Loads the training set (60,000 images)
    download=True,  # downloads the data from the internet if it's not available in root.
    transform=ToTensor()   # specify the feature and label transformations
)
test_data = datasets.FashionMNIST(
    root='data',
    train=False,   # specifies the test dataset (10,000 images)
    download=True,
    transform=ToTensor()
)


# ITERATING & VISUALIZING THE DATASET

# print(training_data.classes)
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3   # 9 images (classes)
for i in range(1, cols*rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()   # picks one random image index from the dataset
    # torch.randint(...) - picks a random index between 0 and len(training_data)-1.
    # size=(1,) - means we are generating a single random number.
    # .item() - converts the tensor to a plain Python integer.
    img, label = training_data[sample_idx]    # Fetches the image and label at the randomly chosen index.
    figure.add_subplot(rows, cols, i)    # Adds a subplot (mini-plot) in the 3x3 grid at position i, e.g., i=1 puts 1st image in top-left, and so on...
    plt.title(labels_map[label])   # Displays the class name on top of the image.
    plt.axis("off")   # Hides the axis ticks and labels
    # print(img.shape)    # torch.Size([1, 28, 28])
    plt.imshow(img.squeeze(), cmap="gray")    # img.squeeze() - removes that single-channel dimension; makes it (28,28) for plotting.
plt.show()


# CREATING A CUSTOM DATASET FOR YOUR FILES

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        # __init__ function - runs when instantiating the Dataset object.
        # Initialize the directory containing images, the annotations file, and both transforms.
        self.img_labels = pd.read_csv(annotations_file)   # This reads labels.csv file and store it in a DataFrame.
        print(self.img_labels)
        self.img_dir = img_dir   # Stores the image folder path.
        # self.transform: Applies transformation to the image data like resizing, converting to tensor, normalizing pixel values, random rotation or flip, etc.
        self.transform = transform
        # self.target_transform: # Applies transformation to the label/class/target like converting to tensor, one-hot encoding, mapping label from one format to another, etc.
        self.target_transform = target_transform

    def __len__(self):
        # Returns the no. of samples in our dataset.
        return len(self.img_labels)

    def __getitem__(self, idx):
        # This function loads and returns a sample from the dataset at the given index idx.
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # self.img_labels is a DataFrame created from the CSV (e.g., labels.csv)
        # .iloc[idx, 0] fetches the filename in the first column of the 'idx'th row
        # os.path.join joins image directory with filename
        image = decode_image(img_path)
        # decode_image reads the image file from disk and converts it into a tensor.
        label = self.img_labels.iloc[idx, 1]
        # Gets the label from column 1 (second column) of the CSV for the given index.
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label    # a tuple: (image tensor, label)


# PREPARING THE DATA FOR TRAINING WITH DATALOADERS

# While training a model, we typically want to pass samples in “minibatches”, and reshuffle the data at every epoch to reduce model overfitting.
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
#  Each iteration returns a batch of train_features and train_labels (containing batch_size=64 features and labels respectively).
#  Because we specified shuffle=True, after we iterate over all batches the data is shuffled.


# ITERATE THROUGH THE DATALOADER

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
# iter(train_dataloader): Creates a Python iterator over the dataloader.
# next(...): Gets the next batch (i.e., the first 64 samples).
print(f"Feature batch shape: {train_features.size()}")    # Output: torch.Size([64, 1, 28, 28]), i.e., [batch size, grayscale channel, height, width]
print(f"Labels batch shape: {train_labels.size()}")    # Output: torch.Size([64])
img = train_features[0].squeeze()   # Removes channel dimension of the first image in the batch.
label = train_labels[0]   # Label for the first image.
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")