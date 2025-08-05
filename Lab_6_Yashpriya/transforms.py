
### REFERENCE: https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
###          : https://docs.pytorch.org/vision/stable/transforms.html

# To manipulate the data such that it becomes suitable for training purpose.
# transform - to modify the features
# target_transform - to modify the labels

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(    # The FashionMNIST features are in PIL Image format, and the labels are integers.
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    # ToTensor() - converts a PIL image or NumPy ndarray into a FloatTensor, and scales the image’s pixel intensity values in the range [0., 1.].
    # For example:
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))    # Convert an integer label into a one-hot encoded tensor.
    # Lambda transforms apply any user-defined lambda function.
    # It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls scatter_ which assigns a value=1 on the index as given by the label y.
)