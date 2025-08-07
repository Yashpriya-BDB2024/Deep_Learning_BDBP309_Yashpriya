
### REFERENCE: https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
###          : https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
###          : https://docs.pytorch.org/docs/stable/nn.html

import torch
from torch import nn    # Provides all the building blocks we need to build our own neural network.


# GET DEVICE FOR TRAINING

# We want to be able to train our model on an accelerator such as CUDA, MPS, MTIA, or XPU.
# If the current accelerator is available, we will use it. Otherwise, we use the CPU.

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# DEFINE THE NEURAL NETWORK CLASS

class NeuralNetwork(nn.Module):
    # This class inherits from nn.Module, the base class for all neural networks.
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()   # Converts a 2D image (28*28) into a 1D vector (784 features/pixels).
        self.linear_relu_stack = nn.Sequential(      # Builds a feed-forward neural network using nn.Sequential.
            # First linear layer: z1 = x @ w1^T + b1
            # Input: x (batch_size, 784)
            # Weight: w1 (512*784), Bias: b1 (512)
            # Output z1: (batch_size, 512)

            # Note: More neurons, better ability to learn complex patterns, but high risk of overfitting & slower training.
            # Fewer neurons, faster but might underfit.
            # So, 512 is a balanced choice. Others are - 64, 128, 256, 1024, etc.

            nn.Linear(28*28, 512),
            nn.ReLU(),   # a1=ReLU(z1) - adds non-linearity

            # Second layer: z2 = a1 @ w2^T + b2
            # w2: (512*512), b2: (512)
            # Output z2: (batch_size, 512)
            nn.Linear(512, 512),
            nn.ReLU(),   # a2=ReLU(z2)

            # Final layer: z3 = a2 @ w3^T + b3
            # w3: (10*512), b3: (10)
            # Output z3: (batch_size, 10) - probability scores for 10 classes.
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)   # Flattens the input image
        logits = self.linear_relu_stack(x)   # Pass through the neural network
        return logits    # Return raw class scores
model = NeuralNetwork().to(device)   # Move model to device (CPU)
print(model)   # Just the model structure

# For testing & debugging the model -
X = torch.rand(1, 28, 28, device=device)   # random image-like tensor of batch_size=1, 28*28
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)   # Softmax converts logits to probabilities
y_pred = pred_probab.argmax(1)    # Get predicted class (index of highest probability).
print(f"Predicted probabilities: {pred_probab}")
print(f"Predicted class: {y_pred}")


# MODEL LAYERS

input_image = torch.rand(3,28,28)   # Random batch of 3 images (each 28x28)
print(input_image.size())

# nn.Flatten
flatten = nn.Flatten()
flat_image = flatten(input_image)   # Flatten all 3 images; shape: (3, 784)
print(flat_image.size())

# nn.Linear
layer1 = nn.Linear(in_features=28*28, out_features=20)   # w: (20, 784), b: (20)
hidden1 = layer1(flat_image)
print(hidden1.size())   # Output shape: (3, 20)

# nn.ReLU
print(f"Before ReLU: {hidden1}\n\n")   # can contain negative values
hidden1 = nn.ReLU()(hidden1)   # set negatives to zero
print(f"After ReLU: {hidden1} \n\n")

# nn.Sequential
seq_modules = nn.Sequential(   # This container executes layers in order
    flatten,   # Flatten the input
    layer1,    # Linear layer (784 --> 20)
    nn.ReLU(),   # Activation
    nn.Linear(20, 10)   # Final layer (20 --> 10 class scores)
)
input_image = torch.rand(3,28,28)    # New random batch of 3 images
logits = seq_modules(input_image)    # Forward pass through sequential layers
print(f"Logits: {logits} \n")

# nn.Softmax
softmax = nn.Softmax(dim=1)    # Softmax across the class dimension (dim=1)
pred_probab = softmax(logits)    # Apply softmax to logits --> output shape: (3, 10)
print(f"Predicted Probabilities: {pred_probab} \n")


# MODEL PARAMETERS

print(f"Model structure: {model}\n\n")
# Loop through each layer's parameters and print their names, shapes, and sample values.
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")