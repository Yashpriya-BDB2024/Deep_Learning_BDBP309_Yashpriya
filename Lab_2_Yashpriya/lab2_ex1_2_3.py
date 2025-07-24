### FORWARD PASS (from scratch)

### Q.1: Consider the network in which W is a matrix, x is a vector, z is a vector, and a is a vector. y^ is a scalar and a final prediction.
### Initialize x, w randomly, z is a dot product of x and w, a is ReLU(z) but for output layer, it's softmax(z). Initialize X and W randomly. Every neuron has a bias term.

### Q.2 & Q.3: Implement the forward pass using vectorized operations. The implementation should not contain any loops.
### Print activation values for each neuron at each layer. Print the loss value (y^).


import numpy as np
from lab1_ex1 import ReLU_function, softmax_function

def initialization(layers):
    W = {}   # weight matrix
    b = {}   # bias vector
    for l in range(1, len(layers)):   # layer-0 - input layer
        W[l] = np.random.randn(layers[l], layers[l - 1])
        b[l] = np.random.randn(layers[l], 1)
    return W, b

def forward_pass(x, W, b, layers):
    activ_func = {}
    a = x
    print(f"Layer 0 (Input):")
    print(a, "\n")
    for l in range(1, len(layers)):
        z = W[l] @ a + b[l]
        if l == len(layers) - 1:
            a = softmax_function(z)
        else:
            a = ReLU_function(z)
        activ_func[l] = a
        print(f"Layer {l} activation values:")
        print(a, "\n")
    return activ_func, a

def main():
    layers = list(map(int, input("Enter number of neurons in each layer (space separated): ").split()))
    x = np.random.randn(layers[0], 1)   # input vector
    W, b = initialization(layers)
    activ_func, y_hat = forward_pass(x, W, b, layers)
    print(f"Final prediction (y_hat): {y_hat}")

if __name__ == "__main__":
    main()
