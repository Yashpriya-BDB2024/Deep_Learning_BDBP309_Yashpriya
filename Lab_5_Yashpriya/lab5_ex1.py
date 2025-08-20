
### GENERALIZED FORWARD PASS

# Implement a 2-layer (input layer, hidden layer and output layer) neural network from scratch for the XOR operation. This includes implementing forward pass from scratch.

import numpy as np
from lab1_ex1 import sigmoid_function, tanh_function, ReLU_function, leaky_ReLU_function, softmax_function

def initialization(layers, method="random"):
    W, b = {}, {}
    for l in range(1, len(layers)):
        if method == "xavier":   # good for sigmoid/tanh
            W[l] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2 / (layers[l] + layers[l - 1]))
        elif method == "he":     # good for ReLU
            W[l] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2 / layers[l - 1])
        elif method == "manual":
            try:
                vals = input(f"Enter weights for layer {l} (shape={layers[l]}x{layers[l-1]}, "
                             f"enter {layers[l]*layers[l-1]} numbers comma-separated, row-wise): ")
                arr = list(map(float, vals.replace(";", " ").replace(",", " ").split()))
                W[l] = np.array(arr).reshape(layers[l], layers[l-1])
            except:
                print(f"Invalid manual weights for layer {l}, using random.")
                W[l] = np.random.randn(layers[l], layers[l-1])
        else:                    # default random
            W[l] = np.random.randn(layers[l], layers[l-1])
        b[l] = np.random.randn(layers[l], 1)
    return W, b

act_funcs = {
    "sigmoid": sigmoid_function,
    "tanh": tanh_function,
    "relu": ReLU_function,
    "leakyrelu": leaky_ReLU_function,
    "softmax": softmax_function
}

def forward_pass(x, W, b, layers, hidden_act, output_act):
    a = x
    print("Layer 0 (input):\n", a, "\n")
    activ = {}
    for l in range(1, len(layers)):
        z = W[l] @ a + b[l]   # linear step
        # choose activation
        if l == len(layers) - 1:   # output layer
            func = act_funcs.get(output_act, softmax_function)
        else:                      # hidden layers
            func = act_funcs.get(hidden_act[l], ReLU_function)
        a = func(z)        # activation step
        activ[l] = a
        print(f"Layer {l} ({'output' if l==len(layers)-1 else 'hidden'}) activation:\n", a, "\n")
    return activ, a

def main():
    try:
        layers = list(map(int, input("Enter neurons in each layer (space-separated, e.g. 2 2 1): ").split()))
        if len(layers) < 2: raise ValueError
    except:
        print("Invalid, using default [3 4 2]")
        layers = [3,4,2]

    try:
        val = input(f"Enter input vector (shape={layers[0]}x1, enter {layers[0]} numbers comma-separated): ")
        if val.strip():
            x = np.array(list(map(float, val.split(",")))).reshape(layers[0],1)
        else:
            x = np.random.randn(layers[0],1)
    except:
        print("Invalid, random input used")
        x = np.random.randn(layers[0],1)

    init = input("Weight init (random/xavier/he/manual): ").strip().lower()
    if init not in ["random", "xavier", "he", "manual"]:
        init = "random"
    W, b = initialization(layers, init)

    if input("Enter biases manually? (y/n): ").lower() == "y":
        for l in range(1,len(layers)):
            try:
                vals = input(f"Bias for layer {l} (shape={layers[l]}x1, enter {layers[l]} numbers comma-separated): ")
                b[l] = np.array(list(map(float, vals.split(",")))).reshape(layers[l],1)
            except:
                print(f"Invalid bias for layer {l}, kept random.")

    hidden_act = {}
    same = input("Same activation for all hidden layers? (y/n): ").lower() or "y"
    if same=="y":
        act = input("Hidden activation (sigmoid/tanh/relu/leakyrelu, default=relu): ").lower() or "relu"
        for l in range(1,len(layers)-1):
            hidden_act[l] = act
    else:
        for l in range(1,len(layers)-1):
            act = input(f"Activation for hidden layer {l}: ").lower() or "relu"
            hidden_act[l] = act

    output_act = input("Output activation (sigmoid/tanh/relu/leakyrelu/softmax, default=softmax): ").lower() or "softmax"

    activ, y_hat = forward_pass(x,W,b,layers,hidden_act,output_act)
    print("Final prediction:\n", y_hat)

if __name__ == "__main__":
    main()


### TESTED FOR XOR IMPLEMENTATION (following is the output of x1=1, x2=1)

# Enter neurons in each layer (space-separated, e.g. 2 2 1): 2 2 1
# Enter input vector (shape=2x1, enter 2 numbers comma-separated): 1,1
# Weight init (random/xavier/he/manual): manual
# Enter weights for layer 1 (shape=2x2, enter 4 numbers comma-separated, row-wise): 1,1,1,1
# Enter weights for layer 2 (shape=1x2, enter 2 numbers comma-separated, row-wise): 1,-2
# Enter biases manually? (y/n): y
# Bias for layer 1 (shape=2x1, enter 2 numbers comma-separated): 0,-1
# Bias for layer 2 (shape=1x1, enter 1 numbers comma-separated): 0
# Same activation for all hidden layers? (y/n): y
# Hidden activation (sigmoid/tanh/relu/leakyrelu, default=relu): relu
# Output activation (sigmoid/tanh/relu/leakyrelu/softmax, default=softmax): relu
# Layer 0 (input):
#  [[1.]
#  [1.]]

# Layer 1 (hidden) activation:
#  [[2.]
#  [1.]]

# Layer 2 (output) activation:
#  [[0.]]

# Final prediction:
#  [[0.]]
