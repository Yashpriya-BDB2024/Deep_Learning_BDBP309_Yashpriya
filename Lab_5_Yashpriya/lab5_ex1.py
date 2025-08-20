import numpy as np
from lab1_ex1 import (
    sigmoid_function, tanh_function, ReLU_function,
    leaky_ReLU_function, softmax_function
)

# -------------------- weight initialization --------------------
def initialization(layers, method="random"):
    """
    Initialize weights and biases for each layer.
    Supports random, Xavier, He, and Manual initialization.
    """
    W, b = {}, {}
    for l in range(1, len(layers)):
        if method == "xavier":   # good for sigmoid/tanh
            W[l] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2 / (layers[l] + layers[l - 1]))
        elif method == "he":     # good for ReLU
            W[l] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2 / layers[l - 1])
        elif method == "manual":
            try:
                vals = input(f"Enter weights for layer {l} ({layers[l]}x{layers[l-1]} comma-separated): ")
                arr = list(map(float, vals.replace(";", " ").replace(",", " ").split()))
                W[l] = np.array(arr).reshape(layers[l], layers[l-1])
            except:
                print(f"Invalid manual weights for layer {l}, using random.")
                W[l] = np.random.randn(layers[l], layers[l-1])
        else:                    # default random
            W[l] = np.random.randn(layers[l], layers[l-1])

        # biases always random initially (can override later)
        b[l] = np.random.randn(layers[l], 1)
    return W, b


# dictionary of activation functions
act_funcs = {
    "sigmoid": sigmoid_function,
    "tanh": tanh_function,
    "relu": ReLU_function,
    "leakyrelu": leaky_ReLU_function,
    "softmax": softmax_function
}

# -------------------- forward pass --------------------
def forward_pass(x, W, b, layers, hidden_act, output_act):
    """
    Perform forward propagation through the network.
    hidden_act: dict of activations per hidden layer
    output_act: activation for output layer
    """
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

# -------------------- main program --------------------
def main():
    # 1. ask for layers
    try:
        layers = list(map(int, input("Enter neurons in each layer: ").split()))
        if len(layers) < 2: raise ValueError
    except:
        print("Invalid, using default [3 4 2]")
        layers = [3,4,2]

    # 2. input vector
    try:
        val = input(f"Enter input vector of size {layers[0]}, or press enter: ")
        if val.strip():
            x = np.array(list(map(float, val.split(",")))).reshape(layers[0],1)
        else:
            x = np.random.randn(layers[0],1)
    except:
        print("Invalid, random input used")
        x = np.random.randn(layers[0],1)

    # 3. weight initialization method
    init = input("Weight init (random/xavier/he/manual): ").strip().lower()
    if init not in ["random", "xavier", "he", "manual"]:
        init = "random"
    W, b = initialization(layers, init)

    # 4. biases (manual entry or random)
    if input("Enter biases manually? (y/n): ").lower() == "y":
        for l in range(1,len(layers)):
            try:
                vals = input(f"Bias for layer {l} ({layers[l]} values): ")
                b[l] = np.array(list(map(float, vals.split(",")))).reshape(layers[l],1)
            except:
                print(f"Invalid bias for layer {l}, kept random.")

    # 5. activations for hidden layers
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

    # 6. activation for output layer
    output_act = input("Output activation (sigmoid/tanh/relu/leakyrelu/softmax, default=softmax): ").lower() or "softmax"

    # 7. forward pass
    activ, y_hat = forward_pass(x,W,b,layers,hidden_act,output_act)
    print("Final prediction:\n", y_hat)

if __name__ == "__main__":
    main()
