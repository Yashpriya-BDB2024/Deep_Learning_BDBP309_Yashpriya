### DATE: 17-07-25
### REFERENCE: Hands-On Machine Learning with Scikit-Learn and TensorFlow, Aurélien Géron

### Generate 100 equally spaced values between -10 and 10. Call this list as z. Implement the activation functions - Sigmoid, tanh, ReLU, Leaky ReLU & Softmax ,
### and its derivative (from scratch). Use z as input and plot both the function outputs and its derivative outputs.

import numpy as np
import matplotlib.pyplot as plt

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    sig = sigmoid_function(z)
    return sig * (1 - sig)

def tanh_function(z):
    numer = np.exp(z) - np.exp(-z)
    denom = np.exp(z) + np.exp(-z)
    return numer / denom

# Alternative expression, tanh(z) = 2 * sig(2 * z) - 1

def tanh_derivative(z):
    tanh = tanh_function(z)
    return 1 - np.power(tanh, 2)

def ReLU_function(z):
    return np.maximum(0, z)

# Note:- max() - can't compare scalar with an array, np.max() - to get maximum value in an array, np.maximum() - element-wise max. value, and np.argmax() - to get the index of max. value.

def ReLU_derivative(z):
    # If z > 0 then 1 , if z < 0 then 0, if z = 0 then undefined.
    return np.where(z > 0, 1, np.where(z < 0, 0, np.nan))

# Note:- np.where(condition, x, y) - acts like an element-wise if-else, i.e., if condition is True, then return x, otherwise y.

def leaky_ReLU_function(z, alpha = 0.01):
    return np.maximum(alpha * z, z)

def leaky_ReLU_derivative(z, alpha=0.01):
    # If z > 0 then 1 , if z < 0 then alpha, if z = 0 then undefined.
    return np.where(z > 0, 1, np.where(z < 0, alpha, np.nan))

# Note:- Hyperparameter 'alpha' defines how much the function leaks; this small slope ensures that leaky ReLUs never die.
# Dying ReLUs means that some neurons stop outputting anything other than 0 (happens when a neuron's weight gets updated such that weighted sum of neuron's inputs is -ve).
# Researchers found out that huge leak (alpha = 0.2) seemed to result in better performance than small leak (alpha=0.01).

def softmax_function(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

# Note:- Normally we subtract np.max(z) from z for numerical stability to prevent overflow in exp(z), especially when z has large values (e.g., > 100).

def softmax_derivative(z):
    sftmx = softmax_function(z)
    n = len(sftmx)
    J = np.zeros((n, n))    # Initializing the Jacobian matrix (a matrix of first order partial derivatives) with zeroes; shape - (n, n).
    for i in range(n):
        for j in range(n):
            if i == j:   # Diagonal elements
                J[i][j] = sftmx[i] * (1 - sftmx[i])
            else:   # i != j - Off-Diagonal elements
                J[i][j] = -sftmx[i] * sftmx[j]
    return J

def plot_all(z):
    funcs = [sigmoid_function, tanh_function, ReLU_function, leaky_ReLU_function]
    derivs = [sigmoid_derivative, tanh_derivative, ReLU_derivative, leaky_ReLU_derivative]
    titles = ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU']
    fig1, axs1 = plt.subplots(1, 4, figsize=(20, 4))
    fig2, axs2 = plt.subplots(1, 4, figsize=(20, 4))
    for i in range(4):
        axs1[i].plot(z, funcs[i](z))
        axs1[i].set_title(titles[i])
        axs2[i].plot(z, derivs[i](z))
        axs2[i].set_title(titles[i] + " Derivative")
    fig1.suptitle("Activation Functions")
    fig2.suptitle("Activation Function Derivatives")
    plt.tight_layout()
    plt.show()

def main():
    z = np.linspace(-10, 10, 100)
    plot_all(z)

if __name__ == "__main__":
    main()
