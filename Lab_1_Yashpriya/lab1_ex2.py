
### Write down the observations from the plot for all the above functions in the code.

import numpy as np
from lab1_ex1 import sigmoid_function, tanh_function, ReLU_function, leaky_ReLU_function

### (a) What are the min and max values for the functions?
### (b) Is the output of the function zero-centered?

def analyze_min_max_zero_centered(z):
    funcs = {
        "Sigmoid": sigmoid_function,
        "Tanh": tanh_function,
        "ReLU": ReLU_function,
        "Leaky ReLU": leaky_ReLU_function
    }
    results = {}
    for name, func in funcs.items():
        output = func(z)
        min_val = np.min(output)
        max_val = np.max(output)
        mean_val = np.mean(output)
        zero_centered = np.isclose(mean_val, 0, atol=1e-3)  # within small tolerance
        results[name] = {
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "zero_centered": zero_centered
        }
    return results


### (c) What happens to the gradient when the input values are too small or too big?
# Please refer the plots generated from lab1_ex1.py file.

# Observations:
# Sigmoid: The gradient is largest at z = 0 (~0.25) & quickly approaches 0 when z is very negative or very positive - vanishing gradient problem in extreme inputs.
# Tanh: Similar behaviour — gradient peaks at z = 0 (value = 1) & drops toward 0 for large z; also prone to vanishing gradients at extremes,
#       but better than sigmoid because it’s zero-centered.
# ReLU: Gradient is 0 for negative inputs, 1 for positive inputs - no vanishing gradient for positive values, but “dead neurons” possible when inputs stay negative.
# Leaky ReLU: Gradient is α (small positive constant) for negative inputs, 1 for positive inputs - avoids dead neurons by allowing small gradient flow in negative region.


### (d) What is the relationship between sigmoid and tanh?
### Please refer the plots generated from lab1_ex1.py file.

# Observations:
# Tanh is a scaled and shifted version of Sigmoid.
# Both are S-shaped and bounded, but:
# Sigmoid outputs in (0, 1) — not zero-centered, whereas Tanh outputs in (-1, 1) — zero-centered, making optimization easier in many cases.

def main():
    z = np.linspace(-10, 10, 100)
    analysis = analyze_min_max_zero_centered(z)
    for name, stats in analysis.items():
        print(f"{name}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, zero-centered={stats['zero_centered']}")

if __name__ == "__main__":
    main()


