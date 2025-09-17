
### Implement vanilla RNN from scratch.
### Reference: https://medium.com/@thisislong/building-a-recurrent-neural-network-from-scratch-ba9b27a42856

import numpy as np
import pandas as pd

def rnn_forward_pass(hidden_state_dim, W_xh, W_hh, W_hy, seq):
    h0 = np.zeros((hidden_state_dim, 1))
    h, y = [], []
    for t, x in enumerate(seq):   # t: time step index , x: actual vector at that time step
        x = np.array(x).reshape(-1, 1)   # array to column vector
        h_t = np.tanh(W_hh @ h0 + W_xh @ x)
        y_t = W_hy @ h_t
        h.append(h_t)
        y.append(y_t)
        h0 = h_t
    return h, y

def main():
    samples_num = int(input("Please enter the number of samples:"))
    representation_vec_dim = int(input("Please enter the dimension of representation vector (same for all samples):"))
    samples_inputs = []
    time_steps_list = []
    for s in range(samples_num):
        T = int(input(f"Enter number of time steps for sample {s+1}: "))
        time_steps_list.append(T)
        seq = []
        for t in range(T):
            x_t = eval(input(f"Please enter the input (in list format) x{t + 1} for sample {s + 1} "
                             f"(dimension = {representation_vec_dim}): "))
            seq.append(x_t)
        samples_inputs.append(seq)
    hidden_state_dim = int(input("Please enter the hidden state (h_t) dimension (int format):"))
    output_dim = int(input("Please enter the output (y_t) dimension:"))

    init_method = input("Initialize weights randomly or manually?")
    if init_method == "random":
        W_xh = np.random.randn(hidden_state_dim, representation_vec_dim) * 0.01   # Scaling (0.01) is done to prevent 'h' from blowing up.
        W_hh = np.random.randn(hidden_state_dim, hidden_state_dim) * 0.01
        W_hy = np.random.randn(output_dim, hidden_state_dim) * 0.01
    else:
        W_xh = np.array(eval(input(f"Enter W_xh of shape ({hidden_state_dim}, {representation_vec_dim}) - in list of list format: ")))
        W_hh = np.array(eval(input(f"Enter W_hh of shape ({hidden_state_dim}, {hidden_state_dim}): ")))
        W_hy = np.array(eval(input(f"Enter W_hy of shape ({output_dim}, {hidden_state_dim}): ")))

    for s, seq in enumerate(samples_inputs):
        print(f"\nForward pass for sample {s + 1} -")
        hs, ys = rnn_forward_pass(hidden_state_dim, W_xh, W_hh, W_hy, seq)
        print("\nHidden states:")
        for t, h in enumerate(hs):
            print(f"h{t + 1}:\n{h}")
        print("\nOutputs:\n")
        for t, y in enumerate(ys):
            print(f"y{t + 1}:\n{y}")

if __name__ == "__main__":
    main()
