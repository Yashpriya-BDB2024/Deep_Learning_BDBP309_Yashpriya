
### Implement maxpool operation from scratch.

import numpy as np
from lab11_ex1 import convolution_operation, get_input_image, get_filters

def max_pooling_operation(input, filter_size=2, stride=2):
    C, H, W = input.shape
    # Output dimensions -
    H_out = (H - filter_size) // stride + 1
    W_out = (W - filter_size) // stride + 1
    output = np.zeros((C, H_out, W_out))   # Will store the max. values for each pooling window.
    for c in range(C):   # Loop over channels independently.
        # i and j are top-left indices of current pooling window.
        for i in range(0, H_out * stride, stride):
            for j in range(0, W_out * stride, stride):
                patch = input[c, i:i+filter_size, j:j+filter_size]    # Extract a small patch of size (filter_size * filter_size) from input channel c.
                # i//stride: converts from input index to output index; thus assigns max. value from that patch to the correct position in O/P.
                output[c, i//stride, j//stride] = np.max(patch)
    return output

if __name__ == "__main__":
    image = get_input_image()
    choice = input("Do you want to apply convolution before Max Pooling? (y/n): ").lower()
    if choice == "y":
        # Get convolution parameters
        k = int(input("Enter kernel size (e.g., 3 for 3x3): "))
        padding = int(input("Enter padding value: "))
        stride = int(input("Enter stride value: "))
        # Get filters and apply convolution
        C = image.shape[0]
        filters = get_filters(C, k)
        conv_out = convolution_operation(image, filters, padding=padding, stride=stride)
        print("Convolution output shape:", conv_out.shape)
        print("Convolution output array:\n", conv_out)
        input_for_pooling = conv_out  # Pooling will use convolution output
    else:
        # Skip convolution, use original image for pooling
        input_for_pooling = image

    pool_size = int(input("Enter pooling filter size (e.g., 2): "))
    pool_stride = int(input("Enter pooling stride (e.g., 2): "))
    pooled_out = max_pooling_operation(input_for_pooling, filter_size=pool_size, stride=pool_stride)
    print("Max Pooling output shape:", pooled_out.shape)
    print("Max Pooling output array:\n", pooled_out)

### Example for testing purpose:

# Choose input type:
# 1: Random pixel values
# 2: Manually enter pixel values
# Enter choice (1/2): 2
# Enter number of channels: 1
# Enter height of an image: 5
# Enter width of an image: 5
# Enter pixel values for channel 1 row-wise, separated by spaces:
# Row 1: 1 3 2 1 3
# Row 2: 2 9 1 1 5
# Row 3: 1 3 2 3 2
# Row 4: 8 3 5 1 0
# Row 5: 5 6 1 2 9
# Manual image entered with shape: (1, 5, 5)
# Do you want to apply convolution before Max Pooling? (y/n): n
# Enter pooling filter size (e.g., 2): 3
# Enter pooling stride (e.g., 2): 1
# Max Pooling output shape: (1, 3, 3)
# Max Pooling output array:
#  [[[9. 9. 5.]
#   [9. 9. 5.]
#   [8. 6. 9.]]]
