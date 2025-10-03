
######### CNN IMPLEMENTATION FROM SCRATCH #########

### Implement convolution operations from scratch. Assume a 3x3 kernel and apply it on an input image of 32x32.

import numpy as np

def apply_padding(matrix, padding_size=1):
    # Applies zero padding around a 2D matrix.
    rows, cols = matrix.shape
    padded = np.zeros((rows + 2*padding_size, cols + 2*padding_size))  # Default=1 (if not specified); accounts for padding on all sides.
    padded[padding_size:padding_size+rows, padding_size:padding_size+cols] = matrix   # Assign matrix to the center of padded.
    return padded

def convolution_operation(input, filters, padding=1, stride=1):
    C, H, W = input.shape   # C: channels, H: Height of image, W: Width of image
    N, _, k, _ = filters.shape  # N: no. of filters, k: kernel/filter size

    # Applies padding to each channel
    padded = np.array([apply_padding(input[c], padding) for c in range(C)])   # np.array(...): converts list of padded 2D arrays into 3D array (C,H_p,W_p).
    H_p, W_p = padded.shape[1], padded.shape[2]   # Extracts height and width of padded image.

    # Output dimensions
    H_out = (H_p - k) // stride + 1
    W_out = (W_p - k) // stride + 1
    output = np.zeros((N, H_out, W_out))    # Initializes output array for each filter.

    for n in range(N):   # loop over filters
        # Loop over sliding window positions.
        for i in range(0, H_out * stride, stride):
            for j in range(0, W_out * stride, stride):
                # Slide the current patch of input (: - all channels, i:i+k - rows from i to i+k-1, j:j+k - columns from j to j+k-1)
                patch = padded[:, i:i+k, j:j+k]   # shape (C, k, k)
                output[n, i//stride, j//stride] = np.sum(patch * filters[n])   # element-wise multiplication, sum all values to get single scalar and then, compute output indices.
    return output

def get_input_image():
    print("Choose input type:")
    print("1: Random pixel values")
    print("2: Manually enter pixel values")
    choice = int(input("Enter choice (1/2): "))

    if choice == 1:
        C = int(input("Enter number of channels: "))
        H = int(input("Enter height of an image: "))
        W = int(input("Enter width of an image: "))
        image = np.random.randint(0, 256, size=(C, H, W))
        print("Random image generated with shape:", image.shape)
        return image

    elif choice == 2:
        C = int(input("Enter number of channels: "))
        H = int(input("Enter height of an image: "))
        W = int(input("Enter width of an image: "))
        image = np.zeros((C, H, W))
        for c in range(C):
            print(f"Enter pixel values for channel {c + 1} row-wise, separated by spaces:")
            for i in range(H):
                row = list(map(float, input(f"Row {i + 1}: ").split()))
                if len(row) != W:
                    raise ValueError(f"Expected {W} values, got {len(row)}")
                image[c, i, :] = row
        print("Manual image entered with shape:", image.shape)
        return image
    else:
        raise ValueError("Invalid choice")

def get_filters(C, k):
    # Predefined 3x3 filters
    predefined = {
        "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        "Blur": (1 / 9) * np.ones((3, 3)),
        "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "Edge Detection (Vertical)": np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
        "Edge Detection (Horizontal)": np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
        "Edge Detection (Diagonal)": np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]]),
        "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    }

    print("\nChoose filter type:")
    print("1: Random filters")
    print("2: Predefined filters")
    print("3: Manual entry")
    choice = int(input("Enter choice (1/2/3): "))

    if choice == 1:
        N = int(input("Enter number of filters: "))
        filters = np.random.randint(-1, 2, size=(N, C, k, k))
        return filters

    elif choice == 2:
        print("\nAvailable Predefined Filters:")
        for i, name in enumerate(predefined.keys(), start=1):
            print(f"{i}: {name}")
        selections = list(map(int, input("Enter filter numbers (comma separated): ").split(",")))

        selected_filters = []
        for sel in selections:
            base = list(predefined.values())[sel - 1]
            if base.shape != (k, k):
                raise ValueError(f"Predefined filter is {base.shape}, expected {k}x{k}")
            # Expand across channels
            expanded = np.stack([base] * C, axis=0)
            selected_filters.append(expanded)
        return np.array(selected_filters)

    elif choice == 3:
        N = int(input("Enter number of filters: "))
        filters = np.zeros((N, C, k, k))
        for n in range(N):
            print(f"Enter values for filter {n + 1}:")
            for i in range(k):
                row = list(map(float, input(f"Row {i + 1}: ").split()))
                if len(row) != k:
                    raise ValueError(f"Expected {k} values, got {len(row)}")
                for c in range(C):
                    filters[n, c, i, :] = row
        return filters
    else:
        raise ValueError("Invalid filter choice")

if __name__ == "__main__":
    image = get_input_image()

    k = int(input("Enter kernel size (e.g., 3 for 3x3): "))
    padding = int(input("Enter padding value: "))
    stride = int(input("Enter stride value: "))

    C = image.shape[0]
    filters = get_filters(C, k)
    print("Filters:\n", filters)

    conv_out = convolution_operation(image, filters, padding=padding, stride=stride)
    print("Convolution output shape:", conv_out.shape)
    print("Convolution output array:\n", conv_out)

### Example for testing purpose:

# Choose input type:
# 1: Random pixel values
# 2: Manually enter pixel values
# Enter choice (1/2): 2
# Enter number of channels: 1
# Enter height of an image: 5
# Enter width of an image: 5
# Enter pixel values for channel 1 row-wise, separated by spaces:
# Row 1: 2 3 7 4 6
# Row 2: 6 6 9 8 7
# Row 3: 3 4 8 3 8
# Row 4: 7 8 3 6 6
# Row 5: 4 2 1 8 3
# Manual image entered with shape: (1, 5, 5)
# Enter kernel size (e.g., 3 for 3x3): 2
# Enter padding value: 1
# Enter stride value: 2
#
# Choose filter type:
# 1: Random filters
# 2: Predefined filters
# 3: Manual entry
# Enter choice (1/2/3): 3
# Enter number of filters: 1
# Enter values for filter 1:
# Row 1: 1 -1
# Row 2: 0 -1
# Filters:
#  [[[[ 1. -1.]
#    [ 0. -1.]]]]
# Convolution output shape: (1, 3, 3)
# Convolution output array:
#  [[[ -2.  -7.  -6.]
#   [ -9. -11.  -7.]
#   [-11.   4.  -3.]]]
