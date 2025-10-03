
######### CNN IMPLEMENTATION FROM SCRATCH #########

### Implement convolution operations from scratch. Assume a 3x3 kernel and apply it on an input image of 32x32.
### Generalized code with multiple input options.

import numpy as np
from PIL import Image
from torchvision import datasets, transforms

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
    print("2: Load local image file")
    print("3: Load image from dataset")
    print("4: Manually enter pixel values")
    choice = int(input("Enter choice (1/2/3/4): "))

    if choice == 1:
        C = int(input("Enter number of channels: "))
        H = int(input("Enter height of an image: "))
        W = int(input("Enter width of an image: "))
        image = np.random.randint(0, 256, size=(C, H, W))
        print("Random image generated with shape:", image.shape)
        return image

    elif choice == 2:
        path = input("Enter image file path: ")
        img = Image.open(path).convert("RGB")  # ensure 3 channels
        img = img.resize((32, 32))  # optional: resize to fixed size
        image = np.array(img).transpose(2, 0, 1)  # (H,W,C) to (C,H,W)
        print("Image loaded with shape:", image.shape)
        return image

    elif choice == 3:   # This can be made generalized later like it should ideally look the presence of datasets folder and ask the user to select accordingly.
        print("Available datasets:\n1: MNIST\n2: CIFAR-10")
        ds_choice = int(input("Enter dataset choice (1/2): "))
        idx = int(input("Enter image index: "))
        if ds_choice == 1:
            dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
        else:
            dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
        img, _ = dataset[idx]  # img is a tensor (C,H,W)
        image = img.numpy()
        print(f"Image loaded from dataset at index {idx}, shape: {image.shape}")
        return image

    elif choice == 4:
        C = int(input("Enter number of channels: "))
        H = int(input("Enter height of an image: "))
        W = int(input("Enter width of an image: "))
        image = np.zeros((C, H, W))
        for c in range(C):
            print(f"Enter pixel values for channel {c+1} row-wise, separated by spaces:")
            for i in range(H):
                row = list(map(float, input(f"Row {i+1}: ").split()))
                if len(row) != W:
                    raise ValueError(f"Expected {W} values, got {len(row)}")
                image[c, i, :] = row
        print("Manual image entered with shape:", image.shape)
        return image
    else:
        raise ValueError("Invalid choice")

if __name__ == "__main__":
    image = get_input_image()

    N = int(input("Enter number of filters: "))
    k = int(input("Enter kernel size (e.g., 3 for 3x3): "))
    padding = int(input("Enter padding value: "))
    stride = int(input("Enter stride value: "))

    C = image.shape[0]
    filters = np.random.randint(-1, 2, size=(N, C, k, k))
    print("Filters:\n", filters)

    conv_out = convolution_operation(image, filters, padding=padding, stride=stride)
    print("Convolution output shape:", conv_out.shape)
    print("Convolution output array:\n", conv_out)
