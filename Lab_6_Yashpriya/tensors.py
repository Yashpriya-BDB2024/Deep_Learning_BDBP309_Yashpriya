
### REFERENCE: https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

### ABOUT TENSORS

# They are a specialized data structure that are very similar to arrays & matrices; used to encode the inputs & outputs of a model, as well as the model’s parameters.
# Can run on GPUs or other hardware accelerators.
# Also, they're optimized  for automatic differentiation (Autograd).

import torch
import numpy as np


### INITIALIZING A TENSOR

# Directly from data -
data = [[1,2], [3,4]]
x_data = torch.tensor(data)
print(f"Tensor initialization directly from data: \n {x_data} \n")   # Output: tensor([[1, 2], [3, 4]])

# From a NumPy array -
np_array = np.array(data)
x_np = torch.from_numpy((np_array))
print(f"Tensor initialization from a NumPy array: \n {x_np} \n")
# Changes in NumPy array reflects in the tensor.
np.add(np_array, 1, out=np_array)
print(f"Tensor: {x_np} \n")
print(f"NumPy array: {np_array} \n")

# From another tensor -
x_ones = torch.ones_like(x_data)    # retains the properties (shape, datatype) of x_data
print(f"Ones Tensor: \n {x_ones} \n")   # Output: tensor([[1, 1], [1, 1]])

x_rand = torch.rand_like(x_data, dtype=torch.float)    # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")   # Output: tensor([[0.3884, 0.5463], [0.0352, 0.4377]])

# With random or constant values -
# 'shape' is a tuple of tensor dimensions.
shape = (2,3,)   # rows=2, columns=3
rand_tensor = torch.rand(shape)
print(f"Random tensor with shape {shape}: \n {rand_tensor} \n")   # Output: tensor([[0.8279, 0.9072, 0.0469], [0.6629, 0.8526, 0.7754]])
ones_tensor = torch.ones(shape)
print(f"Ones tensor with shape {shape}: \n {ones_tensor} \n")   # Output: tensor([[1., 1., 1.], [1., 1., 1.]])
zeros_tensor = torch.zeros(shape)
print(f"Zeros tensor with shape {shape}: \n {zeros_tensor} \n")   # Output: tensor([[0., 0., 0.], [0., 0., 0.]])


### ATTRIBUTES OF A TENSOR

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")   # Output: torch.Size([3, 4])
print(f"Datatype of tensor: {tensor.dtype}")   # Output: torch.float32
print(f"Device tensor is stored on: {tensor.device} \n")   # Output: cpu


### OPERATIONS ON TENSORS

# By default, tensors are created on the CPU. We need to explicitly move tensors to the accelerator using .to method (after checking for accelerator availability).
# Copying large tensors across devices can be expensive in terms of time and memory.

if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

# Standard numpy-like indexing and slicing -
tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")   #  Output: tensor([1., 1., 1., 1.])
print(f"First column: {tensor[:, 0]}")   # Output: tensor([1., 1., 1., 1.])
print(f"Last column: {tensor[..., -1]}")   # Output: tensor([1., 1., 1., 1.])
tensor[:,1] = 0   # 2nd column is set to 0
print(f"{tensor} \n")    # Output: tensor([[1., 0., 1., 1.], [1., 0., 1., 1.], [1., 0., 1., 1.], [1., 0., 1., 1.]])

# Joining tensors -
# torch.cat - to concatenate a sequence of tensors along an existing dimension.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"Concatenated tensors: \n {t1}")
# Output: tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])

# torch.stack() - concatenates a seq. of tensors along a new dimension, but all tensors need to be of the same sizee.
x = torch.randn(2,3)   # rows=2, columns=3
print(x)
# Output: tensor([[ 0.9274, -0.0083, -0.0987],
#         [-1.0199,  1.2365,  0.3985]])

identical_stack = torch.stack((x, x), dim=0)   # stacking two identical copies of x
print(identical_stack)
# Output: tensor([[[ 0.9274, -0.0083, -0.0987],
#          [-1.0199,  1.2365,  0.3985]],
#
#         [[ 0.9274, -0.0083, -0.0987],
#          [-1.0199,  1.2365,  0.3985]]])
print(identical_stack.size())   # We now have a new first axis with 2 items — each item is a [2, 3] matrix.
# Output: torch.Size([2, 2, 3])

stack_row = torch.stack((x,x), dim=1)   # Grouping row-wise
print(stack_row)
# Output: tensor([[[ 0.9274, -0.0083, -0.0987],
#          [ 0.9274, -0.0083, -0.0987]],
#
#         [[-1.0199,  1.2365,  0.3985],
#          [-1.0199,  1.2365,  0.3985]]])
print(stack_row.size())    # Output: torch.Size([2, 2, 3])

num_paired_stack = torch.stack((x,x), dim=2)   # Each no. gets paired.
# Alternative: torch.stack((x,x), dim=-1)
print(num_paired_stack)
# Output: tensor([[[ 0.9274,  0.9274],
#          [-0.0083, -0.0083],
#          [-0.0987, -0.0987]],
#
#         [[-1.0199, -1.0199],
#          [ 1.2365,  1.2365],
#          [ 0.3985,  0.3985]]])
print(f"{num_paired_stack.size()} \n")     # Output: torch.Size([2, 3, 2])

# Arithmetic Operations -

print(f"Transpose of given {tensor} is following: \n\n {tensor.T} \n")   # tensor.T returns the transpose of a tensor
print("Matrix Multiplication: \n")
y1 = tensor @ tensor.T   # Method-1
print(f"{y1} \n")
# Output: tensor([[3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.]])
y2 = tensor.matmul(tensor.T)   # Method-2
y3 = torch.rand_like(y1)   # Method-3
torch.matmul(tensor, tensor.T, out=y3)

print("Element-wise product: \n")
z1 = tensor * tensor   # Method-1
z2 = tensor.mul(tensor)   # Method-2
print(f"{z2} \n")
# Output: tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])
z3 = torch.rand_like(tensor)   # Method-3
torch.mul(tensor, tensor, out=z3)

# Single-element tensors -
# If we have a one-element tensor, for example by aggregating all values of a tensor into one value, you can convert it to a Python numerical value using item()
agg = tensor.sum()
agg_item = agg.item()
print(f"Single-element tensor: {agg_item}, {type(agg_item)} \n")   # Output: 12.0 <class 'float'>

# In-place operations -
# Operations that store the result into the operand are called in-place. They are denoted by a _ suffix.
print(f"{tensor.add_(5)} \n")   # Output: tensor([[6., 5., 6., 6.], [6., 5., 6., 6.], [6., 5., 6., 6.], [6., 5., 6., 6.]])
print(tensor.t_())   # transpose in-place
print(tensor.mul(2))  # multiplication

# Tensor to NumPy array -
t = torch.ones(5)
print(f"tensor: {t} \n")
n = t.numpy()
print(f"tensor to numpy array: {n} \n")
# A change in the tensor reflects in the NumPy array.
t.add(1)
print(f"Updated tensor: {t}")
print(f"Automatic update in numpy array: {n} \n")
