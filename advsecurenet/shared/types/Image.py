import torch

# Create a tensor with shape (batch_size, channels, height, width)
batch_size = 4
channels = 1
height = 256
width = 256
image_tensor = torch.randn(batch_size, channels, height, width)

# Change the number of channels to 3
image_tensor_3_channels = image_tensor.expand(batch_size, 3, height, width)
