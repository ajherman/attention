import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a 2D convolution layer and a matching 2D transposed convolution layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1,bias=False)
conv_layer.weight.data = torch.ones(conv_layer.weight.data.shape)
transpose_conv_layer = nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size=3, stride=1, padding=1,bias=False)
transpose_conv_layer.weight.data = conv_layer.weight.data

# Random input tensor
for test in range(1):
    # input_tensor = torch.randn(1, 3, 32, 32)-0.5
    A = torch.zeros(1, 3, 32, 32)
    A[0,1,10,15] = 1
    B = torch.zeros(1, 3, 32, 32)
    B[0,2,11,14] = 1
    # input_tensor = torch.zeros(1, 3, 32, 32)
    # input_tensor[0,1,10,15] = 1

    # Perform convolution and then transposed convolution
    conv_output = conv_layer(A)
    reconstructed = transpose_conv_layer(conv_output)
    dot = torch.sum(B * reconstructed)
    print(dot)


# # Function to perform manual convolution using unfold (mimics sparse matrix multiplication)
# def manual_conv2d(input, weight, bias=None, stride=1, padding=1):
#     # Unfold the input tensor
#     unfold_input = F.unfold(input, kernel_size=3, dilation=1, padding=padding, stride=stride)
#     # Perform the convolution operation
#     output = unfold_input.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
#     # Add bias if it exists
#     if bias is not None:
#         output += bias.unsqueeze(0).unsqueeze(2)
#     # Reshape to get the final output
#     output_size = (input.size(0), weight.size(0), input.size(2), input.size(3))
#     return output.view(output_size)

# # Perform manual convolution
# manual_conv_output = manual_conv2d(input_tensor, conv_layer.weight, conv_layer.bias, stride=1, padding=1)

# # Check if the results are close
# result_check = torch.allclose(reconstructed, manual_conv_output)

# print('Result check: ', result_check)
