import torch.nn as nn
import torch
from torchvision.datasets import MNIST


# class MNISTlinear(nn.Module):
#     def __init__(self):
#         super(MNISTlinear, self).__init__()
#         self.dropout = nn.Dropout(0.2)
#         self.linear = nn.Linear(1*28*28,10)
#     def forward(self,x):
#         x = x.view(len(x),-1)
#         output = self.dropout(x)
#         output = self.linear(output)
#         return output
#
#
# if __name__ =='__main__':
#     model = MNISTlinear()
#     model.train()
#     x = torch.randn((10,1,28,28))
#     print(x.shape)
#
#     print(model(x).shape)

import torch
import torch.nn as nn

# num_features - num_features from an expected input of size:batch_size*num_features*height*width
# eps:default:1e-5 (公式中为数值稳定性加到分母上的值)
# momentum:动量参数，用于running_mean and running_var计算的值，default：0.1
m = nn.BatchNorm2d(2, affine=True)  # affine参数设为True表示weight和bias将被使用
input = torch.randn(1, 2, 3, 4)
output = m(input)

print(input)
print(m.weight)
print(m.bias)
print(output)
print()