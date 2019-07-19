#https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
import torch
import numpy as np
x = torch.empty(5,3)#构建5×3未初始化的矩阵
print(x)
x = torch.rand(5,3)#构建5×3随机初始化矩阵
print(x)
x = torch.zeros(5,3,dtype=torch.long)#零矩阵
print(x)
x = torch.tensor([5.5,3])#直接通过数组构建矩阵
print(x)

x = x.new_ones(5,3,dtype=torch.double)#将x替换为类型为torch.double的矩阵，如果不指定类型则和原来x同类型
x = torch.randn_like(x,dtype=torch.float)
print(x)

print(x.size())#输出形状

#将torch的tensor类型转化为numpy类型
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

#将numpy类型转为tensor类型
a = np.ones(5)
b = torch.from_numpy(a)

#cuda tensors
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x,device=device)
    x = x.to(device)
    z = x + y
