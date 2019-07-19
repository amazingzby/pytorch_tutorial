#神经网络的训练过程：
#1.定义有可训练参数的神经网络
#2.对输入数据迭代
#3.使用神经网络计算输入
#4.计算loss
#5.计算网络反向传播的梯度
#6.更新权重

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16*6*6,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features*=s
        return num_features
net = Net()
print(net)

input = torch.randn(1,1,32,32)
out = net(input)
print(out)

#zero_grad将网络反向传播梯度设置为0
net.zero_grad()
#10个输出反向传播系数随机
out.backward(torch.randn(1,10))

#note：torch.nn仅支持mini-batches，输入为mini-batches，不支持单样本（没有batch维度的样本）
#如果是单样本，需要使用input.unsqueeze(0)增加batch维度

#简单回顾
#torch.Tensor 多维数组，支持自动求导（autograd）
#nn.Module  神经网络模块，很方便封装参数，将参数迁移至GPU，数据导出，下载等
#nn.Parameter Tensor的一种类型，当将其分配到Module中时自动分配参数
#autograd.Function 自定义操作中执行fordward和backward定义

output = net(input)
target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()

loss = criterion(output,target)
print(loss)

#前向传播过程input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#      -> view -> linear -> relu -> linear -> relu -> linear
#      -> MSELoss
#      -> loss

#当调用loss.backward(),整个网络反向传播，requires_grad=True的tensor有.grad梯度值
#.grad_fn属性保存梯度graph
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])#Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])#Relu

#反向传播
net.zero_grad()
print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)

loss.backward()
print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)

#反向传播公式：weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data*learning_rate)

import torch.optim as optim
optimizer = optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output,target)
loss.backward()
optimizer.step()