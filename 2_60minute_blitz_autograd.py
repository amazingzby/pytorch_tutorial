# autograd:自动求微分
# torch.Tensor 通过将.requires_grad设置为true，将追踪所有对它的操作;
#当完成计算时调用.baclward()自动完成梯度计算，调用.grad属性获得当前tensor的梯度

#使用.detach()方法返回一个永远不需要梯度的变量，也可以使用下面语句：with torch.no_grad():

#Tensor有.grad_fn属性，指向创建tensor的函数

import torch
#创建一个tensor，设置 requires_grad为true
x = torch.ones(2,2,requires_grad=True)
print(x)

y = x + 2
print(y)

#输出 y的梯度
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z,out)

a = torch.randn(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)

out.backward()#计算梯度
print(x.grad)#print gradients d(out)/dx

#Now let’s take a look at an example of vector-Jacobian product:
x = torch.randn(3,requires_grad=True)
y = x*2
while y.data.norm()<1000:
    y = y * 2
print(y)

#关于backward() 如果 y是表量，backward()函数不需要参数
#如果是矩阵，传入的参数为各个标量的系数
v = torch.tensor([0.1,1.0,0.0001],dtype=torch.float)
y.backward(v)
print(x.grad)

#对于requires_grad=True的tensor，如果不想使其保留梯度：
print(x.requires_grad)
print((x**2).requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)