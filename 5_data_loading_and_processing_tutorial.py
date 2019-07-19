import os
import torch
import pandas as pd
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils

import warnings
warnings.filterwarnings("ignore")#Ignore warnings
plt.ion()#interactive mode

landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
n = 65
image_name = landmarks_frame.iloc[n,0]#根据下标取值
landmarks  = landmarks_frame.iloc[n,1:].as_matrix()
landmarks  = landmarks.astype('float').reshape(-1,2)

print("Image name:{}".format(image_name))
print("Landmarks shape.{}".format(landmarks.shape))
print("First 4 Landmarks:{}".format(landmarks[:4]))

def show_landmarks(image,landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:,0],landmarks[:,1],s=10,marker='.',c='r')
    plt.pause(2)
plt.figure()
show_landmarks(io.imread(os.path.join('data/faces',image_name)),landmarks)

#Dataset class
#自定义Dataset class继承DataSet，重写两个属性：
#__len__()  返回dataset总长度
# __getitem__（self,idx） 支持索引类似于dataset[i]
class FaceLandmarksDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform= transform
    def __len__(self):
        return len(self.landmarks_frame)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[idx,0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx,1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1,2)
        sample = {'image':image,'landmarks':landmarks}
        if self.transform:
            sample = self.transform(sample)
        return sample

face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',root_dir='data/faces/')
fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i,sample['image'].shape,sample['landmarks'].shape)
    ax = plt.subplot(1,4,i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)
    if i==3:
        plt.show()
        break

#Transforms 数据处理
#Rescale scale the image
#RandomCrop 将图片随机挖出一片区域
#ToTensor 将numpy格式的图片转换为tensor(改变轴的顺序)
#自定义需要重写:
#__init__
#__call__(self,sample)
#python __call__ 可以调用的对象，允许让类的实例表现的像函数一样

class Rescale(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image,landmarks = sample['image'],sample['landmarks']
        h,w = image.shape[:2]
        if isinstance(self.output_size,int):
            if h>w:
                new_h,new_w = self.output_size*h/w,self.output_size
            else:
                new_h,new_w = self.output_size,self.output_size*w/h
        else:
            new_h,new_w = self.output_size
        new_h,new_w = int(new_h),int(new_w)
        img = transform.resize(image,(new_h,new_w))
        landmarks = landmarks*[new_w/w,new_h/h]
        return {'image':img,'landmarks':landmarks}
class RandomCrop(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size,int):
            self.output_size = (output_size,output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size
    def __call__(self, sample):
        image,landmarks = sample['image'],sample['landmarks']
        h,w = image.shape[:2]
        new_h,new_w = self.output_size

        top = np.random.randint(0,h-new_h)
        left= np.random.randint(0,w-new_w)

        image = image[top:top+new_h,left:left+new_w]
        landmarks = landmarks - [left,top]
        return {'image':image,'landmarks':landmarks}

class ToTensor(object):
    def __call__(self, sample):
        image,landmarks = sample['image'],sample['landmarks']
        image = image.transpose((2,0,1))
        return {'image':torch.from_numpy(image),'landmarks':torch.from_numpy(landmarks)}

#Compose transforms
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),RandomCrop(224)])

fig = plt.figure()
sample = face_dataset[65]
for i ,tsfrm in enumerate([scale,crop,composed]):
    transformed_sample = tsfrm(sample)
    ax = plt.subplot(1,3,i+1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)
plt.show()

#迭代数据
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))
for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i,sample['image'].size(),sample['landmarks'].size())
    if i==3:
        break

#dataloader
dataloader = DataLoader(transformed_dataset,batch_size=4,shuffle=True,num_workers=4)

def show_landmarks_batch(sample_batched):
    image_batch,landmarks_batch = sample_batched['image'],sample_batched['landmarks']
    batch_size = len(image_batch)
    im_size = image_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(image_batch)
    plt.imshow(grid.numpy().transpose((1,2,0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i,:,0].numpy()+i*im_size+(i+1)*grid_border_size,landmarks_batch[i,:,1].numpy()+grid_border_size,s=10,marker='.',c='r')
        plt.title('Batch from dataloader')

for i_batch,sample_batched in enumerate(dataloader):
    print(i_batch,sample_batched['image'].size(),sample_batched['landmarks'].size())
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break

#使用torchvision
import torch
from torchvision import transforms,datasets

data_transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,batch_size=4,shuffle=True,num_workers=4)






















