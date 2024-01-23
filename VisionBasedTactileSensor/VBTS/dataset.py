import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from parameter import label_mean, label_std

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None, label_transform=False):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.label_transform = label_transform
        self.fixed_image_path = "/home/pmh/nvme1/Code/VBTS/images/001.jpg"

    def __len__(self):
        return len(self.images_path)
    
    def label_Normalize(self, list, mean, std):
        list =np.array(list)
        mean=np.array(mean)
        std=np.array(std)
        list = (list - mean) / std
        list = list.tolist()
        return list

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        img_orin = Image.open(self.fixed_image_path).convert('RGB')
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)
            img_orin = self.transform(img_orin)

        if self.label_transform is not False:
            label = self.label_Normalize(label, label_mean, label_std)

        img = torch.cat((img, img_orin), 0)

        return img, label


