import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

# Labels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK", "VA", "ND"]
Labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
# class MultiLabelDataset(Dataset):
#     def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, onlyDefects=False):
#         super(MultiLabelDataset, self).__init__()
#         self.imgRoot = imgRoot
#         self.annRoot = annRoot
#         self.split = split
#
#         self.transform = transform
#         self.loader = default_loader
#
#         self.LabelNames = Labels.copy()
#         self.LabelNames.remove("VA")
#         self.LabelNames.remove("ND")
#         self.onlyDefects = onlyDefects
#
#         self.num_classes = len(self.LabelNames)
#
#         self.loadAnnotations()
#         self.class_weights = self.getClassWeights()
#
#     def loadAnnotations(self):
#         gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
#         gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = self.LabelNames + ["Filename", "Defect"])
#
#         if self.onlyDefects:
#             gt = gt[gt["Defect"] == 1]
#
#         self.imgPaths = gt["Filename"].values
#         self.labels = gt[self.LabelNames].values
#
#     def __len__(self):
#         return len(self.imgPaths)
#
#     def __getitem__(self, index):
#         path = self.imgPaths[index]
#
#         img = self.loader(os.path.join(self.imgRoot, path))
#         if self.transform is not None:
#             img = self.transform(img)
#
#         target = self.labels[index, :]
#
#         return img, target, path
#
#     def getClassWeights(self):
#         data_len = self.labels.shape[0]
#         class_weights = []
#
#         for defect in range(self.num_classes):
#             pos_count = len(self.labels[self.labels[:,defect] == 1])
#             neg_count = data_len - pos_count
#
#             class_weight = neg_count/pos_count if pos_count > 0 else 0
#             class_weights.append(np.asarray([class_weight]))
#         return torch.as_tensor(class_weights).squeeze()
#         # return torch.from_numpy(class_weights, dtype=np.float32).squeeze()



class MultiLabelDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, onlyDefects=False):
        super(MultiLabelDataset, self).__init__()
        self.imgRoot = imgRoot
        # self.annRoot = r"D:\desktop\sewerml"
        self.split = split

        self.transform = transform
        self.loader = loader

        self.LabelNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "11"]  # Label值对应子文件夹的名称
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)

        self.loadAnnotations()
        self.class_weights = self.getClassWeights()
    def loadAnnotations(self):
        if self.split == "Train":
            data_root = self.imgRoot
        elif self.split == "Val":
            data_root = r"F:\sewer\validation_data"
        else:
            raise ValueError(f"无效的 split 参数：{self.split}")

        img_paths = []
        labels = []

        for class_name in self.LabelNames:
            class_path = os.path.join(self.imgRoot, class_name)
            if not os.path.exists(class_path):
                raise FileNotFoundError(f"子文件夹 {class_path} 不存在")

            for img_filename in os.listdir(class_path):
                img_path = os.path.join(class_path, img_filename)
                img_paths.append(img_path)
                print(f"img_filename: {img_filename}")


                # 将子文件夹名称作为标签
                label = [0] * self.num_classes
                label[self.LabelNames.index(class_name)] = 1
                labels.append(label)
                print(f"target shape: {labels}")


                # label = self.LabelNames.index(class_name)
                # print(f"label: {label}")
                # labels.append(label)
                # print(f"label type: {type(label)}")


        self.imgPaths = img_paths
        self.labels = torch.tensor(labels)
        print(f"labels: {labels}")
        print(f"self.labels: {self.labels}")
        # print(f"self.labels111: {self.labels[30, :]}")
        # self.labels = self.LabelNamaes


    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)


        # target = self.labels[index]
        # target = self.labels[index]
        # target = os.path.basename(path)
        # target = int(target)

        # target = self.labels[index]  # 获取第index个元素，表示第index个样本的标签
        # target = [1 if i == target else 0 for i in range(self.num_classes)]  # 转换为独热编码形式
        # target = torch.tensor(target, dtype=torch.float32)  # 转换为浮点型张量
        target = self.labels[index]

        path = path.replace(self.imgRoot, "")
        return img, target, path

    def getClassWeights(self):
        data_len = len(self.labels)
        class_weights = []

        for defect in range(self.num_classes):
            pos_count = len(self.labels[self.labels == defect])
            neg_count = data_len - pos_count

            class_weight = neg_count / pos_count if pos_count > 0 else 0
            class_weights.append(class_weight)

        return torch.tensor(class_weights, dtype=torch.float32)











class MultiLabelDatasetInference(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, onlyDefects=False):
        super(MultiLabelDatasetInference, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)

        self.loadAnnotations()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["Filename"])

        self.imgPaths = gt["Filename"].values
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        return img, path


class BinaryRelevanceDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, defect=None):
        super(BinaryRelevanceDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.defect = defect

        assert self.defect in self.LabelNames

        self.num_classes = 1

        self.loadAnnotations()
        self.class_weights = self.getClassWeights()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["Filename", self.defect])

        self.imgPaths = gt["Filename"].values
        self.labels =  gt[self.defect].values.reshape(self.imgPaths.shape[0], 1)
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index]

        return img, target, path

    def getClassWeights(self):
        pos_count = len(self.labels[self.labels == 1])
        neg_count = self.labels.shape[0] - pos_count
        class_weight = np.asarray([neg_count/pos_count])

        return torch.as_tensor(class_weight)


class BinaryDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader):
        super(BinaryDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.num_classes = 1

        self.loadAnnotations()
        self.class_weights = self.getClassWeights()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["Filename", "Defect"])

        self.imgPaths = gt["Filename"].values
        self.labels =  gt["Defect"].values.reshape(self.imgPaths.shape[0], 1)
        print(self.labels.shape)
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index]

        return img, target, path

    def getClassWeights(self):
        pos_count = len(self.labels[self.labels == 1])
        neg_count = self.labels.shape[0] - pos_count
        class_weight = np.asarray([neg_count/pos_count])

        return torch.as_tensor(class_weight)



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
        transforms.ToTensor()])

    
    train = MultiLabelDataset(annRoot="./annotations", imgRoot="./Data", split="Train", transform=transform)
    train_defect = MultiLabelDataset(annRoot="./annotations", imgRoot="./Data", split="Train", transform=transform, onlyDefects=True)
    binary_train = BinaryDataset(annRoot="./annotations", imgRoot="./Data", split="Train", transform=transform)
    binary_relevance_train = BinaryRelevanceDataset(annRoot="./annotations", imgRoot="./Data", split="Train", transform=transform, defect="RB")

    print(len(train), len(train_defect), len(binary_train), len(binary_relevance_train))
    print(train.class_weights, train_defect.class_weights, binary_train.class_weights, binary_relevance_train.class_weights)

    