import torchvision
from torchvision import transforms as T


def train_transform():
    trainT = T.Compose([
                    # T.RandomHorizontalFlip(), # 随机翻转图像
                    # T.RandomCrop(32, padding=4), # 随机裁剪并在边缘填充
                    # T.RandomRotation(15), # 随机旋转 +/- 15 度
                    # T.Resize((224, 224)), # 修改图像大小
                    # T.ToTensor()
                    # ,T.Normalize(mean = [0.485, 0.456, 0.406]
                    #             ,std = [0.229, 0.224, 0.225])

                    T.Resize((299, 299)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue = 0.1),
                    T.ToTensor(),
                    T.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
                    ])
    return trainT


def test_transform():
    testT = T.Compose([T.CenterCrop(299),
                       T.ToTensor(),
                       T.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
                       ])
    return testT


def train_data():
    train = torchvision.datasets.ImageFolder(root=r"F:\sewer\SewerData_split\train",
                                            transform = train_transform()
                                            )
    return train

def test_data():
    test = torchvision.datasets.ImageFolder(root=r"F:\sewer\SewerData_split\val",
                                            transform = test_transform()
                                            )
    return test