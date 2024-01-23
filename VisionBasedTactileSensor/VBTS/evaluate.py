import os
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
from model.resnet import resnet34
from sklearn.metrics import mean_squared_error
from parameter import label_mean, label_std
from dataset import MyDataSet

def inverse_transform(list, mean, std):
    list =np.array(list)
    mean=np.array(mean)
    std=np.array(std)
    list = list * std + mean
    list = list.tolist()
    return list

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 加载标签
    label_path = "/home/pmh/nvme1/Code/VBTS/data/tensor_data.csv"
    df = pd.read_csv(label_path)
    labels = df.iloc[:, 1:].values

    # create model
    model = resnet34(num_classes=121, include_top=True).to(device)
    # load model weights
    weights_path = r"/home/pmh/nvme1/Code/VBTS/weights/epoch_0.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # model.load_state_dict(torch.load(weights_path, map_location=device))
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    # prediction
    model.eval()

    # 创建数据集
    image_dir = "/home/pmh/nvme1/Code/VBTS/eva"
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
    dataset = MyDataSet(images_path=image_paths, images_class=labels, transform=data_transform, label_transform=True)

    preds = []
    true_labels = []

    with torch.no_grad():
        for img, label in dataset:
            img = img.unsqueeze(0).to(device)
            output = torch.squeeze(model(img)).cpu().numpy()
            output = inverse_transform(output, label_mean, label_std)
            label = inverse_transform(label, label_mean, label_std)
            preds.append(output)
            true_labels.append(label)

    mse = mean_squared_error(true_labels, preds)
    print(f"MSE: {mse}")


if __name__ == '__main__':
    main()
