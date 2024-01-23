import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from parameter import label_mean, label_std
from model import resnet34

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

    # load image
    img_path = r"/home/pmh/nvme1/Code/VBTS/data/pic/448.jpg"
    # img_path = r"/home/pmh/nvme1/Code/VBTS/midud.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    fixed_img_path = r"./images/001.jpg"  # Replace with the path of your second image
    assert os.path.exists(fixed_img_path), "file: '{}' does not exist.".format(fixed_img_path)
    img2 = Image.open(fixed_img_path)


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    

    # [N, C, H, W]
    img = data_transform(img)
    img2 = data_transform(img2)
    # expand batch dimension
    img_concatenated = torch.cat((img, img2), dim=0)
    img_concatenated = torch.unsqueeze(img_concatenated, dim=0)

    # create model
    model = resnet34().to(device)

    # load model weights
    weights_path = r"/home/pmh/nvme1/Code/VBTS/weights/epoch_2.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # model.load_state_dict(torch.load(weights_path, map_location=device))
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img_concatenated.to(device))).cpu()
        # output = inverse_transform(output, label_mean, label_std) 
    print(output)
    plt.subplot(1, 2, 2)
    output_matrix = output.flip(0).reshape(11, 11).numpy()
    plt.imshow(output_matrix, cmap='hot', interpolation='nearest')
    plt.title("Output Heatmap")
    plt.show()


if __name__ == '__main__':
    main()
