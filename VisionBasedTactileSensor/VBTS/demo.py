import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import resnet34
from mpl_toolkits.mplot3d import Axes3D
from parameter import label_mean, label_std

def inverse_transform(list, mean, std):
    list =np.array(list)
    mean=np.array(mean)
    std=np.array(std)
    list = list * std + mean
    list = list.tolist()
    return list

def plot_output(fig, ax, output):
    Z = np.array(output).reshape((11, 11))
    x, y = np.linspace(0, 10, 11), np.linspace(0, 10, 11)
    X, Y = np.meshgrid(x, y)

    ax.cla()
    ax.bar3d(X.flatten(), Y.flatten(), np.zeros_like(Z).flatten(), 1, 1, Z.flatten(), color='skyblue', alpha=0.8)
    ax.plot_surface(X, Y, np.zeros_like(X), color='yellow', alpha=0.5)
    ax.view_init(elev=20, azim=45)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    fixed_img_path = r"/home/pmh/nvme1/Code/VBTS/images/001.jpg"  # Replace with the path of your second image
    assert os.path.exists(fixed_img_path), "file: '{}' does not exist.".format(fixed_img_path)
    img2 = Image.open(fixed_img_path)
    img2_size = img2.size
    img2 = data_transform(img2)

    model = resnet34().to(device)
    weights_path = r"/home/pmh/nvme1/Code/VBTS/weights/epoch_99.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # model.load_state_dict(torch.load(weights_path, map_location=device))
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    plt.ioff()

    video_path = r"/home/pmh/nvme1/Code/VBTS/mov.mp4"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, img2_size)

        img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        img_transformed = data_transform(img)

        img_concatenated = torch.cat((img_transformed, img2), dim=0)
        img_concatenated = torch.unsqueeze(img_concatenated, dim=0).to(device)

        with torch.no_grad():
            output = torch.squeeze(model(img_concatenated)).cpu()
            # output = inverse_transform(output, label_mean, label_std) 
        plot_img = plot_output(fig, ax, output.numpy())
        cv2.imshow('Video Frame', frame)
        cv2.imshow('3D Plot', plot_img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()