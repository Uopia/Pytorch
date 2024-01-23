import os
import argparse
import torch
from torchvision import transforms
from utils import read_split_data, plot_data_loader_image, train_one_epoch, evaluate
from dataset import MyDataSet


os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main(args):

    # set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # split dataset into train and validate
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.annotations_path, args.data_path)

    # data transform
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # instantiate dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"],
                              label_transform=args.label_transform
                              )

    labels = []
    # 收集训练集中的所有标签
    labels = [torch.tensor(label, dtype=torch.float32) for _, label in train_dataset]
    # labels = torch.tensor(labels, dtype=torch.float32)
    labels = torch.stack(labels, dim=0)  # 现在这个操作应该可以正常工作
    label_mean = torch.mean(labels, dim=0)
    label_std = torch.std(labels, dim=0)

    parameter_file_path = r'/home/pmh/nvme1/Code/VBTS/parameter.py'

    with open(parameter_file_path, 'w') as f:
        f.write(f'label_mean = {label_mean.tolist()}\n')
        f.write(f'label_std = {label_std.tolist()}\n')


    print(f"Label Mean: {label_mean}")
    print(f"Label Std: {label_std}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--label_transform', type=bool, default=False)
    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default=r"/home/pmh/nvme1/Code/VBTS/data/pic")
    parser.add_argument('--annotations-path', type=str, 
                        default=r'/home/pmh/nvme1/Code/VBTS/data/tensor_data.csv')

    # parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, 
                        default= r'/home/pmh/nvme1/Code/VBTS/resnet34_pre.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

