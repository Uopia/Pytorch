import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from utils import read_split_data, plot_data_loader_image, train_one_epoch, evaluate
from dataset import MyDataSet
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from model.resnet import resnet34
from model.vit_model import vit_base_patch16_224_in21k as vit_base

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main(args):
    # set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create weights folder
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # init tensorboard
    tb_writer = SummaryWriter()


    # split dataset into train and validate
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.annotations_path, args.data_path)

    # data transform
    data_transform = {
        "train": transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                   }

    # instantiate dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"],
                              label_transform=args.label_transform
                              )
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"],
                            label_transform=args.label_transform
                            )
    
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(val_dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=False, 
                                                  num_workers=nw)

    # plot_data_loader_image(train_loader)

    train_num = len(train_dataset)
    val_num = len(val_dataset)


    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    

    net = resnet34(num_classes=args.num_classes, include_top=True)
    # net = vit_base(num_classes = args.num_classes, has_logits=False)
    print(net)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    
    if args.weights != '':
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        pretrained_dict = torch.load(args.weights, map_location='cpu')
        # print(net.load_state_dict(pretrained_dict, strict=False))
        del pretrained_dict['fc.weight']
        del pretrained_dict['fc.bias']
        # del pretrained_dict['patch_embed.proj.weight']
        # del pretrained_dict['patch_embed.proj.bias']
        # del pretrained_dict['head.weight']
        # del pretrained_dict['head.bias']
        missing_keys, unexpected_keys = net.load_state_dict(pretrained_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    # load checkpoint
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location=device)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print("Resuming the training process from epoch {}...".format(args.start_epoch))
    else:
        args.start_epoch = 0


    # if args.freeze_layers:
    #     for name, para in net.named_parameters():
    #         # 除head, pre_logits外，其他权重全部冻结
    #         if "head" not in name and "pre_logits" not in name:
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))

    net.to(device)

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.start_epoch, args.epochs):
        # train
        train_loss = train_one_epoch(net, 
                                     optimizer,
                                     train_loader, 
                                     device, 
                                     epoch)
        scheduler.step()

        val_loss = evaluate(model=net, 
                            data_loader=validate_loader, 
                            device=device,
                            epoch=epoch)

        tags = ["train_loss", "val_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], val_loss, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]['lr'], epoch)

        state = {
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
        }
        torch.save(state, './weights/epoch_{}.pth'.format(epoch))    
        # torch.save(net.state_dict(), './weights/epoch_{}.pth'.format(epoch))

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=121)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--label_transform', type=bool, default=True)
    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default=r"/home/pmh/nvme1/Code/VBTS/data/pic")
    parser.add_argument('--annotations-path', type=str, 
                        default=r'/home/pmh/nvme1/Code/VBTS/data/tensor_data.csv')

    # parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, 
    #                     default= r'/home/pmh/nvme1/Code/VBTS/pre_train_weight/vit_base_patch16_224_in21k.pth',
    #                     help='initial weights path')
    parser.add_argument('--weights', type=str, 
                        default= r'/home/pmh/nvme1/Code/VBTS/pre_train_weight/resnet34_pre.pth',
                        help='initial weights path')
    
    parser.add_argument('--resume', type=str, default='', 
                    help='Path to resume training from a checkpoint')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')


    opt = parser.parse_args()

    main(opt)

