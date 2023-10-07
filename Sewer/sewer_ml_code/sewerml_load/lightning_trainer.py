import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch import nn

from torchvision import models as torch_models
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_datamodules import MultiLabelDataModule, BinaryDataModule, BinaryRelevanceDataModule
import sewer_models
import ml_models
from collections import OrderedDict

import os

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


class MultiLabelModel(pl.LightningModule):
    # 主要用于获取三个模块（torch_models、sewer_models、ml_models）中所有小写名称、非私有方法（不以双下划线开头）的可调用对象，并将它们按字母顺序排序后组合成一个列表MODEL_NAMES。
    TORCHVISION_MODEL_NAMES = sorted(name for name in torch_models.__dict__ if
                                     name.islower() and not name.startswith("__") and callable(
                                         torch_models.__dict__[name]))
    SEWER_MODEL_NAMES = sorted(name for name in sewer_models.__dict__ if
                               name.islower() and not name.startswith("__") and callable(sewer_models.__dict__[name]))
    MULTILABEL_MODEL_NAMES = sorted(name for name in ml_models.__dict__ if
                                    name.islower() and not name.startswith("__") and callable(ml_models.__dict__[name]))
    MODEL_NAMES = TORCHVISION_MODEL_NAMES + SEWER_MODEL_NAMES + MULTILABEL_MODEL_NAMES

    # 类初始化函数
    def __init__(self, model="resnet18", num_classes=2, learning_rate=1e-2, momentum=0.9, weight_decay=0.0001,
                 criterion=torch.nn.BCEWithLogitsLoss, pretrained_ckpt_path=None, **kwargs):
        super(MultiLabelModel, self).__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes

        # 给self.model赋值
        if model in MultiLabelModel.TORCHVISION_MODEL_NAMES:
            self.model = torch_models.__dict__[model](num_classes=self.num_classes)
        elif model in MultiLabelModel.SEWER_MODEL_NAMES:
            self.model = sewer_models.__dict__[model](num_classes=self.num_classes)
        elif model in MultiLabelModel.MULTILABEL_MODEL_NAMES:
            self.model = ml_models.__dict__[model](num_classes=self.num_classes)
        else:
            raise ValueError("Got model {}, but no such model is in this codebase".format(model))

        if pretrained_ckpt_path is not None:
            # 使用torch.load函数加载预训练的检查点文件。pretrained_ckpt_path是包含预训练模型权重的文件路径。
            # map_location=self.device指定了加载的设备（GPU或CPU），self.device是类中的一个属性，用于指定设备。
            ckpt = torch.load(pretrained_ckpt_path, map_location=self.device)

            # state_dict = ckpt["state_dict"] 和 state_dict = ckpt: 这是根据检查点文件的格式来选择正确的权重字典。
            # 有些检查点文件可能包含state_dict键，而另一些可能直接保存了权重字典。这里首先尝试通过"state_dict"键获取权重字典，如果不存在，则直接使用ckpt中的字典。
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt

            # updated_state_dict = OrderedDict(): 创建一个有序字典updated_state_dict用于存储更新后的权重。
            updated_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "criterion" in k:
                    continue
                name = k.replace("model.", "")
                updated_state_dict[name] = v

            # 将更新后的权重字典updated_state_dict加载到模型的self.model属性中。
            self.model.load_state_dict(updated_state_dict)

            # 在加载预训练模型的权重后，输出一些信息来确认是否成功加载了权重。
            if len(state_dict) > 0:
                print("\nFirst 5 keys and values in the loaded state_dict:")
                for i, (k, v) in enumerate(state_dict.items()):
                    if i < 5:
                        print(f"{k}: {v}")
                    else:
                        break
                print(f"Loaded pretrained weights from: {pretrained_ckpt_path}")
            else:
                print("No pretrained weights were found in the given checkpoint file.")
        else:
            print("No pretrained weights were loaded.")

        # self.aux_logits = hasattr(self.model, "aux_logits"): 这行代码检查self.model是否具有属性"aux_logits"。hasattr()是一个内置函数，
        # 它返回一个布尔值，指示对象是否具有指定的属性。在这里，它用于检查self.model是否有"aux_logits"这个属性。
        # "aux_logits"通常用于指示模型是否具有辅助分类器。

        # self.train_function = self.aux_loss: 如果模型具有辅助分类器（self.aux_logits为True），
        # 则将self.train_function设置为self.aux_loss。这意味着在训练过程中将使用辅助损失函数self.aux_loss。

        # else: self.train_function = self.normal_loss: 如果模型没有辅助分类器（self.aux_logits为False），
        # 则将self.train_function设置为self.normal_loss。这意味着在训练过程中将使用普通的损失函数self.normal_loss。
        self.aux_logits = hasattr(self.model, "aux_logits")
        if self.aux_logits:
            self.train_function = self.aux_loss
        else:
            self.train_function = self.normal_loss

        self.criterion = criterion
        if callable(getattr(self.criterion, "set_device", None)):
            self.criterion.set_device(self.device)

    # 这是一个标准的forward方法，用于定义模型的前向传播过程。
    # x是输入数据（通常是一个批次的样本），通过模型进行前向传播，得到预测的logits（未经过激活函数的输出）。
    # 在这里，模型的前向传播过程是通过self.model(x)来实现的，self.model表示该类中的模型。
    def forward(self, x):
        logits = self.model(x)
        return logits

    # 这是一个用于计算辅助损失（auxiliary loss）的方法。
    # x是输入数据（通常是一个批次的样本），y是对应的标签。
    # 首先将y转换为浮点型（float），因为损失函数可能要求标签为浮点数类型。
    # 然后调用self(x)进行模型的前向传播，得到主分类器输出y_hat和辅助分类器输出y_aux_hat。
    # 计算损失函数，这里使用两部分损失的加权和，主分类器的损失和辅助分类器的损失。
    # 通常，辅助分类器的损失会以较小的权重（0.4）加到主分类器损失上，以辅助训练主分类器。
    # 返回计算得到的总损失。
    def aux_loss(self, x, y):
        y = y.float()
        y_hat, y_aux_hat = self(x)
        loss = self.criterion(y_hat, y) + 0.4 * self.criterion(y_aux_hat, y)

        return loss

    def normal_loss(self, x, y):
        y = y.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        return loss

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.train_function(x, y)

        # .log sends to tensorboard/logger, prog_bar also sends to the progress bar
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.normal_loss(x, y)

        # lightning monitors 'checkpoint_on' to know when to checkpoint (this is a tensor)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, sync_dist=True)
        return result

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.normal_loss(x, y)

        # lightning monitors 'checkpoint_on' to know when to checkpoint (this is a tensor)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('test_loss', loss, sync_dist=True)
        return result

    def configure_optimizers(self):
        # 随机梯度下降法
        optim = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
                                momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        # 这行代码创建一个多步学习率调度器（MultiStepLR）。它接受之前创建的优化器optim作为参数，并根据milestones中指定的训练轮次（epoch）来调整学习率。
        # gamma参数用于设置学习率的乘法因子，当训练轮次在milestones中指定的里程碑时，学习率将乘以gamma。
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[30, 60, 80], gamma=0.1)

        # return [optim], [scheduler]
        return [optim], [{'scheduler': scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.1)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument('--model', type=str, default="resnet18", choices=MultiLabelModel.MODEL_NAMES)
        return parser


def main(args):
    pl.seed_everything(1234567890)

    # Init data with transforms
    # 如果模型名称是 "inception_v3" 或 "chen2018_multilabel"，则将 img_size 设置为 299，否则将其设置为 224。
    img_size = 299 if args.model in ["inception_v3", "chen2018_multilabel"] else 224

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    if args.training_mode == "e2e":
        dm = MultiLabelDataModule(batch_size=args.batch_size, workers=args.workers,
                                  ann_root=args.ann_root, data_root=args.data_root,
                                  train_transform=train_transform, eval_transform=eval_transform,
                                  only_defects=False)
    elif args.training_mode == "defect":
        dm = MultiLabelDataModule(batch_size=args.batch_size, workers=args.workers,
                                  ann_root=args.ann_root, data_root=args.data_root,
                                  train_transform=train_transform, eval_transform=eval_transform,
                                  only_defects=True)
    elif args.training_mode == "binary":
        dm = BinaryDataModule(batch_size=args.batch_size, workers=args.workers,
                              ann_root=args.ann_root, data_root=args.data_root,
                              train_transform=train_transform, eval_transform=eval_transform)
    elif args.training_mode == "binaryrelevance":
        assert args.br_defect is not None, "Training mode is 'binary_relevance', but no 'br_defect' was stated"
        dm = BinaryRelevanceDataModule(batch_size=args.batch_size, workers=args.workers,
                                       ann_root=args.ann_root, data_root=args.data_root,
                                       train_transform=train_transform, eval_transform=eval_transform,
                                       defect=args.br_defect)
    else:
        raise Exception("Invalid training_mode '{}'".format(args.training_mode))

    dm.prepare_data()
    dm.setup("fit")

    # # Init our model
    # pretrained_ckpt_path = r"G:\Pmh\code\log\tresnet_l\defect-version_1\last.ckpt"
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=dm.class_weights)
    # light_model = MultiLabelModel(num_classes=dm.num_classes, criterion= criterion, pretrained_ckpt_path=pretrained_ckpt_path, **vars(args))
    light_model = MultiLabelModel(num_classes=dm.num_classes, criterion=criterion, **vars(args))

    # train
    prefix = "{}-".format(args.training_mode)
    if args.training_mode == "binaryrelevance":
        prefix += args.br_defect

    logger = TensorBoardLogger(save_dir=args.log_save_dir, name=args.model,
                               version=prefix + "version_" + str(args.log_version))

    logger_path = os.path.join(args.log_save_dir, args.model, prefix + "version_" + str(args.log_version))

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(logger_path, '{epoch:02d}-{val_loss:.2f}'),
        save_top_k=5,
        save_last=True,
        verbose=False,
        monitor="val_loss",
        mode='min',
        prefix='',
        period=1
    )

    lr_monitor = LearningRateLogger(logging_interval='step')

    trainer = pl.Trainer.from_argparse_args(args, terminate_on_nan=True, benchmark=True, max_epochs=args.max_epochs,
                                            logger=logger, checkpoint_callback=checkpoint_callback,
                                            callbacks=[lr_monitor])

    try:
        trainer.fit(light_model, dm)
    except Exception as e:
        print(e)
        with open(os.path.join(logger_path, "error.txt"), "w") as f:
            f.write(str(e))


def run_cli():
    # add PROGRAM level args
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='Pytorch-Lightning')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default='./annotations')
    parser.add_argument('--data_root', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=64, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--log_save_dir', type=str, default="./logs")
    parser.add_argument('--log_version', type=int, default=1)
    parser.add_argument('--training_mode', type=str, default="e2e",
                        choices=["e2e", "binary", "binaryrelevance", "defect"])
    parser.add_argument('--br_defect', type=str, default=None,
                        choices=[None, "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO", "GR", "PH",
                                 "PB", "OS", "OP", "OK"])

    # add TRAINER level args
    parser = pl.Trainer.add_argparse_args(parser)

    # add MODEL level args
    parser = MultiLabelModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # Adjust learning rate to amount of nodes/GPUs
    args.workers = max(0, min(8, 4 * args.gpus))
    args.learning_rate = args.learning_rate * (args.gpus * args.num_nodes * args.batch_size) / 256

    main(args)


if __name__ == "__main__":
    run_cli()