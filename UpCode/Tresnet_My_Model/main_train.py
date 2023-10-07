import os
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark=True #用于加速GPU运算的代码
import matplotlib.pyplot as plt #可视化
from time import time #计算时间、记录时间
import random #控制随机性
import numpy as np
import gc #garbage collector 垃圾回收
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from Transform import train_data, test_data
from tresnet_model import TResNet
from EarlyStopping import EarlyStopping
from L1 import L1_Regularization
from Vision import plotloss, plotsample
from torch.optim.lr_scheduler import MultiStepLR


random.seed(42) #random
np.random.seed(42) #numpy.random
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MODEL_NAMES = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(ml_models.__dict__[name]))

def IterOnce(net,criterion,opt,x,y):
    """
    对模型进行一次迭代的函数

    net: 实例化后的架构
    criterion: 损失函数
    opt: 优化算法
    x: 这一个batch中所有的样本
    y: 这一个batch中所有样本的真实标签
    """
    l1_regularization = L1_Regularization(lambda_l1=1e-3)
    # 前向传播
    sigma = net.forward(x)
    # 计算损失
    loss = criterion(sigma,y)
    # loss = criterion(sigma, y) + l1_regularization(net.parameters())
    # 反向传播
    loss.backward()
    # 使用优化算法 opt 执行一步参数更新，将模型的参数按照计算得到的梯度进行更新。
    opt.step()
    #比起设置梯度为0，让梯度为None会更节约内存
    opt.zero_grad(set_to_none=True)

    yhat = torch.max(sigma,1)[1]
    correct = torch.sum(yhat == y)
    return correct, loss

def TestOnce(net,criterion,x,y):
    """
    对一组数据进行测试并输出测试结果的函数

    net: 经过训练后的架构
    criterion：损失函数
    x：要测试的数据的所有样本
    y：要测试的数据的真实标签
    """
    #对测试，一定要阻止计算图追踪
    #这样可以节省很多内存，加速运算
    # l1_regularization = L1_Regularization(lambda_l1=1e-3)
    with torch.no_grad():
        sigma = net.forward(x)
        loss = criterion(sigma,y)
        # loss = criterion(sigma, y) + l1_regularization(net.parameters())
        yhat = torch.max(sigma,1)[1]
        correct = torch.sum(yhat == y)
    return correct,loss


def tresnet_l(num_classes, pretrained=False, **kwargs):
    """ Constructs a large TResnet model.
    """
    model = TResNet(layers=[4, 5, 18, 3], num_classes=num_classes, in_chans=3, width_factor=1.2,
                    remove_aa_jit=True)

    if pretrained:
        pass


    return model


def fit_test(net, batchdata, testdata, criterion, opt, epochs, tol, modelname, PATH):

    SamplePerEpoch = batchdata.dataset.__len__()
    allsamples = SamplePerEpoch * epochs
    trainedsamples = 0
    trainlosslist = []
    testlosslist = []
    early_stopping = EarlyStopping(tol = tol)
    highestacc = None

    for epoch in range(1, epochs + 1):
        net.train()
        loss_train = 0
        correct_train = 0
        for batch_idx, (x, y) in enumerate(batchdata):
            x = x.to(device, non_blocking = True)
            y = y.to(device, non_blocking = True).view(x.shape[0])
            correct, loss = IterOnce(net, criterion, opt, x, y)
            trainedsamples += x.shape[0]
            loss_train += loss
            correct_train += correct

            if (batch_idx + 1) % 10 == 0:
                print('Epoch{}/{}:[{}/{}({:.0f}%)]'.format(epoch
                                                        , epochs + 1
                                                        , trainedsamples
                                                        , allsamples
                                                        , 100*trainedsamples/allsamples))

        # TrainAccThisEpoch = float(correct_train*100)/SamplePerEpoch
        # TrainLossThisEpoch = float(loss_train*100)/SamplePerEpoch
        TrainAccThisEpoch = float(correct_train)/SamplePerEpoch
        TrainLossThisEpoch = float(loss_train)/SamplePerEpoch
        trainlosslist.append(TrainLossThisEpoch)

        del x, y, correct, loss, correct_train, loss_train
        gc.collect()
        torch.cuda.empty_cache()

        net.eval()
        loss_test = 0
        correct_test = 0
        TestSample = testdata.dataset.__len__()

        for x, y in testdata:
            with torch.no_grad():
                x = x.to(device, non_blocking = True)
                y = y.to(device, non_blocking = True).view(x.shape[0])
                correct, loss = TestOnce(net, criterion, x, y)
                loss_test += loss
                correct_test += correct

        # TestAccThisEpoch = float(correct_test*100)/TestSample
        # TestLossThisEpoch = float(loss_test*100)/TestSample
        TestAccThisEpoch = float(correct_test)/TestSample
        TestLossThisEpoch = float(loss_test)/TestSample
        testlosslist.append(TestLossThisEpoch)

        print("\t Train Loss:{:.6f}, Test Loss:{:.6f}, Train Acc:{:.3f}%, Test Acc:{:.3f}%"
              .format(TrainLossThisEpoch ,TestLossThisEpoch, TrainAccThisEpoch * 100, TestAccThisEpoch * 100))

        if highestacc == None:
            highestacc = TestAccThisEpoch
        if highestacc < TestAccThisEpoch:
            highestacc = TestAccThisEpoch
            torch.save(net.state_dict(), os.path.join(PATH, modelname+".pt"))
            print("\t Weight Saved")


        #EarlyStopping
        early_stop = early_stopping(TestLossThisEpoch)
        if early_stop == True:
            break

    print("Mission Complete")
    return trainlosslist,testlosslist


def full_procedure(net, epochs, bs, traindata, testdata, modelname, PATH, lr=0.001,alpha=0.99,gamma=0.,wd=0.,
                   tol=10**(-5), load_initial_weight=False):

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 指定学习率调度步骤
    milestones = [60, 120]
    lr_values = [0.1, 0.01, 0.001]

    #分割数据
    batchdata = DataLoader(traindata,batch_size=bs,shuffle=True
                           ,drop_last=False, pin_memory=True) #线程 - 调度计算资源的最小单位
    testdata = DataLoader(testdata,batch_size=bs,shuffle=False
                          ,drop_last=False, pin_memory=True)

    plotsample(batchdata)

    #损失函数，优化算法
    # criterion = nn.CrossEntropyLoss(reduction="sum") #进行损失函数计算时，最后输出结果的计算模式
    criterion=torch.nn.BCEWithLogitsLoss
    # opt = optim.RMSprop(net.parameters(),lr=lr
    #                     ,alpha=alpha,momentum=gamma,weight_decay=wd)
    opt = torch.optim.SGD(net.parameters(), lr=lr_values[0],
                            momentum=gamma, weight_decay=wd)
    scheduler = MultiStepLR(opt, milestones=milestones, gamma=0.1)  # 在milestones处调整学习率


    if load_initial_weight:
        initial_weight_path = r"D:\desktop\initial_model_weights.pt"  # 设置初始模型权重文件的路径
        if os.path.exists(initial_weight_path):
            net.load_state_dict(torch.load(initial_weight_path))
            print("Initial model weights loaded.")

    #训练与测试
    for epoch in range(1, epochs + 1):
        trainloss, testloss = fit_test(net,batchdata,testdata,criterion,opt,epochs,tol,modelname,PATH)
        scheduler.step()

        return trainloss, testloss


# 建立目录用于存储模型选择结果
PATH = r"F:\sewer\Pmh\results\1"

# 使用函数full_procedure中的默认参数，在模型选择时加入时间计算
# 基于现有显存，batch_size被设置为256，对于CPU而言最好从64开始设置
# MyResNet
avgtime = [] #用来存放每次循环之后获得的训练时间
for i in range(1): # 进行5次训练
    # 设置随机数种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


    # 实例化自定义模型
    model = tresnet_l(16, pretrained= False)
    net = model.to(device,non_blocking=True)

    # 训练
    start = time() # 计算训练时间
    trainloss, testloss = full_procedure(net
                                         , traindata = train_data()
                                         , testdata = test_data()
                                         , epochs=500
                                         , bs=128
                                         , modelname="model_selection_TResnet"
                                         , PATH = PATH
                                         , tol = 10**(-3)
                                         , lr = 0.1
                                         , gamma = 0.9
                                         , wd = 0.0001
                                         )
    avgtime.append(time()-start)
    plotloss(trainloss, testloss)



