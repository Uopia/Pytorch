import torchvision.transforms as T
from Transform import train_data, test_data
import matplotlib.pyplot as plt #可视化

def vision_sample(n):
    # 获取一个样本（图像和标签）
    sample_index = n # 可以选择任意索引
    img, label = train_data[sample_index]
    # 将张量图像转换回PIL图像
    unnormalized_img = T.ToPILImage()(img)
    # 显示图像和标签
    print(f"Label: {label}")
    unnormalized_img.show()


def plotloss(trainloss, testloss):
    plt.figure(figsize=(10, 7))
    plt.plot(trainloss, color="red", label="Trainloss")
    plt.plot(testloss, color="orange", label="Testloss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plotsample(data): #只能够接受tensor格式的图像
    fig, axs = plt.subplots(1,5,figsize=(10,10)) #建立子图
    for i in range(5):
        num = random.randint(0,len(data)-1) #首先选取随机数，随机选取五次
        #抽取数据中对应的图像对象，make_grid函数可将任意格式的图像的通道数升为3，而不改变图像原始的数据
        #而展示图像用的imshow函数最常见的输入格式也是3通道
        npimg = torchvision.utils.make_grid(data[num][0]).numpy()
        nplabel = data[num][1] #提取标签
        #将图像由(3, weight, height)转化为(weight, height, 3)，并放入imshow函数中读取
        axs[i].imshow(np.transpose(npimg, (1, 2, 0)))
        axs[i].set_title(nplabel) #给每个子图加上标签
        axs[i].axis("off") #消除每个子图的坐标轴