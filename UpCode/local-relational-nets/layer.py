import torch


class GeometryPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=0.5):
        super(GeometryPrior, self).__init__()
        self.channels = channels
        self.k = k
        # torch.rand(1, 2, k, k): 使用PyTorch的rand函数生成一个形状为(1, 2, k, k)的张量，其中每个元素都是从0到1的随机数。这里的k是一个未在代码中给出的变量，代表张量的某一维度的大小。
        # 2 * torch.rand(1, 2, k, k): 将上一步生成的张量的每个元素乘以2，这样元素的范围变为0到2。
        # 2 * torch.rand(1, 2, k, k) - 1: 从上一步的结果中减去1，这样元素的范围变为-1到1。
        # requires_grad=True: 这是一个参数，表示创建的张量需要计算梯度。这通常在神经网络中用于参数的更新，意味着这个张量在反向传播过程中会计算其梯度。
        # self.position = ...: 将上述创建的张量赋值给self.position。这里的self通常指的是一个类的实例，在类的方法中使用。
        self.position = 2 * torch.rand(1, 2, k, k, requires_grad=True) - 1
        self.l1 = torch.nn.Conv2d(2, int(multiplier * channels), 1)
        self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)
        
    def forward(self, x):
        x = self.l2(torch.nn.functional.relu(self.l1(self.position)))
        return x.view(1, self.channels, 1, self.k ** 2)



class KeyQueryMap(torch.nn.Module):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = torch.nn.Conv2d(channels, channels // m, 1)
    
    def forward(self, x):
        return self.l(x)


class AppearanceComposability(torch.nn.Module):
    def __init__(self, k, padding, stride):
        super(AppearanceComposability, self).__init__()
        self.k = k
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)
    
    def forward(self, x):
        key_map, query_map = x
        k = self.k
        key_map_unfold = self.unfold(key_map)
        query_map_unfold = self.unfold(query_map)
        key_map_unfold = key_map_unfold.view(
                    key_map.shape[0], key_map.shape[1],
                    -1,
                    key_map_unfold.shape[-2] // key_map.shape[1])
        query_map_unfold = query_map_unfold.view(
                    query_map.shape[0], query_map.shape[1],
                    -1,
                    query_map_unfold.shape[-2] // query_map.shape[1])
        return key_map_unfold * query_map_unfold[:, :, :, k**2//2:k**2//2+1]


def combine_prior(appearance_kernel, geometry_kernel):
    return torch.nn.functional.softmax(appearance_kernel + geometry_kernel,
                                       dim=-1)


class LocalRelationalLayer(torch.nn.Module):
    def __init__(self, channels, k, stride=1, m=None, padding=0):
        super(LocalRelationalLayer, self).__init__()
        self.channels = channels
        self.k = k
        self.stride = stride
        self.m = m or 8
        self.padding = padding
        self.kmap = KeyQueryMap(channels, k)
        self.qmap = KeyQueryMap(channels, k)
        self.ac = AppearanceComposability(k, padding, stride)
        self.gp = GeometryPrior(k, channels//m)
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)
        self.final1x1 = torch.nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        gpk = self.gp(0)
        km = self.kmap(x)
        qm = self.qmap(x)
        ak = self.ac((km, qm))
        ck = combine_prior(ak, gpk)[:, None, :, :, :]
        x_unfold = self.unfold(x)
        x_unfold = x_unfold.view(x.shape[0], self.m, x.shape[1] // m,
                                 -1, x_unfold.shape[-2] // x.shape[1])
        pre_output = (ck * x_unfold).view(x.shape[0], x.shape[1],
                                          -1, x_unfold.shape[-2] // x.shape[1])
        h_out = (x.shape[2] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                                                            self.stride + 1
        w_out = (x.shape[3] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                                                            self.stride + 1                               
        pre_output = torch.sum(pre_output, axis=-1).view(x.shape[0], x.shape[1],
                                                         h_out, w_out)
        return self.final1x1(pre_output)
