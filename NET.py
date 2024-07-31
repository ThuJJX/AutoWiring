from torch import nn


class SimpleNet(nn.Module):
    """
    定义了一个简单的三层全连接神经网络，每一层都是线性的
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer4 = nn.Linear(n_hidden_3, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ActivationNet(nn.Module):
    """
    在上面的simpleNet的基础上，在每层的输出部分添加了激活函数
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(ActivationNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.PReLU(1, init=0.25))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.PReLU(1, init=0.25))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.PReLU(1, init=0.25))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.PReLU(1, init=0.25))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5), nn.PReLU(1, init=0.25))
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim))
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


class BatchNet(nn.Module):
    """
    在上面的Activation_Net的基础上，增加了一个加快收敛速度的方法——批标准化
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(BatchNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.Softplus(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.Softplus(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3), nn.Softplus(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x