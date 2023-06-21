import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):  # 初始化函数
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 定义第一个卷积层，
        self.pool1 = nn.MaxPool2d(2, 2)  # 池化核大小2*2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 120个节点
        self.fc2 = nn.Linear(120, 84)  # 84个节点
        self.fc3 = nn.Linear(84, 10)  # 10个节点

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))  # input(6,12,12),output1(16,8,8) output2(16,4,4)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # 全连接层
        return x


if __name__ == '__main__':
    net = LeNet5()
    print(net)
