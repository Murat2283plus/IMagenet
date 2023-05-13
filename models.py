# models.py 
import torch
import torch.nn as nn
# 我们首先定义了一个 BasicBlock 类，
# 它是卷积神经网络的基本组件，
# 包含两个卷积层、批量归一化（如果启用）和一个可选的残差连接。
# 接下来，我们定义了 ConvNet 类，
# 其中包括一个初始卷积层、批量归一化层（如果启用）、ReLU 激活函数和最大池化层。
# 然后我们创建一个堆叠的 BasicBlock 序列以增加网络深度。
# 最后，我们添加自适应平均池化层、dropout 层（根据指定的 dropout 率）和一个全连接层以输出预测的类别概率。
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm, use_residual):
        super(BasicBlock, self).__init__()
        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_residual:
            identity = self.residual(x)

        out += identity
        out = self.relu(out)

        return out

class ConvNet(nn.Module):
    def __init__(self, num_classes=200, dropout_rate=0.5, use_batch_norm=True, use_residual=True, num_blocks=4):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        in_channels = 64
        out_channels = 64
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            block = BasicBlock(in_channels, out_channels, use_batch_norm, use_residual)
            self.blocks.add_module(f"block{i + 1}", block)
            in_channels = out_channels
            out_channels *= 2

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.blocks(x)

    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.dropout(x)
    #     x = self.fc(x)

    #     return x
    def forward(self, x): 
        x2 = self.conv1(x)
        x3 = self.bn1(x2)
        x4 = self.relu(x3)
        x5 = self.maxpool(x4)

        x6 = self.blocks(x5)

        x7 = self.avgpool(x6)
        x8 = torch.flatten(x7, 1)
        x9 = self.dropout(x8)
        # x9.shape # torch.Size([128, 512])
        x10 = self.fc(x9)

        return x10
        
    