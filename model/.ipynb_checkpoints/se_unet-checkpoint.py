import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """通道注意力模块(Squeeze-and-Excitation)"""
    def __init__(self, channel, reduction=16):
        """
        参数说明：
        - channel: 输入特征图的通道数
        - reduction: 通道压缩比例(16)
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze操作
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # Excitation操作
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        x = self.double_conv(x)
        x = self.se(x)  # 添加SE模块
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_se=use_se)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class SE_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_se=True):
        super(SE_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_se = use_se

        # 编码器部分
        self.inc = DoubleConv(n_channels, 64, use_se=use_se)
        self.down1 = Down(64, 128, use_se=use_se)
        self.down2 = Down(128, 256, use_se=use_se)
        self.down3 = Down(256, 512, use_se=use_se)
        self.down4 = Down(512, 1024, use_se=use_se)

        # 解码器部分
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
