from HelpFunctions import *
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity() #pass the input as is
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv= Conv2dAuto(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict({'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                                    stride=self.downsampling, bias=False),
                                                    'bn': nn.BatchNorm2d(self.expanded_channels)})) if self.should_apply_shortcut else None

        @property
        def expanded_channels(self):
            return self.out_channels * self.expansion

        @property
        def should_apply_shortcut(self):
            return self.in_channels != self.expanded_channels

class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),)

class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
            super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
            self.blocks = nn.Sequential(
                conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
                activation(),
                conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
                activation(),
                conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
            )

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
            out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)])

        def forward(self, x):
            x = self.blocks(x)
            return x