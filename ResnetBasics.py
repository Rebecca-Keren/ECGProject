from HelpFunctions import *
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity() #layers that we want to apply to input
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity() #pass the input as is

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
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling = expansion, downsampling
        self.shortcut = nn.Sequential(nn.Conv1d(self.in_channels, self.expanded_channels, kernel_size=1,
                                                    stride=self.downsampling, bias=False),nn.BatchNorm1d(self.expanded_channels)) if self.should_apply_shortcut else None

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
        #prima volta e' da dim piccola a grande
        #la seconda volta e da dimensione piccola a expanded
        self.blocks = nn.Sequential(nn.Conv1d(self.in_channels, self.out_channels,kernel_size= 3,padding = 1, bias=False, stride=self.downsampling),
                        nn.BatchNorm1d(out_channels),
                        activation(),
                        nn.Conv1d(self.out_channels, self.expanded_channels, kernel_size=3, padding=1,bias=False,stride=self.downsampling),
                        nn.BatchNorm1d(out_channels),)

# class ResNetBottleNeckBlock(ResNetResidualBlock):
#     expansion = 4
#     def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
#             super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
#             self.blocks = nn.Sequential(
#                 conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
#                 activation(),
#                 conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
#                 activation(),
#                 conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
#             )

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