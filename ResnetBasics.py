from HelpFunctions import *
import torch
import torch.nn as nn

class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels,conv, activation='relu',expansion=1, downsampling=1, *args, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels, self.activation,self.expansion, self.downsampling = in_channels , out_channels ,activation,expansion, downsampling
        self.blocks = nn.Sequential(conv(self.in_channels, self.out_channels,kernel_size= 3, padding = 1, bias=False, stride=self.downsampling),
                        nn.BatchNorm1d(out_channels),
                        activation(),
                        conv(self.out_channels, self.expanded_channels, kernel_size=3, padding=1,bias=False),
                        nn.BatchNorm1d(out_channels),)
        self.activate = activation_func(activation)
        self.shortcut = nn.Sequential(conv(self.in_channels, self.expanded_channels, kernel_size=3,
                                                    stride=self.downsampling,padding = 1,bias=False),nn.BatchNorm1d(self.expanded_channels)) if self.should_apply_shortcut else None

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, conv, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels,conv, *args, **kwargs, downsampling=downsampling),
                *[block(out_channels * block.expansion,
            out_channels,conv, downsampling=1, *args, **kwargs) for _ in range(n - 1)])

        def forward(self, x):
            x = self.blocks(x)
            return x