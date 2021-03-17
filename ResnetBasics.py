from HelpFunctions import *
import torch
import torch.nn as nn

import torch
from torch import nn

from functools import partial


class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2,)  # dynamic add padding based on the kernel_size
        # print(self.kernel_size, self.padding)


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, downsampling=1, kernel_size=3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.downsampling, self.conv = downsampling, partial(Conv1dAuto, kernel_size=kernel_size, bias=False)
        self.shortcut = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm1d(self.out_channels))


def conv_bn(in_channels, out_channels, conv, kernel_size, *args, **kwargs):
    return nn.Sequential(
        conv(in_channels, out_channels, kernel_size=kernel_size, *args, **kwargs),
        nn.BatchNorm1d(out_channels)
    )


class ResNet753Block(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3conv/batchnorm/activation
    """

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.in_channels, conv=self.conv, kernel_size=7),
            activation_func(self.activation),
            conv_bn(self.in_channels, self.in_channels, conv=self.conv, kernel_size=5, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.in_channels, out_channels, conv=self.conv, kernel_size=3),
        )


class ResNet333Block(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3conv/batchnorm/activation
    """

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.in_channels, conv=self.conv, kernel_size=3),
            activation_func(self.activation),
            conv_bn(self.in_channels, self.in_channels, conv=self.conv, kernel_size=3, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.in_channels, out_channels, conv=self.conv, kernel_size=3),
        )


class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """

    def __init__(self, in_channels, out_channels, block, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        if n == 1:
            self.blocks = nn.Sequential(
                block(in_channels, out_channels, downsampling=1, *args, **kwargs)
            )
        else:
            self.blocks = nn.Sequential(
                block(in_channels, in_channels, *args, **kwargs, downsampling=downsampling),
                *[block(in_channels, in_channels, downsampling=1, *args, **kwargs) for _ in range(1, n - 1)],
                block(in_channels, out_channels, downsampling=1, *args, **kwargs)
            )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetBasicBlockDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', expansion=1, downsampling=1, output_padding = 0, *args,
                 **kwargs):
        super().__init__()
        self.in_channels, self.out_channels, self.activation, self.expansion, self.downsampling,self.output_padding= in_channels, out_channels, activation, expansion, downsampling, output_padding
        self.blocks = nn.Sequential(
            nn.ConvTranspose1d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False, stride=self.downsampling,output_padding = self.output_padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(self.out_channels, self.expanded_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels), )
        #self.activate = activation_func(activation)
        self.shortcut = nn.Sequential(nn.ConvTranspose1d(self.in_channels, self.expanded_channels, kernel_size=3,
                                           stride=self.downsampling,output_padding = self.output_padding, padding=1, bias=False),
                                      nn.BatchNorm1d(self.expanded_channels)) if self.should_apply_shortcut else None

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


class ResNetLayerDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlockDecoder, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        ouput_padding = 1 if (downsampling == 2) else 0

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling,output_padding= ouput_padding),
            *[block(out_channels,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)])

    def forward(self, x):
        x = self.blocks(x)
        return x