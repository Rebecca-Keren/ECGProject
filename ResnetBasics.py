from HelpFunctions import *
import torch
import torch.nn as nn

class ResNetBasicBlockEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', expansion=1, downsampling=1, *args, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels , out_channels
        self.activation, self.expansion, self.downsampling = activation, expansion, downsampling
        self.blocks = nn.Sequential(nn.Conv1d(self.in_channels, self.out_channels,kernel_size= 3, padding = 1, bias=False, stride=self.downsampling),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(self.out_channels, self.expanded_channels, kernel_size=3, padding=1,bias=False),
                        nn.BatchNorm1d(out_channels),)
        #self.activate = activation_func(activation)
        self.shortcut = nn.Sequential(nn.Conv1d(self.in_channels, self.expanded_channels, kernel_size=3,
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

class ResNetLayerEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlockEncoder, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, downsampling=downsampling, *args, **kwargs),
                *[block(out_channels,
            out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)])

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