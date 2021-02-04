from ResnetNetwork import *
import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
