import torch.nn as nn

class DisentangledModel(nn.Module):
    def __init__(self, latentSize=256):
        super(DisentangledModel, self).__init__()
        self.latentSize = latentSize
        self.latentHalf = self.latentSize/2

    def forward(self, x):
        self.MfirstHalf = x[: self.latentHalf-1].type(x.type())
        self.FsecondHalf = x[self.latentHalf:].type(x.type())

        return self.MfirstHalf, self.FsecondHalf
