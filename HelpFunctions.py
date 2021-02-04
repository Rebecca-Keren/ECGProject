import torch
import torch.nn as nn
import os

from collections import OrderedDict

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
        self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),'bn': nn.BatchNorm2d(out_channels)}))

def simulated_database_list(sim_dir):
    list = []
    for filename in os.listdir(sim_dir):
        if 'mix' not in filename:
            continue
        name = filename[:len(filename) - 4]
        if 'mix0' in filename:
            list.append([filename, name + 'mecg0',name + 'fecg10'])
        elif 'mix1' in filename:
            list.append([filename, name + 'mecg1',name + 'fecg11'])
        else:
            list.append([filename, name + 'mecg2',name + 'fecg12'])
    return list
