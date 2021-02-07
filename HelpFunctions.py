import torch
import torch.nn as nn
import os

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

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
