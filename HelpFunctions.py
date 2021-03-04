import torch
import torch.nn as nn
import os
import numpy as np

def increase_sampling_rate(signal,rate):
    signal_size = len(signal)
    x = [j for j in range(signal_size)]
    y = [signal[i] for i in range(signal_size)]
    xvals = np.linspace(0, signal_size, int(signal_size*rate))
    interpolated_signal = np.interp(xvals, x, y)
    if (rate >= 1):
        interpolated_signal = interpolated_signal[:signal_size]
    return interpolated_signal

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

def criterion_hinge_loss(m_feature,f_feature,delta):
    m_normalized = torch.norm(m_feature)
    f_normalized = torch.norm(f_feature)

    distance = nn.MSELoss()
    return np.max(delta - distance(m_normalized,f_normalized).item(), 0)

def simulated_database_list(sim_dir):
    list = []
    for filename in os.listdir(sim_dir):
        if 'mix' not in filename:
            continue
        name = filename[:(filename.find("mix"))]
        if 'mix0' in filename:
            list.append([filename, name + 'mecg0', name + 'fecg10'])
        elif 'mix1' in filename:
            list.append([filename, name + 'mecg1', name + 'fecg11'])
        else:
            list.append([filename, name + 'mecg2', name + 'fecg12'])
    return list
