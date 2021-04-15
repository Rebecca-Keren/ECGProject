import torch
import torch.nn as nn
import os
import numpy as np
import scipy.stats


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)

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
        ['tanh', nn.Tanh()],
        ['none', nn.Identity()]
    ])[activation]

def criterion_hinge_loss(m_feature,f_feature,delta):
    # m_normalized = torch.norm(m_feature)
    # f_normalized = torch.norm(f_feature)
    #print(f_feature.size())
    distance = nn.functional.mse_loss(m_feature,f_feature)
    #print(str(delta-distance))
    return nn.functional.relu(delta - distance)

def simulated_database_list(sim_dir):
    list = []
    signal_given = []
    for filename in os.listdir(sim_dir):
        if 'mix' not in filename:
            continue
        name = filename[:(filename.find("mix"))]
        if name not in signal_given:
            signal_given.append(name);
            for i in range(73):
                list.append([name + 'mix' + str(i), name + 'mecg' + str(i) , name + 'fecg1' + str(i)])
        else:
            continue
    return list
