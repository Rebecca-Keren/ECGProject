import torch
import torch.nn as nn
import os
import numpy as np
import scipy.stats
from scipy.io import loadmat

SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "simulated_windows")

def check_correlation(orig_signal,changed_signal):
    orig = orig_signal
    signal_to_compare = changed_signal
    correlation, p_value = scipy.stats.pearsonr(orig, signal_to_compare)
    return correlation.real #the correlation of the original signal in comperison to the changed signal

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
    for filename in os.listdir(sim_dir):
        if 'mix' not in filename:
            continue
        name = filename[:(filename.find("mix"))]
        number = filename[(filename.find("mix")) + 3:]
        list.append([name + 'mix' + str(number), name + 'mecg' + str(number), name + 'fecg1' + str(number)])
    return list

def remove_nan_signals(list_signals):
    for idx,signal_tuple in enumerate(list_signals):
        is_nan = False
        for signal_name in signal_tuple:
            path = os.path.join(SIMULATED_DATASET,signal_name)
            signal = loadmat(path)['data']
            is_nan = np.any(np.isnan(signal))
            if(is_nan):
                print(list_signals[idx])
                list_signals = np.delete(list_signals,idx)
                break

