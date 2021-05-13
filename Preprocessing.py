import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io import loadmat
import os
import scipy.io as sio
import scipy.fftpack as function
from SignalPreprocessing.data_preprocess_function import *

SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "simulated_windows_noise")
LOW_FREQUENCY_SIGNAL = "low_frequency_signals"
HIGH_FREQUENCY_SIGNAL = "high_frequency_signals"
FREQUENCY_CLEAN_SIGNAL = "both_frequency_signals"

if not os.path.exists(LOW_FREQUENCY_SIGNAL):
    os.mkdir(LOW_FREQUENCY_SIGNAL)
if not os.path.exists(HIGH_FREQUENCY_SIGNAL):
    os.mkdir(HIGH_FREQUENCY_SIGNAL)
if not os.path.exists(FREQUENCY_CLEAN_SIGNAL):
    os.mkdir(FREQUENCY_CLEAN_SIGNAL)

dir_path = os.path.dirname(os.path.realpath(__file__))
low_path = os.path.join(dir_path,LOW_FREQUENCY_SIGNAL)
high_path = os.path.join(dir_path,HIGH_FREQUENCY_SIGNAL)
both_path = os.path.join(dir_path,FREQUENCY_CLEAN_SIGNAL)

if __name__ == '__main__':
    for filename in os.listdir(SIMULATED_DATASET):
        if "mix" not in filename:
            sio.savemat(os.path.join(low_path, filename), {'data': loadmat(os.path.join(SIMULATED_DATASET,filename))['data']})
            sio.savemat(os.path.join(high_path, filename), {'data': loadmat(os.path.join(SIMULATED_DATASET,filename))['data']})
            sio.savemat(os.path.join(both_path, filename), {'data': loadmat(os.path.join(SIMULATED_DATASET,filename))['data']})
            continue
        print(filename)
        current_signal = loadmat(os.path.join(SIMULATED_DATASET,filename))['data']

        yf, freq, t = transformation('fft', current_signal)

        yf1 = frequency_removal(yf,freq,10000000,20)
        low_frequency_signal = function.ifft(yf1)
        sio.savemat(os.path.join(low_path,filename), {'data': low_frequency_signal})

        yf2 = frequency_removal(yf, freq, 220, 0)
        high_frequency_signal = function.ifft(yf2)
        sio.savemat(os.path.join(high_path,filename), {'data': high_frequency_signal})

        yf3 = frequency_removal(yf, freq, 220, 20)
        both_frequency_signal = function.ifft(yf3)
        sio.savemat(os.path.join(both_path,filename), {'data': both_frequency_signal})


