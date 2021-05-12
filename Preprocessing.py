import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io import loadmat
import os
import scipy.io as sio
import scipy.fftpack as function
from SignalPreprocessing.data_preprocess_function import *

SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SimulatedDatabase")
LOW_FREQUENCY_SIGNAL = "low_frequency_signals"
HIGH_FREQUENCY_SIGNAL = "low_frequency_signals"
FREQUENCY_CLEAN_SIGNAL = "low_frequency_signals"

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
            continue
        print(filename)
        current_signal = loadmat(os.path.join(SIMULATED_DATASET,filename))['data']

        #both_frequency_signal = function.ifft(yf3)

        yf, freq, t = transformation('fft', current_signal)

        plt.plot(freq,np.abs(yf))
        plt.xlim(0)
        plt.show()
        plt.close()

        # Remove sinus and high frequency
        yf1 = yf.copy()
        yf1 = frequency_removal(yf,freq,10000000,50)
        #yf1 = [0 if np.abs(elem) > 0.005 and np.abs(elem) < 0.2 else yf1 for elem, yf1 in zip(freq, yf1)]

        plt.plot(freq, np.abs(yf1))
        plt.xlim(0)
        plt.show()
        plt.close()

        low_frequency_signal = function.ifft(yf1)

        fig, (ax1,ax2) = plt.subplots(2,1)
        ax1.plot(current_signal)
        ax2.plot(low_frequency_signal)
        plt.show()
        plt.close()

        # Remove frequency removal and mean
        yf2 = frequency_removal(yf, freq, 100, 0)

        plt.plot(freq, yf2)
        plt.xlim(0)
        plt.show()
        plt.close()

        high_frequency_signal = function.ifft(yf2)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(current_signal)
        ax2.plot(high_frequency_signal)
        plt.show()
        plt.close()


