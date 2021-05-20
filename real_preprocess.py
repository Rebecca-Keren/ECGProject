import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from scipy import signal
import scipy.io as sio
import scipy.fftpack as function
from SignalPreprocessing.data_preprocess_function import *

REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "RealSignals")
#LOW_FREQUENCY_SIGNAL = "low_frequency_signals"


#if not os.path.exists(LOW_FREQUENCY_SIGNAL):
#    os.mkdir(LOW_FREQUENCY_SIGNAL)

#dir_path = os.path.dirname(os.path.realpath(__file__))
#low_path = os.path.join(dir_path,LOW_FREQUENCY_SIGNAL)
if __name__ == '__main__':
    for filename in os.listdir(REAL_DATASET):
        print(filename)
        current_signal = loadmat(os.path.join(REAL_DATASET, filename))['data'][3000]

        x250 = np.arange(0, len(current_signal), 1/250)  # frequency 250
        x1000 = np.arange(0, len(current_signal), 1/1000)  # frequency 256

        resampled_signal = np.interp(x1000, x250, current_signal)
        plt.plot(current_signal, label="before")
        plt.plot(resampled_signal, label="AfterPreprocess")
        plt.show()
        plt.close()
        """fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(current_signal, label="BeforePreprocess")
        ax2.plot(resampled_signal, label="AfterPreprocess")
        plt.show()
        plt.close()"""

        """plt.plot(current_signal)
        plt.show()
        plt.close()
        print(len(current_signal))"""
