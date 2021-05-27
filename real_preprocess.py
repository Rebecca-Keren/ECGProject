import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from scipy import signal
import scipy.io as sio
import scipy.fftpack as function
from SignalPreprocessing.data_agumentation_function import *
from SignalPreprocessing.data_preprocess_function import *

REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BestSignals")
REAL_WINDOWS = "best_windows"
SIM_WINDOWS = "SimulatedDatabase"

if not os.path.exists(REAL_WINDOWS):
    os.mkdir(REAL_WINDOWS)

dir_path = os.path.dirname(os.path.realpath(__file__))
window_path = os.path.join(dir_path,REAL_WINDOWS)
sim_path = os.path.join(dir_path,SIM_WINDOWS)

if __name__ == '__main__':
    for filename in os.listdir(REAL_DATASET):
        print(filename)
        current_signal = np.ravel(loadmat(os.path.join(REAL_DATASET, filename))['data'])
        current_signal = current_signal - np.mean(current_signal)
        current_signal = [elem * 20 for elem in current_signal]
        # Resampling
        resampled_signal = increase_sampling_rate(current_signal, 0.25)
        # Windowing
        number_of_window = int(len(resampled_signal) / 1024)
        window_size = 1024
        for i in range(number_of_window):
            record = resampled_signal[i * window_size:(i + 1) * window_size]
            sio.savemat(os.path.join(window_path, filename[:(len(filename) - 4)] + '_mix' + str(i)), {'data': record})

    """for filename in os.listdir(REAL_DATASET):
        print(filename)
        current_signal = np.ravel(loadmat(os.path.join(REAL_DATASET, filename))['data'])
        #Preprocess
        signal = remove_beginning_end(current_signal)
        yf, freq, t = transformation('fft', signal)
        yf = [0 if (np.abs(elem) > 0.005 and np.abs(elem) < 0.2) else yf for elem, yf in zip(freq, yf)]
        yf = [0 if np.abs(elem) > 30 else yf for elem, yf in zip(freq, yf)]
        new_signal = function.ifft(yf)
        #Resampling
        resampled_signal = increase_sampling_rate(signal,0.25)
        #Windowing
        number_of_window = int(len(resampled_signal) / 1024)
        window_size = 1024
        for i in range(number_of_window):
            record = resampled_signal[i * window_size:(i + 1) * window_size]
            sio.savemat(os.path.join(window_path, filename[:(len(filename)-4)] + '_mix' + str(i)), {'data': record})"""




