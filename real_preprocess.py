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
REAL_WINDOWS = "real_windows"


if not os.path.exists(REAL_WINDOWS):
    os.mkdir(REAL_WINDOWS)

dir_path = os.path.dirname(os.path.realpath(__file__))
window_path = os.path.join(dir_path,REAL_WINDOWS)

if __name__ == '__main__':
    for filename in os.listdir(REAL_DATASET):
        print(filename)
        current_signal = np.ravel(loadmat(os.path.join(REAL_DATASET, filename))['data'])

        #Preprocess
        signal = remove_beginning_end(current_signal)
        yf, freq, t = transformation('fft', signal)
        yf_new = [0 if (np.abs(elem) >= 0 and np.abs(elem) < 0.6) else yf for elem, yf in zip(freq, yf)]
        yf_new = [0 if np.abs(elem) > 30 else yf_new for elem, yf_new in zip(freq, yf_new)]
        new_signal = function.ifft(yf_new)
        #Resampling
        resampled_signal = increase_sampling_rate(new_signal,0.25)
        #Windowing
        number_of_window = int(len(resampled_signal) / 1024)
        window_size = 1024
        for i in range(number_of_window):
            record = np.array(resampled_signal[i * window_size:(i + 1) * window_size]).real
            sio.savemat(os.path.join(window_path, filename + '_mix' + str(i)), {'data': record})