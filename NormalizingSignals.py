import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io import loadmat
import os
import scipy.io as sio
from SignalPreprocessing.data_agumentation_function import *

REAL_SIGNALS = "SignalsReal"
NEW_SIGNALS = "NormalizedReal"

if not os.path.exists(NEW_SIGNALS):
    os.mkdir(NEW_SIGNALS)

dir_path = os.path.dirname(os.path.realpath(__file__))
real_dir = os.path.join(dir_path,REAL_SIGNALS)
new_dir = os.path.join(dir_path,NEW_SIGNALS)

if __name__ == '__main__':
    # Merging Data,dividing into windows and saving
    for filename in os.listdir(real_dir):
        signal = loadmat(os.path.join(real_dir, filename))['data'][0]
        abs_signal = [abs(signal[i]) for i in range(len(signal))]
        max_signal = max(abs_signal)
        signal_new = [sample / max_signal for sample in signal]
        resampled_signal = increase_sampling_rate(signal_new,0.25)
        number_of_window = int(len(resampled_signal) / 1024)
        window_size = 1024
        for k in range(number_of_window):
            record = np.array(resampled_signal[k * window_size:(k + 1) * window_size])
            sio.savemat(os.path.join(new_dir, filename + str(k)), {'data': record})
