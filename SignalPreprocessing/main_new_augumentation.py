import os
import data_preprocess_function as dt
import data_agumentation_function as da
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as function
import scipy.signal as signals
from scipy.io import loadmat

DATA_FOLDER = "AugumentedWindows"
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path,DATA_FOLDER)

def main():

    for filename in os.listdir(data_path):
        if filename == '.DS_Store':
            continue
        path_window = os.path.join(data_path,filename)
        for window in os.listdir(path_window):
            signal = loadmat(window)['data']
            name = window[::(len(window) - len('.mat'))]
            path_augumentation = os.path.join(path_window,window)
            new_signal = da.add_more_augumentation_with_p(signal,0.1)
            dt.save_mat(new_signal, os.path.join(path_augumentation,'new.mat'))

if __name__ == '__main__':
    main()
