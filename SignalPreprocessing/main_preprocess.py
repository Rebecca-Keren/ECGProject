import os
import data_preprocess_function as dt
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as function
import scipy.signal as signals
from scipy.io import loadmat


files = []
DATA_FOLDER = "Shai_Signals"
WINDOW_FOLDER = 'Window_Folder_FRM'
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path,DATA_FOLDER)
window_path = os.path.join(dir_path,WINDOW_FOLDER)

def main():
        for filename in os.listdir(data_path):
            filename = files[int(number) - 1]
            filename = os.path.join(data_path, filename)
            signal = loadmat(filename)['data']
            signal = dt.remove_beginning_end(signal)
            signal1 = signal.copy()
            signal = signal - np.mean(signal)
            yf,freq,t = dt.transformation('fft',signal)
            yf1 = yf.copy()

            #Remove sinus and high frequency
            yf1 = [0 if np.abs(elem) > 0.005 and np.abs(elem) < 0.2 else yf1 for elem,yf1 in zip(freq,yf1)]

            #Remove frequency removal and mean
            yf2 = dt.frequency_removal(yf,freq,30,0)

            #Preprocess on window
            window3 = dt.get_first_window(name,signal1)
            yf3,freq3,t = dt.transformation('fft',window3)
            yf3 = dt.frequency_removal(yf3,freq3,30,0.15)

if __name__ == '__main__':
    main()

