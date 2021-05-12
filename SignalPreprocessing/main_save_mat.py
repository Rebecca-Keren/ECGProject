import os
import data_preprocess_function as dt
import data_agumentation_function as da
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as function
import scipy.signal as signals
from scipy.io import loadmat

files = []
DATA_FOLDER = "Shai_Signals"
WINDOW_FOLDER = 'Window_Folder_FRM'
FETUS_WINDOWS = "Fetus-Preprocessed Signals"
NO_FETUS_WINDOWS = "NoFetusWindows"
FETUS_PROCESSED_SIGNALS = "FetusPreprocessedSignals"
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path,DATA_FOLDER)
window_path = os.path.join(dir_path,WINDOW_FOLDER)
fetus_path = os.path.join(dir_path,FETUS_WINDOWS)
no_fetus_path = os.path.join(dir_path,NO_FETUS_WINDOWS)

fetus_processed_signal_path = os.path.join(dir_path,FETUS_PROCESSED_SIGNALS)

def main():

    for filename in os.listdir(fetus_path):
        if filename == '.DS_Store':
            continue
        name = filename[:(len(filename) - len('.mat'))]
        print(name)
        path_augumentation = os.path.join(dir_path,name)
        os.mkdir(path_augumentation)
        filename = os.path.join(fetus_path, filename)
        signal = loadmat(filename)['data']

        yf3, freq3, t3 = dt.transformation('fft', signal)

        # switch with 0.1 p and loop 5
        signal = function.ifft(yf3)
        signal_to_be_switched = function.ifft(yf3)
        for i in range(5):
            signal_to_be_switched = da.switch_samples_dist(signal_to_be_switched, 0.1)
        # dt.save_mat(signal_to_be_switched, os.path.join(path_augumentation, 'switch01-5.mat'))

        # switch with 0.1 p and loop 7
        signal = function.ifft(yf3)
        signal_to_be_switched = function.ifft(yf3)
        for i in range(7):
            signal_to_be_switched = da.switch_samples_dist(signal_to_be_switched, 0.5)
        # dt.save_mat(signal_to_be_switched, os.path.join(path_augumentation, 'switch01-7.mat'))

        # switch with 0.1 p and loop 10
        signal = function.ifft(yf3)
        signal_to_be_switched = function.ifft(yf3)
        for i in range(10):
            signal_to_be_switched = da.switch_samples_dist(signal_to_be_switched, 0.5)
        # dt.save_mat(signal_to_be_switched, os.path.join(path_augumentation, 'switch01-10.mat'))

        # switch with 0.3 p and loop 5
        signal = function.ifft(yf3)
        signal_to_be_switched = function.ifft(yf3)
        for i in range(5):
            signal_to_be_switched = da.switch_samples_dist(signal_to_be_switched, 0.3)
        # dt.save_mat(signal_to_be_switched, os.path.join(path_augumentation, 'switch03-5.mat'))

        # switch with 0.3 p and loop 7
        signal = function.ifft(yf3)
        signal_to_be_switched = function.ifft(yf3)
        for i in range(7):
            signal_to_be_switched = da.switch_samples_dist(signal_to_be_switched, 0.3)
        # dt.save_mat(signal_to_be_switched, os.path.join(path_augumentation, 'switch03-7.mat'))

        # switch with 0.3 p and loop 10
        signal = function.ifft(yf3)
        signal_to_be_switched = function.ifft(yf3)
        for i in range(10):
            signal_to_be_switched = da.switch_samples_dist(signal_to_be_switched, 0.3)
        # dt.save_mat(signal_to_be_switched, os.path.join(path_augumentation, 'switch03-10.mat'))

        # noise in time
        noised_signal = da.addnoise(function.ifft(yf3), 0.003)
        # dt.save_mat(noised_signal, os.path.join(path_augumentation, 'noise3.mat'))

        # noise in time
        noised_signal = da.addnoise(function.ifft(yf3), 0.002)
        # dt.save_mat(noised_signal, os.path.join(path_augumentation, 'noise2.mat'))

        # noise in time
        noised_signal = da.addnoise(signal, 0.001)
        # dt.save_mat(noised_signal, os.path.join(path_augumentation, 'noise1.mat'))

        # add frequencies 0.1 probability and 0.1 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.9, 0.1)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0101.mat'))

        # add frequencies 0.1 probability and 0.2 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.9, 0.2)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0102.mat'))

        # add frequencies 0.1 probability and 0.3 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.9, 0.3)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0103.mat'))

        # add frequencies 0.2 probability and 0.1 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.8, 0.1)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0201.mat'))

        # add frequencies 0.2 probability and 0.2 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.8, 0.2)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0202.mat'))

        # add frequencies 0.2 probability and 0.3 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.8, 0.3)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0203.mat'))

        # add frequencies 0.3 probability and 0.1 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.7, 0.1)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0301.mat'))

        # add frequencies 0.3 probability and 0.2 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.7, 0.2)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0302.mat'))

        # add frequencies 0.3 probability and 0.3 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.7, 0.3)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0303.mat'))

        # subtract frequencies 0.1 probability and 0.1 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.9, -0.1)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0101.mat'))

        # subtract frequencies 0.1 probability and 0.2 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.9, -0.2)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0102.mat'))

        # subtract frequencies 0.1 probability and 0.3 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.9, -0.3)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0103.mat'))

        # subtract frequencies 0.2 probability and 0.1 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.8, -0.1)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0201.mat'))

        # subtract frequencies 0.2 probability and 0.2 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.8, -0.2)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0202.mat'))

        # subtract frequencies 0.2 probability and 0.3 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.8, -0.3)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0203.mat'))

        # subtract frequencies 0.3 probability and 0.1 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.7, -0.1)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0301.mat'))

        # subtract frequencies 0.3 probability and 0.2 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.7, -0.2)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0302.mat'))

        # subtract frequencies 0.3 probability and 0.3 percentage
        new_frequencies = da.add_frequencies(function.fft(signal), 0.7, -0.3)
        # dt.save_mat(function.ifft(new_frequencies), os.path.join(path_augumentation, 'addfreq0303.mat'))

        # randomly drop samples from the signal with p = 0.1
        dropped_signal = da.drop_from_signal(function.ifft(yf3), 0.99)
        # dt.save_mat(dropped_signal, os.path.join(path_augumentation, 'drop01.mat'))

        # randomly drop samples from the signal with p = 0.05
        dropped_signal = da.drop_from_signal(function.ifft(yf3), 0.995)
        # dt.save_mat(dropped_signal, os.path.join(path_augumentation, 'drop005.mat'))

        # shuffle in frequency with p 0.1
        yf_shuffle, freq, t = dt.transformation('fft', signal)
        yf_shuffle = da.switch_samples_dist(yf_shuffle, 0.99)
        # dt.save_mat(function.ifft(yf_shuffle), os.path.join(path_augumentation, 'shufflefreq01.mat'))

        # shuffle in frequency with p 0.05
        yf_shuffle, freq, t = dt.transformation('fft', signal)
        yf_shuffle = da.switch_samples_dist(yf_shuffle, 0.995)
        # dt.save_mat(function.ifft(yf_shuffle), os.path.join(path_augumentation, 'shufflefreq005.mat'))

        # increasing sampling rate 5%
        increased_signal = da.increase_sampling_rate(signal, 1.05)
        # dt.save_mat(increased_signal, os.path.join(path_augumentation, 'increase5.mat'))

        # increasing sampling rate 6%
        increased_signal = da.increase_sampling_rate(signal, 1.06)
        # dt.save_mat(increased_signal, os.path.join(path_augumentation, 'increase6.mat'))

        # increasing sampling rate 7%
        increased_signal = da.increase_sampling_rate(signal, 1.07)
        # dt.save_mat(increased_signal, os.path.join(path_augumentation, 'increase7.mat'))

        # increasing sampling rate 8%
        increased_signal = da.increase_sampling_rate(signal, 1.08)
        # dt.save_mat(increased_signal, os.path.join(path_augumentation, 'increase8.mat'))

        # increasing sampling rate 9%
        increased_signal = da.increase_sampling_rate(signal, 1.09)
        # dt.save_mat(increased_signal, os.path.join(path_augumentation, 'increase9.mat'))

        # increasing sampling rate 10%
        increased_signal = da.increase_sampling_rate(signal, 1.1)
        # dt.save_mat(increased_signal, os.path.join(path_augumentation, 'increase10.mat'))

        # decrease sampling rate 5%
        increased_signal = da.increase_sampling_rate(signal, 0.95)
        # dt.save_mat(increased_signal, os.path.join(path_augumentation, 'decrease5.mat'))

        # decrease sampling rate 6%
        increased_signal = da.increase_sampling_rate(signal, 0.94)
        # dt.save_mat(increased_signal, os.path.join(path_augumentation, 'decrease6.mat'))

        # decrease sampling rate 7%
        increased_signal = da.increase_sampling_rate(signal, 0.93)
        # dt.save_mat(increased_signal, os.path.join(path_augumentation, 'decrease7.mat'))

        # decrease sampling rate 8%
        increased_signal = da.increase_sampling_rate(signal, 0.98)
        # dt.save_mat(increased_signal, os.path.join(path_augumentation, 'decrease8.mat'))

        # decrease sampling rate 9%
        increased_signal = da.increase_sampling_rate(signal, 0.91)
        # dt.save_mat(increased_signal, os.path.join(path_augumentation, 'decrease9.mat'))

        # decrease sampling rate 10%
        increased_signal = da.increase_sampling_rate(signal, 0.9)
        # dt.save_mat(increased_signal, os.path.join(path_augumentation, 'decrease10.mat'))

        # reverse signal
        reverse_signal = da.reverse_signal(signal)
        # dt.save_mat(reverse_signal, os.path.join(path_augumentation, 'reverse.mat'))
if __name__ == '__main__':
    main()

