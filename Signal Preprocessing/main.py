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
            files.append(filename)
        while (True):
            print("Choose the number of the signal to view:")
            index = 1
            for f in files:
                name = f[:(len(f) - len('.mat'))]
                print(str(index) + "." + name)
                index += 1
            number = input()

            filename = files[int(number) - 1]
            name = filename[:(len(filename) - len('.mat'))]
            filename = os.path.join(fetus_path, filename)
            signal = loadmat(filename)['data'][0][15000:20000]
            print(len(signal))
            plt.plot(signal)
            plt.show()
            plt.close()
            yf3,freq3,t3 = dt.transformation('fft',signal)


            # #switch with 0.1 p and loop 5
            # signal = function.ifft(yf3)
            # signal_to_be_switched = function.ifft(yf3)
            # for i in range(5):
            #     signal_to_be_switched = da.switch_samples_dist(signal_to_be_switched, 0.1)
            # correlation = da.check_correlation(function.ifft(yf3), signal_to_be_switched)
            # print("switch every two neighbour samples: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('1')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(signal_to_be_switched)
            # plt.show()
            # plt.close()
            #
            # # switch with 0.1 p and loop 7
            # signal = function.ifft(yf3)
            # signal_to_be_switched = function.ifft(yf3)
            # for i in range(7):
            #     signal_to_be_switched = da.switch_samples_dist(signal_to_be_switched, 0.5)
            # correlation = da.check_correlation(function.ifft(yf3), signal_to_be_switched)
            # print("switch every two neighbour samples: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('2')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(signal_to_be_switched)
            # plt.show()
            # plt.close()
            #
            # # switch with 0.1 p and loop 10
            # signal = function.ifft(yf3)
            # signal_to_be_switched = function.ifft(yf3)
            # for i in range(10):
            #     signal_to_be_switched = da.switch_samples_dist(signal_to_be_switched,0.5)
            # correlation = da.check_correlation(function.ifft(yf3), signal_to_be_switched)
            # print("switch every two neighbour samples: ",correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('3')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(signal_to_be_switched)
            # plt.show()
            # plt.close()
            #
            # # switch with 0.3 p and loop 5
            # signal = function.ifft(yf3)
            # signal_to_be_switched = function.ifft(yf3)
            # for i in range(5):
            #     signal_to_be_switched = da.switch_samples_dist(signal_to_be_switched, 0.3)
            # correlation = da.check_correlation(function.ifft(yf3), signal_to_be_switched)
            # print("switch every two neighbour samples: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('4')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(signal_to_be_switched)
            # plt.show()
            # plt.close()
            #
            # # switch with 0.3 p and loop 7
            # signal = function.ifft(yf3)
            # signal_to_be_switched = function.ifft(yf3)
            # for i in range(7):
            #     signal_to_be_switched = da.switch_samples_dist(signal_to_be_switched, 0.3)
            # correlation = da.check_correlation(function.ifft(yf3), signal_to_be_switched)
            # print("switch every two neighbour samples: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('5')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(signal_to_be_switched)
            # plt.show()
            # plt.close()
            #
            # # switch with 0.3 p and loop 10
            # signal = function.ifft(yf3)
            # signal_to_be_switched = function.ifft(yf3)
            # for i in range(10):
            #     signal_to_be_switched = da.switch_samples_dist(signal_to_be_switched, 0.3)
            # correlation = da.check_correlation(function.ifft(yf3), signal_to_be_switched)
            # print("switch every two neighbour samples: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('6')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(signal_to_be_switched)
            # plt.show()
            # plt.close()

            # # noise in time
            # noised_signal = da.addnoise(function.ifft(yf3), 0.0003)
            # # dt.save_mat(noised_signal, os.path.join(path_augumentation, 'noise5.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), noised_signal)
            # print("noise 0.003: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('7')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(noised_signal)
            # plt.show()
            # plt.close()

            # # noise in time
            # noised_signal = da.addnoise(function.ifft(yf3), 0.002)
            # # dt.save_mat(noised_signal, os.path.join(path_augumentation, 'noise5.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), noised_signal)
            # print("noise 0.002: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('8')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(noised_signal)
            # plt.show()
            # plt.close()

            # # noise in time
            # noised_signal = da.addnoise(signal, 0.001)
            # # dt.save_mat(noised_signal, os.path.join(path_augumentation, 'noise3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), noised_signal)
            # print("noise 0.001: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('9')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(noised_signal)
            # plt.show()
            # plt.close()
            #
            # # add frequencies 0.1 probability and 0.1 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal),0.9,0.1)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 1: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('10')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()

            # # add frequencies 0.1 probability and 0.2 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.9,0.2)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 2: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('11')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()

            # # add frequencies 0.1 probability and 0.3 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.9,0.3)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 3: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('12')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # add frequencies 0.2 probability and 0.1 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.8, 0.1)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 1: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('13')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # add frequencies 0.2 probability and 0.2 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.8, 0.2)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 2: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('14')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # add frequencies 0.2 probability and 0.3 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.8, 0.3)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 3: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('15')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # add frequencies 0.3 probability and 0.1 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.7, 0.1)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 1: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('16')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # add frequencies 0.3 probability and 0.2 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.7, 0.2)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 2: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('17')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # add frequencies 0.3 probability and 0.3 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.7, 0.3)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 3: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('18')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()

            # # subtract frequencies 0.1 probability and 0.1 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.9, -0.1)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 1: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('19')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()

            # # subtract frequencies 0.1 probability and 0.2 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.9, -0.2)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 2: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('20')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # subtract frequencies 0.1 probability and 0.3 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.9, -0.3)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 3: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('21')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # subtract frequencies 0.2 probability and 0.1 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.8, -0.1)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 1: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('22')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # subtract frequencies 0.2 probability and 0.2 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.8, -0.2)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 2: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('23')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # subtract frequencies 0.2 probability and 0.3 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.8, -0.3)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 3: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('24')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # subtract frequencies 0.3 probability and 0.1 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.7, -0.1)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 1: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('25')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # subtract frequencies 0.3 probability and 0.2 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.7, -0.2)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 2: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('26')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # subtract frequencies 0.3 probability and 0.3 percentage
            # new_frequencies = da.add_frequencies(function.fft(signal), 0.7, -0.3)
            # # dt.save_mat(new_frequencies, os.path.join(path_augumentation, 'addfreq3.mat'))
            # correlation = da.check_correlation(function.ifft(yf3), new_frequencies)
            # print("add frequencies 3: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('27')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(new_frequencies[100:])
            # plt.show()
            # plt.close()
            #
            # # randomly drop samples from the signal with p = 0.1
            # dropped_signal = da.drop_from_signal(function.ifft(yf3),0.99)
            # correlation = da.check_correlation(function.ifft(yf3), dropped_signal)
            # print("drop samples p=0.02: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('28')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(dropped_signal)
            # plt.show()
            # plt.close()
            #
            # # randomly drop samples from the signal with p = 0.05
            # dropped_signal = da.drop_from_signal(function.ifft(yf3), 0.995)
            # correlation = da.check_correlation(function.ifft(yf3), dropped_signal)
            # print("drop samples p=0.02: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('29')
            # ax1.plot(function.ifft(yf3))
            # ax2.plot(dropped_signal)
            # plt.show()
            # plt.close()

            #
            # # shuffle in frequency with p 0.1
            # yf_shuffle, freq, t = dt.transformation('fft', signal)
            # yf_shuffle = da.switch_samples_dist(yf_shuffle,0.99)
            # correlation = da.check_correlation(signal, function.ifft(yf_shuffle))
            # print("shuffle in frequency with low probability: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('30')
            # ax1.plot(signal)
            # ax2.plot(function.ifft(yf_shuffle))
            # plt.show()
            # plt.close()

            # # shuffle in frequency with p 0.05
            # yf_shuffle, freq, t = dt.transformation('fft', signal)
            # yf_shuffle = da.switch_samples_dist(yf_shuffle, 0.995)
            # correlation = da.check_correlation(signal, function.ifft(yf_shuffle))
            # print("shuffle in frequency with low probability: ", correlation)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('31')
            # ax1.plot(signal)
            # ax2.plot(function.ifft(yf_shuffle))
            # plt.show()
            # plt.close()

            # increasing sampling rate 5%
            increased_signal = da.increase_sampling_rate(signal, 1.05)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('32')
            # ax1.plot(signal)
            # ax2.plot(increased_signal)
            # plt.show()
            # plt.close()
            #
            # # increasing sampling rate 6%
            # increased_signal = da.increase_sampling_rate(signal, 1.06)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('33')
            # ax1.plot(signal)
            # ax2.plot(increased_signal)
            # plt.show()
            # plt.close()
            #
            # # increasing sampling rate 7%
            # increased_signal = da.increase_sampling_rate(signal, 1.07)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('34')
            # ax1.plot(signal)
            # ax2.plot(increased_signal)
            # plt.show()
            # plt.close()
            #
            # # increasing sampling rate 8%
            # increased_signal = da.increase_sampling_rate(signal, 1.08)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('35')
            # ax1.plot(signal)
            # ax2.plot(increased_signal)
            # plt.show()
            # plt.close()
            #
            # # increasing sampling rate 9%
            # increased_signal = da.increase_sampling_rate(signal, 1.09)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('36')
            # print("increase sampling rate 2")
            # ax1.plot(signal)
            # ax2.plot(increased_signal)
            # plt.show()
            # plt.close()
            #
            # # increasing sampling rate 10%
            # increased_signal = da.increase_sampling_rate(signal, 1.1)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('37')
            # ax1.plot(signal)
            # ax2.plot(increased_signal)
            # plt.show()
            # plt.close()
            #
            # # decrease sampling rate 5%
            increased_signal = da.increase_sampling_rate(signal, 0.95)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('38')
            # ax1.plot(signal)
            # ax2.plot(increased_signal)
            # plt.show()
            # plt.close()
            #
            # # decrease sampling rate 6%
            # increased_signal = da.increase_sampling_rate(signal, 0.94)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('39')
            # ax1.plot(signal)
            # ax2.plot(increased_signal)
            # plt.show()
            # plt.close()
            #
            # # decrease sampling rate 7%
            # increased_signal = da.increase_sampling_rate(signal, 0.93)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('40')
            # ax1.plot(signal)
            # ax2.plot(increased_signal)
            # plt.show()
            # plt.close()
            #
            # # decrease sampling rate 8%
            # increased_signal = da.increase_sampling_rate(signal, 0.98)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('41')
            # ax1.plot(signal)
            # ax2.plot(increased_signal)
            # plt.show()
            # plt.close()
            #
            # # decrease sampling rate 9%
            # increased_signal = da.increase_sampling_rate(signal, 0.91)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('42')
            # ax1.plot(signal)
            # ax2.plot(increased_signal)
            # plt.show()
            # plt.close()
            #
            # # decrese sampling rate 10%
            # increased_signal = da.increase_sampling_rate(signal, 0.9)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('43')
            # ax1.plot(signal)
            # ax2.plot(increased_signal)
            # plt.show()
            # plt.close()
            #
            # # reverse signal
            # reverse_signal = da.reverse_signal(signal)
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # fig.suptitle('44')
            # ax1.plot(signal)
            # ax2.plot(reverse_signal)
            # plt.show()
            # plt.close()


if __name__ == '__main__':
    main()

