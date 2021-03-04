import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import scipy.fftpack as function
import scipy.signal as signal
import pandas as pd
import scipy.io as sio

#Function to get desired transformation in frequency plane
def transformation(name_of_transformation,array,size_wavelet = 2000,size_stft = 1000,size_psd = 1000):
    ecg_signal = np.array(array)
    freq = function.fftfreq(len(array),1/1000)
    t = 0
    x = np.array(array).flatten()
    fs = 1/1000
    # number of sample, 1/ number of sampling per second
    if name_of_transformation == "dct":
        yf = function.dct(x)
    if name_of_transformation == "fft":
        yf = function.fft(x)
    if name_of_transformation == "stft":
        freq, t, yf = signal.stft(x,size_stft,window= 'parzen')
    if name_of_transformation == "psd":
        freq, yf = signal.welch(x,size_psd, nperseg=10000000000)
    if name_of_transformation == "wavelet":
        widths = np.arange(1,4000)
        yf = signal.cwt(np.array(ecg_signal).flatten(),signal.ricker,widths)
    return yf,freq,t

#Function to mirror graph over xaxis
def mirror_on_xaxis(array):
    for index in range(array.size):
        array[index] *= -1
    return array

#Function to remove first and last 10000 samples of signal
def remove_beginning_end(array,size = 10000):
    return array[size:(len(array)-size)]


def windowing_dc_one_sample(array,w_size):
    #list_of_windows = []
    size = array.size
    for i in range(0,len(array)-w_size,2):
        if i == (0):
            mean = np.mean(array[i:w_size])
            array[i:int((w_size)/2)] = array[i:int((w_size)/2)] - mean
            #list_of_windows.append(array[i:int((w_size)/2)])
        mean = np.mean(array[i:(w_size)+i])
        array[int(((w_size)+i-1)/2)] = array[int(((w_size)+i-1)/2)] - mean
        #list_of_windows.append(array[i:(w_size)+i])
    return array

def frequency_removal(yf,freq,high_frequency,low_frequency):
    cut_f_signal = yf.copy()
    cut_f_signal[(np.abs(freq) > high_frequency)] = 0
    if (low_frequency > 0):
        cut_f_signal[(np.abs(freq) < low_frequency)] = 0
    return cut_f_signal

#Function to transform np.array to cvs file
def cvsTransform(ecg_signal):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    csv_file = os.path.join(dir_path, "csvTry.csv")
    with open(csv_file, "w+") as f:
        for index in range(ecg_signal.size):
            f.write("%s\n" % ecg_signal[index][0])
    df = pd.read_csv(csv_file)
    df.plot()
    return

#Function that subtracts invididual mean from each graph and returns dictionary with new updated array
def offset_DC(dictionary):
    temporary_dictionary = dict()
    for key in dictionary.keys():
        for elem in dictionary[key]:
            head, tail = os.path.split(elem)
            data = loadmat(elem)
            ecg_signal = mirror_on_xaxis(data['data'])
            removing_only_my_mean = np.copy(ecg_signal)
            tmp_mean = np.average(np.array(ecg_signal).flatten())
            removing_only_my_mean = removing_only_my_mean - tmp_mean
            temporary_dictionary[tail] = [removing_only_my_mean]
    return temporary_dictionary

#Function to zero values over specific size from FFT
def removingPeaksFFT(array ,size = 300):
    yf,freq,t = transformation('fft',array)
    indices = [True if elem < np.abs(size) else False for elem in yf]
    yf = indices * yf
    return yf, freq

#Function to zero values over specific size from DCT
def removingPeaksDCT(array ,size = 300):
    yf,freq,t = transformation('dct',array)
    indices = [True if elem < np.abs(size) else False for elem in yf]
    yf = indices * yf
    return yf, freq

#Function to divide signal into windows of specific size
def windowing(array,w_size,overlapping):
    list_of_windows = []
    size = array.size
    number_of_windows = size // w_size #there are approximately 5 cycles in 2000 samples on average
    list_of_windows.append(array[0:w_size])
    for i in range(1,number_of_windows):
        if i == (number_of_windows-1):
            tmp = array[(i*w_size)-overlapping:]
        else:
            tmp = array[(i*w_size)-overlapping:((i+1)*w_size)]
        list_of_windows.append(tmp)
    return list_of_windows

#Function to calculate and subtracts individual mean from each window, array is a list of windows
def DC_offset_with_windowing(array):
    overall_mean = []
    for i in range(len(array)):
            mean = np.mean(array[i])
            overall_mean.append(mean)
            array[i] = array[i] - mean
    return array, np.mean(overall_mean)

#Function to bring together the signal after processing it into windows
def get_signal_no_window(array,window_size,overlapping_rate):
    overlapping = int(window_size/ overlapping_rate)
    result = []
    for i,elem in enumerate(array):
        if (i == 0):
            result.extend(elem[:(len(elem)-overlapping)])
            result.extend(average_overlapping_elements(elem[(int(len(elem)-overlapping)):],array[i+1][:overlapping]))
        elif (i == len(array)-1):
            result.extend(elem[overlapping:])
        else:
            result.extend(elem[overlapping:(len(elem)-overlapping)])
            result.extend(average_overlapping_elements(elem[(len(elem)-overlapping):],array[i+1][:overlapping]))
    return result

#Function to calculate average of each element in overlapping between two windows
def average_overlapping_elements(array1,array2):
    result = []
    for elem1,elem2 in zip(array1,array2):
        result.append(np.average([elem1,elem2]))
    return result


def plot_stft(freq,t,yf):
    plt.pcolormesh(t, freq, np.abs(yf), vmin=0, shading='gourard')
    plt.ylim([0, 35])
    plt.show()
    plt.close()
    return

def plot_spectrogram(signal,name):
    # window for spectrogram: np.hamming, np.blackman, np.kaiser, np,hanning, np.bartlett (default is hanning)
    Pxx, freqs, bins, im = plt.specgram(np.array(signal).flatten(), sides='onesided', NFFT=2048, Fs=1000,
                                        noverlap=256, window=np.bartlett(2048))
    plt.title("spectrogram frequency limit of under 35 " + name)
    plt.ylim([0, 35])
    plt.show()
    plt.close()
    return

def plot_psd(freq,yf):
    plt.semilogy(freq, yf)
    plt.show()
    plt.close()
    return

def plot_wavelet(yf):
    plt.imshow(yf,cmap='CMRmap', aspect='auto',vmax=abs(yf).max(),vmin=-abs(yf).max())
    plt.show()
    plt.close()
    return

'''def plot_wavelet_git(signal,method):
    if method == 'ricker':
        wa = wavelets.WaveletAnalysis(signal, wavelet=wavelets.Ricker(), dt= 1 / 1000)
    if method == 'morlet':
        wa = wavelets.WaveletAnalysis(signal, wavelet=wavelets.Morlet(), dt= 1 / 1000)
    # wavelet power spectrum
    power = wa.wavelet_power
    # scales
    scales = wa.scales
    # associated time vector
    t = wa.time
    T, S = np.meshgrid(t, scales)
    plt.contourf(T, S, power, 100)
    #zax.set_yscale('log')
    plt.show()
    plt.close()'''

def save_numpy_array(directory_path,filename,array):
    path = os.path.join(directory_path,filename)
    np.save(path,array)
    return

def cutting_good_window(name,signal,window_path):
    if name ==  'pt01_posLat':
        first = signal[290000:320000]
        second = signal [560000:610000]
        sio.savemat(os.path.join(window_path,name + ' 290-310'),{'data':first})
        sio.savemat(os.path.join(window_path,name + ' 560-610'),{'data' :second})
        return
    if name == 'pt01_posM':
        first = signal[50000:110000]
        second = signal[170000:240000]
        third = signal[290000:350000]
        sio.savemat(os.path.join(window_path,name + ' 50-110'), {'data': first})
        sio.savemat(os.path.join(window_path,name + ' 170-240'),{'data' :second})
        sio.savemat(os.path.join(window_path,name + ' 290-350'), {'data': third})
        return
    if name == 'pt02_poslat':
        first = signal[65000:90000]
        second = signal[120000:150000]
        third = signal[2000:32000]
        sio.savemat(os.path.join(window_path,name + ' 65-90'), {'data': first})
        sio.savemat(os.path.join(window_path,name + ' 120-150'),{'data' :second})
        sio.savemat(os.path.join(window_path,name + ' 2-32'), {'data': third})
        return
    if name == 'pt02_poslat2':
        first = signal[32000:61000]
        second = signal[62000:93000]
        third = signal[116000:146000]
        sio.savemat(os.path.join(window_path,name + ' 32-61'), {'data': first})
        sio.savemat(os.path.join(window_path,name + ' 62-93'),{'data' :second})
        sio.savemat(os.path.join(window_path,name + ' 116-146'), {'data': third})
        return
    if name == 'pt05_poslat':
        first = signal[0:20000]
        second = signal[50000:108000]
        sio.savemat(os.path.join(window_path,name + ' 0-20'), {'data': first})
        sio.savemat(os.path.join(window_path,name + ' 50-108'),{'data' :second})
        return
    if name == 'pt06_poslat_mat':
        first = signal[0:32500]
        second = signal[45000:80000]
        sio.savemat(os.path.join(window_path,name + ' 0-32,5'), {'data': first})
        sio.savemat(os.path.join(window_path,name + ' 45-80'),{'data' :second})
        return
    if name == 'pt06_poslat2':
        first = signal[95000:125000]
        sio.savemat(os.path.join(window_path,name + ' 95-125'), {'data': first})
        return
    if name == 'pt13_posLAT2-orig':
        first = signal[20000:50000]
        second = signal[130000:165000]
        sio.savemat(os.path.join(window_path,name + ' 20-50'), {'data': first})
        sio.savemat(os.path.join(window_path,name + ' 130-165'),{'data' :second})
        return
    if name == 'pt13_posLAT-orig':
        first = signal[0:25000]
        second = signal[55000:80000]
        sio.savemat(os.path.join(window_path,name + ' 0-25'), {'data': first})
        sio.savemat(os.path.join(window_path,name + ' 55-80'),{'data' :second})
        return
    if name == 'pt14_posLAT2-orig':
        first = signal[10000:65000]
        second = signal[102500:120000]
        third = signal[191000:210000]
        sio.savemat(os.path.join(window_path,name + ' 10-65'), {'data': first})
        sio.savemat(os.path.join(window_path,name + ' 102,5-120'),{'data' :second})
        sio.savemat(os.path.join(window_path,name + ' 181-210'), {'data': third})
        return
    if name == 'pt14_posLAT-orig':
        first = signal[0:25000]
        second = signal[42500:85000]
        third = signal[130000:160000]
        sio.savemat(os.path.join(window_path,name + ' 0-25'), {'data': first})
        sio.savemat(os.path.join(window_path,name + ' 42,5-85'),{'data' :second})
        sio.savemat(os.path.join(window_path,name + ' 130-160'), {'data': third})
        return
    if name == 'pt15_pos_LAT-orig':
        first = signal[0:20000]
        second = signal[130000:160000]
        sio.savemat(os.path.join(window_path,name + ' 0-20'), {'data': first})
        sio.savemat(os.path.join(window_path,name + ' 130-160'),{'data' :second})
        return
    if name == 'pt16_pos_LAT2a-orig':
        first = signal[12000:24000]
        second = signal[77000:83000]
        sio.savemat(os.path.join(window_path,name + ' 12-24'), {'data': first})
        sio.savemat(os.path.join(window_path,name + ' 77-83'),{'data' :second})
    return

def get_first_window(name,signal):
    if name == 'pt01_posLat':
        first = signal[290000:320000]
    if name == 'pt01_posM':
        first = signal[50000:110000]
    if name == 'pt02_poslat':
        first = signal[65000:90000]
    if name == 'pt02_poslat2':
       first = signal[32000:61000]
    if name == 'pt05_poslat':
        first = signal[0:20000]
    if name == 'pt06_poslat_mat':
        first = signal[0:32500]
    if name == 'pt06_poslat2':
        first = signal[95000:125000]
    if name == 'pt13_posLAT2-orig':
        first = signal[20000:50000]
    if name == 'pt13_posLAT-orig':
        first = signal[0:25000]
    if name == 'pt14_posLAT2-orig':
        first = signal[10000:60000]
    if name == 'pt14_posLAT-orig':
        first = signal[0:25000]
    if name == 'pt15_pos_LAT-orig':
        first = signal[0:20000]
    if name == 'pt16_pos_LAT2a-orig':
        first = signal[12000:24000]
    return first
