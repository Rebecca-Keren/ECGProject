import numpy as np
import random as rd
import scipy
from itertools import combinations
import scipy.fftpack as function

#Not to use
def average_correlation_signal(list_windows):
    # A Python program to print all
    # combinations of given length
    # Get all combinations of [1, 2, 3]
    # and length 2
    correlation_list = []
    comb = combinations(list_windows, 2)
    # Print the obtained combinations
    for item in list(comb):
        signal1 = item[0]
        signal2 = item[1]

        if (len(signal1) != len(signal2)):
            continue
        correlation = check_correlation(signal1,signal2)

        correlation_list.append(correlation)
    return np.mean(correlation_list)
#FATTO
def increase_sampling_rate(signal,rate):
    signal_size = len(signal)
    x = [j for j in range(signal_size)]
    y = [signal[i] for i in range(signal_size)]
    xvals = np.linspace(0, signal_size, int(signal_size*rate))
    interpolated_signal = np.interp(xvals, x, y)
    if (rate >= 1):
        interpolated_signal = interpolated_signal[:signal_size]
    return interpolated_signal

#Not to use
def switch_samples(signal):
    signal_size = len(signal)
    switched_signal = signal
    for i in range(signal_size-1):
        prob = rd.randint(0,1)#returns 0 or 1 and decides which samples to switch with 50% probability
        if(prob):#switch between the two successive samples
            temp = switched_signal[i+1]
            switched_signal[i+1]=switched_signal[i]
            switched_signal[i]=temp
    return switched_signal
#FATTO
def switch_samples_dist(signal,p,jump = True,window_jump = 10):
    signal_size = len(signal)
    rand_list = [0,1]
    distribution = [p,1-p]
    switched_signal = signal
    for i in range(signal_size-1):
        if ( jump==True ):
            jump = rd.randint(1,window_jump)
            i = jump
            if (i >= signal_size - 1):
                break
        random_number = rd.choices(rand_list, distribution)
        if(random_number[0]):#switch between the two successive samples
            temp = switched_signal[i+1]
            switched_signal[i+1]=switched_signal[i]
            switched_signal[i]=temp
    return switched_signal
#FATTO
def check_correlation(orig_signal,changed_signal):
    orig = orig_signal
    signal_to_compare = changed_signal
    correlation, p_value = scipy.stats.pearsonr(orig, signal_to_compare)
    return correlation.real #the correlation of the original signal in comperison to the changed signal
#FATTO
def addnoise(array,percentage):
    i = len(array)
    # create 1D numpy data:
    npdata = np.asarray(array).reshape((i))
    # add uniform noise:
    u = npdata + np.random.normal(0, 1, i) * percentage
    return u
#FATTO
def add_frequencies(signal,p,percentage,jump= True,window_jump = 10):
    signal_size = len(signal)
    rand_list = [0,1]
    distribution = [p,1-p]
    switched_signal = signal
    for i in range(signal_size - 1):
        if ( jump==True ):
            jump = rd.randint(1,window_jump)
            i = jump
            if (i >= signal_size - 1):
                break
        random_number = rd.choices(rand_list, distribution)
        if (random_number[0] and switched_signal[i]!=0):# increase the frequency by a specific factor if not zero
            switched_signal[i]=switched_signal[i]+(switched_signal[i]*percentage)
    return scipy.fftpack.ifft(switched_signal)
#Not to use
def concatenating(signal1,signal2):
    return np.concatenate((signal1,signal2), axis = None),np.concatenate((signal2,signal1), axis = None)
#FATTO
def drop_from_signal(signal,p,jump = True,window_jump = 10):
    signal_size = len(signal)
    rand_list = [0, 1]
    distribution = [p, 1 - p]
    drop_signal = signal
    for i in range(signal_size - 1):
        if ( jump==True ):
            jump = rd.randint(1,window_jump)
            i = jump
            if (i >= signal_size - 1):
                break
        random_number = rd.choices(rand_list, distribution)
        if (random_number[0]):
            drop_signal[i]=0
    return drop_signal
#FATTO
def reverse_signal(signal):
    result = signal[::-1]
    return signal
#FATTO
def add_more_augumentation_with_p(signal,p):
    rand_list = [0,1]
    distribution = [p,1-p]
    random_number = rd.choices(rand_list, distribution)
    new_signal = signal.copy()
    if (random_number[0]):
        rate_list = list(range(-0.9,1.11),0.01)
        rate = rd.choices(rate_list)
        new_signal = increase_sampling_rate(new_signal, rate)
    random_number = rd.choices(rand_list, distribution)
    if (random_number[0]):
        prob_list = [0.1,0.2,0.3]
        prob =  rd.choices(prob_list)
        new_signal = switch_samples_dist(new_signal, prob, jump=True, window_jump=10)
    random_number = rd.choices(rand_list, distribution)
    if (random_number[0]):
        prob_list = [0.1,0.05]
        prob =  rd.choices(prob_list)
        new_signal = function.ifft(switch_samples_dist(function.fft(new_signal), prob, jump=True, window_jump=10))
    random_number = rd.choices(rand_list, distribution)
    if (random_number[0]):
        percentage_list = [0.003,0.002,0.001]
        percentage = rd.choices(percentage_list)
        new_signal = addnoise(new_signal, percentage)
    random_number = rd.choices(rand_list, distribution)
    if (random_number[0]):
        percentage_list = [0.1,0.2,0.3]
        percentage = rd.choices(percentage_list)
        prob = rd.choices(percentage_list)
        new_signal = function.ifft(add_frequencies(function.fft(new_signal), prob, percentage, jump=True, window_jump=10))
    random_number = rd.choices(new_signal, distribution)
    if (random_number[0]):
        prob_list = [0.1,0.05]
        prob = rd.choices(prob_list)
        new_signal = drop_from_signal(new_signal, prob, jump=True, window_jump=10)
    random_number = rd.choices(rand_list, distribution)
    if (random_number[0]):
        new_signal = reverse_signal(new_signal)
    return new_signal