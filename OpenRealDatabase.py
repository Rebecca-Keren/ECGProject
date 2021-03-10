import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io import loadmat
import os
import scipy.io as sio

from HelpFunctions import *

REAL_DATASET = "Real Database"
dir_path = os.path.dirname(os.path.realpath(__file__))
real_dir = os.path.join(dir_path, REAL_DATASET)

if __name__ == '__main__':
    print("ciao")
    data = loadmat(os.path.join(real_dir, "pt02_posSucc0.mat"))['data'][0]
    print(len(data))
    data = increase_sampling_rate(data, 0.9)
    print(len(data))
    plt.plot(data[:2048])
    plt.show()
