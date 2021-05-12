import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from data_agumentation_function import  *
FETUS_PROCESSED_SIGNALS = "Preprocessed Signals"

dir_path = os.path.dirname(os.path.realpath(__file__))

fetus_processed_signal_path = os.path.join(dir_path,FETUS_PROCESSED_SIGNALS)

# con 2048 ho 2/3 peimot almeno, dopo aver fatto decreasing maximal
def main():
    for filename in os.listdir(fetus_processed_signal_path):
        if filename == '.DS_Store':
            continue
        filename = os.path.join(fetus_processed_signal_path,filename)
        signal = loadmat(filename)['data'][0]
        signal = increase_sampling_rate(signal, 0.9)
        print(len(signal))
        plt.plot(signal[:2048])
        plt.show()

if __name__ == '__main__':
    main()

