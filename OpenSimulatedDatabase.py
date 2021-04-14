import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io import loadmat
import os
import scipy.io as sio
from HelpFunctions import *

REAL_DATASET = "Real Database"
SIMULATED_SIGNAL_MAT = "Simulated Database Dat"
MAT_SIMULATED_SIGNAL = "Simulated Database Mat"
MERGED_SIMULATED_SIGNAL = "SimulatedDatabase"
SIMULATED_SIGNAL = "Simulated Database"

dir_path = os.path.dirname(os.path.realpath(__file__))
simulated_dir = os.path.join(dir_path,SIMULATED_SIGNAL)
new_dir = os.path.join(dir_path,MAT_SIMULATED_SIGNAL)
merged_dir = os.path.join(dir_path,MERGED_SIMULATED_SIGNAL)
last_dir = os.path.join(dir_path,SIMULATED_SIGNAL)
real_dir = os.path.join(dir_path,REAL_DATASET)


if __name__ == '__main__':
    # Transforming from dat to mat
    #channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
    #for filename in os.listdir(simulated_dir):
        #if not(filename.endswith(".dat")):
            #continue
        #print(filename[:len(filename)-4])
        #record,fields = wfdb.rdsamp(os.path.join(simulated_dir,filename[:len(filename)-4]), channels=[20])
        #sio.savemat(os.path.join(new_dir,filename[:len(filename)-4]),{'data': record})



    #Merging Data,dividing into windows and saving
    # files = os.listdir(new_dir).copy()
    # for filename in os.listdir(new_dir):
    #     if 'fecg1' not in filename:
    #         continue
    #     files.remove(filename)
    #     name = filename[:len(filename)-5]
    #     signals = [filename]
    #     for filename in files:
    #         if name in filename:
    #             signals.append(filename)
    #     size = len(signals)
    #     window_size = 1024
    #     number_of_window = 73
    #     num_of_signal_to_remove = 248
    #
    #     if size == 2:
    #         sig1 = loadmat(os.path.join(new_dir, signals[0]))['data'][num_of_signal_to_remove:]
    #         # print(len(sig1))
    #         sig2 = loadmat(os.path.join(new_dir, signals[1]))['data'][num_of_signal_to_remove:]
    #         for i in range(number_of_window):
    #             record = [a + b  for a, b in zip(sig1[i*window_size:(i+1)*window_size], sig2[i*window_size:(i+1)*window_size])]
    #             # print(len(record))
    #             sio.savemat(os.path.join(merged_dir, name + 'mix' + str(i)), {'data': record})
    #             for elem in signals:
    #                 if ('fecg1' in elem or 'mecg' in elem):
    #                     data = loadmat(os.path.join(new_dir, elem))['data'][i*window_size:(i+1)*window_size]
    #                     # print(len(data))
    #                     sio.savemat(os.path.join(merged_dir, elem + str(i)), {'data':data})
    #
    #     elif size == 4:
    #         sig1 = loadmat(os.path.join(new_dir,signals[0]))['data'][num_of_signal_to_remove:]
    #         # print(len(sig1))
    #         sig2 = loadmat(os.path.join(new_dir,signals[1]))['data'][num_of_signal_to_remove:]
    #         sig3 = loadmat(os.path.join(new_dir,signals[2]))['data'][num_of_signal_to_remove:]
    #         sig4 = loadmat(os.path.join(new_dir,signals[3]))['data'][num_of_signal_to_remove:]
    #         for i in range(number_of_window):
    #             record = [a + b + c + d for a, b, c, d in zip(sig1[i*window_size:(i+1)*window_size], sig2[i*window_size:(i+1)*window_size], sig3[i*window_size:(i+1)*window_size], sig4[i*window_size:(i+1)*window_size])]
    #             print(len(record))
    #             sio.savemat(os.path.join(merged_dir, name + 'mix' + str(i)), {'data': record})
    #             for elem in signals:
    #                 if ('fecg1' in elem or 'mecg' in elem):
    #                     data = loadmat(os.path.join(new_dir, elem))['data'][i * window_size:(i + 1) * window_size]
    #                     # print(len(data))
    #                     sio.savemat(os.path.join(merged_dir, elem + str(i)),{'data': data})
    #
    #     elif (size == 5):
    #         sig1 = loadmat(os.path.join(new_dir, signals[0]))['data'][num_of_signal_to_remove:]
    #         print(len(sig1))
    #         sig2 = loadmat(os.path.join(new_dir, signals[1]))['data'][num_of_signal_to_remove:]
    #         sig3 = loadmat(os.path.join(new_dir, signals[2]))['data'][num_of_signal_to_remove:]
    #         sig4 = loadmat(os.path.join(new_dir, signals[3]))['data'][num_of_signal_to_remove:]
    #         sig5 = loadmat(os.path.join(new_dir, signals[4]))['data'][num_of_signal_to_remove:]
    #         for i in range(number_of_window):
    #             record = [a + b + c + d + e for a, b, c, d, e in zip(sig1[i*window_size:(i+1)*window_size], sig2[i*window_size:(i+1)*window_size], sig3[i*window_size:(i+1)*window_size], sig4[i*window_size:(i+1)*window_size], sig5[i*window_size:(i+1)*window_size])]
    #             # print(len(record))
    #             sio.savemat(os.path.join(merged_dir, name + 'mix' + str(i)), {'data': record})
    #             for elem in signals:
    #                 if ('fecg1' in elem or 'mecg' in elem):
    #                     data = loadmat(os.path.join(new_dir, elem))['data'][i * window_size:(i + 1) * window_size]
    #                     # print(len(data))
    #                     sio.savemat(os.path.join(merged_dir, elem + str(i)),{'data':data})
    #
    #     else:
    #         print('ciao')






    #Check for number of beats
    sig = loadmat(os.path.join(real_dir,'pt01_posAP0'))['data'][0][:2000]
    # plt.plot(sig)
    # plt.show()
    # print(len(sig))
    # sig5 = np.array((loadmat(os.path.join(new_dir,'sub01_snr03dB_l1_c0_mecg'))['data'][:1024]))
    # sig2 = loadmat(os.path.join(new_dir,'sub01_snr03dB_l1_c0_fecg1'))['data'][:1024]
    # sig3 = [a + b for a , b in zip(sig2,sig5)]
    #
    # fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
    #
    # ax1.plot(sig5)
    # ax2.plot(sig2)
    # ax3.plot(sig3)
    # plt.show()
    # plt.close()









