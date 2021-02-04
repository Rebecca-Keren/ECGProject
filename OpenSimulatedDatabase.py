import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io import loadmat
import os
import scipy.io as sio

SIMULATED_SIGNAL_MAT = "Simulated Database Dat"
MAT_SIMULATED_SIGNAL = "Simulated Database Mat"
MERGED_SIMULATED_SIGNAL = "Merged Simulated Database"
SIMULATED_SIGNAL = "Simulated Database"

dir_path = os.path.dirname(os.path.realpath(__file__))
simulated_dir = os.path.join(dir_path,SIMULATED_SIGNAL)
new_dir = os.path.join(dir_path,MAT_SIMULATED_SIGNAL)
merged_dir = os.path.join(dir_path,MERGED_SIMULATED_SIGNAL)
last_dir = os.path.join(dir_path,SIMULATED_SIGNAL)

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
    files = os.listdir(new_dir).copy()
    for filename in os.listdir(new_dir):
        if 'fecg1' not in filename:
            continue
        files.remove(filename)
        name = filename[:len(filename)-5]
        signals = [filename]
        for filename in files:
            if name in filename:
                signals.append(filename)
        size = len(signals)

        if size == 2:
            sig1 = loadmat(os.path.join(new_dir, signals[0]))['data']
            sig2 = loadmat(os.path.join(new_dir, signals[1]))['data']
            for i in range(3):
                record = [a + b  for a, b in zip(sig1[i*25000:(i+1)*25000], sig2[i*25000:(i+1)*25000])]
                sio.savemat(os.path.join(merged_dir, name + 'mix' + str(i)), {'data': record})
                for elem in signals:
                    if ('fecg1' in elem or 'mecg' in elem):
                        sio.savemat(os.path.join(merged_dir, elem + str(i)), {'data': loadmat(os.path.join(new_dir, elem))['data'][i*25000:(i+1)*25000]})

        elif size == 4:
            sig1 = loadmat(os.path.join(new_dir,signals[0]))['data']
            sig2 = loadmat(os.path.join(new_dir,signals[1]))['data']
            sig3 = loadmat(os.path.join(new_dir,signals[2]))['data']
            sig4 = loadmat(os.path.join(new_dir,signals[3]))['data']
            for i in range(3):
                record = [a + b + c + d for a, b, c, d in zip(sig1[i*25000:(i+1)*25000], sig2[i*25000:(i+1)*25000], sig3[i*25000:(i+1)*25000], sig4[i*25000:(i+1)*25000])]
                sio.savemat(os.path.join(merged_dir, name + 'mix' + str(i)), {'data': record})
                for elem in signals:
                    if ('fecg1' in elem or 'mecg' in elem):
                        sio.savemat(os.path.join(merged_dir, elem + str(i)),{'data': loadmat(os.path.join(new_dir, elem))['data'][i * 25000:(i + 1) * 25000]})

        elif (size == 5):
            sig1 = loadmat(os.path.join(new_dir, signals[0]))['data']
            sig2 = loadmat(os.path.join(new_dir, signals[1]))['data']
            sig3 = loadmat(os.path.join(new_dir, signals[2]))['data']
            sig4 = loadmat(os.path.join(new_dir, signals[3]))['data']
            sig5 = loadmat(os.path.join(new_dir, signals[4]))['data']
            for i in range(3):
                record = [a + b + c + d + e for a, b, c, d, e in zip(sig1[i*25000:(i+1)*25000], sig2[i*25000:(i+1)*25000], sig3[i*25000:(i+1)*25000], sig4[i*25000:(i+1)*25000], sig5[i*25000:(i+1)*25000])]
                sio.savemat(os.path.join(merged_dir, name + 'mix' + str(i)), {'data': record})
                for elem in signals:
                    if ('fecg1' in elem or 'mecg' in elem):
                        sio.savemat(os.path.join(merged_dir, elem + str(i)),
                                    {'data': loadmat(os.path.join(new_dir, elem))['data'][i * 25000:(i + 1) * 25000]})

        else:
            print('ciao')








