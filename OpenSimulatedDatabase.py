import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os
import scipy.io as sio

SIMULATED_SIGNAL = "Simulated Database Dat"
MAT_SIMULATED_SIGNAL = "Simulated Database"
dir_path = os.path.dirname(os.path.realpath(__file__))
simulated_dir = os.path.join(dir_path,SIMULATED_SIGNAL)
new_dir = os.path.join(dir_path,MAT_SIMULATED_SIGNAL)

if __name__ == '__main__':
    #channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
    #for filename in os.listdir(simulated_dir):
        #if not(filename.endswith(".dat")):
            #continue
        #print(filename[:len(filename)-4])
        #record,fields = wfdb.rdsamp(os.path.join(simulated_dir,filename[:len(filename)-4]), channels=[20])
        #sio.savemat(os.path.join(new_dir,filename[:len(filename)-4]),{'data': record})
        #ann = wfdb.rdann('sub01_snr00dB_l1_c0_mecg','qrs',sampto=3000)
        #wfdb.plot_items(signal=record,
                     #ann_samp=[ann.sample, ann.sample])
        #wfdb.plot_wfdb(record = record,plot_sym=True, figsize=(10,4))
        record,fields = wfdb.rdsamp(os.path.join(simulated_dir,"sub01_snr03dB_l1_c1_mecg"), channels=[20],sampto=3000)
        record1,fields = wfdb.rdsamp(os.path.join(simulated_dir,"sub01_snr03dB_l1_c0_mecg"), channels=[20],sampto=3000)
        #record2, fields = wfdb.rdsamp(os.path.join(simulated_dir, "sub01_snr03dB_l1_c1_noise1"), channels=[20],
                                               #sampto=3000)
        #record3, fields = wfdb.rdsamp(os.path.join(simulated_dir, "sub01_snr03dB_l1_c1_noise2"), channels=[20],
                                               #sampto=3000)

        #record4 = [a + b + c  for a, b, c in zip(record,record1,record3)]

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(record[:1000])
        ax2.plot(record1[:1000])
        plt.show()
        plt.close()
