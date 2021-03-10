import numpy as np
import struct
import ctypes
import matplotlib.pyplot as plt
import codecs
import csv
import json
import wfdb
import os
import pyedflib
if __name__ == '__main__':
    # signal = list()
    # #f = pyedflib.open('r01.edf',)
    # f = pyedflib.EdfReader('r01.edf')
    # n = f.signals_in_file
    # x = f.readSignal(0)
    # # plt.plot(x)
    # # plt.show()
    # # plt.close()
    # # x = f.readSignal(1)
    # # plt.plot(x)
    # # plt.show()
    # # plt.close()
    # # x = f.readSignal(2)
    # # plt.plot(x)
    # # plt.show()
    # # plt.close()
    # # x = f.readSignal(3)
    # # plt.plot(x)
    # # plt.show()
    # # plt.close()
    # # x = f.readSignal(4)
    # # plt.plot(x)
    # # plt.show()
    # # plt.close()
    # ann = f.readAnnotations()[0]
    # for elem in ann:
    #     print(elem)


    #record,fields = wfdb.rdsamp('sub01_snr00dB_l1_c0_mecg', channels=[32,33])
    #channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
    record,fields = wfdb.rdsamp('sub01_snr00dB_l1_c0_mecg', channels=[32],sampto=3000)
    ann = wfdb.rdann('sub01_snr00dB_l1_c0_mecg','qrs',sampto=3000)
    wfdb.plot_items(signal=record,
                     ann_samp=[ann.sample, ann.sample])

    # edf_record = wfdb.rdrecord('r01.edf')
    # print(edf_record.p_signal)
    # print(edf_record.d_signal)
    # print(edf_record.record_name)
    # print(edf_record.n_sig)
    # print(edf_record.sig_len)


    #for elem in record:
    #     signal.append(elem[16])
    # plt.plot(signal[:2000])
    # plt.show()

    #or elem in record:
    #    print(len(elem))
    # record = wfdb.rdrecord('sub01_snr00dB_l1_c0_fecg1',sampto=3000)
    # print(record)
    #wfdb.plot_wfdb(record = record,plot_sym=True, figsize=(10,4))
    # print(len(signal))
    # plt.plot(signal[:30000])
    # plt.show()


    # file = open("filename.dat", "rb")
    # data = [line.decode("utf-16", "ignore") for line in file]
    # print(data)
    # #
    # # fp = codecs.open("filename.dat", "r", "utf-16")
    # # lines = fp.readlines()
    # # for line in lines:
    # #     print(line)
    # #     break
    # # with open("filename.dat",'rb') as f:
    # #     for line in f:
    # #         print(line)
    # #         line = bytes(filter(None, line))
    # #         print(json.loads(line.decode('utf16')))
    # #         #for elem in line:
    # #            #print(int.from_bytes(elem,'little'))
    # #
    # #         #print(line)
    # #         break
    # #         result = bytes(filter(None,line))
    # #         out_hex = ['{:02}'.format(b) for b in result]
    # #         for j in range(len(out_hex)):
    # #             if int(out_hex[j]) > 150:
    # #               samples.append(int(out_hex[j]))
    # #     prova = list()
    # #     for i in range(200):
    # #         if int(samples[i]) < 20:
    # #             prova.append(int(samples[i]))
    #
    #
    #     # plt.plot(samples[0:30000])
    #     # plt.show()
    #
    #
    # #datContent = [i.strip().split() for i in open("filename.dat",'b').readlines()]
    # # print(len(datContent))
    # # i = 0
    # # for elem in datContent:
    # #     if i == 20:
    # #         break
    # #     print(elem)
    # #     i = i + 1
    # #
    # # samples = list()
    # # for elem in datContent:
    # #     if len(elem) == 0:
    # #         continue
    # #     for i in range(len(elem)):
    # #         tmp = bytes(elem[i], 'utf-8')
    # #         for j in range(len(tmp)):
    # #             samples.append(tmp[j])
    # # #print(samples)
    # # plt.plot(samples)
    # # plt.show()
    #
