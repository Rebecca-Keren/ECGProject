import pyedflib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import scale

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3, axis=1):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = filtfilt(b, a, data, axis=axis)
	return y

def ecgSignals(fileNames,index):
	f = pyedflib.EdfReader(fileNames[index])
	n = f.signals_in_file
	ann = f.readAnnotations()
	#print(ann)
	ann = np.array(ann[0])
	#print(ann)
	for i in range(len(ann)):
		ann[i] = int(ann[i])
	print(ann)
	abdECG = np.zeros((n - 1, f.getNSamples()[0]))
	fetalECG = np.zeros((1, f.getNSamples()[0]))
	fetalECG[0, :] = f.readSignal(0)
	fetalECG[0, :] = scale(butter_bandpass_filter(fetalECG, 10, 50, 1000), axis=1)
	for i in np.arange(1, n):
		abdECG[i - 1, :] = f.readSignal(i)
	abdECG = scale(butter_bandpass_filter(abdECG, 10, 50, 1000), axis=1)
	abdECG = signal.resample(abdECG, int(abdECG.shape[1] / 5), axis=1)
	fetalECG = signal.resample(fetalECG, int(fetalECG.shape[1] / 5), axis=1)
	return abdECG, fetalECG,ann

if __name__ == '__main__':
	fileNames = ["r01.edf"]
	for i in range(1):
		abdECG, fetalECG,ann = ecgSignals(fileNames,i)
		# for j in range(4):
		# 	plt.title("abdominal ecg" + str(i) +"number"+str(j))
		# 	plt.plot(abdECG[j])
		# 	plt.show()
		#print(ann)
		#plt.title("fetal ecg"+ str(i))
		x = fetalECG[0]
		#plt.plot(x,ann,marker = "*")
		#plt.show()