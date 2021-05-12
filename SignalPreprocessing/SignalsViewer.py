import os
from tkinter import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fftpack import fft,dct,fftfreq,rfft
import numpy as np

DATA_FOLDER = "RawData"

top = Tk()

top.geometry("300x350")

lbl = Label(top, text="signals:")

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, DATA_FOLDER)

def showFigure(evt):
    w = evt.widget
    index = int(w.curselection()[0])
    value = w.get(index)
    file_path = os.path.join(data_path, value)
    data = loadmat(file_path)
    ecg_signal = data['data']
    for index in range(ecg_signal.size):
        ecg_signal[index] *= -1
    plt.title(value)
    plt.plot(ecg_signal)
    #freq = fftfreq([ecg_signal.size][0], 1 / 1000)
    #yf = dct(np.array(ecg_signal).flatten())
    #plt.plot(freq, np.abs(yf))
    plt.show()

listbox = Listbox(top,width=50, height=20)
listbox.bind('<<ListboxSelect>>', showFigure)

index = 1

for filename in os.listdir(data_path):
    if os.path.splitext(filename)[1] == ".mat":
        listbox.insert(index, filename)
        index+= 1

lbl.pack()
listbox.pack()

top.mainloop()




