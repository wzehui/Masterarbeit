"""
Compare extracted features
------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import pandas as pd

CsvPath = '/Users/wzehui/Documents/MA/Daten/index/index.csv'
FilePath = '/Users/wzehui/Documents/MA/Daten/quellcode/sounddb/'

csvData = pd.read_csv(CsvPath)

FeatureIndex = 3

file_names = []
labels = []
folder_names = []
ff = []

for i in range(0, 1):
    row_element = csvData.iloc[i, 0]
    row_element = row_element.split(";")
    file_names.append(row_element[0])
    folder_names.append(row_element[1])
    labels.append(int(row_element[2]))

    [Fs, x] = audioBasicIO.readAudioFile(FilePath + folder_names[i] + '/' + file_names[i] + '_m.wav')

    F, f_names = audioFeatureExtraction.stFeatureExtraction(x[:,0], Fs, 0.050*Fs, 0.025*Fs)

# Bass: green, Bariton: blue, Tenor: yellow, Sopran: red

    if labels[i] == 0:
        plt.plot(F[FeatureIndex,:], color='green', linewidth=0.5)
    elif labels[i] == 1:
        plt.plot(F[FeatureIndex, :], color='blue', linewidth=0.5)
    elif labels[i] == 2:
        plt.plot(F[FeatureIndex, :], color='orange', linewidth=0.5)
    elif labels[i] == 3:
        plt.plot(F[FeatureIndex, :], color='red', linewidth=0.5)

plt.xlabel('Frame no'); plt.ylabel(f_names[FeatureIndex])
plt.show()