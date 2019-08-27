"""
Cut the original audio file into samples according to relevant *.nfo
---------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import math
from pydub import AudioSegment

csv_path = '/Users/wzehui/Documents/MA/Daten/quellcode/index.csv'
file_path = '/Users/wzehui/Documents/MA/Daten/quellcode/sounddb/'

csvData = pd.read_csv(csv_path)

file_names = []
folder_names = []
t = []
t_min = 1000
t_max = 500

for i in range(0, len(csvData)):
    # row_element = csvData.iloc[i, 0]
    # row_element = row_element.split(";")
    file_names = csvData.iloc[i, 0] # row_element[0]
    folder_names = csvData.iloc[i, 1] #row_element[1]

    if folder_names.find(os.sep):
        folder_names = folder_names.split('\\')
        folder_names = os.path.join(folder_names[1], folder_names[2])

    # info_path = file_path + str(folder_names) + os.sep + file_names + ".nfo"
    # audio_pth = file_path + str(folder_names) + os.sep + file_names + ".vis"

    info_path = os.path.join(file_path, folder_names + '.nfo')
    audio_path = os.path.join(file_path, folder_names + '.vis')

    with open(info_path, 'rb') as f:

        line = f.readlines()
        if type(line[0]) == bytes:
            for j in range(0, line.__len__()):
                line[j] = line[j].decode('utf8','ignore')

        line = list(map(str.strip, line))

        for j in range(0, line.__len__()):
            temp = re.split(":", line[j])
            if temp[0] == 'vstart':
                time_b = float(temp[1])
            elif temp[0] == 'vstop':
                time_s = float(temp[1])

    if time_s - time_b > t_max:
        t_max = time_s - time_b
    if time_s - time_b < t_min:
        t_min = time_s - time_b
    t.append(time_s - time_b)
    # print(time_b, time_s)

    # add surplus
    # time_b = float(time_b) + 1
    # time_s = float(time_s) + 0.1

    # audio_temp = AudioSegment.from_file(audio_path)
    # audio_temp = audio_temp[time_b:time_s]
    # audio_temp.export(file_path + str(folder_names) + os.sep + file_names + "_m" + ".wav", format="wav")


# Duration Distribution Analysis

r = np.arange(math.floor(min(t)/100)*100, max(t)+500, 400)
n, bins, patches = plt.hist(x=t, bins=8, alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Dauer (ms)')
plt.ylabel('Anzahl des Dateien')
# plt.title('Sopran')
maxfreq = n.max()
# 设置y轴的上限
plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.savefig('/Users/wzehui/Documents/MA/Plot/distribution.pdf')
plt.show()
