"""
Cut the original audio file into samples according to relevant *.nfo
---------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

import pandas as pd
import os
import re
from pydub import AudioSegment

csv_path = '/Users/wzehui/Documents/MA/Daten/index/index.csv'
file_path = '/Users/wzehui/Documents/MA/Daten/quellcode/sounddb/'

csvData = pd.read_csv(csv_path)

file_names = []
folder_names = []

for i in range(0, len(csvData)):
    row_element = csvData.iloc[i, 0]
    row_element = row_element.split(";")
    file_names = row_element[0]
    folder_names = row_element[1]

    info_path = file_path + str(folder_names) + os.sep + file_names + ".nfo"
    audio_pth = file_path + str(folder_names) + os.sep + file_names + ".vis"

    with open(info_path, 'rb') as f:
        line = f.readlines()
        if type(line[0]) == bytes:
            for j in range(0, line.__len__()):
                line[j] = line[j].decode('utf8','ignore')

        line = list(map(str.strip, line))

        for j in range(0, line.__len__()):
            temp = re.split(":", line[j])
            if temp[0] == 'vstart':
                time_b = temp[1]
            elif temp[0] == 'vstop':
                time_s = temp[1]

    # print(time_b, time_s)
    audio_temp = AudioSegment.from_file(audio_pth)
    audio_temp = audio_temp[float(time_b):float(time_s)]
    audio_temp.export(file_path + str(folder_names) + os.sep + file_names + "_m" + ".wav", format="wav")