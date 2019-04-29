"""
Load raw audio files in time domain into dataset for training or testing
------------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

import torch
import torchaudio
import pandas as pd
import os


class PreprocessData(object):

    def __init__(self, csv_path, file_path, rate):
        csv_data = pd.read_csv(csv_path)
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folder_names = []

        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(int(rate[0] * len(csv_data)), int(rate[1] * len(csv_data))):
            row_element = csv_data.iloc[i, 0]
            row_element = row_element.split(";")
            self.file_names.append(row_element[0])
            self.folder_names.append(row_element[1])
            self.labels.append(int(row_element[2]))

        self.file_path = file_path
        self.mixer = torchaudio.transforms.DownmixMono()  # uses two channels, this will convert them to one

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.file_path + str(self.folder_names[index]) + os.sep + self.file_names[index] + "_m" + ".wav"
        sound = torchaudio.load(path, out=None, normalization=True)
        # load returns a tensor with the sound data and the sampling frequency
        sound_data = self.mixer(sound[0])

        temp_data = torch.zeros([1, 120000])  # tempData accounts for audio clips that are too short
        if sound_data.numel() < 120000:
            temp_data[0, :sound_data.numel()] = sound_data[0, :]
        else:
            temp_data[0, :] = sound_data[0, :120000]

        sound_data = temp_data
        return sound_data, self.labels[index]

    def __len__(self):
        return len(self.file_names)
