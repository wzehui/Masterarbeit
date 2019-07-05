"""
Load raw audio files in time domain into dataset for training or testing
Load extracted feature files into dataset for training or testing
feature format :
                |0         |1          |2        |3               |4              |5        |
                |Name      |Label      |Voice    |Start of formant|Stop of formant|PHE      |
                ----------------------------------------------------------------------------
                |6         |7          |8        |9               |10             |11       |
                |FHE       |Vibratofreq|Extent   |Jitter          |Shimmer        |Strength |
                ----------------------------------------------------------------------------
                |12        |13         |14       |15              |16             |         |
                |Centroid  |Variance   |Skewness |Kurtosis        |Tonklasse      |         |
---------------------------------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

import torch
import torchaudio
import pandas as pd
import os
import numpy as np
import random

from utils.MelFrequency import mel_freq


def process_index(csv_path, test_rate, val_rate, repeat_time):
    len_csv = len(pd.read_csv(csv_path))
    index = [n for n in range(len_csv)]
    random.seed(5)  # reproducibility

####
    # # hold-out method
    # test_index = random.sample(index, round(test_rate * len(index)))
    # train_index = list(set(index)-set(test_index))
    # val_index = random.sample(train_index, round(val_rate * len(train_index)))
    # train_index = list(set(train_index)-set(val_index))

    # bootstrapping method
    train_index = []
    test_index = random.sample(index, round(test_rate * len(index)))
    temp_index = list(set(index) - set(test_index))
    train_index += random.choices(temp_index, k=len(temp_index))
    val_index = list(set(temp_index) - set(train_index))

    # batch training method
    len_o = len(train_index)
    for i in range(0, repeat_time):
        temp_index = random.sample(train_index, len_o)
        train_index.extend(temp_index)

    return train_index, val_index, test_index


class PreprocessData(object):

    def __init__(self, csv_path, file_path, index):
        csv_data = pd.read_csv(csv_path)
        self.index = index
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folder_names = []

        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(len(csv_data)):
            row_element = csv_data.iloc[i, 0]
            row_element = row_element.split(";")
            self.file_names.append(row_element[0])
            self.folder_names.append(row_element[1])
            self.labels.append(int(row_element[2]))

        self.file_path = file_path
        self.mixer = torchaudio.transforms.DownmixMono()  # uses two channels, this will convert them to one

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.file_path + str(self.folder_names[self.index[index]]) + os.sep + self.file_names[self.index[index]] + "_m" + ".wav"
        sound = torchaudio.load(path, out=None, normalization=True)
        # load returns a tensor with the sound data and the sampling frequency
        sound_data = self.mixer(sound[0])

        # Pre-emphasis
        pre_emphasis = 0.97
        sound_data = np.append(sound_data[0, 0].numpy(),
                                 (sound_data[0, 1:] - pre_emphasis * sound_data[0, :-1]).numpy())
        sound_data = torch.from_numpy(sound_data)
        sound_data = sound_data.unsqueeze(0)

        # 300 ms Redundacy at beginning
        temp_data = torch.zeros([1, 6600])
        temp_data = torch.cat([temp_data, sound_data], 1)
        sound_data = temp_data

        # Zero-Padding to 4000 ms
        temp_data = torch.zeros([1, 90000])  # tempData accounts for audio clips that are too short
        if sound_data.numel() < 90000:
            temp_data[0, :sound_data.numel()] = sound_data[0, :]
        else:
            temp_data[0, :] = sound_data[0, :90000]
        sound_data = temp_data

        sound_data = mel_freq(sound_data, sound[1])
        sound_data = sound_data.unsqueeze(0)  # expand dimension from [*,nmel,nframe] to [*,1,nmel,nframe]
        return sound_data, self.labels[self.index[index]]

    def __len__(self):
        return len(self.index)


class PreprocessFeature(object):

    def __init__(self, feature_path, feature_index, index):
        self.feature_data = pd.read_csv(feature_path)
        self.feature_index = feature_index
        self.index = index
        self.labels = []

        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(len(self.feature_data)):
            self.labels.append(int(self.feature_data.Label[i]))

    def __getitem__(self, index):
        # label = int(self.feature_data['Label'][self.index[index]])
        temp = self.feature_data.iloc[self.index[index]]

        feature = []
        for i in range(0, len(self.feature_index)):
            feature.append(temp[self.feature_index[i]])
        feature = torch.FloatTensor(feature)    # transform list to tensor
        # expand dimension from tensor([*feature_number]) to tensor([[1, *feature_number]])
        feature = torch.unsqueeze(feature, 0)
        return feature, self.labels[self.index[index]]

    def __len__(self):
        return len(self.index)
