"""
Load audio files into dataset for training or testing
------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

import torch
import torchaudio
import pandas as pd
import os


class PreprocessData(object):
    # Argument List
    #  path to the UrbanSound8K csv file
    #  path to the UrbanSound8K audio files
    #  list of folders to use in the dataset

    def __init__(self, csv_path, file_path, rate):
        csvData = pd.read_csv(csv_path)
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folder_names = []
        # loop through the csv entries and only add entries from folders in the folder list
        if rate >= 0.5:
            for i in range(0,int(rate * len(csvData))):
                row_element = csvData.iloc[i, 0]
                row_element = row_element.split(";")
                self.file_names.append(row_element[0])
                self.folder_names.append(row_element[1])
                self.labels.append(row_element[2])
        else:
            for i in range(int((1-rate) * len(csvData)), len(csvData)):
                row_element = csvData.iloc[i, 0]
                row_element = row_element.split(";")
                self.file_names.append(row_element[0])
                self.folder_names.append(row_element[1])
                self.labels.append(row_element[2])

        self.file_path = file_path
        self.mixer = torchaudio.transforms.DownmixMono()  # uses two channels, this will convert them to one
        # self.folderList = folderList

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.file_path + str(self.folder_names[index]) + os.sep + self.file_names[index] + "_m" + ".wav"
        sound = torchaudio.load(path, out = None, normalization = True)
        # load returns a tensor with the sound data and the sampling frequency
        soundData = self.mixer(sound[0])

        tempData = torch.zeros([1, 120000]) # tempData accounts for audio clips that are too short
        if soundData.numel() < 120000:
            tempData[0, :soundData.numel()] = soundData[0, :]
        else:
            tempData[0, :] = soundData[0, :120000]

        soundData = tempData
        return soundData, self.labels[index]

    def __len__(self):
        return len(self.file_names)
