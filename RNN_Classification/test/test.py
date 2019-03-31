import re
import os
import pandas as pd
import torch
import torchaudio
import pandas as pd
import os

path = '/Users/wzehui/Documents/MA/Daten/quellcode/13230.mp3'

class PreprocessData(object):
    def __init__(self, path):
        self.path = path
        self.mixer = torchaudio.transforms.DownmixMono() #UrbanSound8K uses two channels, this will convert them to one

    def __getitem__(self, index):
        #format the file path and load the file
        sound = torchaudio.load(path, out = None, normalization = True)
        #load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
        soundData = self.mixer(sound[0])
        print(soundData.shape)
        print(soundData.numel())
        #downsample the audio to ~8kHz

        tempData = torch.zeros([1, 3000000]) #tempData accounts for audio clips that are too short
        if soundData.numel() < 3000000:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData = soundData

        soundData = tempData
        # soundFormatted = torch.zeros([32000, 1])
        # soundFormatted[:32000] = soundData[::5] #take every fifth sample of soundData
        # soundFormatted = soundFormatted.permute(1, 0)
        return soundData

a = PreprocessData(path)
b = a[0]
# test_set = UrbanSoundDataset(csv_path, file_path)
# a = train_set[2]
