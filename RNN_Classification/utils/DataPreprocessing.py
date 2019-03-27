import torch
import torchaudio
import pandas as pd
import os

class UrbanSoundDataset(Dataset):
#rapper for the UrbanSound8K dataset
    # Argument List
    #  path to the UrbanSound8K csv file
    #  path to the UrbanSound8K audio files
    #  list of folders to use in the dataset

    def __init__(self, csv_path, file_path, folderList):
        csvData = pd.read_csv(csv_path)
        #initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folder_names = []
        #loop through the csv entries and only add entries from folders in the folder list
        for i in range(0,len(csvData)):
            row_element = csvData.iloc[i, 0]
            row_element = row_element.split(";")
            self.file_names.append(row_element[0])
            self.folder_names.append(row_element[1])
            self.labels.append(row_element[2])

        self.file_path = file_path
        # self.mixer = torchaudio.transforms.DownmixMono() #UrbanSound8K uses two channels, this will convert them to one
        # self.folderList = folderList

    def __getitem__(self, index):
        #format the file path and load the file
        path = self.file_path + str(self.folder_names[index]) + os.sep + self.file_names[index] + ".vis"
        sound = torchaudio.load(path, out = None, normalization = True)
        #load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
        soundData = self.mixer(sound[0])
        #downsample the audio to ~8kHz
        tempData = torch.zeros([160000, 1]) #tempData accounts for audio clips that are too short
        if soundData.numel() < 160000:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:160000]

        soundData = tempData
        soundFormatted = torch.zeros([32000, 1])
        soundFormatted[:32000] = soundData[::5] #take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        return soundFormatted, self.labels[index]

    def __len__(self):
        return len(self.file_names)


csv_path = '~/PycharmProjects/Masterarbeit/Daten/index/Bariton.csv'
file_path = '~/PycharmProjects/Masterarbeit/Daten/quellcode/sounddb/'

train_set = UrbanSoundDataset(csv_path, file_path, range(1,10))
test_set = UrbanSoundDataset(csv_path, file_path, [10])
print("Train set size: " + str(len(train_set)))
print("Test set size: " + str(len(test_set)))

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 128, shuffle = True, **kwargs)
