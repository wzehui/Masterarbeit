"""
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
--------------------------------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

import pandas as pd
import torch


class PreprocessFeature(object):

    def __init__(self, csv_path, feature_path, rate, feature_index):
        csvData = pd.read_csv(csv_path)
        self.featureData = pd.read_csv(feature_path)
        self.feature_index = feature_index
        self.file_names = []
        # loop through the csv entries and only add entries from folders in the folder list
        if rate >= 0.5:
            for i in range(0, int(rate * len(csvData))):
                row_element = csvData.iloc[i, 0]
                row_element = row_element.split(";")
                self.file_names.append(row_element[0])
        else:
            for i in range(int((1 - rate) * len(csvData)), len(csvData)):
                row_element = csvData.iloc[i, 0]
                row_element = row_element.split(";")
                self.file_names.append(row_element[0])

    def __getitem__(self, index):
        idx = self.featureData['Name'].to_list().index(self.file_names[index])

        label = int(self.featureData['Label'].to_list()[0])
        temp = self.featureData.iloc[idx].to_list()
        feature = []

        for i in range(0, len(self.feature_index)):
            feature.append(temp[self.feature_index[i]])
        feature = torch.FloatTensor(feature)    # transform list to tensor
        # expand dimension from tensor([*feature_number]) to tensor([[1, *feature_number]])
        feature = torch.unsqueeze(feature, 0)
        return feature, label

    def __len__(self):
        return len(self.file_names)
