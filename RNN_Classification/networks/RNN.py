"""
Define the structure of Convolutional Neural Network
---------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# x = torch.ones(8, 1, 64, 51)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 256, [9, 9], 1)
        self.pool1 = nn.MaxPool2d([3, 1])
        self.conv2 = nn.Conv2d(256, 256, [4, 3], 1)

        self.linear1 = nn.Linear(256, 1)

        self.lstm = nn.LSTM(input_size=15, hidden_size=832, num_layers=2, batch_first=True)

        self.linear2 = nn.Linear(832, 2)

        self.linear3 = nn.Linear(536, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)

        x = x.permute(0, 2, 3, 1)
        x = self.linear1(x)
        x = x.squeeze()

        x = x.permute(0, 2, 1)
        x = self.lstm(x)

        x = self.linear2(x[0])

        x = F.log_softmax(x, dim=2)

        x = x.permute(0, 2, 1)

        x = self.linear3(x)

        x = x.unsqueeze(1)

        x = x.permute(0, 1, 3, 2)

        return x

