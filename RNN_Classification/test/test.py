import torch
import torch.nn as nn
import torch.nn.functional as F


x = torch.ones(8, 1, 64, 546)

conv1 = nn.Conv2d(1, 256, [9, 9], 1)
pool1 = nn.MaxPool2d([3,1])
conv2 = nn.Conv2d(256, 256, [4, 3], 1)

linear1 = nn.Linear(256, 1)

lstm=nn.LSTM(input_size=15, hidden_size=832, num_layers=2, batch_first=True)

linear2 = nn.Linear(832, 2)

linear3 = nn.Linear(536, 1)


x = conv1(x)
x = pool1(x)
x = conv2(x)

x = x.permute(0, 2, 3, 1)
x = linear1(x)
x = x.squeeze()

x = x.permute(0, 2, 1)
x = lstm(x)

x = linear2(x[0])

x = F.log_softmax(x, dim=2)

x = x.permute(0, 2, 1)

x = linear3(x)

x = x.unsqueeze(1)

x = x.permute(0, 1, 3, 2)
# print(lstm_seq.weight_hh_l0.size())
# print(lstm_seq.weight_ih_l0.size())
