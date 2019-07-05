"""
+---------------------------------------------------------------+
| Main function/script                                          |
+---------------------------------------------------------------+
------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

# import
import yaml
import sys
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import random_split

from utils.DataPreprocessing import *
from networks.CNN import Net
from networks.RNN import CLDNN
# from networks.test import Net

from utils.fBankPlot import mel_plot

import visdom
vis = visdom.Visdom(env=u'train')  # Environment：train $ python -m visdom.server


def fParseConfig(sFile):
    # get config file
    with open(sFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg


######################################################################
# Training and Testing the Network
# --------------------------------
#
# Now let’s define a training function that will feed our training data
# into the model and perform the backward pass and optimization steps.
#


def train(model, epoch):
    # tells layers like dropout, batchnorm etc. that you are training the model

    # training phase
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #
        # 将求解器中的参数置零
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)

        # set requires_grad to True for training
        data = data.requires_grad_()
        # 前向传播
        output = model(data)
        output = output.permute(1, 2, 0, 3)  # original output dimensions are batchSizex1x2

        # 计算损失  The negative log likelihood loss：负对数似然损失 nll_loss
        # loss = F.nll_loss(output[0, 0], target)  # the loss functions expects a batchSizex2 input
        loss = F.cross_entropy(output[0, 0], target)
        # 损失反传
        loss.backward()
        # 优化求解
        optimizer.step()

        if batch_idx % log_interval == 0:  # print training stats
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data) * (batch_idx + 1 != len(train_loader)) + len(train_loader.dataset) * (batch_idx + 1 == len(train_loader)),
                len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss))

    # validation phase
    model.eval()
    tp = 0; tn = 0; fp = 0; fn = 0;
    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        output = output.permute(1, 2, 0, 3)
        pred = output.max(3)[1]  # get the index of the max log-probability

        # correct += pred.eq(target).cpu().sum().item()
        correct = pred.eq(target)
        wrong = pred.ne(target)

        tp += (correct & pred[0][0].byte()).cpu().sum().item()
        tn += (correct & ~pred[0][0].byte()).cpu().sum().item()
        fp += (wrong & pred[0][0].byte()).cpu().sum().item()
        fn += (wrong & ~pred[0][0].byte()).cpu().sum().item()

    # # calculate F1 Score to select a model
    # accuracy = correct / len(test_loader.dataset)
    p = tp / (tp + fp + sys.float_info.epsilon)
    r = tp / (tp + fn + sys.float_info.epsilon)
    f1 = (2 * p * r) / (p + r + sys.float_info.epsilon)

    print('\nF1 Score of Validation set : {:.3f}'.format(f1))

    return f1, model.state_dict(), loss

######################################################################
# Now that we have a training function, we need to make one for testing
# the networks accuracy. We will set the model to ``eval()`` mode and then
# run inference on the test dataset. Calling ``eval()`` sets the training
# variable in all modules in the network to false. Certain layers like
# batch normalization and dropout layers behave differently during
# training so this step is crucial for getting correct results.
#


def test(model):
    model.eval()
    tp = 0; tn = 0; fp = 0; fn = 0;

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        output = output.permute(1, 2, 0, 3)
        pred = output.max(3)[1]  # get the index of the max log-probability
        # correct += pred.eq(target).cpu().sum().item()

        correct = pred.eq(target)
        wrong = pred.ne(target)

        tp += (correct & pred[0][0].byte()).cpu().sum().item()
        tn += (correct & ~pred[0][0].byte()).cpu().sum().item()
        fp += (wrong & pred[0][0].byte()).cpu().sum().item()
        fn += (wrong & ~pred[0][0].byte()).cpu().sum().item()

    # accuracy = correct / len(test_loader.dataset)
    p = tp / (tp + fp + sys.float_info.epsilon)
    r = tp / (tp + fn + sys.float_info.epsilon)
    ber = 0.5*(fp/(tn + fp + sys.float_info.epsilon) + fn/(tp + fn + sys.float_info.epsilon))
    f1 = (2*p*r) / (p + r + sys.float_info.epsilon)
    print('\ntp:{}, tn:{}, fp:{}, fn:{}'.format(tp, tn, fp, fn))
    print('\nPrecision/Recall of Test set : {:.1f}%/{:.1f}%'.format(p*100, r*100))
    print('\nBER of Test set : {:.3f}'.format(ber))
    print('\nF1 Score of Test set : {:.3f}'.format(f1))


# detect device type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDevice: " + device.type)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

# load parameter file
cfg = fParseConfig('param.yml')

try:
    with open(cfg['DataPath'], 'rb') as f:
        index_all = pickle.load(f)
        print("\nLoad Data")
        training_index = index_all[0]
        val_index = index_all[1]
        test_index = index_all[2]

except(FileNotFoundError):
    training_index, val_index, test_index = \
    process_index(cfg['IndexPath'], cfg['TestSplitRate'], cfg['ValSplitRate'], cfg['DataRepeat'])

    if cfg['lSave']:
        index_all = [training_index, val_index, test_index]
        with open(cfg['DataPath'], 'wb') as f:
            pickle.dump(index_all, f)

if cfg['FeatureExtraction']:
    train_set = PreprocessFeature(cfg['FeaturePath'], cfg['FeatureSelection'], training_index)
    # a = train_set[0]
    val_set = PreprocessFeature(cfg['FeaturePath'], cfg['FeatureSelection'], val_index)
    test_set = PreprocessFeature(cfg['FeaturePath'], cfg['FeatureSelection'], test_index)


else:
    train_set = PreprocessData(cfg['IndexPath'], cfg['FilePath'], training_index)
    # a = train_set[0]
    val_set = PreprocessData(cfg['IndexPath'], cfg['FilePath'], val_index)
    test_set = PreprocessData(cfg['IndexPath'], cfg['FilePath'], test_index)

print("\nTrain set size: " + str(len(train_set)))
print("\nValidation set size: " + str(len(val_set)))
print("\nTest set size: " + str(len(test_set)))

kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg['BatchSize'], shuffle=False, **kwargs)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg['BatchSize'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg['BatchSize'], shuffle=True, **kwargs)

# load network structure
# model = Net()
model = CLDNN()
# model = Net()

param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

model.to(device)

try:
    model.load_state_dict(torch.load(cfg['BestModelPath']))
    model.to(device)
    print("\nLoad Best Model.")
except(FileNotFoundError):
    pass

# print(model)

# We will use the same optimization technique used in the paper, an Adam
# optimizer with weight decay set to 0.0001. At first, we will train with
# a learning rate of 0.01, but we will use a ``scheduler`` to decrease it
# to 0.001 during training.

# optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=0.0001)
optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# Print model's state_dict
print("\nModel's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("\nOptimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

log_interval = 1
f1_b = 0
n = 0  # validation accuracy doesn't improve

if cfg['lTrain']:
    for epoch in range(1, cfg['epoch']+1):
        f1_cur, model_state, loss = train(model, epoch)

        # visualization
        vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([f1_cur]), win='F1 Score', update='append' if epoch > 1 else None,
             opts={'title': 'F1 Score'})
        vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([loss]), win='Loss', update='append' if epoch > 1 else None,
             opts={'title': 'Loss'})

        if f1_cur > f1_b:
            torch.save(model_state, cfg['BestModelPath'])
            print("\nBest Model has been updated.")
            f1_b = f1_cur
        else:
            n += 1

        if n == 10:
            print("reduce learn rate to 10%")
            scheduler.step()

else:
    test(model)

# if __name__ == '__main__':
#     main()
