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
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import random_split

from utils.DataPreprocessing import PreprocessData
from utils.FeaturePreprocessing import PreprocessFeature
from networks.CNN import Net


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
        correct = 0
        # 将求解器中的参数置零
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)

        # set requires_grad to True for training
        data = data.requires_grad_()
        # 前向传播
        output = model(data)
        output = output.permute(1, 0, 2)  # original output dimensions are batchSizex1x2

        # choose a larger log-probability (log-softmax)
        # predict = torch.ge(output[0][0:, 1], output[0][0:, 0], out=None)
        # correct = (predict.long() == target).float().sum()

        # 计算损失  The negative log likelihood loss：负对数似然损失 nll_loss
        loss = F.nll_loss(output[0], target)  # the loss functions expects a batchSizex2 input
        # 损失反传
        loss.backward()
        # 优化求解
        optimizer.step()

        if batch_idx % log_interval == 0:  # print training stats
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss))

    # validation phase
    model.eval()
    correct = 0
    for data, target in val_loader:
        data = data.requires_grad_()
        output = model(data)
        output = output.permute(1, 0, 2)
        pred = output.max(2)[1]  # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()

    accuracy = correct / len(test_loader.dataset)
    print('\nValidation set Accuracy: {:.3f}'.format(accuracy))

    return accuracy, model.state_dict()

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
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        output = output.permute(1, 0, 2)
        pred = output.max(2)[1]  # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()
    print('\nTest set Accuracy: {:.0f}%'.format(100. * correct / len(test_loader.dataset)))


# detect device type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDevice: " + device.type)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

# load parameter file
cfg = fParseConfig('param.yml')

if cfg['FeatureExtraction']:
    train_set = PreprocessFeature(cfg['IndexPath'], cfg['FeaturePath'], cfg['TrainSplitRate'], cfg['FeatureSelection'])
    test_set = PreprocessFeature(cfg['IndexPath'], cfg['FeaturePath'], cfg['TestSplitRate'], cfg['FeatureSelection'])

else:
    train_set = PreprocessData(cfg['IndexPath'], cfg['FilePath'], cfg['TrainSplitRate'])
    val_set = PreprocessData(cfg['IndexPath'], cfg['FilePath'], cfg['ValSplitRate'])
    test_set = PreprocessData(cfg['IndexPath'], cfg['FilePath'], cfg['TestSplitRate'])

print("\nTrain set size: " + str(len(train_set)))
print("\nValidation set size: " + str(len(val_set)))
print("\nTest set size: " + str(len(test_set)))

# a = train_set[0]

kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg['BatchSize'], shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg['BatchSize'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg['BatchSize'], shuffle=True, **kwargs)

# load network structure
model = Net()
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
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
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
accuracy_b = 0
n = 0  # validation accuracy doesn't improve

if cfg['lTrain']:
    for epoch in range(1, 11):
        accuracy_cur, model_state = train(model, epoch)
        if accuracy_cur > accuracy_b:
            torch.save(model_state, cfg['BestModelPath'])
            print("\nBest Model has been updated.")
            accuracy_b = accuracy_cur
        else:
            n += 1

        if n == 10:
            print("reduce learn rate to 10%")
            scheduler.step()

else:
    test(model)
