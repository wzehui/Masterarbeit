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

from utils.DataPreprocessing import PreprocessData
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
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将求解器中的参数置零
        optimizer.zero_grad()
        # if torch.cuda.is_available():
        data = data.to(device)
        target = target.to(device)
        data = data.requires_grad_()  # set requires_grad to True for training
        # 前向传播
        output = model(data)
        output = output.permute(1, 0, 2)  # original output dimensions are batchSizex1x10
        # 计算损失  The negative log likelihood loss：负对数似然损失 nll_loss
        loss = F.nll_loss(output[0], target)  # the loss functions expects a batchSizex10 input
        # 损失反传
        loss.backward()
        # 优化求解
        optimizer.step()
        if batch_idx % log_interval == 0:  # print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))


######################################################################
# Now that we have a training function, we need to make one for testing
# the networks accuracy. We will set the model to ``eval()`` mode and then
# run inference on the test dataset. Calling ``eval()`` sets the training
# variable in all modules in the network to false. Certain layers like
# batch normalization and dropout layers behave differently during
# training so this step is crucial for getting correct results.
#

def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        output = output.permute(1, 0, 2)
        pred = output.max(2)[1]  # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# detect device type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + device.type)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))


# load parameter file
cfg = fParseConfig('param.yml')

train_set = PreprocessData(cfg['CsvPath'], cfg['FilePath'], cfg['TrainRate'])
test_set = PreprocessData(cfg['CsvPath'], cfg['FilePath'], cfg['TestRate'])
print("Train set size: " + str(len(train_set)))
# a = train_set[2]
print("Test set size: " + str(len(test_set)))

kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg['BatchSize'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg['BatchSize'], shuffle=True, **kwargs)

#load network structure
model = Net()
model.to(device)
print(model)

# We will use the same optimization technique used in the paper, an Adam
# optimizer with weight decay set to 0.0001. At first, we will train with
# a learning rate of 0.01, but we will use a ``scheduler`` to decrease it
# to 0.001 during training.
optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)


log_interval = 20
for epoch in range(1, 11):
    if epoch == 31:
        print("First round of training complete. Setting learn rate to 0.001.")
    scheduler.step()
    train(model, epoch)
    test(model, epoch)