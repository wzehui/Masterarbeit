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
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #
        correct = 0
        # 将求解器中的参数置零
        optimizer.zero_grad()
        #
        # accuracy_b = 0
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

        # better method
        pred = output.max(2)[1]  # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()

        accuracy_cur = correct / output.size()[1]   # correct / batchsize

        # 计算损失  The negative log likelihood loss：负对数似然损失 nll_loss
        loss = F.nll_loss(output[0], target)  # the loss functions expects a batchSizex2 input
        # 损失反传
        loss.backward()
        # 优化求解
        optimizer.step()

        if batch_idx % log_interval == 0:  # print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))
            print("Accuracy: {:.3f}".format(accuracy_cur))

    torch.save(model.state_dict(), '/Users/wzehui/Documents/MA/Model/bestModel')

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

if cfg['FeatureExtraction']:
    train_set = PreprocessFeature(cfg['IndexPath'], cfg['FeaturePath'], cfg['TrainSplitRate'], cfg['FeatureSelection'])
    test_set = PreprocessFeature(cfg['IndexPath'], cfg['FeaturePath'], cfg['TestSplitRate'], cfg['FeatureSelection'])

else:
    train_set = PreprocessData(cfg['IndexPath'], cfg['FilePath'], cfg['TrainSplitRate'])
    test_set = PreprocessData(cfg['IndexPath'], cfg['FilePath'], cfg['TestSplitRate'])

print("Train set size: " + str(len(train_set)))
# a = train_set[0]
print("Test set size: " + str(len(test_set)))

kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg['BatchSize'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg['BatchSize'], shuffle=True, **kwargs)

# load network structure
model = Net()
model.to(device)
# print(model)

# We will use the same optimization technique used in the paper, an Adam
# optimizer with weight decay set to 0.0001. At first, we will train with
# a learning rate of 0.01, but we will use a ``scheduler`` to decrease it
# to 0.001 during training.
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

log_interval = 1
for epoch in range(1, 11):
    if epoch == 31:
        print("First round of training complete. Setting learn rate to 0.001.")
    scheduler.step()
    try:
        model = torch.load(cfg['BestModelPath'])
    except(FileNotFoundError):
        model = Net()

    model.to(device)
    train(model, epoch)
    # test(model, epoch)
