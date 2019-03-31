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

from utils.DataPreprocessing import PreprocessData


def fParseConfig(sFile):
    # get config file
    with open(sFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg


# detect device type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

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


